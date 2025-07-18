from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from openai import RateLimitError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from desktop.views import DesktopStateHistory
from controller.registry.views import ActionModel
from dom.history_tree_processor.views import DOMHistoryElement
from dom.views import DOMElementNode, SelectorMap


@dataclass
class AgentStepInfo:
    step_number: int
    max_steps: int


class ActionResult(BaseModel):
    """Result of executing an action"""

    is_done: Optional[bool] = False
    extracted_content: Optional[str] = None
    error: Optional[str] = None
    include_in_memory: bool = False  # whether to include in past messages as context or not
    success: Optional[bool] = None  # Success flag for the done action


class AgentBrain(BaseModel):
    """Current state of the agent"""

    evaluation_previous_goal: str
    memory: str
    next_goal: str


class AgentOutput(BaseModel):
    """Output model for agent"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    current_state: AgentBrain
    action: list[ActionModel]

    @staticmethod
    def type_with_custom_actions(custom_actions: Type[ActionModel]) -> Type['AgentOutput']:
        """Extend actions with custom actions"""
        return create_model(
            'AgentOutput',
            __base__=AgentOutput,
            action=(list[custom_actions], Field(...)),  # Properly annotated field with no default
            __module__=AgentOutput.__module__,
        )


class AgentHistory(BaseModel):
    """History item for agent actions"""

    model_output: Optional[AgentOutput] = None
    result: list[ActionResult]
    state: DesktopStateHistory

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    @staticmethod
    def get_interacted_element(model_output: AgentOutput, selector_map: SelectorMap) -> list[DOMHistoryElement | None]:
        elements = []
        for action in model_output.action:
            index = action.get_index()
            if index and index in selector_map:
                el: DOMElementNode = selector_map[index]
                from dom.history_tree_processor.service import HistoryTreeProcessor
                elements.append(HistoryTreeProcessor.convert_xml_element_to_history_element(el))
            else:
                elements.append(None)
        return elements

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Custom serialization handling circular references"""

        # Handle action serialization
        model_output_dump = None
        if self.model_output:
            action_dump = [action.model_dump(exclude_none=True) for action in self.model_output.action]
            model_output_dump = {
                'current_state': self.model_output.current_state.model_dump(),
                'action': action_dump,  # This preserves the actual action data
            }

        return {
            'model_output': model_output_dump,
            'result': [r.model_dump(exclude_none=True) for r in self.result],
            'state': self.state.to_dict(),
        }


class AgentHistoryList(BaseModel):
    """List of agent history items"""

    history: list[AgentHistory]

    def save_to_file(self, filepath: str | Path) -> None:
        """Save history to JSON file with proper serialization"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            data = self.model_dump()
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            raise e

    def is_done(self) -> bool:
        """Check if the agent is done"""
        if self.history and len(self.history[-1].result) > 0 and self.history[-1].result[-1].is_done:
            return self.history[-1].result[-1].is_done
        return False

    def errors(self) -> list[str]:
        """Get all errors from history"""
        errors = []
        for h in self.history:
            errors.extend([r.error for r in h.result if r.error])
        return errors


class AgentError:
    """Container for agent error handling"""

    VALIDATION_ERROR = 'Invalid model output format. Please follow the correct schema.'
    RATE_LIMIT_ERROR = 'Rate limit reached. Waiting before retry.'
    NO_VALID_ACTION = 'No valid action found'

    @staticmethod
    def format_error(error: Exception, include_trace: bool = False) -> str:
        """Format error message based on error type and optionally include trace"""
        message = ''
        if isinstance(error, ValidationError):
            return f'{AgentError.VALIDATION_ERROR}\nDetails: {str(error)}'
        if isinstance(error, RateLimitError):
            return AgentError.RATE_LIMIT_ERROR
        if include_trace:
            return f'{str(error)}\nStacktrace:\n{traceback.format_exc()}'
        return f'{str(error)}'