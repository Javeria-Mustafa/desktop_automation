from __future__ import annotations
import datetime
from dataclasses import asdict
import json
import asyncio
import base64
import io
import json
import logging
import os
import platform
import re
import datetime
import textwrap
import uuid
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
)
from lmnr import observe
from openai import RateLimitError
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ValidationError

from agent.message_manager.service import MessageManager
from agent.prompts import AgentMessagePrompt, SystemPrompt
from agent.views import (
    ActionResult,
    AgentError,
    AgentHistory,
    AgentHistoryList,
    AgentOutput,
    AgentStepInfo,
)
from desktop.desktop import Desktop
from desktop.context import DesktopContext
from desktop.views import DesktopState, DesktopStateHistory
from controller.registry.views import ActionModel
from controller.service import Controller
from dom.history_tree_processor.service import HistoryTreeProcessor
from dom.history_tree_processor.views import DOMHistoryElement

from telemetry.service import ProductTelemetry
from telemetry.views import (
    AgentEndTelemetryEvent,
    AgentRunTelemetryEvent,
    AgentStepTelemetryEvent,
)
from utils import time_execution_async

load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
T = TypeVar('T', bound=BaseModel)


class Agent:
    def __init__(
        self,
        task: str,
        llm: BaseChatModel,
        desktop: Desktop | None = None,
        desktop_context: DesktopContext | None = None,
        controller: Controller = Controller(),
        use_screenshot: bool = True,
        save_conversation_path: Optional[str] = None,
        save_conversation_path_encoding: Optional[str] = 'utf-8',
        max_failures: int = 3,
        retry_delay: int = 10,
        system_prompt_class: Type[SystemPrompt] = SystemPrompt,
        max_input_tokens: int = 128000,
        validate_output: bool = False,
        message_context: Optional[str] = None,
        generate_gif: bool | str = True,
        sensitive_data: Optional[Dict[str, str]] = None,
        include_attributes: list[str] = [
            "ClassName", 
            "ControlType", 
            "Name", 
            "AutomationId"
        ],
        max_error_length: int = 400,
        max_actions_per_step: int = 10,
        tool_call_in_content: bool = True,
        initial_actions: Optional[List[Dict[str, Dict[str, Any]]]] = None,
        # Cloud Callbacks
        register_new_step_callback: Callable[['DesktopState', 'AgentOutput', int], None] | None = None,
        register_done_callback: Callable[['AgentHistoryList'], None] | None = None,
        tool_calling_method: Optional[str] = 'auto',
        page_extraction_llm: Optional[BaseChatModel] = None,
    ):
        self.agent_id = str(uuid.uuid4())  # unique identifier for the agent
        self.sensitive_data = sensitive_data
        if not page_extraction_llm:
            self.page_extraction_llm = llm
        else:
            self.page_extraction_llm = page_extraction_llm

        self.task = task
        self.use_screenshot = use_screenshot
        self.llm = llm
        self.save_conversation_path = save_conversation_path
        self.save_conversation_path_encoding = save_conversation_path_encoding
        self._last_result = None
        self.include_attributes = include_attributes
        self.max_error_length = max_error_length
        self.generate_gif = generate_gif

        # Controller setup
        self.controller = controller
        self.max_actions_per_step = max_actions_per_step

        # Desktop setup
        self.injected_desktop = desktop is not None
        self.injected_desktop_context = desktop_context is not None
        self.message_context = message_context

        # Initialize desktop first if needed
        self.desktop = desktop if desktop is not None else (None if desktop_context else Desktop())

        # Initialize desktop context
        if desktop_context:
            self.desktop_context = desktop_context
        elif self.desktop:
            self.desktop_context = DesktopContext(desktop=self.desktop, config=self.desktop.config.new_context_config)
        else:
            # If neither is provided, create both new
            self.desktop = Desktop()
            self.desktop_context = DesktopContext(desktop=self.desktop)

        self.system_prompt_class = system_prompt_class

        # Telemetry setup
        self.telemetry = ProductTelemetry()

        # Action and output models setup
        self._setup_action_models()
        self._set_version_and_source()
        self.max_input_tokens = max_input_tokens

        self._set_model_names()

        self.tool_calling_method = self.set_tool_calling_method(tool_calling_method)

        # Test write permissions to logs directory
        self._test_logs_directory()
        
        # Clear previous logs
        self._clear_previous_logs()

        self.message_manager = MessageManager(
            llm=self.llm,
            task=self.task,
            action_descriptions=self.controller.registry.get_prompt_description(),
            system_prompt_class=self.system_prompt_class,
            max_input_tokens=self.max_input_tokens,
            include_attributes=self.include_attributes,
            max_error_length=self.max_error_length,
            max_actions_per_step=self.max_actions_per_step,
            message_context=self.message_context,
            sensitive_data=self.sensitive_data,
        )

        # Step callback
        self.register_new_step_callback = register_new_step_callback
        self.register_done_callback = register_done_callback

        # Tracking variables
        self.history: AgentHistoryList = AgentHistoryList(history=[])
        self.n_steps = 1
        self.consecutive_failures = 0
        self.max_failures = max_failures
        self.retry_delay = retry_delay
        self.validate_output = validate_output
        self.initial_actions = self._convert_initial_actions(initial_actions) if initial_actions else None
        if save_conversation_path:
            logger.info(f'Saving conversation to {save_conversation_path}')

        self._paused = False
        self._stopped = False
    
    def _test_logs_directory(self) -> None:
        """Test write permissions for logs directory"""
        try:
            os.makedirs("logs", exist_ok=True)
            test_file = os.path.join("logs", "test_write.txt")
            with open(test_file, "w") as f:
                f.write("Test")
            os.remove(test_file)
            logger.info("Write permissions to logs directory confirmed")
        except Exception as e:
            logger.error(f"Cannot write to logs directory: {e}")
    
    def _clear_previous_logs(self) -> None:
        """Archive previous logs to avoid confusion between sessions"""
        try:
            log_dir = "logs"
            if os.path.exists(log_dir) and os.listdir(log_dir):
                # Create archive directory with timestamp
                archive_dir = f"logs_archive_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.makedirs(archive_dir, exist_ok=True)
                
                # Move all files from logs to archive
                for file_name in os.listdir(log_dir):
                    if file_name.endswith('.json'):
                        shutil.move(
                            os.path.join(log_dir, file_name),
                            os.path.join(archive_dir, file_name)
                        )
                logger.info(f"Previous logs archived to {archive_dir}")
        except Exception as e:
            logger.warning(f"Could not archive previous logs: {e}")

    def _set_version_and_source(self) -> None:
        try:
            import pkg_resources

            version = pkg_resources.get_distribution('desktop-use').version
            source = 'pip'
        except Exception:
            try:
                import subprocess

                version = subprocess.check_output(['git', 'describe', '--tags']).decode('utf-8').strip()
                source = 'git'
            except Exception:
                version = 'unknown'
                source = 'unknown'
        logger.debug(f'Version: {version}, Source: {source}')
        self.version = version
        self.source = source

    def _set_model_names(self) -> None:
        self.chat_model_library = self.llm.__class__.__name__
        if hasattr(self.llm, 'model_name'):
            self.model_name = self.llm.model_name  # type: ignore
        elif hasattr(self.llm, 'model'):
            self.model_name = self.llm.model  # type: ignore
        else:
            self.model_name = 'Unknown'

    def _setup_action_models(self) -> None:
        """Setup dynamic action models from controller's registry"""
        # Get the dynamic action model from controller's registry
        self.ActionModel = self.controller.registry.create_action_model()
        # Create output model with the dynamic actions
        self.AgentOutput = AgentOutput.type_with_custom_actions(self.ActionModel)

    def set_tool_calling_method(self, tool_calling_method: Optional[str]) -> Optional[str]:
        if tool_calling_method == 'auto':
            if self.chat_model_library == 'ChatGoogleGenerativeAI':
                return None
            elif self.chat_model_library == 'ChatOpenAI':
                return 'function_calling'
            elif self.chat_model_library == 'AzureChatOpenAI':
                return 'function_calling'
            else:
                return None
        else:
            return tool_calling_method

    def add_new_task(self, new_task: str) -> None:
        self.message_manager.add_new_task(new_task)
        # Reset step counter when adding a new task
        self.n_steps = 1
        # Clear logs for new task
        self._clear_previous_logs()

    def _save_input_messages_to_json(self, input_messages: list[BaseMessage]) -> None:
        """
        Save input messages to individual JSON files for each step.
        Creates separate files named input1.json, input2.json, etc.
        """
        logger.debug(f"Saving input messages to input{self.n_steps}.json")
        
        try:
            # Convert messages to a serializable format
            serializable_messages = []
            for msg in input_messages:
                # Add timestamp for when this message was saved
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                if isinstance(msg.content, list):
                    # Handle multi-modal content
                    content_list = []
                    for item in msg.content:
                        if isinstance(item, dict):
                            # For text items, just include the text
                            if item.get('type') == 'text':
                                content_list.append({"type": "text", "text": item.get('text', '')})
                            # For image items, note there was an image but don't include data
                            elif item.get('type') == 'image_url':
                                # Check if screenshot is actually present
                                has_data = bool(item.get('image_url', {}).get('url', ''))
                                content_list.append({
                                    "type": "image", 
                                    "has_data": has_data,
                                    "note": "Image data present but not included in log file"
                                })
                        else:
                            content_list.append(str(item))
                    serialized_content = content_list
                else:
                    # Handle string content
                    serialized_content = msg.content
                
                serializable_messages.append({
                    "type": msg.__class__.__name__,
                    "content": serialized_content,
                    "timestamp": timestamp,
                    "step": self.n_steps
                })
            
            # Define the output directory
            output_dir = "logs"
            
            # Create the directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Define unique filename for this step
            input_file = os.path.join(output_dir, f"input{self.n_steps}.json")
            
            # Convert to JSON with pretty printing
            json_input = json.dumps(serializable_messages, indent=2)
            
            # Write the input to its own file - overwrite any existing file
            with open(input_file, "w", encoding="utf-8") as f:
                f.write(json_input)
                
            logger.debug(f"Input messages saved to {input_file}")
                
        except Exception as e:
            logger.error(f"Failed to save input messages to JSON: {str(e)}")

    def _save_model_output_to_json(self, model_output: AgentOutput) -> None:
        """
        Save model output to individual JSON files for each step.
        Creates separate files named output1.json, output2.json, etc.
        """
        logger.debug(f"Saving model output to output{self.n_steps}.json")
        
        try:
            # Get model data
            model_dump = model_output.model_dump()
            
            # Add metadata
            output_with_metadata = {
                "model_output": model_dump,
                "metadata": {
                    "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "step": self.n_steps,
                    "task": self.task
                }
            }
            
            # Define the output directory
            output_dir = "logs"
            
            # Create the directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Define unique filename for this step
            output_file = os.path.join(output_dir, f"output{self.n_steps}.json")
            
            # Convert the output to JSON with pretty printing
            json_output = json.dumps(output_with_metadata, indent=2)
            
            # Write the output to its own file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(json_output)
                
            logger.debug(f"Model output saved to {output_file}")
                
        except Exception as e:
            logger.error(f"Failed to save model output to JSON: {str(e)}")
    
    def serialize_state(self, state):
        """
        Serialize the desktop state to a JSON file.
        
        Args:
            state: The desktop state to serialize
        """
        def safe_value(value):
            try:
                json.dumps(value)
                return value
            except:
                return str(value)
        
        state_dict = {
            key: safe_value(getattr(state, key))
            for key in dir(state)
            if not key.startswith('_') and not callable(getattr(state, key))
        }
        
        # Add indication of screenshot presence/size but don't include entire data
        if hasattr(state, 'screenshot') and state.screenshot:
            state_dict['screenshot_present'] = True
            state_dict['screenshot_size'] = len(state.screenshot)
        else:
            state_dict['screenshot_present'] = False
        
        # Add metadata
        state_dict['metadata'] = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'step': self.n_steps,
            'task': self.task
        }
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Save to step-specific state file
        filename = f"logs/state{self.n_steps}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, ensure_ascii=False, indent=4)
        
        logger.debug(f"Desktop state saved to {filename}")
        return filename  
    
    @time_execution_async('--step')
    async def step(self, step_info: Optional[AgentStepInfo] = None) -> None:
        """Execute one step of the task with enhanced monitoring"""
        logger.info(f'📍 Step {self.n_steps}')
        logger.info(f'🔍 Monitoring and validation enabled for this step')
        
        state = None
        model_output = None
        result: list[ActionResult] = []
        step_monitoring_data = {
            "step": self.n_steps,
            "timestamp_start": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "actions_planned": None,
            "actions_executed": None,
            "validation_result": None,
            "state_before": None,
            "state_after": None
        }

        try:
            # Get state before action
            state = await self.desktop_context.get_state()
            step_monitoring_data["state_before"] = {
                "window_title": state.window_title,
                "application_name": state.application_name,
                "app_title": state.app_title,
                "xml_hash": state.xml_hash,
                "xml_length": len(state.xml_content or "")
            }
            # REMOVED: state_file = self.serialize_state(state)  # This line is removed
            
            # Check if screenshot is present
            if hasattr(state, 'screenshot') and state.screenshot and len(state.screenshot) > 0:
                logger.info(f"Screenshot captured successfully: {len(state.screenshot)} chars")
            else:
                logger.warning("Screenshot is empty or missing in state")

            print("Window Title:", state.window_title)
            print("Application Name:", state.application_name) 
            print("App Title:", state.app_title)
            print("XML Hash:", state.xml_hash)
            print("XML Content Length:", len(state.xml_content or ""))

            if self._stopped or self._paused:
                logger.debug('Agent paused after getting state')
                raise InterruptedError

            # Add message with state information and verify screenshot inclusion
            logger.info(f"Adding state message with use_screenshot={self.use_screenshot}")
            self.message_manager.add_state_message(state, self._last_result, step_info, self.use_screenshot)
            input_messages = self.message_manager.get_messages()
            
            # Debug check for screenshot in messages
            screenshot_present = False
            for msg in input_messages:
                if isinstance(msg.content, list):
                    for item in msg.content:
                        if isinstance(item, dict) and item.get('type') == 'image_url':
                            screenshot_present = True
                            image_url = item.get('image_url', {}).get('url', '')
                            if image_url:
                                logger.info(f"Screenshot found in input messages: {len(image_url)} chars")
                            else:
                                logger.warning("Screenshot entry found but URL is empty")
                            break
            if not screenshot_present and self.use_screenshot:
                logger.warning("No screenshot found in input messages despite use_screenshot=True!")
            
            # Save input messages to a JSON file
            self._save_input_messages_to_json(input_messages)

            try:
                logger.debug(f"Getting next action from model for step {self.n_steps}")
                model_output = await self.get_next_action(input_messages)
                
                # Record planned actions in monitoring data
                if model_output:
                    step_monitoring_data["actions_planned"] = [
                        action.model_dump(exclude_unset=True) for action in model_output.action
                    ]
                
                # Save model output to a JSON file
                self._save_model_output_to_json(model_output)

                if self.register_new_step_callback:
                    self.register_new_step_callback(state, model_output, self.n_steps)

                self._save_conversation(input_messages, model_output)
                self.message_manager._remove_last_state_message()  # we dont want the whole state in the chat history

                if self._stopped or self._paused:
                    logger.debug('Agent paused after getting next action')
                    raise InterruptedError

                self.message_manager.add_model_output(model_output)
            except Exception as e:
                # model call failed, remove last state message from history
                logger.error(f"Error in model output processing: {str(e)}")
                self.message_manager._remove_last_state_message()
                step_monitoring_data["error"] = f"Model output error: {str(e)}"
                self._save_step_monitoring_data(step_monitoring_data)
                raise e

            # Execute actions
            result: list[ActionResult] = await self.controller.multi_act(
                model_output.action,
                self.desktop_context,
                page_extraction_llm=self.page_extraction_llm,
                sensitive_data=self.sensitive_data,
            )
            self._last_result = result
            
            # Record executed actions in monitoring data
            step_monitoring_data["actions_executed"] = [
                {
                    "extracted_content": r.extracted_content,
                    "error": r.error,
                    "is_done": r.is_done
                } for r in result
            ]

            # Get state after action to record changes
            try:
                after_state = await self.desktop_context.get_state()
                step_monitoring_data["state_after"] = {
                    "window_title": after_state.window_title,
                    "application_name": after_state.application_name,
                    "app_title": after_state.app_title,
                    "xml_hash": after_state.xml_hash,
                    "xml_length": len(after_state.xml_content or "")
                }
            except Exception as e:
                logger.warning(f"Could not capture state after action: {e}")
                step_monitoring_data["state_after"] = {"error": str(e)}

            if len(result) > 0 and result[-1].is_done:
                logger.info(f'📄 Result: {result[-1].extracted_content}')

            self.consecutive_failures = 0
            
            # Validate after every step if validate_output is True
            if self.validate_output:
                logger.info(f"🔍 Validating step {self.n_steps}")
                validation_result = await self._validate_output()
                step_monitoring_data["validation_result"] = validation_result
                logger.info(f"🔍 Step {self.n_steps} validation result: {validation_result}")
            else:
                logger.warning(f"⚠️ Validation skipped for step {self.n_steps} - validate_output is False")
                
            # Save comprehensive monitoring data for this step
            self._save_step_monitoring_data(step_monitoring_data)
            
            # Increment step counter
            self.n_steps += 1

        except InterruptedError:
            logger.debug('Agent paused')
            return
        except Exception as e:
            logger.error(f"Error in step execution: {str(e)}")
            step_monitoring_data["error"] = str(e)
            self._save_step_monitoring_data(step_monitoring_data)
            result = await self._handle_step_error(e)
            self._last_result = result
            
            # Still increment step counter even on error
            self.n_steps += 1

        finally:
            # Update end timestamp
            step_monitoring_data["timestamp_end"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self._save_step_monitoring_data(step_monitoring_data)
            
            # Capture telemetry for this step
            actions = [a.model_dump(exclude_unset=True) for a in model_output.action] if model_output else []
            self.telemetry.capture(
                AgentStepTelemetryEvent(
                    agent_id=self.agent_id,
                    step=self.n_steps-1,  # Use the step we just completed
                    actions=actions,
                    consecutive_failures=self.consecutive_failures,
                    step_error=[r.error for r in result if r.error] if result else ['No result'],
                )
            )
            
            if not result:
                return

            if state:
                self._make_history_item(model_output, state, result)
                
    def _save_step_monitoring_data(self, monitoring_data: dict) -> None:
        """Save comprehensive monitoring data for a step"""
        try:
            # Create logs directory if it doesn't exist
            os.makedirs("logs", exist_ok=True)
            
            # Define the monitoring file path
            monitoring_file = f"logs/monitoring{monitoring_data['step']}.json"
            
            # Write to file with pretty printing
            with open(monitoring_file, "w", encoding="utf-8") as f:
                json.dump(monitoring_data, f, ensure_ascii=False, indent=4)
            
            logger.info(f"Step monitoring data saved to {monitoring_file}")
        except Exception as e:
            logger.error(f"Failed to save step monitoring data to JSON: {str(e)}")
                
    async def _handle_step_error(self, error: Exception) -> list[ActionResult]:
        """Handle all types of errors that can occur during a step"""
        include_trace = logger.isEnabledFor(logging.DEBUG)
        error_msg = AgentError.format_error(error, include_trace=include_trace)
        prefix = f'❌ Result failed {self.consecutive_failures + 1}/{self.max_failures} times:\n '

        if isinstance(error, (ValidationError, ValueError)):
            logger.error(f'{prefix}{error_msg}')
            if 'Max token limit reached' in error_msg:
                # cut tokens from history
                self.message_manager.max_input_tokens = self.max_input_tokens - 500
                logger.info(f'Cutting tokens from history - new max input tokens: {self.message_manager.max_input_tokens}')
                self.message_manager.cut_messages()
            elif 'Could not parse response' in error_msg:
                # give model a hint how output should look like
                error_msg += '\n\nReturn a valid JSON object with the required fields.'

            self.consecutive_failures += 1
        elif isinstance(error, RateLimitError) or isinstance(error, ResourceExhausted):
            logger.warning(f'{prefix}{error_msg}')
            await asyncio.sleep(self.retry_delay)
            self.consecutive_failures += 1
        else:
            logger.error(f'{prefix}{error_msg}')
            self.consecutive_failures += 1

        return [ActionResult(error=error_msg, include_in_memory=True)]

    def _make_history_item(
        self,
        model_output: AgentOutput | None,
        state: DesktopState,
        result: list[ActionResult],
    ) -> None:
        """Create and store history item"""
        interacted_element = None
        len_result = len(result)

        if model_output:
            interacted_elements = AgentHistory.get_interacted_element(model_output, state.selector_map)
        else:
            interacted_elements = [None]

        state_history = DesktopStateHistory(
            window_title=state.window_title,
            title=state.title,
            open_windows=state.open_windows,
            interacted_element=interacted_elements,
            screenshot=state.screenshot,
        )

        history_item = AgentHistory(model_output=model_output, result=result, state=state_history)

        self.history.history.append(history_item)

    THINK_TAGS = re.compile(r'<think>.*?</think>', re.DOTALL)

    def _remove_think_tags(self, text: str) -> str:
        """Remove think tags from text"""
        return re.sub(self.THINK_TAGS, '', text)

    @time_execution_async('--get_next_action')
    async def get_next_action(self, input_messages: list[BaseMessage]) -> AgentOutput:
        """Get next action from LLM based on current state"""
        # Debug check for screenshot in messages right before sending to model
        screenshot_present = False
        for msg in input_messages:
            if isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, dict) and item.get('type') == 'image_url':
                        screenshot_present = True
                        logger.info("Screenshot confirmed present before sending to model")
                        break
        if not screenshot_present and self.use_screenshot:
            logger.warning("Warning: No screenshot found in messages before sending to model!")
            
        if self.model_name == 'deepseek-reasoner' or self.model_name.startswith('deepseek-r1'):
            converted_input_messages = self.message_manager.convert_messages_for_non_function_calling_models(input_messages)
            merged_input_messages = self.message_manager.merge_successive_human_messages(converted_input_messages)
            output = self.llm.invoke(merged_input_messages)
            output.content = self._remove_think_tags(output.content)
            # TODO: currently invoke does not return reasoning_content, we should override invoke
            try:
                parsed_json = self.message_manager.extract_json_from_model_output(output.content)
                parsed = self.AgentOutput(**parsed_json)
            except (ValueError, ValidationError) as e:
                logger.warning(f'Failed to parse model output: {output} {str(e)}')
                raise ValueError('Could not parse response.')
        elif self.tool_calling_method is None:
            structured_llm = self.llm.with_structured_output(self.AgentOutput, include_raw=True)
            response: dict[str, Any] = await structured_llm.ainvoke(input_messages)  # type: ignore
            parsed: AgentOutput | None = response['parsed']
        else:
            structured_llm = self.llm.with_structured_output(self.AgentOutput, include_raw=True, method=self.tool_calling_method)
            response: dict[str, Any] = await structured_llm.ainvoke(input_messages)  # type: ignore
            parsed: AgentOutput | None = response['parsed']

        if parsed is None:
            raise ValueError('Could not parse response.')

        # cut the number of actions to max_actions_per_step
        parsed.action = parsed.action[: self.max_actions_per_step]
        self._log_response(parsed)

        return parsed

    def _log_response(self, response: AgentOutput) -> None:
        """Log the model's response"""
        if 'Success' in response.current_state.evaluation_previous_goal:
            emoji = '👍'
        elif 'Failed' in response.current_state.evaluation_previous_goal:
            emoji = '⚠'
        else:
            emoji = '🤷'

        logger.info(f'{emoji} Eval: {response.current_state.evaluation_previous_goal}')
        logger.info(f'🧠 Memory: {response.current_state.memory}')
        logger.info(f'🎯 Next goal: {response.current_state.next_goal}')
        for i, action in enumerate(response.action):
            logger.info(f'🛠️  Action {i + 1}/{len(response.action)}: {action.model_dump_json(exclude_unset=True)}')

    def _save_conversation(self, input_messages: list[BaseMessage], response: Any) -> None:
        """Save conversation history to file if path is specified"""
        if not self.save_conversation_path:
            return

        # create folders if not exists
        os.makedirs(os.path.dirname(self.save_conversation_path), exist_ok=True)

        with open(
            self.save_conversation_path + f'_{self.n_steps}.txt',
            'w',
            encoding=self.save_conversation_path_encoding,
        ) as f:
            f.write(f"Task: {self.task}\n")
            f.write(f"Step: {self.n_steps}\n")
            f.write(f"Timestamp: {datetime.datetime.now()}\n\n")
            self._write_messages_to_file(f, input_messages)
            self._write_response_to_file(f, response)

    def _write_messages_to_file(self, f: Any, messages: list[BaseMessage]) -> None:
        """Write messages to conversation file"""
        for message in messages:
            f.write(f' {message.__class__.__name__} \n')

            if isinstance(message.content, list):
                for item in message.content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        f.write(item['text'].strip() + '\n')
                    elif isinstance(item, dict) and item.get('type') == 'image_url':
                        f.write('[SCREENSHOT INCLUDED]\n')
            elif isinstance(message.content, str):
                try:
                    content = json.loads(message.content)
                    f.write(json.dumps(content, indent=2) + '\n')
                except json.JSONDecodeError:
                    f.write(message.content.strip() + '\n')

            f.write('\n')

    def _write_response_to_file(self, f: Any, response: Any) -> None:
        """Write model response to conversation file"""
        f.write(' RESPONSE\n')
        f.write(json.dumps(json.loads(response.model_dump_json(exclude_unset=True)), indent=2))

    def _log_agent_run(self) -> None:
        """Log the agent run"""
        logger.info(f'🚀 Starting task: {self.task}')
        logger.info(f'Using screenshots: {self.use_screenshot}')
        logger.debug(f'Version: {self.version}, Source: {self.source}')
        self.telemetry.capture(
            AgentRunTelemetryEvent(
                agent_id=self.agent_id,
                use_vision=self.use_screenshot,
                task=self.task,
                model_name=self.model_name,
                chat_model_library=self.chat_model_library,
                version=self.version,
                source=self.source,
            )
        )

    @observe(name='agent.run')
    async def run(self, max_steps: int = 100) -> AgentHistoryList:
        """Execute the task with maximum number of steps"""
        try:
            self._log_agent_run()
            
            # Clear previous logs at the start of a new run
            self._clear_previous_logs()

            # Execute initial actions if provided
            if self.initial_actions:
                result = await self.controller.multi_act(
                    self.initial_actions,
                    self.desktop_context,
                    check_for_new_elements=False,
                    page_extraction_llm=self.page_extraction_llm,
                )
                self._last_result = result

            for step in range(max_steps):
                if self._too_many_failures():
                    break

                # Check control flags before each step
                if not await self._handle_control_flags():
                    break

                await self.step()

                if self.history.is_done():
                    if self.validate_output and step < max_steps - 1:
                        if not await self._validate_output():
                            continue
 
                    logger.info('✅ Task completed successfully')
                    if self.register_done_callback:
                        self.register_done_callback(self.history)
                    break
            else:
                logger.info('❌ Failed to complete task in maximum steps')

            return self.history
        finally:
            self.telemetry.capture(
                AgentEndTelemetryEvent(
                    agent_id=self.agent_id,
                    success=self.history.is_done(),
                    steps=self.n_steps,
                    max_steps_reached=self.n_steps >= max_steps,
                    errors=self.history.errors(),
                )
            )

            if not self.injected_desktop_context:
                await self.desktop_context.close()

            if not self.injected_desktop and self.desktop:
                await self.desktop.close()

            if self.generate_gif:
                output_path: str = 'agent_history.gif'
                if isinstance(self.generate_gif, str):
                    output_path = self.generate_gif

                self.create_history_gif(output_path=output_path)

    def _too_many_failures(self) -> bool:
        """Check if we should stop due to too many failures"""
        if self.consecutive_failures >= self.max_failures:
            logger.error(f'❌ Stopping due to {self.max_failures} consecutive failures')
            return True
        return False

    async def _handle_control_flags(self) -> bool:
        """Handle pause and stop flags. Returns True if execution should continue."""
        if self._stopped:
            logger.info('Agent stopped')
            return False

        while self._paused:
            await asyncio.sleep(0.2)  # Small delay to prevent CPU spinning
            if self._stopped:  # Allow stopping while paused
                return False
        return True

    async def _validate_output(self) -> bool:
        """Validate the output of the last action is what the user wanted and save to JSON"""
        # Log that we're starting validation
        logger.info(f"Starting validation for step {self.n_steps}")
        
        validation_data = {
            "step": self.n_steps,
            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "task": self.task,
            "is_valid": None,  # Will be updated later
            "reason": None,    # Will be updated later
            "validation_performed": False,
            "has_screenshot": False
        }

        system_msg = (
            f'You are a validator of an agent who interacts with desktop applications. '
            f'Validate if the output of last action is what the user wanted and if the task is completed. '
            f'If the task is unclear defined, you can let it pass. But if something is missing or the image does not show what was requested dont let it pass. '
            f'Try to understand the screen and help the model with suggestions like scroll, do x, ... to get the solution right. '
            f'Task to validate: {self.task}. Return a JSON object with 2 keys: is_valid and reason. '
            f'is_valid is a boolean that indicates if the output is correct. '
            f'reason is a string that explains why it is valid or not.'
            f' example: {{"is_valid": false, "reason": "The user wanted to enable Bluetooth, but the agent only opened the settings page and did not toggle the Bluetooth switch."}}'
        )
        
        validation_data["validation_prompt"] = system_msg
        
        # Save initial validation file before processing to ensure we have a log
        self._save_validation_to_json(validation_data)

        if not self.desktop_context.session:
            # If no desktop session, we can't validate the output
            validation_data["reason"] = "No desktop session available, validation skipped"
            self._save_validation_to_json(validation_data)
            logger.warning("Validation skipped: No desktop session available")
            return True

        try:
            state = await self.desktop_context.get_state()
            content = AgentMessagePrompt(
                state=state,
                result=self._last_result,
                include_attributes=self.include_attributes,
                max_error_length=self.max_error_length,
            )
            msg = [SystemMessage(content=system_msg), content.get_user_message(self.use_screenshot)]
            
            # Debug check for screenshot in validation message
            validation_has_screenshot = False
            if self.use_screenshot:
                for m in msg:
                    if isinstance(m.content, list):
                        for item in m.content:
                            if isinstance(item, dict) and item.get('type') == 'image_url':
                                validation_has_screenshot = True
                                logger.info("Screenshot included in validation message")
                                break
                if not validation_has_screenshot:
                    logger.warning("No screenshot in validation message despite use_screenshot=True!")
            
            validation_data["has_screenshot"] = validation_has_screenshot if self.use_screenshot else False
            
            # Update validation file with screenshot info
            self._save_validation_to_json(validation_data)

            class ValidationResult(BaseModel):
                is_valid: bool
                reason: str

            validator = self.llm.with_structured_output(ValidationResult, include_raw=True)
            response: dict[str, Any] = await validator.ainvoke(msg)  # type: ignore
            parsed: ValidationResult = response['parsed']
            is_valid = parsed.is_valid
            
            # Update validation data
            validation_data["is_valid"] = is_valid
            validation_data["reason"] = parsed.reason
            validation_data["validation_performed"] = True
            
            # Save final validation result
            self._save_validation_to_json(validation_data)
            
            if not is_valid:
                logger.info(f'❌ Validator decision: {parsed.reason}')
                msg = f'The output is not yet correct. {parsed.reason}.'
                self._last_result = [ActionResult(extracted_content=msg, include_in_memory=True)]
            else:
                logger.info(f'✅ Validator decision: {parsed.reason}')
            return is_valid
            
        except Exception as e:
            # Catch any errors during validation
            error_msg = f"Validation error: {str(e)}"
            logger.error(error_msg)
            validation_data["reason"] = error_msg
            validation_data["is_valid"] = False
            self._save_validation_to_json(validation_data)
            return False
        
    def _save_validation_to_json(self, validation_data: dict) -> None:
        """Save validation data to a JSON file"""
        try:
            # Create logs directory if it doesn't exist
            os.makedirs("logs", exist_ok=True)
            
            # Define the validation file path
            validation_file = f"logs/validation{validation_data['step']}.json"
            
            # Write to file with pretty printing
            with open(validation_file, "w", encoding="utf-8") as f:
                json.dump(validation_data, f, ensure_ascii=False, indent=4)
            
            logger.info(f"Validation data saved to {validation_file}")
        except Exception as e:
            logger.error(f"Failed to save validation data to JSON: {str(e)}")

    async def rerun_history(
        self,
        history: AgentHistoryList,
        max_retries: int = 3,
        skip_failures: bool = True,
        delay_between_actions: float = 2.0,
    ) -> list[ActionResult]:
        """
        Rerun a saved history of actions with error handling and retry logic.

        Args:
                history: The history to replay
                max_retries: Maximum number of retries per action
                skip_failures: Whether to skip failed actions or stop execution
                delay_between_actions: Delay between actions in seconds

        Returns:
                List of action results
        """
        # Execute initial actions if provided
        if self.initial_actions:
            await self.controller.multi_act(self.initial_actions, self.desktop_context, check_for_new_elements=False)

        results = []

        for i, history_item in enumerate(history.history):
            goal = history_item.model_output.current_state.next_goal if history_item.model_output else ''
            logger.info(f'Replaying step {i + 1}/{len(history.history)}: goal: {goal}')

            if (
                not history_item.model_output
                or not history_item.model_output.action
                or history_item.model_output.action == [None]
            ):
                logger.warning(f'Step {i + 1}: No action to replay, skipping')
                results.append(ActionResult(error='No action to replay'))
                continue

            retry_count = 0
            while retry_count < max_retries:
                try:
                    result = await self._execute_history_step(history_item, delay_between_actions)
                    results.extend(result)
                    break

                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        error_msg = f'Step {i + 1} failed after {max_retries} attempts: {str(e)}'
                        logger.error(error_msg)
                        if not skip_failures:
                            results.append(ActionResult(error=error_msg))
                            raise RuntimeError(error_msg)
                    else:
                        logger.warning(f'Step {i + 1} failed (attempt {retry_count}/{max_retries}), retrying...')
                        await asyncio.sleep(delay_between_actions)

        return results

    async def _execute_history_step(self, history_item: AgentHistory, delay: float) -> list[ActionResult]:
        """Execute a single step from history with element validation"""

        state = await self.desktop_context.get_state()
        if not state or not history_item.model_output:
            raise ValueError('Invalid state or model output')
        updated_actions = []
        for i, action in enumerate(history_item.model_output.action):
            updated_action = await self._update_action_indices(
                history_item.state.interacted_element[i],
                action,
                state,
            )
            updated_actions.append(updated_action)

            if updated_action is None:
                raise ValueError(f'Could not find matching element {i} in current screen')

        result = await self.controller.multi_act(
            updated_actions, self.desktop_context, page_extraction_llm=self.page_extraction_llm
        )

        await asyncio.sleep(delay)
        return result

    async def _update_action_indices(
        self,
        historical_element: Optional[DOMHistoryElement],
        action: ActionModel,
        current_state: DesktopState,
    ) -> Optional[ActionModel]:
        """
        Update action indices based on current screen state.
        Returns updated action or None if element cannot be found.
        """
        if not historical_element or not current_state.element_tree:
            return action

        current_element = HistoryTreeProcessor.find_history_element_in_tree(historical_element, current_state.element_tree)

        if not current_element or current_element.highlight_index is None:
            return None

        old_index = action.get_index()
        if old_index != current_element.highlight_index:
            action.set_index(current_element.highlight_index)
            logger.info(f'Element moved in XML, updated index from {old_index} to {current_element.highlight_index}')

        return action

    async def load_and_rerun(self, history_file: Optional[str | Path] = None, **kwargs) -> list[ActionResult]:
        """
        Load history from file and rerun it.

        Args:
                history_file: Path to the history file
                **kwargs: Additional arguments passed to rerun_history
        """
        if not history_file:
            history_file = 'AgentHistory.json'
        history = AgentHistoryList.load_from_file(history_file, self.AgentOutput)
        return await self.rerun_history(history, **kwargs)

    def save_history(self, file_path: Optional[str | Path] = None) -> None:
        """Save the history to a file"""
        if not file_path:
            file_path = 'AgentHistory.json'
        self.history.save_to_file(file_path)

    def create_history_gif(
        self,
        output_path: str = 'agent_history.gif',
        duration: int = 3000,
        show_goals: bool = True,
        show_task: bool = True,
        show_logo: bool = False,
        font_size: int = 40,
        title_font_size: int = 56,
        goal_font_size: int = 44,
        margin: int = 40,
        line_spacing: float = 1.5,
    ) -> None:
        """Create a GIF from the agent's history with overlaid task and goal text."""
        if not self.history.history:
            logger.warning('No history to create GIF from')
            return

        images = []
        # if history is empty or first screenshot is None, we can't create a gif
        if not self.history.history or not self.history.history[0].state.screenshot:
            logger.warning('No screenshot in history to create GIF from')
            if not self.history.history or not self.history.history[0].state.screenshot:
                logger.warning('No history or first screenshot to create GIF from')
                return

        # Try to load nicer fonts
        try:
            # Try different font options in order of preference
            font_options = ['Helvetica', 'Arial', 'DejaVuSans', 'Verdana']
            font_loaded = False

            for font_name in font_options:
                try:
                    if platform.system() == 'Windows':
                        # Need to specify the abs font path on Windows
                        font_name = os.path.join(os.getenv('WIN_FONT_DIR', 'C:\\Windows\\Fonts'), font_name + '.ttf')
                    regular_font = ImageFont.truetype(font_name, font_size)
                    title_font = ImageFont.truetype(font_name, title_font_size)
                    goal_font = ImageFont.truetype(font_name, goal_font_size)
                    font_loaded = True
                    break
                except OSError:
                    continue

            if not font_loaded:
                raise OSError('No preferred fonts found')

        except OSError:
            regular_font = ImageFont.load_default()
            title_font = ImageFont.load_default()
            goal_font = regular_font

        # Load logo if requested
        logo = None
        if show_logo:
            try:
                logo = Image.open('./static/desktop-use.png')
                # Resize logo to be small (e.g., 40px height)
                logo_height = 150
                aspect_ratio = logo.width / logo.height
                logo_width = int(logo_height * aspect_ratio)
                logo = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
            except Exception as e:
                logger.warning(f'Could not load logo: {e}')

        # Create task frame if requested
        if show_task and self.task:
            task_frame = self._create_task_frame(
                self.task,
                self.history.history[0].state.screenshot,
                title_font,
                regular_font,
                logo,
                line_spacing,
            )
            images.append(task_frame)

        # Process each history item
        for i, item in enumerate(self.history.history, 1):
            if not item.state.screenshot:
                continue

            # Convert base64 screenshot to PIL Image
            img_data = base64.b64decode(item.state.screenshot)
            image = Image.open(io.BytesIO(img_data))

            if show_goals and item.model_output:
                image = self._add_overlay_to_image(
                    image=image,
                    step_number=i,
                    goal_text=item.model_output.current_state.next_goal,
                    regular_font=regular_font,
                    title_font=title_font,
                    margin=margin,
                    logo=logo,
                )

            images.append(image)

        if images:
            # Save the GIF
            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=0,
                optimize=False,
            )
            logger.info(f'Created GIF at {output_path}')
        else:
            logger.warning('No images found in history to create GIF')

    def _create_task_frame(
        self,
        task: str,
        first_screenshot: str,
        title_font: ImageFont.FreeTypeFont,
        regular_font: ImageFont.FreeTypeFont,
        logo: Optional[Image.Image] = None,
        line_spacing: float = 1.5,
    ) -> Image.Image:
        """Create initial frame showing the task."""
        img_data = base64.b64decode(first_screenshot)
        template = Image.open(io.BytesIO(img_data))
        image = Image.new('RGB', template.size, (0, 0, 0))
        draw = ImageDraw.Draw(image)

        # Calculate vertical center of image
        center_y = image.height // 2

        # Draw task text with increased font size
        margin = 140  # Increased margin
        max_width = image.width - (2 * margin)
        larger_font = ImageFont.truetype(regular_font.path, regular_font.size + 16)  # Increase font size more
        wrapped_text = self._wrap_text(task, larger_font, max_width)

        # Calculate line height with spacing
        line_height = larger_font.size * line_spacing

        # Split text into lines and draw with custom spacing
        lines = wrapped_text.split('\n')
        total_height = line_height * len(lines)

        # Start position for first line
        text_y = center_y - (total_height / 2) + 50  # Shifted down slightly

        for line in lines:
            # Get line width for centering
            line_bbox = draw.textbbox((0, 0), line, font=larger_font)
            text_x = (image.width - (line_bbox[2] - line_bbox[0])) // 2

            draw.text(
                (text_x, text_y),
                line,
                font=larger_font,
                fill=(255, 255, 255),
            )
            text_y += line_height

        # Add logo if provided (top right corner)
        if logo:
            logo_margin = 20
            logo_x = image.width - logo.width - logo_margin
            image.paste(logo, (logo_x, logo_margin), logo if logo.mode == 'RGBA' else None)

        return image

    def _add_overlay_to_image(
        self,
        image: Image.Image,
        step_number: int,
        goal_text: str,
        regular_font: ImageFont.FreeTypeFont,
        title_font: ImageFont.FreeTypeFont,
        margin: int,
        logo: Optional[Image.Image] = None,
        display_step: bool = True,
        text_color: tuple[int, int, int, int] = (255, 255, 255, 255),
        text_box_color: tuple[int, int, int, int] = (0, 0, 0, 255),
    ) -> Image.Image:
        """Add step number and goal overlay to an image."""
        image = image.convert('RGBA')
        txt_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(txt_layer)
        if display_step:
            # Add step number (bottom left)
            step_text = str(step_number)
            step_bbox = draw.textbbox((0, 0), step_text, font=title_font)
            step_width = step_bbox[2] - step_bbox[0]
            step_height = step_bbox[3] - step_bbox[1]

            # Position step number in bottom left
            x_step = margin + 10  # Slight additional offset from edge
            y_step = image.height - margin - step_height - 10  # Slight offset from bottom

            # Draw rounded rectangle background for step number
            padding = 20  # Increased padding
            step_bg_bbox = (
                x_step - padding,
                y_step - padding,
                x_step + step_width + padding,
                y_step + step_height + padding,
            )
            draw.rounded_rectangle(
                step_bg_bbox,
                radius=15,  # Add rounded corners
                fill=text_box_color,
            )

            # Draw step number
            draw.text(
                (x_step, y_step),
                step_text,
                font=title_font,
                fill=text_color,
            )

        # Draw goal text (centered, bottom)
        max_width = image.width - (4 * margin)
        wrapped_goal = self._wrap_text(goal_text, title_font, max_width)
        goal_bbox = draw.multiline_textbbox((0, 0), wrapped_goal, font=title_font)
        goal_width = goal_bbox[2] - goal_bbox[0]
        goal_height = goal_bbox[3] - goal_bbox[1]

        # Center goal text horizontally, place above step number
        x_goal = (image.width - goal_width) // 2
        y_goal = y_step - goal_height - padding * 4  # More space between step and goal

        # Draw rounded rectangle background for goal
        padding_goal = 25  # Increased padding for goal
        goal_bg_bbox = (
            x_goal - padding_goal,  # Remove extra space for logo
            y_goal - padding_goal,
            x_goal + goal_width + padding_goal,
            y_goal + goal_height + padding_goal,
        )
        draw.rounded_rectangle(
            goal_bg_bbox,
            radius=15,  # Add rounded corners
            fill=text_box_color,
        )

        # Draw goal text
        draw.multiline_text(
            (x_goal, y_goal),
            wrapped_goal,
            font=title_font,
            fill=text_color,
            align='center',
        )

        # Add logo if provided (top right corner)
        if logo:
            logo_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
            logo_margin = 20
            logo_x = image.width - logo.width - logo_margin
            logo_layer.paste(logo, (logo_x, logo_margin), logo if logo.mode == 'RGBA' else None)
            txt_layer = Image.alpha_composite(logo_layer, txt_layer)

        # Composite and convert
        result = Image.alpha_composite(image, txt_layer)
        return result.convert('RGB')

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
        """
        Wrap text to fit within a given width.

        Args:
            text: Text to wrap
            font: Font to use for text
            max_width: Maximum width in pixels

        Returns:
            Wrapped text with newlines
        """
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            current_line.append(word)
            line = ' '.join(current_line)
            bbox = font.getbbox(line)
            if bbox[2] > max_width:
                if len(current_line) == 1:
                    lines.append(current_line.pop())
                else:
                    current_line.pop()
                    lines.append(' '.join(current_line))
                    current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))

        return '\n'.join(lines)

    def pause(self) -> None:
        """Pause the agent before the next step"""
        logger.info('🔄 pausing Agent ')
        self._paused = True

    def resume(self) -> None:
        """Resume the agent"""
        logger.info('▶️ Agent resuming')
        self._paused = False

    def stop(self) -> None:
        """Stop the agent"""
        logger.info('⏹️ Agent stopping')
        self._stopped = True

    def _convert_initial_actions(self, actions: List[Dict[str, Dict[str, Any]]]) -> List[ActionModel]:
        """Convert dictionary-based actions to ActionModel instances"""
        converted_actions = []
        action_model = self.ActionModel
        for action_dict in actions:
            # Each action_dict should have a single key-value pair
            action_name = next(iter(action_dict))
            params = action_dict[action_name]

            # Get the parameter model for this action from registry
            action_info = self.controller.registry.registry.actions[action_name]
            param_model = action_info.param_model

            # Create validated parameters using the appropriate param model
            validated_params = param_model(**params)

            # Create ActionModel instance with the validated parameters
            action_model = self.ActionModel(**{action_name: validated_params})
            converted_actions.append(action_model)

        return converted_actions