from typing import Callable, Dict, Type, List, Optional

import uiautomation as auto
from pydantic import BaseModel, ConfigDict


class RegisteredAction(BaseModel):
    """Model for a registered action"""

    name: str
    description: str
    function: Callable
    param_model: Type[BaseModel]

    # filters: provide specific windows or a function to determine whether the action should be available on the given window or not
    windows: list[str] | None = None  # e.g. ['WindowsForms*', 'Notepad', 'Chrome_WidgetWin*']
    window_filter: Callable[[auto.WindowControl], bool] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def prompt_description(self) -> str:
        """Get a description of the action for the prompt"""
        skip_keys = ['title']
        s = f'{self.description}: \n'
        s += '{' + str(self.name) + ': '
        s += str(
            {
                k: {sub_k: sub_v for sub_k, sub_v in v.items() if sub_k not in skip_keys}
                for k, v in self.param_model.model_json_schema()['properties'].items()
            }
        )
        s += '}'
        return s


class ActionModel(BaseModel):
    """Base model for dynamically created action models"""

    # this will have all the registered actions, e.g.
    # click_element = param_model = ClickElementParams
    # done = param_model = None
    #
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_index(self) -> int | None:
        """Get the index of the action - kept for compatibility"""
        return self.get_highlight_index()
    
    def get_highlight_index(self) -> int | None:
        """Get the highlight index of the action"""
        # {'clicked_element': {'highlight_index':5}}
        params = self.model_dump(exclude_unset=True).values()
        if not params:
            return None
        for param in params:
            if param is not None and 'highlight_index' in param:
                return param['highlight_index']
        return None

    def set_index(self, highlight_index: int) -> None:
        """Set the index of the action - kept for compatibility"""
        self.set_highlight_index(highlight_index)
        
    def set_highlight_index(self, highlight_index: int) -> None:
        """Overwrite the highlight index of the action"""
        # Get the action name and params
        action_data = self.model_dump(exclude_unset=True)
        action_name = next(iter(action_data.keys()))
        action_params = getattr(self, action_name)

        # Update the highlight_index directly on the model
        if hasattr(action_params, 'highlight_index'):
            action_params.highlight_index = highlight_index


class ActionRegistry(BaseModel):
    """Model representing the action registry"""

    actions: Dict[str, RegisteredAction] = {}

    @staticmethod
    def _match_windows(windows: list[str] | None, window_class: str) -> bool:
        """
        Match a list of window class patterns against a window class name.

        Args:
            windows: A list of window class patterns that can include glob patterns (* wildcard)
            window_class: The window class name to match against

        Returns:
            True if the window class matches the pattern, False otherwise
        """

        if windows is None or not window_class:
            return True

        import fnmatch

        for window_pattern in windows:
            if fnmatch.fnmatch(window_class, window_pattern):  # Perform glob *.matching.*
                return True
        return False

    @staticmethod
    def _match_window_filter(window_filter: Callable[[auto.WindowControl], bool] | None, window: auto.WindowControl) -> bool:
        """Match a window filter against a window"""
        if window_filter is None:
            return True
        return window_filter(window)

    def get_prompt_description(self, window: auto.WindowControl | None = None) -> str:
        """Get a description of all actions for the prompt

        Args:
            window: If provided, filter actions by window using window_filter and windows.

        Returns:
            A string description of available actions.
            - If window is None: return only actions with no window_filter and no windows (for system prompt)
            - If window is provided: return only filtered actions that match the current window (excluding unfiltered actions)
        """
        if window is None:
            # For system prompt (no window provided), include only actions with no filters
            return '\n'.join(
                action.prompt_description()
                for action in self.actions.values()
                if action.window_filter is None and action.windows is None
            )

        # only include filtered actions for the current window
        filtered_actions = []
        for action in self.actions.values():
            if not (action.windows or action.window_filter):
                # skip actions with no filters, they are already included in the system prompt
                continue

            window_class_matches = self._match_windows(action.windows, window.ClassName)
            filter_matches = self._match_window_filter(action.window_filter, window)

            if window_class_matches and filter_matches:
                filtered_actions.append(action)

        return '\n'.join(action.prompt_description() for action in filtered_actions)