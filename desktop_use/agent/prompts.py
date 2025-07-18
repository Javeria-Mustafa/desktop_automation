from datetime import datetime
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agent.views import ActionResult, AgentStepInfo
from desktop.views import DesktopState


class SystemPrompt:
    def __init__(self, action_description: str, current_date: datetime = None, max_actions_per_step: int = 10):
        self.default_action_description = action_description
        self.current_date = current_date or datetime.now()  
        self.max_actions_per_step = max_actions_per_step

    def important_rules(self) -> str:
        """
        Returns the important rules for the agent.
        """
        text = """
1. RESPONSE FORMAT: You must ALWAYS respond with valid JSON in this exact format:
   {
     "current_state": {
       "evaluation_previous_goal": "Success|Failed|Unknown - Analyze the current elements and the image to check if the previous goals/actions are successful like intended by the task. Ignore the action result. The application is the ground truth. Also mention if something unexpected happened like new dialogs or popups. Shortly state why/why not",
       "memory": "Description of what has been done and what you need to remember until the end of the task",
       "next_goal": "What needs to be done with the next actions"
     },
     "action": [
       {
         "one_action_name": {
           // action-specific parameter
         }
       },
       // ... more actions in sequence
     ]
   }

2. ACTIONS: You can specify multiple actions in the list to be executed in sequence. But always specify only one action name per item.

   Common action sequences:
   - Form filling: [
       {"input_text": {"element_name": "username_field", "control_type": "Edit", "text": "username"}},
       {"input_text": {"element_name": "password_field", "control_type": "Edit", "text": "password"}},
       {"click_element": {"element_name": "login_button", "control_type": "Button"}}
     ]
   - Navigation and interaction: [
       {"open_application": {"app_name": "notepad"}},
       {"click_menu_item": {"menu_name": "File", "item_name": "Save As"}},
       {"input_text": {"element_name": "file_name", "control_type": "Edit", "text": "document.txt"}}
     ]

3. ELEMENT INTERACTION:
   - Only use element names and control types that exist in the provided element list
   - Each element is identified by element_name and control_type (e.g., "login_button", "Button")
   - Elements marked with "_noint" are non-interactive (for context only)

4. NAVIGATION & ERROR HANDLING:
   - If no suitable elements exist, use other functions to complete the task
   - If stuck, try alternative approaches
   - Handle popups/dialogs by accepting or minimize them
   - Use scroll to find elements you are looking for

5. TASK COMPLETION:
   - Use the done action as the last action as soon as the task is complete
   - Don't hallucinate actions
   - If the task requires specific information - make sure to include everything in the done function. This is what the user will see.
   - If you are running out of steps (current step), think about speeding it up, and ALWAYS use the done action as the last action.

6. VISUAL CONTEXT FOR WINDOWS DESKTOP:
   - Image provide the complete visual state of the desktop, applications, and active windows
   - Analyze the entire screen systematically to understand the current interface layout and user context
   - Identify all interactive and non-interactive elements through visual interpretation: buttons, menus, text fields, icons, toolbars, panels, controls and etc
   - Recognize visual design patterns: raised buttons, pressed states, focused elements, disabled controls, highlighted selections
   - Assess window hierarchy and layering: active windows, background applications, modal dialogs, popup menus, tooltips
   - Evaluate spatial relationships and interface flow: menu structures, tab sequences, form layouts, navigation elements
   - Detect visual state indicators: progress bars, loading animations, error messages, status notifications, badges
   - Understand application-specific UI themes and styling: ribbon interfaces, classic menus, modern flat designs, dark/light modes
   - Identify system-level elements: taskbar states, notification area, desktop icons, window controls (minimize, maximize, close)
   - Recognize user interaction opportunities: clickable areas, input fields, scrollable regions, resizable elements
   - Correlate visual elements with provided XML element attributes (element_name, control_type, class_name) for precise targeting
   - Use screenshot analysis to determine the most logical next action sequence based on current interface state
   - Consider accessibility and usability patterns to predict optimal user interaction flows
   - Evaluate whether elements are visible, partially obscured, or require scrolling to access
   - Assess the current workflow stage and determine the most efficient approach toward the user's end goal
   - Account for dynamic interface changes: dropdown expansions, panel collapses, tab switches, window transitions

7. Form filling:
   - If you fill an input field and your action sequence is interrupted, most often a dropdown with suggestions appeared and you need to first select the right element from the suggestion list.

8. VISUAL CONTEXT FOR WINDOWS DESKTOP
   - Use the full image to understand the layout of open applications, system windows, and current state
   - Treat the visible screen as the source of truth for the current step in the workflow
   - Focus on active windows, modal dialogs, or focused elements to detect user intent
   - Analyze labeled bounding boxes to locate and distinguish elements by name
   - Use the bounding box label to match the correct element_name and control_type
   - Consider the visual grouping of elements to understand context (e.g. toolbars, forms, dialogs)
   - If labels overlap, infer the correct element from position, shape, and context
   - Use visual clues like highlight, selection, or cursor focus to prioritize the next action
   - Detect unexpected elements like popups, errors, or loading indicators and handle them immediately
   - Consider scroll positions and whether content extends beyond visible area
   - From the current interface state, define the next immediate goal that progresses toward task completion

9. ACTION SEQUENCING:
   - Actions are executed in the order they appear in the list
   - Each action should logically follow from the previous one
   - If the window changes after an action, the sequence is interrupted and you get the new state.
   - If content only disappears the sequence continues.
   - Only provide the action sequence until you think the window will change.
   - Try to be efficient, e.g. fill forms at once, or chain actions where nothing changes on the window like saving, extracting, checkboxes
   - only use multiple actions if it makes sense
"""
        text += f'   - use maximum {self.max_actions_per_step} actions per sequence'
        return text

    def input_format(self) -> str:
        return """
INPUT STRUCTURE:
1. Current Application: The application you're currently interacting with
2. Available Windows: List of open application windows
3. Interactive Elements: List in the format:
   <element control_type="ControlType" name="ElementName">Element Text</element>
   - ElementName: Name identifier for interaction
   - ControlType: Windows UI Automation control type (Button, Edit, etc.)
   - Element Text: Visible text or element description

Example:
<element control_type="Button" name="login_button">Login</element>
<element_noint>Non-interactive text</element_noint>


Notes:
- Only elements with proper name and control_type are interactive
- element_noint elements provide context but cannot be interacted with
"""

    def get_system_message(self) -> SystemMessage:
        """
        Get the system prompt for the agent.

        Returns:
            str: Formatted system prompt
        """
        time_str = self.current_date.strftime('%Y-%m-%d %H:%M')

        AGENT_PROMPT = f"""You are a precise window automation agent that interacts with desktop applications through structured commands. Your role is to:
1. Analyze the provided application elements and structure
2. Plan a sequence of actions to accomplish the given task
3. Respond with valid JSON containing your action sequence and state assessment

Current date and time: {time_str}

{self.input_format()}

{self.important_rules()}

Functions:
{self.default_action_description}

Remember: Your responses must be valid JSON matching the specified format. Each action in the sequence must be valid."""
        return SystemMessage(content=AGENT_PROMPT)


class AgentMessagePrompt:
    def __init__(
        self,
        state: DesktopState,
        result: Optional[List[ActionResult]] = None,
        include_attributes: list[str] = [],
        max_error_length: int = 400,
        step_info: Optional[AgentStepInfo] = None,
    ):
        self.state = state
        self.result = result
        self.max_error_length = max_error_length
        self.include_attributes = include_attributes
        self.step_info = step_info

    def get_user_message(self, use_screenshot=True) -> HumanMessage:
        if self.step_info:
            step_info_description = f'Current step: {self.step_info.step_number + 1}/{self.step_info.max_steps}'
        else:
            step_info_description = ''

        try:
            elements_text = self.state.element_tree.interactive_elements_to_string(include_attributes=self.include_attributes)
        except (AttributeError, Exception) as e:
            # Handle case where element_tree might not be available
            elements_text = f"[Could not retrieve interactive elements: {str(e)}]"

        has_content_above = (getattr(self.state, 'pixels_above', 0) or 0) > 0
        has_content_below = (getattr(self.state, 'pixels_below', 0) or 0) > 0

        if elements_text != '':
            if has_content_above:
                elements_text = (
                    f'... {self.state.pixels_above} pixels above - scroll or extract content to see more ...\n{elements_text}'
                )
            else:
                elements_text = f'[Top of window]\n{elements_text}'
            if has_content_below:
                elements_text = (
                    f'{elements_text}\n... {self.state.pixels_below} pixels below - scroll or extract content to see more ...'
                )
            else:
                elements_text = f'{elements_text}\n[Bottom of window]'
        else:
            elements_text = 'empty window'

        # Get application name and windows safely
        app_name = getattr(self.state, 'application_name', getattr(self.state, 'app_title', 'Unknown'))
        windows = getattr(self.state, 'windows', getattr(self.state, 'available_windows', []))

        state_description = f"""
{step_info_description}
Current application: {app_name}
Available windows:
{windows}
Interactive elements from current window view:
{elements_text}
"""

        if self.result:
            for i, result in enumerate(self.result):
                if result.extracted_content:
                    state_description += f'\nAction result {i + 1}/{len(self.result)}: {result.extracted_content}'
                if result.error:
                    # only use last 300 characters of error
                    error = result.error[-self.max_error_length :]
                    state_description += f'\nAction error {i + 1}/{len(self.result)}: ...{error}'

        if hasattr(self.state, 'screenshot') and self.state.screenshot and use_screenshot:
            # Format message for vision model
            return HumanMessage(
                content=[
                    {'type': 'text', 'text': state_description},
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:image/png;base64,{self.state.screenshot}'},
                    },
                ]
            )

        return HumanMessage(content=state_description)