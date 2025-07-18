import asyncio
import enum
import json
import logging
from typing import Dict, Generic, Optional, Type, TypeVar
import uiautomation as auto
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
import pyautogui
from pydantic import BaseModel
from agent.views import ActionModel,ActionResult
from desktop.context import DesktopContext
from controller.registry.service import Registry
from controller.views import (
    ClickElementAction,
    RightClickElementAction,
    DoubleClickElementAction,
    DoneAction,
    InputTextAction,
    PressEnterAction,
    SelectTextAction,
    CopyTextAction,
    PasteTextAction,
    ScrollAction,
    SendKeysAction,
    LaunchApplicationAction,
    NoParamsAction,
    FindElementByPropertiesAction,
    SliderAction
)
from utils import time_execution_async, time_execution_sync

logger = logging.getLogger(__name__)

Context = TypeVar('Context')


class Controller(Generic[Context]):
    def __init__(
        self,
        exclude_actions: list[str] = [],
        output_model: Optional[Type[BaseModel]] = None,
    ):
        self.exclude_actions = exclude_actions
        self.output_model = output_model
        self.registry = Registry[Context](exclude_actions)
        self._register_default_actions()

    def _register_default_actions(self):
        """Register all default desktop actions"""
        self._register_done_action()
        self._register_basic_app_actions()
        self._register_element_interaction_actions()
        self._register_navigation_actions()

    def _register_done_action(self):
        """Register the done action with appropriate model"""
        if self.output_model is not None:
            class ExtendedOutputModel(BaseModel):
                success: bool = True
                data: self.output_model #type: ignore

            @self.registry.action(
                'Complete task - with return text and if the task is finished (success=True) or not yet completely finished (success=False), because last step is reached',
                param_model=ExtendedOutputModel,
            )
            async def done(params: ExtendedOutputModel):
                # Exclude success from the output JSON since it's an internal parameter
                output_dict = params.data.model_dump()

                # Enums are not serializable, convert to string
                for key, value in output_dict.items():
                    if isinstance(value, enum.Enum):
                        output_dict[key] = value.value

                return ActionResult(is_done=True, success=params.success, extracted_content=json.dumps(output_dict))
        else:
            @self.registry.action(
                'Complete task - with return text and if the task is finished (success=True) or not yet completely finished (success=False), because last step is reached',
                param_model=DoneAction,
            )
            async def done(params: DoneAction):
                return ActionResult(is_done=True, success=params.success, extracted_content=params.text)

    def _register_basic_app_actions(self):
        """Register basic application actions"""
        @self.registry.action(
            'Launch a desktop application by name',
            param_model=LaunchApplicationAction
        )
        async def launch_application(params: LaunchApplicationAction, dom_context: DesktopContext):
            try:
                msg = await dom_context.start_application(params.app_name)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                error_msg = f'❌ Error launching application "{params.app_name}": {str(e)}'
                logger.error(error_msg)
                return ActionResult(error=error_msg)

        @self.registry.action('Wait for x seconds default 3')
        async def wait(seconds: int = 3):
            msg = f'🕒 Waiting for {seconds} seconds'
            logger.info(msg)
            await asyncio.sleep(seconds)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action(
            'Take a screenshot and save it to a file',
            param_model=NoParamsAction,
        )
        async def take_screenshot(_, dom_context: DesktopContext):
            try:
                screenshot = await dom_context.take_screenshot()
                timestamp = asyncio.get_event_loop().time()
                filename = f"screenshot_{int(timestamp)}.png"
                filepath = f"screenshots/{filename}"
                
                # Ensure directory exists
                import os
                os.makedirs("screenshots", exist_ok=True)
                
                # Save screenshot
                import base64
                with open(filepath, "wb") as f:
                    f.write(base64.b64decode(screenshot))
                
                return ActionResult(extracted_content=f"📸 Screenshot saved to {filepath}", include_in_memory=True)
            except Exception as e:
                return ActionResult(error=f"Screenshot failed: {str(e)}")

        @self.registry.action(
            'Close the current application window',
            param_model=NoParamsAction,
        )
        async def close_application(_, dom_context: DesktopContext):
            try:
                window = await dom_context.get_current_window()
                app_name = window.Name
                window.Close()
                await asyncio.sleep(1)
                if window.Exists():
                    pyautogui.hotkey('alt', 'f4')
                return ActionResult(extracted_content=f"❌ Closed application: {app_name}", include_in_memory=True)
            except Exception as e:
                return ActionResult(error=f"Close failed: {str(e)}")

    def _register_element_interaction_actions(self):
        """Register element interaction actions"""
        
        # Enhanced click element with slider support
        @self.registry.action('Click element by properties with optional slider value', param_model=ClickElementAction)
        async def click_element(params: ClickElementAction, dom_context: DesktopContext):
            try:
                result = await dom_context.click_element(params)
                return result
            except Exception as e:
                logger.warning(f'Click failed: {str(e)}')
                return ActionResult(error=str(e))

        # Right-click element
        @self.registry.action('Right-click element by properties', param_model=RightClickElementAction)
        async def right_click_element(params: RightClickElementAction, dom_context: DesktopContext):
            try:
                element_name = getattr(params, 'element_name', None)
                control_type = getattr(params, 'control_type', None) 
                class_name = getattr(params, 'class_name', None)
                
                result = await dom_context.right_click_element(element_name, control_type, class_name)
                
                if result:
                    return ActionResult(extracted_content=f"🖱️ Right-clicked element: {element_name or 'unnamed'}", include_in_memory=True)
                return ActionResult(error=f"Failed to right-click element with properties")
            except Exception as e:
                logger.warning(f'Right-click failed: {str(e)}')
                return ActionResult(error=str(e))

        # Double-click element
        @self.registry.action('Double-click element by properties', param_model=DoubleClickElementAction)
        async def double_click_element(params: DoubleClickElementAction, dom_context: DesktopContext):
            try:
                element_name = getattr(params, 'element_name', None)
                control_type = getattr(params, 'control_type', None)
                class_name = getattr(params, 'class_name', None)
                
                result = await dom_context.double_click_element(element_name, control_type, class_name)
                
                if result:
                    return ActionResult(extracted_content=f"🖱️ Double-clicked element: {element_name or 'unnamed'}", include_in_memory=True)
                return ActionResult(error=f"Failed to double-click element with properties")
            except Exception as e:
                logger.warning(f'Double-click failed: {str(e)}')
                return ActionResult(error=str(e))

        # Type text using the new method
        @self.registry.action('Type text into an element by properties', param_model=InputTextAction)
        async def type_text(params: InputTextAction, dom_context: DesktopContext, has_sensitive_data: bool = False):
            try:
                element_name = getattr(params, 'element_name', None)
                control_type = getattr(params, 'control_type', None)
                class_name = getattr(params, 'class_name', None)
                text = params.text
                
                result = await dom_context.type_text(element_name, control_type, class_name, text)
                
                if result:
                    msg = f'⌨️ Typed text into element: {element_name or "unnamed"}'
                    if has_sensitive_data:
                        msg = f'⌨️ Typed sensitive data into element'
                    return ActionResult(extracted_content=msg, include_in_memory=True)
                return ActionResult(error=f"Failed to type text into element with properties")
            except Exception as e:
                logger.warning(f'Type text failed: {str(e)}')
                return ActionResult(error=str(e))

        # Press Enter
        @self.registry.action('Press Enter key on an element by properties', param_model=PressEnterAction)
        async def press_enter(params: PressEnterAction, dom_context: DesktopContext):
            try:
                element_name = getattr(params, 'element_name', None)
                control_type = getattr(params, 'control_type', None)
                class_name = getattr(params, 'class_name', None)
                
                result = await dom_context.press_enter(element_name, control_type, class_name)
                
                if result:
                    return ActionResult(extracted_content=f"⏎ Pressed Enter on element: {element_name or 'unnamed'}", include_in_memory=True)
                return ActionResult(error=f"Failed to press Enter on element with properties")
            except Exception as e:
                logger.warning(f'Press Enter failed: {str(e)}')
                return ActionResult(error=str(e))

        # Select text
        @self.registry.action('Select all text in an element by properties', param_model=SelectTextAction)
        async def select_text(params: SelectTextAction, dom_context: DesktopContext):
            try:
                element_name = getattr(params, 'element_name', None)
                control_type = getattr(params, 'control_type', None)
                class_name = getattr(params, 'class_name', None)
                
                result = await dom_context.select_text(element_name, control_type, class_name)
                
                if result:
                    return ActionResult(extracted_content=f"📝 Selected text in element: {element_name or 'unnamed'}", include_in_memory=True)
                return ActionResult(error=f"Failed to select text in element with properties")
            except Exception as e:
                logger.warning(f'Select text failed: {str(e)}')
                return ActionResult(error=str(e))

        # Copy text
        @self.registry.action('Copy text from an element by properties', param_model=CopyTextAction)
        async def copy_text(params: CopyTextAction, dom_context: DesktopContext):
            try:
                element_name = getattr(params, 'element_name', None)
                control_type = getattr(params, 'control_type', None)
                class_name = getattr(params, 'class_name', None)
                
                result = await dom_context.copy_text(element_name, control_type, class_name)
                
                if result:
                    return ActionResult(extracted_content=f"📋 Copied text from element: {element_name or 'unnamed'}", include_in_memory=True)
                return ActionResult(error=f"Failed to copy text from element with properties")
            except Exception as e:
                logger.warning(f'Copy text failed: {str(e)}')
                return ActionResult(error=str(e))

        # Paste text
        @self.registry.action('Paste text into an element by properties', param_model=PasteTextAction)
        async def paste_text(params: PasteTextAction, dom_context: DesktopContext):
            try:
                element_name = getattr(params, 'element_name', None)
                control_type = getattr(params, 'control_type', None)
                class_name = getattr(params, 'class_name', None)
                
                result = await dom_context.paste_text(element_name, control_type, class_name)
                
                if result:
                    return ActionResult(extracted_content=f"📄 Pasted text into element: {element_name or 'unnamed'}", include_in_memory=True)
                return ActionResult(error=f"Failed to paste text into element with properties")
            except Exception as e:
                logger.warning(f'Paste text failed: {str(e)}')
                return ActionResult(error=str(e))

        # Enhanced input text (maintains backward compatibility)
        @self.registry.action(
            'Input text into an interactive element',
            param_model=InputTextAction,
        )
        async def input_text(params: InputTextAction, dom_context: DesktopContext, has_sensitive_data: bool = False):
            try:
                element_name = params.element_name if hasattr(params, 'element_name') else ""
                control_type = params.control_type if hasattr(params, 'control_type') else ""
                class_name = params.class_name if hasattr(params, 'class_name') else ""
                text = params.text
                
                # Call the enhanced input_text_by_properties method
                result = await dom_context.input_text_by_properties(element_name, control_type, class_name, text)
                
                if not result:
                    return ActionResult(error=f"Failed to input text into element with name='{element_name}', type='{control_type}'")

                msg = f'⌨️ Input "{text}" into element with name="{element_name}", type="{control_type}"'
                if has_sensitive_data:
                    msg = f'⌨️ Input sensitive data into element with properties'

                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                return ActionResult(error=str(e))

        # Find element by properties
        @self.registry.action('Find an element by its properties and return its index', param_model=FindElementByPropertiesAction)
        async def find_element_by_properties(params: FindElementByPropertiesAction, dom_context: DesktopContext):
            try:
                class_name = getattr(params, 'class_name', None)
                element_name = getattr(params, 'element_name', None)
                control_type = getattr(params, 'control_type', None)
                
                index = await dom_context.find_element_by_properties(class_name, element_name, control_type)
                
                return ActionResult(extracted_content=f"🔍 Found element at index: {index}", include_in_memory=True)
            except Exception as e:
                logger.warning(f'Find element failed: {str(e)}')
                return ActionResult(error=str(e))

        # Extract content
        @self.registry.action(
            'Extract content from the current screen to retrieve specific information',
        )
        async def extract_content(goal: str, dom_context: DesktopContext, page_extraction_llm: BaseChatModel):
            try:
                state = await dom_context.get_state()

                if not state.xml_content:
                    raise Exception("No XML content available")

                prompt = PromptTemplate( 
                    input_variables=['goal', 'xml'],
                    template='Extract info from desktop XML UI. Goal: {goal}, XML: {xml}'
                )
                response = page_extraction_llm.invoke(prompt.format(goal=goal, xml=state.xml_content))
                return ActionResult(extracted_content=response.content, include_in_memory=True)
            except Exception as e:
                return ActionResult(error=f"Extraction failed: {str(e)}")

        # Send keys
        @self.registry.action(
            'Send keyboard keys like Escape, Backspace, Insert, PageDown, Delete, Enter, or shortcuts such as Control+o',
            param_model=SendKeysAction,
        )
        async def send_keys(params: SendKeysAction, dom_context: DesktopContext):
            try:
                keys = params.keys.lower().replace('control', 'ctrl').replace('alt', 'alt').replace('shift', 'shift').replace('win', 'win')

                if '+' in keys:
                    key_parts = [k.strip() for k in keys.split('+')]
                    pyautogui.hotkey(*key_parts)
                else:
                    special_keys = {'escape': 'esc', 'return': 'enter', 'delete': 'del'}
                    pyautogui.press(special_keys.get(keys, keys))

                msg = f'⌨️ Sent keys: {params.keys}'
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                return ActionResult(error=f'Error sending keys: {str(e)}')

    def _register_navigation_actions(self):
        """Register navigation actions"""
        @self.registry.action(
            'Scroll the current view up or down',
            param_model=ScrollAction,
        )
        async def scroll(params: ScrollAction):
            try:
                direction = params.direction or 'down'
                amount = params.amount if params.amount is not None else 3
                pyautogui.scroll(amount if direction == 'down' else -amount)
                return ActionResult(extracted_content=f"🔍 Scrolled {direction} by {amount}", include_in_memory=True)
            except Exception as e:
                return ActionResult(error=f"Scroll failed: {str(e)}")

    def action(self, description: str, **kwargs):
        """Decorator for registering custom actions

        @param description: Describe the LLM what the function does (better description == better function calling)
        """
        return self.registry.action(description, **kwargs)

    @time_execution_async('--multi-act')
    async def multi_act(
        self,
        actions: list[ActionModel],
        dom_context: DesktopContext,
        check_for_new_elements: bool = True,
        page_extraction_llm: Optional[BaseChatModel] = None,
        sensitive_data: Optional[Dict[str, str]] = None,
    ) -> list[ActionResult]:
        """Execute multiple actions"""
        print(f"\n🚀 DEBUG: Starting multi_act with {len(actions)} actions")
        print(f"   🔍 check_for_new_elements: {check_for_new_elements}")
        print(f"   🤖 page_extraction_llm: {page_extraction_llm is not None}")
        print(f"   🔐 sensitive_data: {sensitive_data is not None}")
        
        # Detailed actions list
        print(f"\n📋 ACTIONS LIST:")
        for idx, action in enumerate(actions):
            print(f"   [{idx}] {type(action).__name__}: {action}")
            # Try to get more details about the action
            if hasattr(action, '__dict__'):
                action_attrs = {k: v for k, v in action.__dict__.items() if not k.startswith('_')}
                for key, value in action_attrs.items():
                    print(f"       {key}: {value}")
            if hasattr(action, 'get_highlight_index'):
                print(f"       highlight_index: {action.get_highlight_index()}")
            print()
        
        results = []
        print(f"   📊 Initialized empty results list")

        # Get initial session state
        print(f"\n🔄 Getting DOM session...")
        session = await dom_context.get_session()
        print(f"   ✅ Session obtained: {type(session).__name__}")
        
        cached_selector_map = session.cached_state.selector_map
        cached_xml_hash = session.cached_state.xml_hash
        print(f"   📋 Cached selector_map keys: {list(cached_selector_map.keys()) if cached_selector_map else 'None'}")
        print(f"   🔗 Cached XML hash: {cached_xml_hash}")
        
        print(f"\n🎨 Removing highlights...")
        await dom_context.remove_highlights()
        print(f"   ✅ Highlights removed")

        # Main action loop
        print(f"\n🔄 Starting action loop...")
        for i, action in enumerate(actions):
            print(f"\n{'='*50}")
            print(f"🎯 PROCESSING ACTION {i + 1}/{len(actions)}")
            print(f"   Action: {action}")
            print(f"   Type: {type(action).__name__}")
            
            # Check highlight index
            highlight_index = action.get_highlight_index()
            print(f"   🎨 Highlight index: {highlight_index}")
            
            if highlight_index is not None and i != 0:
                print(f"   🔍 Action has highlight index and not first action - checking for new elements...")
                
                if check_for_new_elements:
                    print(f"   📊 Getting new DOM state...")
                    new_state = await dom_context.get_state()
                    new_xml_hash = new_state.xml_hash
                    print(f"   🔗 New XML hash: {new_xml_hash}")
                    print(f"   🔗 Cached XML hash: {cached_xml_hash}")
                    
                    if new_xml_hash != cached_xml_hash:
                        print(f"   ⚠️  DOM CHANGED! New elements detected after action {i}")
                        print(f"   🛑 Breaking out of action loop early")
                        logger.info(f'Something new appeared after action {i} / {len(actions)}')
                        break
                    else:
                        print(f"   ✅ DOM unchanged - continuing with action")
                else:
                    print(f"   ⏭️  Skipping new element check (disabled)")

            # Execute the action
            print(f"   🚀 Executing action...")
            try:
                action_result = await self.act(action, dom_context, page_extraction_llm, sensitive_data)
                results.append(action_result)
                print(f"   ✅ Action executed successfully")
                print(f"   📊 Result: {action_result}")
                print(f"   ✅ is_done: {action_result.is_done}")
                print(f"   ❌ error: {action_result.error}")
            except Exception as e:
                print(f"   💥 ACTION EXECUTION FAILED: {str(e)}")
                print(f"   📚 Exception type: {type(e).__name__}")
                raise

            logger.debug(f'Executed action {i + 1} / {len(actions)}')
            
            # Check if we should break early
            should_break = results[-1].is_done or results[-1].error or i == len(actions) - 1
            print(f"   🔍 Break conditions:")
            print(f"      is_done: {results[-1].is_done}")
            print(f"      has_error: {bool(results[-1].error)}")
            print(f"      is_last_action: {i == len(actions) - 1}")
            print(f"      should_break: {should_break}")
            
            if should_break:
                if results[-1].is_done:
                    print(f"   ✅ Action marked as done - breaking")
                elif results[-1].error:
                    print(f"   ❌ Action had error - breaking")
                elif i == len(actions) - 1:
                    print(f"   🏁 Last action completed - breaking")
                break

            # Wait between actions
            print(f"   ⏱️  Calculating wait time...")
            
            # Default wait time
            wait_time = 0.5
            print(f"      Default wait time: {wait_time}s")
            
            # Check config for custom wait time
            if hasattr(dom_context.config, 'wait_between_actions'):
                wait_time = dom_context.config.wait_between_actions
                print(f"      Found wait_between_actions attribute: {wait_time}s")
            elif isinstance(dom_context.config, dict) and 'wait_between_actions' in dom_context.config:
                wait_time = dom_context.config['wait_between_actions']
                print(f"      Found wait_between_actions in dict: {wait_time}s")
            else:
                print(f"      No custom wait time found, using default")
                
            print(f"   ⏳ Waiting {wait_time} seconds before next action...")
            await asyncio.sleep(wait_time)
            print(f"   ✅ Wait completed")

        print(f"\n{'='*50}")
        print(f"🏁 MULTI-ACT COMPLETED")
        print(f"   📊 Total actions processed: {len(results)}")
        print(f"   ✅ Successful actions: {sum(1 for r in results if not r.error)}")
        print(f"   ❌ Failed actions: {sum(1 for r in results if r.error)}")
        print(f"   🎯 Done actions: {sum(1 for r in results if r.is_done)}")
        print(f"   📋 Final results: {results}")
        
        return results

    @time_execution_sync('--act')
    async def act(
        self,
        action: ActionModel,
        dom_context: DesktopContext,
        page_extraction_llm: Optional[BaseChatModel] = None,
        sensitive_data: Optional[Dict[str, str]] = None,
    ) -> ActionResult:
        """Execute an action"""
        try:
            for action_name, params in action.model_dump(exclude_unset=True).items():
                if params is not None:
                    # remove highlights
                    result = await self.registry.execute_action(
                        action_name,
                        params,
                        dom_context=dom_context,
                        page_extraction_llm=page_extraction_llm,
                        sensitive_data=sensitive_data,
                    )
                    if isinstance(result, str):
                        return ActionResult(extracted_content=result)
                    elif isinstance(result, ActionResult):
                        return result
                    elif result is None:
                        return ActionResult()
                    else:
                        raise ValueError(f'Invalid action result type: {type(result)} of {result}')
            return ActionResult()
        except Exception as e:
            raise e