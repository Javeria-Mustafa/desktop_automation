import asyncio
import base64
import gc
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import pyautogui 
import uiautomation as auto
import psutil
import xml.etree.ElementTree as ET
import subprocess
import datetime

# Desktop specific views
from desktop.views import DesktopState, ElementMap
# DOM service and views
from dom.service import DomService 
from dom.views import DOMElementNode 

from controller.views import ClickElementAction
from agent.views import ActionResult
from utils import time_execution_async, time_execution_sync

if TYPE_CHECKING:
    from desktop.desktop import Desktop

logger = logging.getLogger(__name__)

@dataclass
class DesktopContextState:
    window_id: str | None = None

class DesktopContextConfig(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class DesktopSession:
    def __init__(self, cached_state: DesktopState | None = None):
        if cached_state is None:
            cached_state = DesktopState()
            cached_state.xml_hash = None
            cached_state.element_map = {}
        self.cached_state = cached_state

class DesktopContext:
    def __init__(self, desktop: 'Desktop', config: Dict[str, Any] = None, state: Optional[DesktopContextState] = None):
        self.context_id = str(uuid.uuid4())
        logger.debug(f'Initializing new desktop context with id: {self.context_id}')

        self.desktop = desktop
        
        # optimized default timings
        default_config = {
            "wait_for_idle_action_time": 0.1,      
            "minimum_wait_action_time": 0.05,       
            "maximum_wait_action_time": 2.0,       
            "screenshot_delay": 0.05,              
            "focus_delay": 0.05,                   
            "post_action_delay": 0.1,              
            "ui_check_delay": 0.05,               
            "app_launch_delay": 1.5,               
            "text_input_delay": 0.05,              
        }
        
        self.config = {**default_config, **(config or {})}
        self.state = state or DesktopContextState()
        self.session = None
        self.current_window: Optional[auto.Control] = None
        self.dom_service: Optional[DomService] = None

    async def __aenter__(self):
        await self._initialize_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @time_execution_async('--close')
    async def close(self):
        logger.debug('Closing desktop context')
        try:
            if self.session is None: return
            if self.config.get("settings_file"): await self.save_settings()
        finally:
            self.session = None
            self.current_window = None
            gc.collect()

    def __del__(self):
        try:
            if not self.config.get('_force_keep_context_alive', False) and self.session is not None:
                logger.debug('DesktopContext was not properly closed before destruction')
                self.session = None; self.current_window = None; gc.collect()
        except Exception as e: logger.warning(f'Failed to clean up desktop context: {e}')

    @time_execution_async('--initialize_session')
    async def _initialize_session(self):
        logger.debug('Initializing desktop context session')
        if self.config.get("app_path"):
            await self._launch_application(self.config.get("app_path"))

        self.session = DesktopSession(cached_state=None)
        self.dom_service = DomService()
        self._setup_window_monitors()
        return self.session

    async def _launch_application(self, app_path: str):
        logger.debug(f'Internal launch: Launching Settings application instead of: {app_path}')
        try:
            cmd = 'explorer.exe ms-settings:'
            logger.info(f"Running: {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            logger.info(f"Launch result: returncode={result.returncode}")
            if result.stderr: logger.warning(f"Launch stderr: {result.stderr}")
            await asyncio.sleep(self.config.get("app_launch_delay", 1.5))
            foreground = auto.GetForegroundControl()
            if foreground: logger.info(f"Active window after launch: {foreground.Name} ({foreground.ClassName})")
            else: logger.warning("Could not get foreground window after launch.")
        except Exception as e:
            logger.error(f'Failed to launch Settings app: {e}'); raise RuntimeError(f'Could not launch Settings app: {e}')

    def _setup_window_monitors(self): pass

    async def get_session(self) -> DesktopSession:
        if self.session is None: await self._initialize_session()
        assert self.session is not None
        return self.session

    async def get_current_window(self) -> Optional[auto.Control]:
        await self.get_session()
        return await self._get_current_window()

    async def _get_current_window(self) -> Optional[auto.Control]:
        try:
            if self.current_window and self.current_window.Exists(): return self.current_window
            if self.state.window_id:
                try:
                    class_name, window_name = self.state.window_id.split('|', 1)
                    win = auto.WindowControl(searchDepth=1, ClassName=class_name, Name=window_name)
                    if win.Exists(): win.SetFocus(); self.current_window = win; return win
                except Exception as e: logger.debug(f"Error finding window by ID '{self.state.window_id}': {e}")
            
            fg_win = auto.GetForegroundControl()
            if fg_win and fg_win.Exists(): self.current_window = fg_win; return fg_win
            
            children = auto.GetRootControl().GetChildren()
            for child in children:
                if child.Exists() and child.ClassName and child.Name:
                    child.SetFocus(); self.current_window = child; return child
            logger.warning("No valid windows found after all checks.")
            return None
        except Exception as e:
            logger.error(f"Error in _get_current_window: {e}"); return None

    @time_execution_sync('--get_state')
    async def get_state(self) -> DesktopState:
        await self._wait_for_idle()
        session = await self.get_session()
        session.cached_state = await self._update_state()
        if self.config.get("settings_file"): asyncio.create_task(self.save_settings())
        
        return session.cached_state

    async def _wait_for_idle(self, timeout_overwrite: Optional[float] = None):
        start_time = time.time()
        try:
            await self._wait_for_ui_idle()
            window = await self.get_current_window()
            if window: await self._check_and_handle_application(window)
            else: logger.warning("No current window to check application in _wait_for_idle.")
        except Exception as e: logger.warning(f'Action completion check failed: {e}')
        elapsed = time.time() - start_time
        min_wait = self.config.get("minimum_wait_action_time", 0.05)
        remaining = max((timeout_overwrite or min_wait) - elapsed, 0)
        logger.debug(f'--UI stabilized in {elapsed:.2f}s, waiting for additional {remaining:.2f}s')
        if remaining > 0: await asyncio.sleep(remaining)

    async def _wait_for_ui_idle(self):
        start_time = time.time()
        idle_wait = self.config.get("wait_for_idle_action_time", 0.1)
        max_wait = self.config.get("maximum_wait_action_time", 2.0)
        ui_check_delay = self.config.get("ui_check_delay", 0.05)
        
        while True:
            try:
                idle = True; wb = auto.GetForegroundControl(); 
                await asyncio.sleep(ui_check_delay)
                wa = auto.GetForegroundControl()
                if (wb and wa and (getattr(wb, 'AutomationId', None) != getattr(wa, 'AutomationId', None) or wb.Name != wa.Name or wb.ClassName != wa.ClassName)): idle = False
                if idle and (time.time() - start_time) >= idle_wait: break
                if (time.time() - start_time) > max_wait: logger.debug(f'UI idle wait timed out after {max_wait}s'); break
            except Exception as e: logger.debug(f'Error while waiting for UI idle: {e}'); break

    async def _update_state(self) -> DesktopState:
        try:
            current_win = await self.get_current_window()
            if not current_win:
                logger.error("Cannot update state: No current window found.")
                return DesktopState(app_title="Error: No Window", title="Error: No Window Found",
                                    element_map={}, xml_content="Error: Could not retrieve window content.",
                                    element_tree=None, available_windows=[], screenshot="")

            if not self.dom_service:
                logger.warning("DomService not initialized, initializing now.")
                self.dom_service = DomService()
            
            self.dom_service.window = current_win

            logger.info(f"Building element data for window: '{current_win.Name}' ({current_win.ClassName}) via DomService")
            
            element_map, xml_content = await self.dom_service._build_element_data(
                highlight_elements=self.config.get("highlight_elements", False),
                focus_element=self.config.get("focus_element", 0),
                viewport_expansion=self.config.get("viewport_expansion", 0)
            )
            logger.info(f"Element map built with {len(element_map)} elements by DomService.")

            constructed_element_tree: Optional[DOMElementNode] = None
            if xml_content and xml_content.strip():
                try:
                    xml_et_root = ET.fromstring(xml_content)
                    constructed_element_tree = self.dom_service._xml_to_dom_tree(xml_et_root)
                    logger.info(f"Constructed element_tree from xml_content via DomService: Type {type(constructed_element_tree)}")
                    if constructed_element_tree and not hasattr(constructed_element_tree, 'interactive_elements_to_string'):
                        logger.error("FATAL: Constructed element_tree LACKS interactive_elements_to_string method!")
                        constructed_element_tree = None
                except ET.ParseError as e_xml_parse:
                    logger.error(f"Failed to parse xml_content (len: {len(xml_content)}): {e_xml_parse}. XML: '{xml_content[:200]}...'")
                except Exception as e_tree:
                    logger.error(f"Failed to build element_tree from xml_content: {e_tree}", exc_info=True)
            else:
                logger.warning("xml_content is empty, cannot build element_tree.")
                
            if constructed_element_tree is None:
                logger.warning("Final element_tree is None.")

            window_title = "Unknown Window"
            application_name = "Unknown"
            
            if hasattr(current_win, 'Name') and current_win.Name:
                window_title = current_win.Name
            
            if hasattr(current_win, 'ProcessId') and current_win.ProcessId:
                try:
                    import psutil
                    process = psutil.Process(current_win.ProcessId)
                    application_name = process.name()
                    logger.info(f"Got application name from process: {application_name}")
                except Exception as e_proc:
                    logger.debug(f"Could not get process name: {e_proc}")
                    
            if application_name == "Unknown" and hasattr(current_win, 'ClassName') and current_win.ClassName:
                application_name = current_win.ClassName
                
            logger.info(f"Window info: Title='{window_title}', App='{application_name}'")

            available_windows_list = []
            try:
                desktop_root_ctrl = auto.GetRootControl()
                for win_child_ctrl in desktop_root_ctrl.GetChildren():
                    if hasattr(win_child_ctrl, 'Name') and win_child_ctrl.Name:
                        available_windows_list.append(f"{win_child_ctrl.Name}")
            except Exception as e_aw:
                logger.error(f"Could not get available windows: {e_aw}")

            state_to_return = DesktopState(
                window_title=window_title,
                application_name=application_name,
                app_title=current_win.ClassName if hasattr(current_win, 'ClassName') else "Unknown",
                title=current_win.Name if hasattr(current_win, 'Name') else "Unknown",
                element_map=element_map,
                xml_content=xml_content,
                element_tree=constructed_element_tree,
                xml_hash=None, 
                available_windows=available_windows_list, 
                screenshot=await self.take_screenshot()
            )
            
            if self.session:
                self.session.cached_state = state_to_return
            
            return state_to_return
        except Exception as e:
            logger.error(f"Critical error in _update_state: {str(e)}", exc_info=True)
            return DesktopState(
                window_title="Unknown (critical error in state update)",
                application_name="Unknown (critical error in state update)",
                app_title="Unknown (critical error in state update)",
                title="Unknown (critical error in state update)", 
                element_map={},
                xml_content="Error: XML content not retrieved due to critical error.",
                element_tree=None, 
                xml_hash="", 
                available_windows=[], 
                screenshot=""
            )

    async def _is_application_allowed(self, window: Optional[auto.Control]) -> bool:
        return True

    async def _check_and_handle_application(self, window: Optional[auto.Control]):
        if not window: return
        if not await self._is_application_allowed(window):
            app_identifier = window.ClassName if hasattr(window, 'ClassName') and window.ClassName else "Unknown App"
            if hasattr(window, 'ProcessId'):
                try:
                    app_identifier = psutil.Process(window.ProcessId).name()
                except: pass
            logger.warning(f'Non-allowed application: {window.Name} ({app_identifier})')
            try: await self._handle_disallowed_application()
            except Exception as e: logger.error(f'Failed to handle non-allowed app: {e}')
            raise RuntimeError(f'Switched from non-allowed app: {window.Name}')

    async def _handle_disallowed_application(self):
        logger.info("Attempting to switch from non-allowed app.")
        desktop = auto.GetRootControl()
        for win_ctrl in desktop.GetChildren():
            if win_ctrl.Exists() and await self._is_application_allowed(win_ctrl):
                logger.info(f"Switching to allowed window: {win_ctrl.Name}")
                win_ctrl.SetFocus(); self.current_window = win_ctrl; 
                await asyncio.sleep(self.config.get("window_switch_delay", 0.1))
                return
        logger.warning("No allowed window found. Launching fallback."); await self._launch_application("settings")

    @time_execution_async('--take_screenshot')
    async def take_screenshot(self) -> str:
        try:
            window = await self.get_current_window()
            if not window: logger.warning("Cannot take screenshot, no current window."); return ""
            window.SetFocus(); 
            await asyncio.sleep(self.config.get("screenshot_delay", 0.05))
            rect = window.BoundingRectangle
            if rect and rect.width() > 0 and rect.height() > 0 :
                screenshot = pyautogui.screenshot(region=(rect.left, rect.top, rect.width(), rect.height()))
            else:
                logger.warning(f"Invalid rect for '{window.Name}'. Full screenshot."); screenshot = pyautogui.screenshot()
            import io; buffer = io.BytesIO(); 
            screenshot.save(buffer, format='PNG', optimize=True, compress_level=1)  
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e: logger.error(f'Screenshot failed: {e}', exc_info=True); return ""

    @time_execution_async('--save_settings')
    async def save_settings(self):
        settings_file = self.config.get("settings_file")
        if self.session and settings_file:
            try:
                settings = {"window_id": self.state.window_id, "timestamp": time.time(), "context_id": self.context_id}
                logger.debug(f'Saving settings to {settings_file}')
                dirname = os.path.dirname(settings_file); 
                if dirname: os.makedirs(dirname, exist_ok=True)
                with open(settings_file, 'w', encoding='utf-8') as f: json.dump(settings, f, indent=2)
            except Exception as e: logger.warning(f'Failed to save settings: {e}')

    @time_execution_async('--get_windows_info')
    async def get_windows_info(self) -> list[Dict[str, Any]]:
        try:
            infos = []; fg_win = auto.GetForegroundControl()
            for win in auto.GetRootControl().GetChildren():
                if win.Exists() and hasattr(win, 'ClassName') and win.ClassName and hasattr(win, 'Name') and win.Name:
                    infos.append({"title": win.Name, "class": win.ClassName, 
                                  "is_foreground": win == fg_win, "handle": win.NativeWindowHandle})
            return infos
        except Exception as e: logger.error(f'Failed to get windows info: {e}'); return []

    @time_execution_async('--switch_to_window')
    async def switch_to_window(self, window_info: Dict[str, str]) -> None:
        title = window_info.get('title'); cls = window_info.get('class')
        if not title or not cls: raise ValueError("Window info must contain 'title' and 'class'")
        try:
            for win in auto.GetRootControl().GetChildren():
                if win.Exists() and win.Name == title and win.ClassName == cls:
                    if not await self._is_application_allowed(win):
                        raise RuntimeError(f'Cannot switch to non-allowed app: {win.Name}')
                    logger.info(f"Switching to window: {win.Name}"); win.SetFocus()
                    if hasattr(win, 'SetTopmost'): 
                        win.SetTopmost(True); 
                        await asyncio.sleep(0.02)
                        win.SetTopmost(False)
                    self.current_window = win; self.state.window_id = f"{win.ClassName}|{win.Name}"
                    await asyncio.sleep(self.config.get("window_switch_delay", 0.1))
                    return
            raise RuntimeError(f"No window found: title='{title}', class='{cls}'")
        except Exception as e:
            logger.error(f'Failed to switch to window: {e}', exc_info=True)
            raise RuntimeError(f'Failed to switch to window ({title}): {e}')

    async def remove_highlights(self): pass

    # **UPDATED CLICK_ELEMENT METHOD WITH AUTOMATIC MENU/SAVE HANDLING**
    async def click_element(self, params) -> ActionResult:
        """Improved click element with automatic menu handling and fallbacks"""
        name = getattr(params, 'element_name', "")
        ctype = getattr(params, 'control_type', "")
        class_name = getattr(params, 'class_name', "")
        slider_value = getattr(params, 'slider_value', None)
        file_path = getattr(params, 'file_path', None)
        
        logger.info(f"Clicking: name='{name}', type='{ctype}', class='{class_name}'")
        
        # Handle file opening directly
        if ctype == "File" and file_path:
            try:
                logger.info(f"Opening file directly: {file_path}")
                subprocess.run(['start', '', file_path], shell=True, check=True)
                await asyncio.sleep(self.config.get("app_launch_delay", 1.5))
                return ActionResult(
                    extracted_content=f"Opened file: {name} from {file_path}", 
                    include_in_memory=True
                )
            except Exception as e:
                logger.error(f"Failed to open file: {e}")
                return ActionResult(error=f"Could not open file: {name}")
        
        # Get current window
        window = await self.get_current_window()
        if not window: 
            return ActionResult(error="Cannot click: No current window.")
        
        # Get window info for smart handling
        window_name = getattr(window, 'Name', '').lower()
        
        # **AUTOMATIC SMART HANDLING FOR COMMON UI PATTERNS**
        
        # Auto-detect and handle menu items with keyboard shortcuts
        if "MenuItem" in ctype or (name in ["File", "Edit", "View", "Help", "Tools", "Format", "Save", "Save As"] and not ctype):
            return await self._handle_menu_automatically(window, name, ctype, window_name)
        
        # Auto-detect save operations in any text editor
        if name and ("save" in name.lower() or name in ["Save", "Save As"]):
            return await self._handle_save_automatically(window, name, window_name)
        
        # **ENHANCED ELEMENT FINDING WITH MULTIPLE STRATEGIES**
        
        try:
            element = await self._find_element_smart(window, name, ctype, class_name)
            
            if element and element.Exists():
                # Handle slider
                if ctype == "Slider" and slider_value is not None:
                    try:
                        pattern = element.GetRangeValuePattern()
                        if pattern:
                            pattern.SetValue(int(slider_value))
                            await asyncio.sleep(self.config.get("post_action_delay", 0.1))
                            return ActionResult(extracted_content=f"Set slider to {slider_value}", include_in_memory=True)
                    except Exception as e:
                        logger.error(f"Slider operation failed: {e}")
                        return ActionResult(error=f"Failed to set slider value: {e}")
                
                # Enhanced clicking with multiple fallback methods
                return await self._click_element_with_fallbacks(element, name)
            else:
                return ActionResult(error=f"Element not found: {name} (type: {ctype})")
                
        except Exception as e:
            logger.error(f"Click operation failed: {str(e)}", exc_info=True)
            return ActionResult(error=f"Click failed: {str(e)}")

    # **HELPER METHODS FOR ENHANCED CLICKING**
    
    async def _handle_menu_automatically(self, window, name, ctype, window_name):
        """Automatically handle menu clicks using keyboard shortcuts"""
        logger.info(f"Auto-handling menu item: {name}")
        
        try:
            window.SetFocus()
            await asyncio.sleep(0.1)
            
            # Define menu shortcuts
            menu_shortcuts = {
                "File": ("alt", "f"),
                "Edit": ("alt", "e"),
                "View": ("alt", "v"),
                "Help": ("alt", "h"),
                "Tools": ("alt", "t"),
                "Format": ("alt", "o"),
                "Save": ("ctrl", "s"),
                "Save As": ("ctrl", "shift", "s")
            }
            
            # Try direct keyboard shortcut first
            if name in menu_shortcuts:
                keys = menu_shortcuts[name]
                if len(keys) == 2:
                    pyautogui.hotkey(keys[0], keys[1])
                elif len(keys) == 3:
                    pyautogui.hotkey(keys[0], keys[1], keys[2])
                
                # Special handling for Save As
                if name == "Save As":
                    await asyncio.sleep(1.5)  # Wait for dialog
                    # If it's a text editor and no dialog appeared, try Alt+F,A
                    if "notepad" in window_name or "text" in window_name:
                        try:
                            save_dialog = auto.WindowControl(Name="Save As")
                            if not save_dialog.Exists():
                                logger.info("Save As dialog not found, trying Alt+F,A")
                                pyautogui.press('alt')
                                await asyncio.sleep(0.3)
                                pyautogui.press('f')
                                await asyncio.sleep(0.3)
                                pyautogui.press('a')
                        except:
                            pass
                else:
                    await asyncio.sleep(0.5)
                
                return ActionResult(extracted_content=f"Executed {name} via keyboard", include_in_memory=True)
            
            # Fallback: Try Alt + first letter for menu items
            if name in ["File", "Edit", "View", "Help", "Tools", "Format"]:
                pyautogui.press('alt')
                await asyncio.sleep(0.3)
                pyautogui.press(name[0].lower())
                await asyncio.sleep(0.5)
                return ActionResult(extracted_content=f"Opened {name} menu", include_in_memory=True)
            
            # If it's a submenu item, try letter shortcuts
            submenu_letters = {
                "Save As": "a",
                "Save": "s",
                "Copy": "c",
                "Paste": "v",
                "Cut": "x"
            }
            
            if name in submenu_letters:
                pyautogui.press(submenu_letters[name])
                await asyncio.sleep(0.5)
                return ActionResult(extracted_content=f"Selected {name}", include_in_memory=True)
            
            return ActionResult(error=f"No automatic handling available for menu item: {name}")
            
        except Exception as e:
            logger.error(f"Auto menu handling failed: {e}")
            return ActionResult(error=f"Menu handling failed: {str(e)}")

    async def _handle_save_automatically(self, window, name, window_name):
        """Automatically handle save operations"""
        logger.info(f"Auto-handling save operation: {name}")
        
        try:
            window.SetFocus()
            await asyncio.sleep(0.1)
            
            # Try Ctrl+S first for regular save
            if "save as" not in name.lower():
                pyautogui.hotkey('ctrl', 's')
                await asyncio.sleep(1.5)
                
                # If Save As dialog appears, handle it
                try:
                    save_dialog = auto.WindowControl(Name="Save As")
                    if save_dialog.Exists():
                        # Type a default filename
                        default_filename = "document.txt"
                        if "notepad" in window_name:
                            default_filename = "nature_poem.txt"
                        
                        pyautogui.write(default_filename)
                        await asyncio.sleep(0.3)
                        pyautogui.press('enter')
                        await asyncio.sleep(1.0)
                        
                        return ActionResult(extracted_content=f"File saved as {default_filename}", include_in_memory=True)
                except:
                    pass
                
                return ActionResult(extracted_content="Save command executed", include_in_memory=True)
            
            # Handle Save As specifically
            else:
                # Try Ctrl+Shift+S first
                pyautogui.hotkey('ctrl', 'shift', 's')
                await asyncio.sleep(1.5)
                
                # If no dialog, try Alt+F,A
                try:
                    save_dialog = auto.WindowControl(Name="Save As")
                    if not save_dialog.Exists():
                        pyautogui.press('alt')
                        await asyncio.sleep(0.3)
                        pyautogui.press('f')
                        await asyncio.sleep(0.3)
                        pyautogui.press('a')
                        await asyncio.sleep(1.5)
                except:
                    pass
                
                return ActionResult(extracted_content="Save As dialog opened", include_in_memory=True)
            
        except Exception as e:
            logger.error(f"Auto save handling failed: {e}")
            return ActionResult(error=f"Save handling failed: {str(e)}")

    async def _find_element_smart(self, window, name, ctype, class_name):
        """Smart element finding with multiple strategies"""
        element = None
        
        # Strategy 1: Clean control type and try method
        if ctype and ctype != "File":
            try:
                clean_ctype = ctype.replace(" Control", "").replace("Control", "").replace("MenuItem", "").strip()
                if clean_ctype:
                    method_name = f"{clean_ctype}Control"
                    
                    if hasattr(window, method_name):
                        method = getattr(window, method_name)
                        
                        if name:
                            try:
                                element = method(Name=name)
                                if element and element.Exists():
                                    logger.debug(f"Found element using {method_name} with name")
                                    return element
                            except:
                                pass
                        
                        try:
                            element = method()
                            if element and element.Exists():
                                logger.debug(f"Found element using {method_name}")
                                return element
                        except:
                            pass
            except Exception as e:
                logger.debug(f"Control type strategy failed: {e}")
        
        # Strategy 2: Search by name
        if name:
            try:
                element = window.Control(Name=name)
                if element and element.Exists():
                    logger.debug(f"Found element by name: {name}")
                    return element
            except:
                pass
        
        # Strategy 3: Search by class
        if class_name:
            try:
                element = window.Control(ClassName=class_name)
                if element and element.Exists():
                    logger.debug(f"Found element by class: {class_name}")
                    return element
            except:
                pass
        
        # Strategy 4: Partial name matching in children
        if name:
            try:
                for child in window.GetChildren():
                    if child.Exists():
                        child_name = getattr(child, 'Name', '')
                        if child_name and name.lower() in child_name.lower():
                            logger.debug(f"Found element by partial name: {child_name}")
                            return child
            except:
                pass
        
        return None

    async def _click_element_with_fallbacks(self, element, name):
        """Click element using multiple fallback methods"""
        
        # Validate element first
        try:
            if not element.Exists():
                return ActionResult(error=f"Element no longer exists: {name}")
            
            if hasattr(element, 'IsEnabled') and not element.IsEnabled:
                logger.warning(f"Element is disabled but attempting click anyway: {name}")
        except:
            pass
        
        # Try different click methods
        methods = [
            ("invoke_pattern", self._try_invoke_click),
            ("standard_click", self._try_standard_click),
            ("position_click", self._try_position_click),
            ("keyboard_enter", self._try_keyboard_click)
        ]
        
        for method_name, method_func in methods:
            try:
                logger.debug(f"Trying {method_name} for {name}")
                
                # Focus element
                try:
                    element.SetFocus()
                    await asyncio.sleep(self.config.get("focus_delay", 0.05))
                except:
                    pass
                
                # Try the click method
                success = await method_func(element)
                if success:
                    await asyncio.sleep(self.config.get("post_action_delay", 0.1))
                    logger.info(f"Successfully clicked {name} using {method_name}")
                    return ActionResult(extracted_content=f"Clicked: {name}", include_in_memory=True)
                    
            except Exception as e:
                logger.debug(f"{method_name} failed for {name}: {e}")
                continue
        
        return ActionResult(error=f"All click methods failed for: {name}")

    async def _try_invoke_click(self, element):
        """Try invoke pattern click"""
        try:
            if hasattr(element, 'GetInvokePattern'):
                pattern = element.GetInvokePattern()
                if pattern:
                    pattern.Invoke()
                    return True
        except:
            pass
        return False

    async def _try_standard_click(self, element):
        """Try standard click"""
        try:
            element.Click()
            return True
        except:
            pass
        return False

    async def _try_position_click(self, element):
        """Try position-based click"""
        try:
            rect = element.BoundingRectangle
            if rect and rect.width() > 0 and rect.height() > 0:
                center_x = rect.left + rect.width() // 2
                center_y = rect.top + rect.height() // 2
                
                # Ensure on screen
                screen_width, screen_height = pyautogui.size()
                center_x = max(0, min(center_x, screen_width - 1))
                center_y = max(0, min(center_y, screen_height - 1))
                
                pyautogui.click(center_x, center_y)
                return True
        except:
            pass
        return False

    async def _try_keyboard_click(self, element):
        """Try keyboard activation"""
        try:
            pyautogui.press('enter')
            return True
        except:
            pass
        return False

    def _is_interactive_element(self, element) -> bool:
        """Check if element is interactive or clickable - more flexible version"""
        try:
            # Primary interactive control types
            interactive_types = [
                'Button', 'CheckBox', 'RadioButton', 'ComboBox', 'Edit', 
                'Slider', 'Tab', 'MenuItem', 'ListItem', 'TreeItem', 'Hyperlink'
            ]
            
            control_type = getattr(element, 'ControlTypeName', '')
            
            # Always allow these primary interactive types
            if control_type in interactive_types:
                return True
            
            # Check for clickable patterns (Text elements that are actually clickable)
            if control_type in ['Text', 'Static', 'Group', 'Pane']:
                # Check if element can be invoked (clickable)
                if hasattr(element, 'IsInvokePatternAvailable') and element.IsInvokePatternAvailable():
                    return True
                
                # Check if element is keyboard focusable
                if hasattr(element, 'IsKeyboardFocusable') and element.IsKeyboardFocusable:
                    return True
                
                # Check if element has a name (likely clickable if it has a meaningful name)
                element_name = getattr(element, 'Name', '')
                if element_name and len(element_name.strip()) > 0:
                    return True
            
            # Check UI patterns that indicate interactivity
            patterns = [
                'InvokePattern', 'SelectionItemPattern', 'ExpandCollapsePattern',
                'ValuePattern', 'RangeValuePattern', 'ScrollItemPattern'
            ]
            
            for pattern in patterns:
                pattern_getter = f"Get{pattern}"
                if hasattr(element, pattern_getter):
                    try:
                        pattern_obj = getattr(element, pattern_getter)()
                        if pattern_obj:
                            return True
                    except Exception:
                        continue
            
            return False
        except Exception:
            return False

    async def right_click_element(self, target_name: str, target_control_type: str, target_class: str) -> bool:
        """Right-click element - optimized timings"""
        logger.info(f"Right-clicking: name='{target_name}', type='{target_control_type}'")
        window = await self.get_current_window()
        if not window:
            return False
        
        try:
            element = None
            
            # Try control type first
            if target_control_type:
                method = getattr(window, f"{target_control_type}Control", None)
                if method:
                    if target_name:
                        element = method(Name=target_name)
                    else:
                        element = method()
            
            # Try by name
            if not element or not element.Exists():
                if target_name:
                    element = window.Control(Name=target_name)
            
            if element and element.Exists():
                # Special case for desktop
                if target_name == "Desktop":
                    pyautogui.FAILSAFE = False
                    pyautogui.rightClick(0, 0)
                else:
                    if hasattr(element, 'RightClick'):
                        element.RightClick()
                    else:
                        rect = element.BoundingRectangle
                        if rect:
                            pyautogui.rightClick(rect.left + rect.width()//2, rect.top + rect.height()//2)
                
                await asyncio.sleep(self.config.get("post_action_delay", 0.1))
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Right-click failed: {str(e)}")
            return False

    async def double_click_element(self, target_name: str, target_control_type: str, target_class: str) -> bool:
        """Double-click element - optimized timings"""
        logger.info(f"Double-clicking: name='{target_name}', type='{target_control_type}'")
        window = await self.get_current_window()
        if not window:
            return False
        
        try:
            element = None
            
            # Try control type first
            if target_control_type:
                method = getattr(window, f"{target_control_type}Control", None)
                if method:
                    if target_name:
                        element = method(Name=target_name)
                    else:
                        element = method()
            
            # Try by name
            if not element or not element.Exists():
                if target_name:
                    element = window.Control(Name=target_name)
            
            if element and element.Exists():
                if hasattr(element, 'DoubleClick'):
                    element.DoubleClick()
                else:
                    rect = element.BoundingRectangle
                    if rect:
                        pyautogui.doubleClick(rect.left + rect.width()//2, rect.top + rect.height()//2)
                
                await asyncio.sleep(self.config.get("post_action_delay", 0.1))
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Double-click failed: {str(e)}")
            return False

    async def type_text(self, target_name: str, target_control_type: str, target_class: str, text: str) -> bool:
        """Type text - optimized timings"""
        logger.info(f"Typing text: '{text}' into name='{target_name}', type='{target_control_type}'")
        window = await self.get_current_window()
        if not window:
            return False
        
        try:
            element = None
            
            # Try Edit control first
            if target_control_type == "Edit" or not target_control_type:
                if target_name:
                    element = window.EditControl(Name=target_name)
                else:
                    element = window.EditControl()
            
            # Try by name if not found
            if not element or not element.Exists():
                if target_name:
                    element = window.Control(Name=target_name)
            
            if element and element.Exists():
                element.SetFocus()
                await asyncio.sleep(self.config.get("text_input_delay", 0.05))
                pyautogui.write(text, interval=0.01)
                await asyncio.sleep(self.config.get("post_action_delay", 0.1))
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Type text failed: {str(e)}")
            return False

    async def press_enter(self, target_name: str, target_control_type: str, target_class: str) -> bool:
        """Press Enter - optimized timings"""
        logger.info(f"Pressing Enter: name='{target_name}', type='{target_control_type}'")
        window = await self.get_current_window()
        if not window:
            return False
        
        try:
            element = None
            
            # Try control type first
            if target_control_type:
                method = getattr(window, f"{target_control_type}Control", None)
                if method:
                    if target_name:
                        element = method(Name=target_name)
                    else:
                        element = method()
            
            # Try by name
            if not element or not element.Exists():
                if target_name:
                    element = window.Control(Name=target_name)
            
            if element and element.Exists():
                element.SetFocus()
                await asyncio.sleep(self.config.get("focus_delay", 0.05))
                if hasattr(element, 'SendKeys'):
                    element.SendKeys('{Enter}')
                else:
                    pyautogui.press('enter')
                await asyncio.sleep(self.config.get("post_action_delay", 0.1))
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Press Enter failed: {str(e)}")
            return False

    async def select_text(self, target_name: str, target_control_type: str, target_class: str) -> bool:
        """Select text - optimized timings"""
        logger.info(f"Selecting text: name='{target_name}', type='{target_control_type}'")
        window = await self.get_current_window()
        if not window:
            return False
        
        try:
            element = None
            
            # Try Edit control first
            if target_control_type == "Edit" or not target_control_type:
                if target_name:
                    element = window.EditControl(Name=target_name)
                else:
                    element = window.EditControl()
            
            # Try by name
            if not element or not element.Exists():
                if target_name:
                    element = window.Control(Name=target_name)
            
            if element and element.Exists():
                element.SetFocus()
                await asyncio.sleep(self.config.get("focus_delay", 0.05))
                if hasattr(element, 'SendKeys'):
                    element.SendKeys('{Ctrl}a')
                else:
                    pyautogui.hotkey('ctrl', 'a')
                await asyncio.sleep(0.1)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Select text failed: {str(e)}")
            return False

    async def copy_text(self, target_name: str, target_control_type: str, target_class: str) -> bool:
        """Copy text - optimized timings"""
        logger.info(f"Copying text: name='{target_name}', type='{target_control_type}'")
        window = await self.get_current_window()
        if not window:
            return False
        
        try:
            element = None
            
            # Try Edit control first
            if target_control_type == "Edit" or not target_control_type:
                if target_name:
                    element = window.EditControl(Name=target_name)
                else:
                    element = window.EditControl()
            
            # Try by name
            if not element or not element.Exists():
                if target_name:
                    element = window.Control(Name=target_name)
            
            if element and element.Exists():
                element.SetFocus()
                await asyncio.sleep(self.config.get("focus_delay", 0.05))
                if hasattr(element, 'SendKeys'):
                    element.SendKeys('{Ctrl}c')
                else:
                    pyautogui.hotkey('ctrl', 'c')
                await asyncio.sleep(0.1)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Copy text failed: {str(e)}")
            return False

    async def paste_text(self, target_name: str, target_control_type: str, target_class: str) -> bool:
        """Paste text - optimized timings"""
        logger.info(f"Pasting text: name='{target_name}', type='{target_control_type}'")
        window = await self.get_current_window()
        if not window:
            return False
        
        try:
            element = None
            
            # Try Edit control first
            if target_control_type == "Edit" or not target_control_type:
                if target_name:
                    element = window.EditControl(Name=target_name)
                else:
                    element = window.EditControl()
            
            # Try by name
            if not element or not element.Exists():
                if target_name:
                    element = window.Control(Name=target_name)
            
            if element and element.Exists():
                element.SetFocus()
                await asyncio.sleep(self.config.get("focus_delay", 0.05))
                if hasattr(element, 'SendKeys'):
                    element.SendKeys('{Ctrl}v')
                else:
                    pyautogui.hotkey('ctrl', 'v')
                await asyncio.sleep(0.1)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Paste text failed: {str(e)}")
            return False

    # **ESSENTIAL METHODS KEPT**
    
    async def input_text_by_properties(self, element_name: str, control_type: str, class_name: str, text: str) -> bool:
        """Enhanced input text method"""
        logger.info(f"Input text: '{text}' into (name='{element_name}', type='{control_type}')")
        window = await self.get_current_window()
        if not window: 
            logger.error("Cannot input: No current window.")
            return False
        
        try:
            element = None
            
            # Try Edit control first
            if control_type == "Edit" or not control_type:
                if element_name:
                    element = window.EditControl(Name=element_name)
                else:
                    element = window.EditControl()
            
            # Try by name
            if not element or not element.Exists():
                if element_name:
                    element = window.Control(Name=element_name)
            
            if element and element.Exists():
                logger.info(f"Found element for input: {getattr(element, 'Name', 'unnamed')}")
                element.SetFocus()
                await asyncio.sleep(self.config.get("text_input_delay", 0.05))
                
                # Try to clear existing content
                try:
                    if hasattr(element, 'Select') and callable(element.Select):
                        element.Select()
                        await asyncio.sleep(0.02)
                        element.SendKeys('{BACKSPACE}')
                    elif hasattr(element, 'ValuePattern') and element.ValuePattern.IsSupported:
                        element.ValuePattern.SetValue('')
                    else:
                        element.SendKeys('^a{BACKSPACE}')
                except Exception as e_clear:
                    logger.debug(f"Could not clear element content: {e_clear}")
                
                await asyncio.sleep(0.05)
                element.SendKeys(text, interval=0.005)
                await asyncio.sleep(self.config.get("post_action_delay", 0.1))
                logger.info("Text input successful.")
                return True
            
            logger.warning(f"Could not find element for input (Name='{element_name}', Type='{control_type}')")
            return False
            
        except Exception as e:
            logger.error(f"Input text failed: {str(e)}")
            return False

    @time_execution_async('--find_element_by_properties')
    async def find_element_by_properties(self, class_name: Optional[str] = None, element_name: Optional[str] = None, control_type: Optional[str] = None) -> int:
        """Find element by properties in element map"""
        window = await self.get_current_window()
        if not window: 
            raise RuntimeError("Cannot find element: No current window.")
        
        session = await self.get_session()
        if not session.cached_state or not session.cached_state.element_map:
            await self.get_state()
            session = await self.get_session()
            if not session.cached_state or not session.cached_state.element_map:
                raise RuntimeError("Failed to get element map.")
        
        for index, el_info in session.cached_state.element_map.items():
            if not isinstance(el_info, dict): 
                continue
            
            cls_match = not class_name or el_info.get("class name") == class_name
            name_match = not element_name or el_info.get("element name") == element_name
            type_match = not control_type or el_info.get("control type") == control_type
            
            if cls_match and name_match and type_match:
                logger.info(f"Found element in map at index {index}")
                return int(index)
        
        raise RuntimeError(f"No element in map for: Class='{class_name}', Name='{element_name}', Type='{control_type}'")

    @time_execution_async('--start_application')
    async def start_application(self, app_name: str) -> str:
        """Start an application by name using Win+R method"""
        try:
            logger.info(f"Launching application using Win+R: {app_name}")
            
            pyautogui.hotkey('win', 'r')
            await asyncio.sleep(0.5)
            pyautogui.write(app_name)
            await asyncio.sleep(0.2)
            pyautogui.press('enter')
            await asyncio.sleep(self.config.get("app_launch_delay", 1.5))

            target_window = None
            desktop = auto.GetRootControl()
            app_name_simple = os.path.splitext(os.path.basename(app_name))[0].lower()

            # Try to find the window
            for attempt in range(3):
                windows = desktop.GetChildren()
                logger.debug(f"Searching for '{app_name_simple}' window, attempt {attempt+1}. Found {len(windows)} top-level windows.")
                
                # Check foreground window first
                foreground_win = auto.GetForegroundControl()
                if foreground_win and foreground_win.Exists() and hasattr(foreground_win, 'Name') and app_name_simple in foreground_win.Name.lower():
                    target_window = foreground_win
                    logger.info(f"Found matching foreground window: {target_window.Name}")
                    break

                for win_ctrl in windows:
                    if win_ctrl.Exists() and hasattr(win_ctrl, 'Name'):
                        if app_name_simple in win_ctrl.Name.lower():
                            if win_ctrl.ControlTypeName == "Window" and win_ctrl.IsWindowPatternAvailable():
                                target_window = win_ctrl
                                logger.info(f"Found matching window by title: {target_window.Name}")
                                break
                    
                    # Check by process name
                    if hasattr(win_ctrl, 'ProcessId'):
                        try:
                            import psutil
                            process = psutil.Process(win_ctrl.ProcessId)
                            if app_name_simple in process.name().lower():
                                target_window = win_ctrl
                                logger.info(f"Found matching window by process name '{process.name()}': {target_window.Name}")
                                break
                        except (psutil.NoSuchProcess, psutil.AccessDenied, ImportError):
                            pass
                
                if target_window:
                    break
                await asyncio.sleep(0.5)

            if not target_window:
                target_window = auto.GetForegroundControl()
                if target_window:
                    logger.info(f"Could not specifically find window for '{app_name}'. Using current foreground window: {target_window.Name}")
                else:
                    logger.warning(f"Could not find any window after launching '{app_name}'.")
            
            if target_window and target_window.Exists():
                logger.info(f"Activating window: {target_window.Name}")
                target_window.SetFocus()
                if hasattr(target_window, 'SetTopmost'):
                    target_window.SetTopmost(True)
                    await asyncio.sleep(0.02)
                    target_window.SetTopmost(False)

                self.current_window = target_window
                self.state.window_id = f"{getattr(target_window, 'ClassName', 'UnknownClass')}|{getattr(target_window, 'Name', 'UnknownWindowName')}"
                
                if self.session: 
                    self.session.cached_state = None
                if self.dom_service: 
                    self.dom_service.window = target_window
                
                await self.get_state()
                return f"Launched {app_name} using Win+R and switched focus to '{target_window.Name}'."
            else:
                logger.error(f"Could not find or focus window after launching {app_name} using Win+R.")
                return f"Attempted to launch {app_name} using Win+R but could not locate or focus its window."
                
        except Exception as e:
            logger.error(f'Failed to start application {app_name} using Win+R: {str(e)}', exc_info=True)
            raise RuntimeError(f"Failed to start application {app_name} using Win+R: {str(e)}")