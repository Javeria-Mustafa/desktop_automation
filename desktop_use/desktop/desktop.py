import asyncio
import gc
import logging
import os
import time
import base64
from io import BytesIO
from typing import Dict, List, Optional, Any

import pyautogui
import uiautomation as auto
import xml.etree.ElementTree as ET

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from desktop.views import DesktopState, ElementMap
from desktop.context import DesktopContext, DesktopContextConfig
from utils import time_execution_async

load_dotenv()

logger = logging.getLogger(__name__)


class DesktopConfig(BaseModel):
    """
    Configuration for Desktop automation.

    Default values:
        enable_screenshots: True
            Whether to take screenshots during automation
            
        screenshot_dir: "screenshots"
            Directory to save screenshots
            
        include_attributes: ["class name", "element name", "control type"]
            Attributes to include in the XML representation
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='ignore',
        populate_by_name=True,
        from_attributes=True,
        validate_assignment=True,
    )

    enable_screenshots: bool = True
    screenshot_dir: str = "screenshots"
    include_attributes: List[str] = Field(default_factory=lambda: ["class name", "element name", "control type"])
    new_context_config: Dict[str, Any] = Field(default_factory=lambda: {
        "enable_screenshots": True,
    })


class Desktop:
    """
    Desktop automation framework.

    This is the main entry point for desktop automation. It provides methods to interact
    with desktop applications and extract information from the UI.
    """

    def __init__(
        self,
        config: DesktopConfig = None,
    ):
        logger.debug('🖥️ Initializing desktop automation')
        self.config = config or DesktopConfig()
        
        # Create directories if they don't exist - still useful for other saved files
        os.makedirs(self.config.screenshot_dir, exist_ok=True)

    async def new_context(self, config: Dict[str, Any] = None) -> DesktopContext:
        """Create a desktop context"""
        return DesktopContext(desktop=self, config=config or self.config.new_context_config)

    async def get_available_windows(self) -> List[str]:
        """Get list of available windows"""
        windows = []
        try:
            desktop = auto.GetRootControl()
            for win in desktop.GetChildren():
                if win.ClassName:  # Only include windows with class names
                    windows.append(f"{win.Name} ({win.ClassName})")
        except Exception as e:
            logger.error(f"Error getting available windows: {str(e)}")
        
        return windows
    
    async def take_screenshot(self) -> str:
        """Take a screenshot and return base64 encoded data"""
        try:
            if not self.config.enable_screenshots:
                return ""
                
            # Take screenshot in memory
            screen = pyautogui.screenshot()
            
            # Convert to base64 without saving to disk
            buffered = BytesIO()
            screen.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return img_str
        except Exception as e:
            logger.error(f"Error taking screenshot: {str(e)}")
            return ""
            
    @time_execution_async('--get_state (desktop)')
    async def get_state(self, window: auto.WindowControl = None) -> DesktopState:
        """
        Get the current state of the desktop
        
        Args:
            window: Window to get state for (None for current foreground window)
            
        Returns:
            DesktopState object with current state
        """
        try:
            # Get current window if not provided
            if window is None:
                window = auto.GetForegroundControl()
            
            # Get window properties
            app_title = window.ClassName if window.ClassName else "Unknown"
            title = window.Name if window.Name else "Unknown"
            
            # Take screenshot if enabled
            screenshot = None
            if self.config.enable_screenshots:
                screenshot = await self.take_screenshot()
            
            # Get available windows
            available_windows = await self.get_available_windows()
            
            # Extract UI elements and build element map
            element_map = await self._build_element_map(window)
            
            # Create and return state
            state = DesktopState(
                app_title=app_title,
                title=title,
                screenshot=screenshot,
                element_map=element_map,
                available_windows=available_windows
            )
            
            return state
        
        except Exception as e:
            logger.error(f"Error getting desktop state: {str(e)}")
            raise RuntimeError(f"Failed to get desktop state: {str(e)}")
    
    async def _build_element_map(self, window: auto.WindowControl) -> ElementMap:
        """
        Build a map of interactive elements from a window
        
        Args:
            window: Window to extract elements from
            
        Returns:
            ElementMap with interactive elements
        """
        element_map = ElementMap()
        index = 1
        
        try:
            # Process the window and its children recursively
            await self._process_element(window, element_map, index)
        except Exception as e:
            logger.error(f"Error building element map: {str(e)}")
        
        return element_map
    
    async def _process_element(self, element: auto.Control, element_map: ElementMap, index: int) -> int:
        """
        Process an element and its children recursively  
        
        Args:
            element: Element to process
            element_map: Map to add elements to
            index: Current index
            
        Returns:
            Next index to use
        """
        try:
            # Skip invisible elements
            if not hasattr(element, 'IsOffscreen') or not element.IsOffscreen:
                return index
            
            # Check if element is interactive
            is_interactive = self._is_interactive(element)
            
            if is_interactive:
                # Add to element map
                props = {} 
                
                if hasattr(element, 'ClassName') and element.ClassName:
                    props["class name"] = element.ClassName
                
                if hasattr(element, 'Name') and element.Name:
                    props["element name"] = element.Name
                    
                if hasattr(element, 'ControlTypeName') and element.ControlTypeName:
                    props["control type"] = element.ControlTypeName
                
                element_map[index] = props
                index += 1
            
            # Process children
            try:
                if hasattr(element, 'GetChildren'):
                    for child in element.GetChildren():
                        index = await self._process_element(child, element_map, index)
            except Exception as child_error:
                logger.debug(f"Error processing children: {child_error}")
            
            return index
            
        except Exception as e:
            logger.debug(f"Error processing element: {e}")
            return index
    
    def _is_interactive(self, element: auto.Control) -> bool:
        """Determine if an element is interactive - more generous version"""
        try:
            # Always consider these control types interactive
            if hasattr(element, 'ControlTypeName') and element.ControlTypeName in self.interactive_control_types:
                return True
                
            # Consider elements with a name to be potentially interactive
            if hasattr(element, 'Name') and element.Name:
                return True
                
            # Check keyboard focusability
            if hasattr(element, 'IsKeyboardFocusable') and element.IsKeyboardFocusable:
                return True
                
            # Check for any UI Automation patterns
            patterns = [
                "InvokePattern", "SelectionItemPattern", "ExpandCollapsePattern",
                "ValuePattern", "RangeValuePattern", "ScrollItemPattern"
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
                        
            # As a last resort, check if element has children - parent elements often interactive
            if hasattr(element, 'GetChildren') and element.GetChildren():
                return True
                
            return False
        except Exception:
            return False
    
    async def close(self):
        """Clean up resources"""
        # Not much to clean up for desktop automation
        # But we keep the method for API compatibility with the browser version
        gc.collect()