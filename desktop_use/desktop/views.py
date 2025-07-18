from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from dom.history_tree_processor.views import DOMHistoryElement  # Fix: Changed DomHistoryElement to DOMHistoryElement
from dom.views import DOMState


# Pydantic
class WindowInfo(BaseModel):
    """Represents information about a desktop window"""

    window_id: int
    title: str


# ElementMap is a specialized dictionary mapping element indices to their properties
class ElementMap(Dict[int, Dict[str, Any]]):
    """Mapping of element indices to their properties"""
    pass


@dataclass
class DesktopState(DOMState):
    """Represents the state of the desktop at a point in time"""
    
    window_title: str = "Unknown Window"
    title: str = "Unknown Title"
    open_windows: List[WindowInfo] = field(default_factory=list)
    element_tree: Optional[Any] = None
    selector_map: Dict[int, Any] = field(default_factory=dict)
    screenshot: Optional[str] = None
    pixels_above: int = 0
    pixels_below: int = 0
    desktop_errors: List[str] = field(default_factory=list)
    
    # Additional fields for desktop-specific state
    element_map: ElementMap = field(default_factory=ElementMap)
    available_windows: List[str] = field(default_factory=list)
    xml_content: Optional[str] = None
    xml_hash: Optional[str] = None 
    scroll_position_y: int = 0
    scroll_height: int = 0
    app_title: Optional[str] = None
    windows: List[str] = field(default_factory=list)
    application_name: str = "Unknown"


@dataclass
class DesktopState(DOMState):
    """Represents the state of the desktop at a point in time"""
    
    window_title: str = "Unknown Window"
    title: str = "Unknown Title"
    open_windows: List[WindowInfo] = field(default_factory=list)
    element_tree: Optional[Any] = None
    selector_map: Dict[int, Any] = field(default_factory=dict)
    screenshot: Optional[str] = None
    pixels_above: int = 0
    pixels_below: int = 0
    desktop_errors: List[str] = field(default_factory=list)
    
    # Additional fields for desktop-specific state
    element_map: ElementMap = field(default_factory=ElementMap)
    available_windows: List[str] = field(default_factory=list)
    xml_content: Optional[str] = None
    xml_hash: Optional[str] = None  # Make sure this is added
    scroll_position_y: int = 0
    scroll_height: int = 0
    app_title: Optional[str] = None
    windows: List[str] = field(default_factory=list)
    application_name: str = "Unknown"
    
@dataclass
class DesktopStateHistory:
    """Represents historical state information for desktop automation"""
    
    window_title: str = "Unknown Window"
    title: str = "Unknown Title"
    open_windows: List[Dict[str, Any]] = field(default_factory=list)
    interacted_element: List[Optional[Dict[str, Any]]] = field(default_factory=list)
    screenshot: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        data = {}
        data['open_windows'] = self.open_windows
        data['screenshot'] = self.screenshot
        data['interacted_element'] = self.interacted_element
        data['window_title'] = self.window_title
        data['title'] = self.title
        return data

class DesktopError(Exception):
    """Base class for all desktop automation errors"""
    pass


class AppNotAllowedError(DesktopError):
    """Error raised when an application is not allowed"""
    pass


class ElementNotFoundError(DesktopError):
    """Error raised when an element cannot be found"""
    pass


class ActionFailedError(DesktopError):
    """Error raised when an action fails"""
    pass


class DesktopStateError(DesktopError):
    """Error raised when the UI state is inconsistent or unexpected"""
    pass