from typing import Literal, Optional

from pydantic import BaseModel, Field


class NoParamsAction(BaseModel):
    """Action with no parameters"""
    pass

class LaunchApplicationAction(BaseModel):
    app_name: str

class DoneAction(BaseModel):
    """Complete task with return text and success flag"""
    
    text: str = Field(
        ...,
        description="Return all information you gathered or output you want to report for this task",
    )
    success: bool = Field(
        ...,
        description="Set to true if the task was successfully completed, false otherwise",
    )

class ClickElementAction(BaseModel):
    """Click on an UI element by properties"""
    
    element_name: Optional[str] = Field(
        None,
        description="Name of the element to click",
    )
    control_type: Optional[str] = Field(
        None,
        description="Control type of the element (e.g., Button, Edit, etc.)",
    )
    class_name: Optional[str] = Field(
        None,
        description="Class name of the element",
    )
    slider_value: Optional[int] = Field(
        None,
        description="Value to set if the element is a slider control",
    )

class RightClickElementAction(BaseModel):
    """Right-click on an UI element by properties"""
    
    element_name: Optional[str] = Field(
        None,
        description="Name of the element to right-click",
    )
    control_type: Optional[str] = Field(
        None,
        description="Control type of the element (e.g., Button, Edit, etc.)",
    )
    class_name: Optional[str] = Field(
        None,
        description="Class name of the element",
    )

class DoubleClickElementAction(BaseModel):
    """Double-click on an UI element by properties"""
    
    element_name: Optional[str] = Field(
        None,
        description="Name of the element to double-click",
    )
    control_type: Optional[str] = Field(
        None,
        description="Control type of the element (e.g., Button, Edit, etc.)",
    )
    class_name: Optional[str] = Field(
        None,
        description="Class name of the element",
    )

class InputTextAction(BaseModel):
    """Input text into an UI element"""
    
    element_name: Optional[str] = Field(
        None,
        description="Name of the element to input text into",
    )
    control_type: Optional[str] = Field(
        None,
        description="Control type of the element (e.g., Edit, ComboBox, etc.)",
    )
    class_name: Optional[str] = Field(
        None,
        description="Class name of the element",
    )
    text: str = Field(
        ...,
        description="Text to input into the element. Do not use this for keyboard shortcuts (like Ctrl+C) - use the send_keys action instead.",
    )

class PressEnterAction(BaseModel):
    """Press Enter key on an element"""
    
    element_name: Optional[str] = Field(
        None,
        description="Name of the element to press Enter on",
    )
    control_type: Optional[str] = Field(
        None,
        description="Control type of the element",
    )
    class_name: Optional[str] = Field(
        None,
        description="Class name of the element",
    )

class SelectTextAction(BaseModel):
    """Select all text in an element"""
    
    element_name: Optional[str] = Field(
        None,
        description="Name of the element to select text in",
    )
    control_type: Optional[str] = Field(
        None,
        description="Control type of the element",
    )
    class_name: Optional[str] = Field(
        None,
        description="Class name of the element",
    )

class CopyTextAction(BaseModel):
    """Copy text from an element"""
    
    element_name: Optional[str] = Field(
        None,
        description="Name of the element to copy text from",
    )
    control_type: Optional[str] = Field(
        None,
        description="Control type of the element",
    )
    class_name: Optional[str] = Field(
        None,
        description="Class name of the element",
    )

class PasteTextAction(BaseModel):
    """Paste text into an element"""
    
    element_name: Optional[str] = Field(
        None,
        description="Name of the element to paste text into",
    )
    control_type: Optional[str] = Field(
        None,
        description="Control type of the element",
    )
    class_name: Optional[str] = Field(
        None,
        description="Class name of the element",
    )

class SendKeysAction(BaseModel):
    """Send keyboard keys or shortcuts"""
    
    keys: str = Field(
        ...,
        description="Key or key combination to send (e.g., 'Enter', 'Escape', 'Control+c', 'Alt+Tab', 'Win+r'). For shortcuts, separate keys with '+'. Special keys include: Enter, Escape, Backspace, Tab, Space, PageUp, PageDown, Home, End, Insert, Delete, F1-F12, Alt, Control, Shift, Win.",
    )

class WaitAction(BaseModel):
    """Wait for a specified amount of time"""
    
    seconds: int = Field(
        3,
        description="Number of seconds to wait",
    )

class ScrollAction(BaseModel):
    """Scroll the current view"""
    
    direction: Optional[str] = Field(
        "down",
        description="Direction to scroll: 'up' or 'down'",
    )
    amount: Optional[int] = Field(
        3,
        description="Amount to scroll in clicks (default: 3)",
    )

class FindElementByPropertiesAction(BaseModel):
    """Find an element by its class name, element name, or control type"""
    
    class_name: Optional[str] = Field(
        None,
        description="Class name of the element to find",
    )
    element_name: Optional[str] = Field(
        None,
        description="Name of the element to find",
    )
    control_type: Optional[str] = Field(
        None,
        description="Control type of the element to find",
    )

class SliderAction(BaseModel):
    """Set slider value"""
    
    element_name: Optional[str] = Field(
        None,
        description="Name of the slider element",
    )
    control_type: Optional[str] = Field(
        "Slider",
        description="Control type of the element (should be Slider)",
    )
    class_name: Optional[str] = Field(
        None,
        description="Class name of the element",
    )
    value: int = Field(
        ...,
        description="Value to set the slider to",
    )