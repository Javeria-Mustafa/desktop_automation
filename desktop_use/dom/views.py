from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List, Optional, Any, Union, ForwardRef

# Define HashedElement directly in this file instead of importing it
@dataclass
class HashedElement:
    """Hashed representation of a DOM element for comparison"""
    class_name_hash: str
    element_name_hash: str
    control_type_hash: str
    attributes_hash: str

from utils import time_execution_sync

# Remove circular imports - instead use string type annotations (PEP 563)
DOMElementNodeRef = ForwardRef('DOMElementNode')

@dataclass(frozen=False)
class DOMBaseNode:
    is_visible: bool = True
    # Use string type annotation for forward references
    parent: Optional['DOMElementNode'] = None


@dataclass(frozen=False)
class DOMTextNode(DOMBaseNode):
    text: str = ""
    type: str = 'TEXT_NODE'

    def has_parent_with_highlight_index(self) -> bool:
        current = self.parent
        while current is not None:
            # stop if the element has an index (will be handled separately)
            if current.highlight_index is not None:
                return True

            current = current.parent
        return False

    def is_parent_in_viewport(self) -> bool:
        if self.parent is None:
            return False
        return self.parent.is_in_viewport

    def is_parent_top_element(self) -> bool:
        if self.parent is None:
            return False
        return self.parent.is_top_element


@dataclass(frozen=False)
class DOMElementNode(DOMBaseNode):
    """
    xpath: the xpath of the element from the last root node.
    To properly reference the element we need to recursively switch the root node.
    """

    tag_name: str = ""  # Will store control_type for desktop elements
    xpath: str = ""      # Will store automation_id or XML path for desktop elements
    attributes: Dict[str, str] = None
    children: List[Any] = None  # Will contain DOMBaseNode instances but avoid circular refs
    is_interactive: bool = False
    is_top_element: bool = False
    is_in_viewport: bool = False
    highlight_index: Optional[int] = None  # This replaces element_index for consistency

    def __post_init__(self):
        # Initialize default values for attributes that can't be set directly in the dataclass
        if self.attributes is None:
            self.attributes = {}
        if self.children is None:
            self.children = []

    def __repr__(self) -> str:
        tag_str = f'<{self.tag_name}'

        # Add key attributes for desktop element properties
        class_name = self.attributes.get('class name', '')
        element_name = self.attributes.get('element name', '')
        control_type = self.attributes.get('control type', '')
        
        if class_name:
            tag_str += f' class="{class_name}"'
        if element_name:
            tag_str += f' name="{element_name}"'
        if control_type and control_type != self.tag_name:
            tag_str += f' controlType="{control_type}"'
            
        tag_str += '>'

        # Add extra info
        extras = []
        if self.is_interactive:
            extras.append('interactive')
        if self.is_top_element:
            extras.append('top')
        if self.highlight_index is not None:
            extras.append(f'highlight:{self.highlight_index}')
        if self.is_in_viewport:
            extras.append('in-viewport')

        if extras:
            tag_str += f' [{", ".join(extras)}]'

        return tag_str

    @cached_property
    def hash(self) -> HashedElement:
        # Using our locally defined HashedElement class
        import hashlib

        def _string_hash(text: str) -> str:
            """Create a hash from a string"""
            return hashlib.sha256(text.encode()).hexdigest()

        def _attributes_hash(attributes: Dict[str, str]) -> str:
            """Create a hash from element attributes"""
            # Sort keys to ensure consistent hashing
            attributes_string = ''.join(f'{key}={value}' for key, value in sorted(attributes.items()))
            return hashlib.sha256(attributes_string.encode()).hexdigest()
        
        return HashedElement(
            class_name_hash=_string_hash(self.attributes.get('class name', '')),
            element_name_hash=_string_hash(self.attributes.get('element name', '')),
            control_type_hash=_string_hash(self.attributes.get('control type', '') or self.tag_name),
            attributes_hash=_attributes_hash(self.attributes)
        )

    def get_all_text_till_next_clickable_element(self, max_depth: int = -1) -> str:
        text_parts = []

        def collect_text(node: DOMBaseNode, current_depth: int) -> None:
            if max_depth != -1 and current_depth > max_depth:
                return

            # Skip this branch if we hit a highlighted element (except for the current node)
            if isinstance(node, DOMElementNode) and node != self and node.highlight_index is not None:
                return

            if isinstance(node, DOMTextNode):
                text_parts.append(node.text)
            elif isinstance(node, DOMElementNode):
                for child in node.children:
                    collect_text(child, current_depth + 1)

        collect_text(self, 0)
        return '\n'.join(text_parts).strip()

    @time_execution_sync('--clickable_elements_to_string')
    def clickable_elements_to_string(self, include_attributes: list[str] | None = None) -> str:
        """Convert the processed DOM content to a string."""
        formatted_text = []

        def process_node(node: DOMBaseNode, depth: int) -> None:
            if isinstance(node, DOMElementNode):
                # Add element with highlight_index
                if node.highlight_index is not None:
                    attributes_str = ''
                    text = node.get_all_text_till_next_clickable_element()
                    
                    if include_attributes:
                        attributes = []
                        for key in include_attributes:
                            value = node.attributes.get(key)
                            if value and value != node.tag_name and value not in [node.attributes.get('class name', ''), 
                                                                            node.attributes.get('element name', ''),
                                                                            node.attributes.get('control type', '')]:
                                attributes.append(f'{key}="{value}"')
                        
                        if attributes:
                            attributes_str = ' ' + ' '.join(attributes)
                    
                    # Get class name, element name, and control type
                    class_name = node.attributes.get('class name', '')
                    element_name = node.attributes.get('element name', '')
                    control_type = node.attributes.get('control type', '') or node.tag_name
                    
                    line = f'{node.highlight_index}[:]<{control_type}'
                    
                    if class_name:
                        line += f' class="{class_name}"'
                    if element_name:
                        line += f' name="{element_name}"'
                    
                    line += attributes_str
                    
                    if text:
                        line += f'>{text}</{control_type}>'
                    else:
                        line += ' />'
                        
                    formatted_text.append(line)

                # Process children regardless
                for child in node.children:
                    process_node(child, depth + 1)

            elif isinstance(node, DOMTextNode):
                # Add text only if it doesn't have a highlighted parent
                if not node.has_parent_with_highlight_index() and node.is_visible:
                    formatted_text.append(f'_[:]{node.text}')

        process_node(self, 0)
        return '\n'.join(formatted_text)

    def interactive_elements_to_string(self, include_attributes: list[str] | None = None) -> str:
        """Alias for clickable_elements_to_string for compatibility"""
        return self.clickable_elements_to_string(include_attributes)

    def get_file_upload_element(self, check_siblings: bool = True) -> Optional['DOMElementNode']:
        # For desktop, look for file dialog launcher buttons
        control_type = self.attributes.get('control type', '') or self.tag_name
        element_name = self.attributes.get('element name', '')
        
        # Check if current element is a file dialog control
        if control_type == 'Button' and element_name and any(s in element_name.lower() 
                                                       for s in ['open', 'browse', 'upload', 'file']):
            return self

        # Check children
        for child in self.children:
            if isinstance(child, DOMElementNode):
                result = child.get_file_upload_element(check_siblings=False)
                if result:
                    return result

        # Check siblings only for the initial call
        if check_siblings and self.parent:
            for sibling in self.parent.children:
                if sibling is not self and isinstance(sibling, DOMElementNode):
                    result = sibling.get_file_upload_element(check_siblings=False)
                    if result:
                        return result

        return None


# Update forward references to work with the class now that it's defined
DOMElementNode.__annotations__['parent'] = Optional[DOMElementNode]
DOMElementNode.__annotations__['children'] = List[Union[DOMElementNode, DOMTextNode]]

SelectorMap = Dict[int, DOMElementNode]


@dataclass
class DOMState:
    element_tree: Optional[DOMElementNode] = None
    selector_map: Optional[Dict[int, DOMElementNode]] = None

    def __post_init__(self):
        if self.selector_map is None:
            self.selector_map = {}


import gc
import logging
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import uiautomation as auto

logger = logging.getLogger(__name__)

class ElementService:
    def __init__(self, window=None):
        self.window = window
        self.element_index = 1
        self.element_cache = {}
        
        # Common interactive control types for desktop automation
        self.interactive_control_types = [
            "Button", "CheckBox", "ComboBox", "Edit", "Hyperlink", 
            "ListItem", "MenuItem", "RadioButton", "Slider", "Spinner",
            "SplitButton", "Tab", "Text", "ToolBar", "Tree", "TreeItem"
        ]
    
    async def _build_element_data(
        self,
        highlight_elements: bool = False,
        focus_element: Optional[int] = None,
        viewport_expansion: int = 0
    ) -> Tuple[Dict[int, Dict[str, str]], str]:
        """Build element data for desktop automation"""
        self.element_index = 1
        self.xml_root = ET.Element("UI")
        
        if self.window:
            self.xml_root.set("WindowTitle", getattr(self.window, 'Name', '') or '')
            self.xml_root.set("WindowClass", getattr(self.window, 'ClassName', '') or '')
        
        element_map: Dict[int, Dict[str, str]] = {}
        
        try:
            if self.window:
                # Process the window and its children
                await self._process_element_tree(
                    self.window,
                    self.xml_root,
                    element_map,
                    highlight_elements,
                    focus_element,
                    viewport_expansion
                )
            
            xml_string = ET.tostring(self.xml_root, encoding="utf-8").decode("utf-8")
            return element_map, xml_string
            
        except Exception:
            return {}, "<UI></UI>"
        finally:
            self.element_cache.clear()
            gc.collect()
    
    async def _process_element_tree(
        self,
        element: auto.Control,
        xml_parent: ET.Element,
        element_map: Dict[int, Dict[str, str]],
        highlight_elements: bool,
        focus_element: Optional[int],
        viewport_expansion: int
    ) -> None:
        """Process desktop UI element tree"""
        try:
            if not element or not hasattr(element, 'Exists') or not element.Exists():
                return
                
            element_name = getattr(element, 'Name', '')
            element_type = getattr(element, 'ControlTypeName', '')
                
            xml_element = ET.SubElement(xml_parent, "Element")
            
            # Add element properties
            props_to_add = [
                ('ControlTypeName', 'ControlType'),
                ('ClassName', 'ClassName'),
                ('Name', 'Name'),
                ('AutomationId', 'AutomationId')
            ]
            
            for attr, xml_attr in props_to_add:
                if hasattr(element, attr):
                    value = getattr(element, attr)
                    if value:
                        xml_element.set(xml_attr, str(value))
            
            # Add position and size
            center_x = None
            center_y = None
            try:
                rect = element.BoundingRectangle
                if rect:
                    xml_element.set("X", str(rect.left))
                    xml_element.set("Y", str(rect.top))
                    xml_element.set("Width", str(rect.width()))
                    xml_element.set("Height", str(rect.height()))
                    
                    # Calculate center point for position-based clicking
                    center_x = rect.left + rect.width() // 2
                    center_y = rect.top + rect.height() // 2
            except Exception:
                pass
            
            # Determine interactivity based on element properties and behavior
            is_interactive = False
            
            # Check if element has a name (making it potentially interactive)
            if element_name:
                is_interactive = True
            
            # Check if element has an interactive control type
            if element_type and element_type in self.interactive_control_types:
                is_interactive = True
            
            # Check if element has keyboard focus capability
            try:
                if hasattr(element, 'IsKeyboardFocusable') and element.IsKeyboardFocusable:
                    is_interactive = True
            except Exception:
                pass
            
            # Check for interaction patterns
            patterns_to_check = [
                "InvokePattern", "SelectionItemPattern", "ExpandCollapsePattern",
                "ValuePattern", "RangeValuePattern", "ScrollItemPattern"
            ]
            
            for pattern in patterns_to_check:
                pattern_getter = f"Get{pattern}"
                if hasattr(element, pattern_getter):
                    try:
                        pattern_obj = getattr(element, pattern_getter)()
                        if pattern_obj:
                            is_interactive = True
                            break
                    except Exception:
                        continue
            
            xml_element.set("IsInteractive", str(is_interactive).lower())
            
            if is_interactive:
                xml_element.set("Index", str(self.element_index))
                
                element_info = {
                    "class name": getattr(element, 'ClassName', '') or '',
                    "element name": element_name or '',
                    "control type": element_type or '',
                    "automation_id": getattr(element, 'AutomationId', '') or ''
                }
                
                # Add position for fallback clicking
                if center_x is not None and center_y is not None:
                    element_info["position"] = (center_x, center_y)
                
                element_map[self.element_index] = element_info
                self.element_index += 1
                
                if highlight_elements:
                    self._try_highlight_element(element, self.element_index == focus_element)
            
            # Process children
            try:
                children = element.GetChildren()
                for child in children:
                    await self._process_element_tree(
                        child,
                        xml_element,
                        element_map,
                        highlight_elements,
                        focus_element,
                        viewport_expansion
                    )
            except Exception:
                pass
                
        except Exception:
            pass
    
    def _is_interactive(self, element: auto.Control) -> bool:
        """Determine if an element is interactive"""
        try:
            # Check control type
            if hasattr(element, 'ControlTypeName') and element.ControlTypeName in self.interactive_control_types:
                return True
                
            # Check keyboard focusability
            if hasattr(element, 'IsKeyboardFocusable') and element.IsKeyboardFocusable:
                return True
                
            # Check UI patterns
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
            
            return False
        except Exception:
            return False
    
    def _try_highlight_element(self, element: auto.Control, is_focus: bool = False) -> None:
        """Highlight an element for debugging"""
        try:
            if hasattr(element, 'DrawHighlight'):
                element.DrawHighlight()
        except Exception:
            pass
    
    async def _get_available_windows(self) -> List[str]:
        """Get list of available windows"""
        try:
            import uiautomation as auto
            windows = []
            
            desktop = auto.GetRootControl()
            for win in desktop.GetChildren():
                if win.Name:
                    windows.append(f"{win.Name}")
            
            return windows
        except Exception:
            return []