import gc
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import time
import uiautomation as auto
import xml.etree.ElementTree as ET

from dom.views import (
    DOMBaseNode,
    DOMElementNode,
    DOMState,
    DOMTextNode,
    SelectorMap,
)

logger = logging.getLogger(__name__)

@dataclass
class ElementViewportInfo:
    width: int
    height: int

@dataclass
class ElementInfo:
    control_type: str
    class_name: str
    element_name: str
    automation_id: Optional[str] = None
    is_visible: bool = True
    is_interactive: bool = False
    is_top_element: bool = False
    is_in_viewport: bool = True
    rect: Optional[Dict[str, int]] = None
    children: List[int] = None
    index: Optional[int] = None
    xml_path: Optional[str] = None
    attributes: Dict[str, str] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.attributes is None:
            self.attributes = {}

class DomService:
    def __init__(self, window: Optional[auto.WindowControl] = None):
        self.window = window
        self.element_cache = {}
        self.xml_root = None
        self.element_index = 1
        
        # Common interactive control types for desktop automation
        self.interactive_control_types = [
            "Button", "CheckBox", "ComboBox", "Edit", "Hyperlink", 
            "ListItem", "MenuItem", "RadioButton", "Slider", "Spinner",
            "SplitButton", "Tab", "Text", "ToolBar", "Tree", "TreeItem"
        ]

    async def get_clickable_elements(
        self,
        highlight_elements: bool = True,
        focus_element: int = -1,
        viewport_expansion: int = 0,
    ) -> DOMState:
        """Get interactive elements from the desktop window"""
        element_map, xml_string = await self._build_element_data(
            highlight_elements, focus_element, viewport_expansion
        )
        
        # Convert XML to DOM structure
        element_tree = self._xml_to_dom_tree(ET.fromstring(xml_string))
        selector_map = self._create_selector_map(element_tree)
        
        return DOMState(element_tree=element_tree, selector_map=selector_map)

    def _xml_to_dom_tree(self, xml_element: ET.Element, parent: Optional[DOMElementNode] = None) -> DOMElementNode:
        """Convert XML element to DOM tree structure"""
        attributes = dict(xml_element.attrib)
        
        # Extract class name, element name, and control type
        control_type = attributes.get('ControlType', '')
        class_name = attributes.get('ClassName', '')
        element_name = attributes.get('Name', '')
        
        # Store these explicitly in attributes for easier access
        attributes['class name'] = class_name
        attributes['element name'] = element_name
        attributes['control type'] = control_type
        
        # Create DOM element node
        element_node = DOMElementNode(
            tag_name=control_type,  # Using control type as tag name
            xpath=attributes.get('AutomationId', '') or self._get_xml_path(xml_element),
            attributes=attributes,
            children=[],
            is_visible=attributes.get('IsVisible', 'true').lower() == 'true',
            is_interactive=attributes.get('IsInteractive', 'false').lower() == 'true',
            is_top_element=attributes.get('IsTopElement', 'false').lower() == 'true',
            highlight_index=int(attributes.get('Index', -1)) if 'Index' in attributes else None,
            parent=parent
        )
        
        # Process children
        for child in xml_element:
            if child.tag == 'Element':
                child_node = self._xml_to_dom_tree(child, parent=element_node)
                element_node.children.append(child_node)
            elif child.tag == 'Text':
                text_node = DOMTextNode(
                    text=child.text or '',
                    is_visible=True,
                    parent=element_node
                )
                element_node.children.append(text_node)
        
        return element_node

    async def _build_element_data(
        self,
        highlight_elements: bool = False,
        focus_element: int = 0,
        viewport_expansion: int = 0,
    ) -> Tuple[dict[int, dict[str, str]], str]:
        """Build element data for desktop automation"""
        self.element_index = 1
        self.xml_root = ET.Element("UI")
        
        if self.window:
            self.xml_root.set("WindowTitle", getattr(self.window, 'Name', '') or '')
            self.xml_root.set("WindowClass", getattr(self.window, 'ClassName', '') or '')
        
        element_map: dict[int, dict[str, str]] = {}
        
        try:
            if self.window:
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
            
        except Exception as e:
            logger.error(f"Error building element data: {str(e)}")
            return {}, ""
        finally:
            self.element_cache.clear()
            gc.collect()

    async def _process_element_tree(
        self,
        element: auto.Control,
        xml_parent: ET.Element,
        element_map: dict[int, dict[str, str]],
        highlight_elements: bool,
        focus_element: int,
        viewport_expansion: int
    ) -> None:
        """Process desktop UI element tree"""
        try:
            if not element or not hasattr(element, 'Exists') or not element.Exists():
                return
                
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
            
            # Add position and size - CRITICAL for fallback clicking
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
            except Exception as e:
                logger.debug(f"Could not get bounding rectangle: {e}")
                center_x = None
                center_y = None
            
            # Check if element is interactive
            is_interactive = self._is_interactive(element)
            xml_element.set("IsInteractive", str(is_interactive).lower())
            
            if is_interactive:
                xml_element.set("Index", str(self.element_index))
                
                element_info = {
                    "class name": getattr(element, 'ClassName', '') or '',
                    "element name": getattr(element, 'Name', '') or '',
                    "control type": getattr(element, 'ControlTypeName', '') or '',
                    "automation_id": getattr(element, 'AutomationId', '') or ''
                }
                
                # Add position for fallback clicking - CRITICAL
                if center_x is not None and center_y is not None:
                    element_info["position"] = (center_x, center_y)
                
                element_map[self.element_index] = element_info
                
                if highlight_elements:
                    self._try_highlight_element(element, self.element_index == focus_element)
                
                self.element_index += 1
            
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
            except Exception as e:
                logger.warning(f"Error processing child elements: {e}")
                
        except Exception as e:
            logger.error(f"Error processing element: {e}")

    def _is_interactive(self, element: auto.Control) -> bool:
        """Determine if a desktop element is interactive"""
        if hasattr(element, 'ControlTypeName') and element.ControlTypeName in self.interactive_control_types:
            return True
            
        if hasattr(element, 'IsKeyboardFocusable') and element.IsKeyboardFocusable:
            return True
            
        # Check for UI Automation patterns that indicate interactivity
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

    def _get_xml_path(self, element: ET.Element) -> str:
        """Get XML path for desktop element"""
        path_parts = []
        current = element
        
        while current is not None:
            tag = current.tag
            if 'Name' in current.attrib:
                tag += f"[@Name='{current.attrib['Name']}']"
            elif 'ControlType' in current.attrib:
                tag += f"[@ControlType='{current.attrib['ControlType']}']"
                
            path_parts.append(tag)
            current = current.getparent() if hasattr(current, 'getparent') else None
        
        path_parts.reverse()
        return '/'.join(path_parts)

    def _try_highlight_element(self, element: auto.Control, is_focus: bool = False) -> None:
        """Highlight desktop element"""
        try:
            if not element or not hasattr(element, 'DrawHighlight'):
                return
                
            # Use UIAutomation's built-in highlight functionality
            if is_focus:
                element.DrawHighlight(color=0x0000FF)  # Blue for focus
            else:
                element.DrawHighlight()  # Default color
                
            # Remove highlight after a short time
            time.sleep(0.2)
            
        except Exception as e:
            logger.debug(f"Highlight failed: {str(e)}")

    def _create_selector_map(self, element_tree: DOMElementNode) -> SelectorMap:
        """Create selector map for desktop elements"""
        selector_map = {}
        
        def process_node(node: DOMBaseNode):
            if isinstance(node, DOMElementNode):
                if node.highlight_index is not None:
                    selector_map[node.highlight_index] = node
                
                for child in node.children:
                    process_node(child)
        
        process_node(element_tree)
        return selector_map