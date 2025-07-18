import hashlib
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

# Fix: Use fully qualified import path
from dom.history_tree_processor.views import ElementHistoryElement, DOMHistoryElement
from dom.views import DOMElementNode

# Define HashedElement locally to avoid import issues
@dataclass
class HashedDomElement:
    """Hashed representation of a DOM element for comparison"""
    class_name_hash: str
    element_name_hash: str
    control_type_hash: str
    attributes_hash: str


class HistoryTreeProcessor:
    """
    Operations on the UI elements for desktop automation
    
    Handles hashing and comparison of UI elements to track them between states
    """

    @staticmethod
    def convert_xml_element_to_history_element(element: DOMElementNode) -> DOMHistoryElement:
        """Convert a DOM element to a history element"""
        return DOMHistoryElement(
            tag_name=element.tag_name,
            xpath=element.xpath,
            highlight_index=element.highlight_index,
            entire_parent_branch_path=[],  # Simplified for this example
            attributes=element.attributes,
            class_name=element.attributes.get('class name', ''),
            element_name=element.attributes.get('element name', ''),
            control_type=element.attributes.get('control type', '')
        )

    @staticmethod
    def find_history_element_in_tree(history_element: DOMHistoryElement, 
                                    element_tree: DOMElementNode) -> Optional[DOMElementNode]:
        """Find a history element in the current element tree"""
        # Simple implementation to match by highlight_index
        def search_node(node):
            if node.highlight_index == history_element.highlight_index:
                return node
                
            for child in node.children:
                if isinstance(child, DOMElementNode):
                    result = search_node(child)
                    if result:
                        return result
            return None
            
        return search_node(element_tree)

    @staticmethod
    def convert_element_to_history_element(element: Dict[str, str]) -> ElementHistoryElement:
        """Convert an element from the element map to a history element"""
        return ElementHistoryElement(
            class_name=element.get('class name', ''),
            element_name=element.get('element name', ''),
            control_type=element.get('control type', ''),
            xpath=element.get('xpath', ''),
            element_index=element.get('element_index', None),
            attributes=element
        )

    @staticmethod
    def find_history_element_in_map(history_element: ElementHistoryElement, 
                                  element_map: Dict[int, Dict[str, str]]) -> Optional[int]:
        """Find a history element in the current element map"""
        hashed_history_element = HistoryTreeProcessor._hash_history_element(history_element)
        
        for index, element in element_map.items():
            hashed_element = HistoryTreeProcessor._hash_element(element)
            if hashed_element == hashed_history_element:
                return index
                
        return None

    @staticmethod
    def compare_history_element_and_element(history_element: ElementHistoryElement, 
                                          element: Dict[str, str]) -> bool:
        """Compare a history element with a current element"""
        hashed_history_element = HistoryTreeProcessor._hash_history_element(history_element)
        hashed_element = HistoryTreeProcessor._hash_element(element)
        
        return hashed_history_element == hashed_element

    @staticmethod
    def _hash_history_element(history_element: ElementHistoryElement) -> HashedDomElement:
        """Create a hash representation of a history element"""
        class_name_hash = HistoryTreeProcessor._string_hash(history_element.class_name)
        element_name_hash = HistoryTreeProcessor._string_hash(history_element.element_name)
        control_type_hash = HistoryTreeProcessor._string_hash(history_element.control_type)
        attributes_hash = HistoryTreeProcessor._attributes_hash(history_element.attributes)
        
        return HashedDomElement(class_name_hash, element_name_hash, control_type_hash, attributes_hash)

    @staticmethod
    def _hash_element(element: Dict[str, str]) -> HashedDomElement:
        """Create a hash representation of an element"""
        class_name_hash = HistoryTreeProcessor._string_hash(element.get('class name', ''))
        element_name_hash = HistoryTreeProcessor._string_hash(element.get('element name', ''))
        control_type_hash = HistoryTreeProcessor._string_hash(element.get('control type', ''))
        attributes_hash = HistoryTreeProcessor._attributes_hash(element)
        
        return HashedDomElement(class_name_hash, element_name_hash, control_type_hash, attributes_hash)

    @staticmethod
    def _string_hash(text: str) -> str:
        """Create a hash from a string"""
        return hashlib.sha256(text.encode()).hexdigest()

    @staticmethod
    def _attributes_hash(attributes: Dict[str, str]) -> str:
        """Create a hash from element attributes"""
        # Sort keys to ensure consistent hashing
        attributes_string = ''.join(f'{key}={value}' for key, value in sorted(attributes.items()))
        return hashlib.sha256(attributes_string.encode()).hexdigest()
    
    @staticmethod
    def _hash_dom_element(element) -> HashedDomElement:
        """Create a hash from a DOMElementNode"""
        class_name_hash = HistoryTreeProcessor._string_hash(element.attributes.get('class', ''))
        element_name_hash = HistoryTreeProcessor._string_hash(element.attributes.get('name', ''))
        control_type_hash = HistoryTreeProcessor._string_hash(element.tag_name)
        attributes_hash = HistoryTreeProcessor._attributes_hash(element.attributes)
        
        return HashedDomElement(class_name_hash, element_name_hash, control_type_hash, attributes_hash)
    
    @staticmethod
    def create_xml_hash(xml_content: str) -> str:
        """Create a hash from XML content"""
        return hashlib.sha256(xml_content.encode()).hexdigest()
    
    @staticmethod
    def compare_xml_hashes(hash1: str, hash2: str) -> bool:
        """Compare two XML hashes"""
        return hash1 == hash2