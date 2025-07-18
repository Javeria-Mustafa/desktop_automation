from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from pydantic import BaseModel


@dataclass
class HashedDomElement:
	"""Hashed representation of a DOM element for comparison"""
	class_name_hash: str
	element_name_hash: str
	control_type_hash: str
	attributes_hash: str


@dataclass
class ElementHistoryElement:
	"""Historical representation of a DOM element for storing in history"""
	
	class_name: str
	element_name: str
	control_type: str
	automation_id: Optional[str] = None
	xpath: Optional[str] = None
	element_index: Optional[int] = None
	attributes: Dict[str, str] = field(default_factory=dict)
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary for serialization"""
		return {
			"class_name": self.class_name,
			"element_name": self.element_name,
			"control_type": self.control_type,
			"automation_id": self.automation_id,
			"xpath": self.xpath,
			"element_index": self.element_index,
			"attributes": self.attributes
		}
	
	@classmethod
	def from_dict(cls, data: Dict[str, Any]) -> 'ElementHistoryElement':
		"""Create from dictionary"""
		return cls(
			class_name=data.get("class_name", ""),
			element_name=data.get("element_name", ""),
			control_type=data.get("control_type", ""),
			automation_id=data.get("automation_id"),
			xpath=data.get("xpath"),
			element_index=data.get("element_index"),
			attributes=data.get("attributes", {})
		)


class Coordinates(BaseModel):
	x: int
	y: int


class CoordinateSet(BaseModel):
	top_left: Coordinates
	top_right: Coordinates
	bottom_left: Coordinates
	bottom_right: Coordinates
	center: Coordinates
	width: int
	height: int


class BoundingBox(BaseModel):
	left: int
	top: int
	width: int
	height: int


@dataclass
class DOMHistoryElement:
	tag_name: str
	xpath: str
	highlight_index: Optional[int]
	entire_parent_branch_path: list[str]
	attributes: dict[str, str]
	shadow_root: bool = False
	css_selector: Optional[str] = None
	page_coordinates: Optional[CoordinateSet] = None
	viewport_coordinates: Optional[CoordinateSet] = None
	viewport_info: Optional[BoundingBox] = None
	
	# Desktop specific fields
	class_name: Optional[str] = None
	element_name: Optional[str] = None
	control_type: Optional[str] = None
	automation_id: Optional[str] = None
	runtime_id: Optional[str] = None

	def to_dict(self) -> dict:
		page_coordinates = self.page_coordinates.model_dump() if self.page_coordinates else None
		viewport_coordinates = self.viewport_coordinates.model_dump() if self.viewport_coordinates else None
		viewport_info = self.viewport_info.model_dump() if self.viewport_info else None

		return {
			'tag_name': self.tag_name,
			'xpath': self.xpath,
			'highlight_index': self.highlight_index,
			'entire_parent_branch_path': self.entire_parent_branch_path,
			'attributes': self.attributes,
			'shadow_root': self.shadow_root,
			'css_selector': self.css_selector,
			'page_coordinates': page_coordinates,
			'viewport_coordinates': viewport_coordinates,
			'viewport_info': viewport_info,
			'class_name': self.class_name,
			'element_name': self.element_name,
			'control_type': self.control_type,
			'automation_id': self.automation_id,
			'runtime_id': self.runtime_id
		}