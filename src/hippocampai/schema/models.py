"""Schema definition models for custom entity and relationship types."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class AttributeType(str, Enum):
    """Supported attribute types."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    LIST = "list"
    DICT = "dict"
    ANY = "any"


class AttributeDefinition(BaseModel):
    """Definition of an attribute for an entity or relationship.

    Attributes:
        name: Attribute name
        type: Attribute type
        required: Whether the attribute is required
        default: Default value if not provided
        description: Human-readable description
        constraints: Optional constraints (min, max, pattern, enum, etc.)
    """

    name: str
    type: AttributeType = AttributeType.STRING
    required: bool = False
    default: Optional[Any] = None
    description: str = ""
    constraints: dict[str, Any] = Field(default_factory=dict)

    def validate_value(self, value: Any) -> tuple[bool, str]:
        """Validate a value against this attribute definition.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required
        if value is None:
            if self.required:
                return False, f"Attribute '{self.name}' is required"
            return True, ""

        # Type validation
        type_valid, type_error = self._validate_type(value)
        if not type_valid:
            return False, type_error

        # Constraint validation
        constraint_valid, constraint_error = self._validate_constraints(value)
        if not constraint_valid:
            return False, constraint_error

        return True, ""

    def _validate_type(self, value: Any) -> tuple[bool, str]:
        """Validate value type."""
        if self.type == AttributeType.ANY:
            return True, ""

        type_map: dict[AttributeType, type | tuple[type, ...]] = {
            AttributeType.STRING: str,
            AttributeType.INTEGER: int,
            AttributeType.FLOAT: (int, float),
            AttributeType.BOOLEAN: bool,
            AttributeType.LIST: list,
            AttributeType.DICT: dict,
        }

        if self.type == AttributeType.DATETIME:
            if isinstance(value, datetime):
                return True, ""
            if isinstance(value, str):
                try:
                    datetime.fromisoformat(value.replace("Z", "+00:00"))
                    return True, ""
                except ValueError:
                    return False, f"Attribute '{self.name}' must be a valid datetime"
            return False, f"Attribute '{self.name}' must be a datetime"

        expected_type = type_map.get(self.type)
        if expected_type and not isinstance(value, expected_type):
            return False, f"Attribute '{self.name}' must be of type {self.type.value}"

        return True, ""

    def _validate_constraints(self, value: Any) -> tuple[bool, str]:
        """Validate value against constraints."""
        if not self.constraints:
            return True, ""

        # Min/max for numbers
        if self.type in (AttributeType.INTEGER, AttributeType.FLOAT):
            if "min" in self.constraints and value < self.constraints["min"]:
                return False, f"Attribute '{self.name}' must be >= {self.constraints['min']}"
            if "max" in self.constraints and value > self.constraints["max"]:
                return False, f"Attribute '{self.name}' must be <= {self.constraints['max']}"

        # Min/max length for strings
        if self.type == AttributeType.STRING:
            if "min_length" in self.constraints and len(value) < self.constraints["min_length"]:
                return False, f"Attribute '{self.name}' must have at least {self.constraints['min_length']} characters"
            if "max_length" in self.constraints and len(value) > self.constraints["max_length"]:
                return False, f"Attribute '{self.name}' must have at most {self.constraints['max_length']} characters"
            if "pattern" in self.constraints:
                import re
                if not re.match(self.constraints["pattern"], value):
                    return False, f"Attribute '{self.name}' must match pattern {self.constraints['pattern']}"

        # Enum constraint
        if "enum" in self.constraints:
            if value not in self.constraints["enum"]:
                return False, f"Attribute '{self.name}' must be one of {self.constraints['enum']}"

        return True, ""


class EntityTypeDefinition(BaseModel):
    """Definition of a custom entity type.

    Attributes:
        name: Entity type name (e.g., "person", "organization")
        description: Human-readable description
        attributes: List of attribute definitions
        required_attributes: List of required attribute names
        parent_type: Optional parent type for inheritance
    """

    name: str
    description: str = ""
    attributes: list[AttributeDefinition] = Field(default_factory=list)
    parent_type: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_attribute(self, name: str) -> Optional[AttributeDefinition]:
        """Get attribute definition by name."""
        for attr in self.attributes:
            if attr.name == name:
                return attr
        return None

    def get_required_attributes(self) -> list[str]:
        """Get list of required attribute names."""
        return [attr.name for attr in self.attributes if attr.required]


class RelationshipTypeDefinition(BaseModel):
    """Definition of a custom relationship type.

    Attributes:
        name: Relationship type name (e.g., "works_at", "knows")
        description: Human-readable description
        from_entity_types: Allowed source entity types (empty = any)
        to_entity_types: Allowed target entity types (empty = any)
        attributes: List of attribute definitions for the relationship
        bidirectional: Whether the relationship is bidirectional
    """

    name: str
    description: str = ""
    from_entity_types: list[str] = Field(default_factory=list)
    to_entity_types: list[str] = Field(default_factory=list)
    attributes: list[AttributeDefinition] = Field(default_factory=list)
    bidirectional: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    def allows_from_type(self, entity_type: str) -> bool:
        """Check if this relationship allows the given source type."""
        if not self.from_entity_types:
            return True
        return entity_type in self.from_entity_types

    def allows_to_type(self, entity_type: str) -> bool:
        """Check if this relationship allows the given target type."""
        if not self.to_entity_types:
            return True
        return entity_type in self.to_entity_types


class SchemaDefinition(BaseModel):
    """Complete schema definition with entity and relationship types.

    Attributes:
        name: Schema name
        version: Schema version
        description: Human-readable description
        entity_types: List of entity type definitions
        relationship_types: List of relationship type definitions
        created_at: When the schema was created
        metadata: Additional metadata
    """

    name: str = "default"
    version: str = "1.0.0"
    description: str = ""
    entity_types: list[EntityTypeDefinition] = Field(default_factory=list)
    relationship_types: list[RelationshipTypeDefinition] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_entity_type(self, name: str) -> Optional[EntityTypeDefinition]:
        """Get entity type definition by name."""
        for et in self.entity_types:
            if et.name == name:
                return et
        return None

    def get_relationship_type(self, name: str) -> Optional[RelationshipTypeDefinition]:
        """Get relationship type definition by name."""
        for rt in self.relationship_types:
            if rt.name == name:
                return rt
        return None

    def has_entity_type(self, name: str) -> bool:
        """Check if entity type exists."""
        return self.get_entity_type(name) is not None

    def has_relationship_type(self, name: str) -> bool:
        """Check if relationship type exists."""
        return self.get_relationship_type(name) is not None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SchemaDefinition":
        """Create schema from dictionary."""
        entity_types = []
        for et_data in data.get("entity_types", []):
            attributes = [
                AttributeDefinition(**attr_data)
                for attr_data in et_data.get("attributes", [])
            ]
            et_data["attributes"] = attributes
            entity_types.append(EntityTypeDefinition(**et_data))

        relationship_types = []
        for rt_data in data.get("relationship_types", []):
            attributes = [
                AttributeDefinition(**attr_data)
                for attr_data in rt_data.get("attributes", [])
            ]
            rt_data["attributes"] = attributes
            relationship_types.append(RelationshipTypeDefinition(**rt_data))

        return cls(
            name=data.get("name", "default"),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            entity_types=entity_types,
            relationship_types=relationship_types,
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert schema to dictionary."""
        return self.model_dump(mode="json")
