"""Schema validation for entities and relationships."""

import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from hippocampai.schema.models import (
    EntityTypeDefinition,
    RelationshipTypeDefinition,
    SchemaDefinition,
)

logger = logging.getLogger(__name__)


class ValidationError(BaseModel):
    """A single validation error."""

    field: str
    message: str
    value: Optional[Any] = None


class ValidationResult(BaseModel):
    """Result of schema validation."""

    valid: bool
    errors: list[ValidationError] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def add_error(self, field: str, message: str, value: Any = None) -> None:
        """Add a validation error."""
        self.errors.append(ValidationError(field=field, message=message, value=value))
        self.valid = False

    def add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(message)


class SchemaValidator:
    """Validates entities and relationships against a schema."""

    def __init__(self, schema: SchemaDefinition) -> None:
        """Initialize validator with a schema.

        Args:
            schema: Schema definition to validate against
        """
        self.schema = schema

    def validate_entity(
        self,
        entity_type: str,
        attributes: dict[str, Any],
        strict: bool = False,
    ) -> ValidationResult:
        """Validate an entity against the schema.

        Args:
            entity_type: Type of the entity
            attributes: Entity attributes to validate
            strict: If True, reject unknown attributes

        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult(valid=True)

        # Get entity type definition
        type_def = self.schema.get_entity_type(entity_type)
        if type_def is None:
            if strict:
                result.add_error("entity_type", f"Unknown entity type: {entity_type}")
            else:
                result.add_warning(f"Entity type '{entity_type}' not defined in schema")
            return result

        # Validate attributes
        self._validate_attributes(type_def.attributes, attributes, result, strict)

        return result

    def validate_relationship(
        self,
        relationship_type: str,
        from_entity_type: str,
        to_entity_type: str,
        attributes: Optional[dict[str, Any]] = None,
        strict: bool = False,
    ) -> ValidationResult:
        """Validate a relationship against the schema.

        Args:
            relationship_type: Type of the relationship
            from_entity_type: Source entity type
            to_entity_type: Target entity type
            attributes: Relationship attributes to validate
            strict: If True, reject unknown types/attributes

        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult(valid=True)
        attributes = attributes or {}

        # Get relationship type definition
        type_def = self.schema.get_relationship_type(relationship_type)
        if type_def is None:
            if strict:
                result.add_error(
                    "relationship_type",
                    f"Unknown relationship type: {relationship_type}",
                )
            else:
                result.add_warning(
                    f"Relationship type '{relationship_type}' not defined in schema"
                )
            return result

        # Validate endpoint types
        if not type_def.allows_from_type(from_entity_type):
            result.add_error(
                "from_entity_type",
                f"Entity type '{from_entity_type}' not allowed as source for '{relationship_type}'",
                from_entity_type,
            )

        if not type_def.allows_to_type(to_entity_type):
            result.add_error(
                "to_entity_type",
                f"Entity type '{to_entity_type}' not allowed as target for '{relationship_type}'",
                to_entity_type,
            )

        # Validate attributes
        self._validate_attributes(type_def.attributes, attributes, result, strict)

        return result

    def _validate_attributes(
        self,
        attr_defs: list,
        attributes: dict[str, Any],
        result: ValidationResult,
        strict: bool,
    ) -> None:
        """Validate attributes against definitions."""
        defined_attrs = {attr.name for attr in attr_defs}

        # Check for unknown attributes
        if strict:
            for attr_name in attributes:
                if attr_name not in defined_attrs:
                    result.add_error(
                        attr_name,
                        f"Unknown attribute: {attr_name}",
                        attributes[attr_name],
                    )

        # Validate each defined attribute
        for attr_def in attr_defs:
            value = attributes.get(attr_def.name)

            # Use default if not provided
            if value is None and attr_def.default is not None:
                continue

            # Validate
            valid, error = attr_def.validate_value(value)
            if not valid:
                result.add_error(attr_def.name, error, value)

    def validate_memory_payload(
        self,
        payload: dict[str, Any],
        strict: bool = False,
    ) -> ValidationResult:
        """Validate a memory payload that may contain entities and relationships.

        Args:
            payload: Memory payload with optional entities/relationships
            strict: If True, reject unknown types

        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True)

        # Validate entities if present
        entities = payload.get("entities", {})
        for entity_type, entity_list in entities.items():
            if not isinstance(entity_list, list):
                entity_list = [entity_list]

            for entity in entity_list:
                if isinstance(entity, dict):
                    entity_result = self.validate_entity(
                        entity_type,
                        entity.get("attributes", {}),
                        strict,
                    )
                    result.errors.extend(entity_result.errors)
                    result.warnings.extend(entity_result.warnings)
                    if not entity_result.valid:
                        result.valid = False

        # Validate relationships if present
        relationships = payload.get("relationships", [])
        for rel in relationships:
            if isinstance(rel, dict):
                rel_result = self.validate_relationship(
                    rel.get("type", "unknown"),
                    rel.get("from_type", "unknown"),
                    rel.get("to_type", "unknown"),
                    rel.get("attributes", {}),
                    strict,
                )
                result.errors.extend(rel_result.errors)
                result.warnings.extend(rel_result.warnings)
                if not rel_result.valid:
                    result.valid = False

        return result


def create_default_schema() -> SchemaDefinition:
    """Create the default schema with common entity and relationship types.

    Returns:
        Default SchemaDefinition
    """
    from hippocampai.schema.models import AttributeDefinition, AttributeType

    # Default entity types
    entity_types = [
        EntityTypeDefinition(
            name="person",
            description="A person entity",
            attributes=[
                AttributeDefinition(name="name", type=AttributeType.STRING, required=True),
                AttributeDefinition(name="email", type=AttributeType.STRING),
                AttributeDefinition(name="title", type=AttributeType.STRING),
            ],
        ),
        EntityTypeDefinition(
            name="organization",
            description="An organization entity",
            attributes=[
                AttributeDefinition(name="name", type=AttributeType.STRING, required=True),
                AttributeDefinition(name="industry", type=AttributeType.STRING),
                AttributeDefinition(name="location", type=AttributeType.STRING),
            ],
        ),
        EntityTypeDefinition(
            name="location",
            description="A location entity",
            attributes=[
                AttributeDefinition(name="name", type=AttributeType.STRING, required=True),
                AttributeDefinition(name="type", type=AttributeType.STRING),
                AttributeDefinition(name="country", type=AttributeType.STRING),
            ],
        ),
        EntityTypeDefinition(
            name="product",
            description="A product entity",
            attributes=[
                AttributeDefinition(name="name", type=AttributeType.STRING, required=True),
                AttributeDefinition(name="category", type=AttributeType.STRING),
            ],
        ),
        EntityTypeDefinition(
            name="skill",
            description="A skill or technology",
            attributes=[
                AttributeDefinition(name="name", type=AttributeType.STRING, required=True),
                AttributeDefinition(name="category", type=AttributeType.STRING),
                AttributeDefinition(name="proficiency", type=AttributeType.STRING),
            ],
        ),
    ]

    # Default relationship types
    relationship_types = [
        RelationshipTypeDefinition(
            name="works_at",
            description="Person works at organization",
            from_entity_types=["person"],
            to_entity_types=["organization"],
            attributes=[
                AttributeDefinition(name="role", type=AttributeType.STRING),
                AttributeDefinition(name="start_date", type=AttributeType.DATETIME),
                AttributeDefinition(name="end_date", type=AttributeType.DATETIME),
            ],
        ),
        RelationshipTypeDefinition(
            name="located_in",
            description="Entity is located in a location",
            from_entity_types=["person", "organization"],
            to_entity_types=["location"],
        ),
        RelationshipTypeDefinition(
            name="knows",
            description="Person knows another person",
            from_entity_types=["person"],
            to_entity_types=["person"],
            bidirectional=True,
        ),
        RelationshipTypeDefinition(
            name="has_skill",
            description="Person has a skill",
            from_entity_types=["person"],
            to_entity_types=["skill"],
            attributes=[
                AttributeDefinition(name="proficiency", type=AttributeType.STRING),
                AttributeDefinition(name="years", type=AttributeType.INTEGER),
            ],
        ),
        RelationshipTypeDefinition(
            name="part_of",
            description="Entity is part of another entity",
            from_entity_types=[],  # Any
            to_entity_types=[],  # Any
        ),
        RelationshipTypeDefinition(
            name="related_to",
            description="General relationship between entities",
            from_entity_types=[],  # Any
            to_entity_types=[],  # Any
            bidirectional=True,
        ),
    ]

    return SchemaDefinition(
        name="default",
        version="1.0.0",
        description="Default HippocampAI schema with common entity and relationship types",
        entity_types=entity_types,
        relationship_types=relationship_types,
    )
