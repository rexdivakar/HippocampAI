"""Custom schema support for entity and relationship types."""

from hippocampai.schema.models import (
    AttributeDefinition,
    AttributeType,
    EntityTypeDefinition,
    RelationshipTypeDefinition,
    SchemaDefinition,
)
from hippocampai.schema.registry import SchemaRegistry
from hippocampai.schema.validator import SchemaValidator, ValidationError, ValidationResult

__all__ = [
    # Models
    "AttributeDefinition",
    "AttributeType",
    "EntityTypeDefinition",
    "RelationshipTypeDefinition",
    "SchemaDefinition",
    # Registry
    "SchemaRegistry",
    # Validator
    "SchemaValidator",
    "ValidationError",
    "ValidationResult",
]
