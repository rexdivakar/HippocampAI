"""Schema registry for managing custom schemas."""

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

import yaml  # type: ignore[import-untyped]

from hippocampai.schema.models import (
    EntityTypeDefinition,
    RelationshipTypeDefinition,
    SchemaDefinition,
)
from hippocampai.schema.validator import SchemaValidator, ValidationResult, create_default_schema

logger = logging.getLogger(__name__)


class SchemaRegistry:
    """Registry for managing custom schemas.

    The registry maintains schemas and provides validation capabilities.
    It supports loading schemas from files (JSON/YAML) or dictionaries.
    """

    def __init__(self, default_schema: Optional[SchemaDefinition] = None) -> None:
        """Initialize schema registry.

        Args:
            default_schema: Default schema to use (creates standard if None)
        """
        self._schemas: dict[str, SchemaDefinition] = {}
        self._active_schema_name: str = "default"

        # Register default schema
        if default_schema:
            self.register_schema(default_schema)
        else:
            self.register_schema(create_default_schema())

    @property
    def active_schema(self) -> SchemaDefinition:
        """Get the currently active schema."""
        return self._schemas[self._active_schema_name]

    def register_schema(self, schema: SchemaDefinition) -> None:
        """Register a schema.

        Args:
            schema: Schema to register
        """
        self._schemas[schema.name] = schema
        logger.info(f"Registered schema: {schema.name} v{schema.version}")

    def set_active_schema(self, name: str) -> None:
        """Set the active schema by name.

        Args:
            name: Schema name

        Raises:
            KeyError: If schema not found
        """
        if name not in self._schemas:
            raise KeyError(f"Schema not found: {name}")
        self._active_schema_name = name
        logger.info(f"Active schema set to: {name}")

    def get_schema(self, name: str) -> Optional[SchemaDefinition]:
        """Get a schema by name.

        Args:
            name: Schema name

        Returns:
            Schema or None if not found
        """
        return self._schemas.get(name)

    def list_schemas(self) -> list[str]:
        """List all registered schema names."""
        return list(self._schemas.keys())

    def load_schema_from_file(
        self,
        path: Union[str, Path],
        set_active: bool = False,
    ) -> SchemaDefinition:
        """Load a schema from a JSON or YAML file.

        Args:
            path: Path to schema file
            set_active: If True, set as active schema

        Returns:
            Loaded schema

        Raises:
            FileNotFoundError: If file not found
            ValueError: If file format not supported or invalid
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Schema file not found: {path}")

        # Determine format from extension
        suffix = path.suffix.lower()

        with open(path, "r", encoding="utf-8") as f:
            if suffix in (".yaml", ".yml"):
                try:
                    data = yaml.safe_load(f)
                except Exception as e:
                    raise ValueError(f"Invalid YAML in {path}: {e}")
            elif suffix == ".json":
                try:
                    data = json.load(f)
                except Exception as e:
                    raise ValueError(f"Invalid JSON in {path}: {e}")
            else:
                raise ValueError(f"Unsupported file format: {suffix}")

        # Validate and create schema
        schema = self._parse_schema_data(data, str(path))

        # Register
        self.register_schema(schema)

        if set_active:
            self.set_active_schema(schema.name)

        return schema

    def load_schema_from_dict(
        self,
        data: dict[str, Any],
        set_active: bool = False,
    ) -> SchemaDefinition:
        """Load a schema from a dictionary.

        Args:
            data: Schema data dictionary
            set_active: If True, set as active schema

        Returns:
            Loaded schema
        """
        schema = self._parse_schema_data(data, "dict")
        self.register_schema(schema)

        if set_active:
            self.set_active_schema(schema.name)

        return schema

    def _parse_schema_data(
        self,
        data: dict[str, Any],
        source: str,
    ) -> SchemaDefinition:
        """Parse schema data into a SchemaDefinition.

        Args:
            data: Raw schema data
            source: Source identifier for error messages

        Returns:
            Parsed SchemaDefinition

        Raises:
            ValueError: If schema data is invalid
        """
        if not isinstance(data, dict):
            raise ValueError(f"Schema must be a dictionary, got {type(data)}")

        try:
            return SchemaDefinition.from_dict(data)
        except Exception as e:
            raise ValueError(f"Invalid schema from {source}: {e}")

    def validate_entity(
        self,
        entity_type: str,
        attributes: dict[str, Any],
        schema_name: Optional[str] = None,
        strict: bool = False,
    ) -> ValidationResult:
        """Validate an entity against a schema.

        Args:
            entity_type: Entity type name
            attributes: Entity attributes
            schema_name: Schema to use (active if None)
            strict: If True, reject unknown attributes

        Returns:
            ValidationResult
        """
        schema = self._get_schema_for_validation(schema_name)
        validator = SchemaValidator(schema)
        return validator.validate_entity(entity_type, attributes, strict)

    def validate_relationship(
        self,
        relationship_type: str,
        from_entity_type: str,
        to_entity_type: str,
        attributes: Optional[dict[str, Any]] = None,
        schema_name: Optional[str] = None,
        strict: bool = False,
    ) -> ValidationResult:
        """Validate a relationship against a schema.

        Args:
            relationship_type: Relationship type name
            from_entity_type: Source entity type
            to_entity_type: Target entity type
            attributes: Relationship attributes
            schema_name: Schema to use (active if None)
            strict: If True, reject unknown types

        Returns:
            ValidationResult
        """
        schema = self._get_schema_for_validation(schema_name)
        validator = SchemaValidator(schema)
        return validator.validate_relationship(
            relationship_type,
            from_entity_type,
            to_entity_type,
            attributes,
            strict,
        )

    def validate_memory_payload(
        self,
        payload: dict[str, Any],
        schema_name: Optional[str] = None,
        strict: bool = False,
    ) -> ValidationResult:
        """Validate a memory payload.

        Args:
            payload: Memory payload with entities/relationships
            schema_name: Schema to use (active if None)
            strict: If True, reject unknown types

        Returns:
            ValidationResult
        """
        schema = self._get_schema_for_validation(schema_name)
        validator = SchemaValidator(schema)
        return validator.validate_memory_payload(payload, strict)

    def _get_schema_for_validation(
        self,
        schema_name: Optional[str],
    ) -> SchemaDefinition:
        """Get schema for validation."""
        if schema_name:
            schema = self.get_schema(schema_name)
            if schema is None:
                raise KeyError(f"Schema not found: {schema_name}")
            return schema
        return self.active_schema

    def add_entity_type(
        self,
        entity_type: EntityTypeDefinition,
        schema_name: Optional[str] = None,
    ) -> None:
        """Add an entity type to a schema.

        Args:
            entity_type: Entity type definition
            schema_name: Schema to modify (active if None)
        """
        schema = self._get_schema_for_validation(schema_name)

        # Check for duplicates
        if schema.has_entity_type(entity_type.name):
            logger.warning(f"Replacing existing entity type: {entity_type.name}")
            schema.entity_types = [et for et in schema.entity_types if et.name != entity_type.name]

        schema.entity_types.append(entity_type)
        logger.info(f"Added entity type '{entity_type.name}' to schema '{schema.name}'")

    def add_relationship_type(
        self,
        relationship_type: RelationshipTypeDefinition,
        schema_name: Optional[str] = None,
    ) -> None:
        """Add a relationship type to a schema.

        Args:
            relationship_type: Relationship type definition
            schema_name: Schema to modify (active if None)
        """
        schema = self._get_schema_for_validation(schema_name)

        # Check for duplicates
        if schema.has_relationship_type(relationship_type.name):
            logger.warning(f"Replacing existing relationship type: {relationship_type.name}")
            schema.relationship_types = [
                rt for rt in schema.relationship_types if rt.name != relationship_type.name
            ]

        schema.relationship_types.append(relationship_type)
        logger.info(f"Added relationship type '{relationship_type.name}' to schema '{schema.name}'")

    def export_schema(
        self,
        path: Union[str, Path],
        schema_name: Optional[str] = None,
        format: str = "yaml",
    ) -> None:
        """Export a schema to a file.

        Args:
            path: Output file path
            schema_name: Schema to export (active if None)
            format: Output format ("yaml" or "json")
        """
        schema = self._get_schema_for_validation(schema_name)
        path = Path(path)

        data = schema.to_dict()

        with open(path, "w", encoding="utf-8") as f:
            if format == "yaml":
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            else:
                json.dump(data, f, indent=2, default=str)

        logger.info(f"Exported schema '{schema.name}' to {path}")


# Global registry instance
_global_registry: Optional[SchemaRegistry] = None


def get_schema_registry() -> SchemaRegistry:
    """Get the global schema registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = SchemaRegistry()
    return _global_registry


def set_schema_registry(registry: SchemaRegistry) -> None:
    """Set the global schema registry instance."""
    global _global_registry
    _global_registry = registry
