"""Tests for custom schema support."""

import json
import tempfile

import pytest

from hippocampai.schema.models import (
    AttributeDefinition,
    AttributeType,
    EntityTypeDefinition,
    RelationshipTypeDefinition,
    SchemaDefinition,
)
from hippocampai.schema.registry import SchemaRegistry
from hippocampai.schema.validator import (
    SchemaValidator,
    ValidationResult,
    create_default_schema,
)


class TestAttributeDefinition:
    """Tests for AttributeDefinition."""

    def test_string_attribute(self) -> None:
        """Test string attribute validation."""
        attr = AttributeDefinition(
            name="name",
            type=AttributeType.STRING,
            required=True,
        )

        valid, error = attr.validate_value("John")
        assert valid is True

        valid, error = attr.validate_value(123)
        assert valid is False
        assert "string" in error.lower()

    def test_integer_attribute(self) -> None:
        """Test integer attribute validation."""
        attr = AttributeDefinition(
            name="age",
            type=AttributeType.INTEGER,
            constraints={"min": 0, "max": 150},
        )

        valid, error = attr.validate_value(25)
        assert valid is True

        valid, error = attr.validate_value(-1)
        assert valid is False

        valid, error = attr.validate_value(200)
        assert valid is False

    def test_required_attribute(self) -> None:
        """Test required attribute validation."""
        attr = AttributeDefinition(
            name="name",
            type=AttributeType.STRING,
            required=True,
        )

        valid, error = attr.validate_value(None)
        assert valid is False
        assert "required" in error.lower()

    def test_optional_attribute(self) -> None:
        """Test optional attribute validation."""
        attr = AttributeDefinition(
            name="nickname",
            type=AttributeType.STRING,
            required=False,
        )

        valid, error = attr.validate_value(None)
        assert valid is True

    def test_enum_constraint(self) -> None:
        """Test enum constraint validation."""
        attr = AttributeDefinition(
            name="status",
            type=AttributeType.STRING,
            constraints={"enum": ["active", "inactive", "pending"]},
        )

        valid, error = attr.validate_value("active")
        assert valid is True

        valid, error = attr.validate_value("unknown")
        assert valid is False
        assert "one of" in error.lower()

    def test_string_length_constraints(self) -> None:
        """Test string length constraints."""
        attr = AttributeDefinition(
            name="code",
            type=AttributeType.STRING,
            constraints={"min_length": 3, "max_length": 10},
        )

        valid, error = attr.validate_value("ABC")
        assert valid is True

        valid, error = attr.validate_value("AB")
        assert valid is False

        valid, error = attr.validate_value("ABCDEFGHIJK")
        assert valid is False


class TestEntityTypeDefinition:
    """Tests for EntityTypeDefinition."""

    def test_create_entity_type(self) -> None:
        """Test creating an entity type."""
        entity_type = EntityTypeDefinition(
            name="person",
            description="A person entity",
            attributes=[
                AttributeDefinition(name="name", type=AttributeType.STRING, required=True),
                AttributeDefinition(name="age", type=AttributeType.INTEGER),
            ],
        )

        assert entity_type.name == "person"
        assert len(entity_type.attributes) == 2
        assert entity_type.get_attribute("name") is not None
        assert entity_type.get_attribute("unknown") is None

    def test_get_required_attributes(self) -> None:
        """Test getting required attributes."""
        entity_type = EntityTypeDefinition(
            name="person",
            attributes=[
                AttributeDefinition(name="name", required=True),
                AttributeDefinition(name="email", required=True),
                AttributeDefinition(name="phone", required=False),
            ],
        )

        required = entity_type.get_required_attributes()
        assert "name" in required
        assert "email" in required
        assert "phone" not in required


class TestRelationshipTypeDefinition:
    """Tests for RelationshipTypeDefinition."""

    def test_create_relationship_type(self) -> None:
        """Test creating a relationship type."""
        rel_type = RelationshipTypeDefinition(
            name="works_at",
            from_entity_types=["person"],
            to_entity_types=["organization"],
        )

        assert rel_type.name == "works_at"
        assert rel_type.allows_from_type("person") is True
        assert rel_type.allows_from_type("organization") is False
        assert rel_type.allows_to_type("organization") is True

    def test_any_endpoint_types(self) -> None:
        """Test relationship with any endpoint types."""
        rel_type = RelationshipTypeDefinition(
            name="related_to",
            from_entity_types=[],  # Any
            to_entity_types=[],  # Any
        )

        assert rel_type.allows_from_type("anything") is True
        assert rel_type.allows_to_type("anything") is True


class TestSchemaDefinition:
    """Tests for SchemaDefinition."""

    def test_create_schema(self) -> None:
        """Test creating a schema."""
        schema = SchemaDefinition(
            name="test_schema",
            version="1.0.0",
            entity_types=[
                EntityTypeDefinition(name="person"),
                EntityTypeDefinition(name="organization"),
            ],
            relationship_types=[
                RelationshipTypeDefinition(name="works_at"),
            ],
        )

        assert schema.name == "test_schema"
        assert schema.has_entity_type("person") is True
        assert schema.has_entity_type("unknown") is False
        assert schema.has_relationship_type("works_at") is True

    def test_from_dict(self) -> None:
        """Test creating schema from dictionary."""
        data = {
            "name": "custom",
            "version": "2.0.0",
            "entity_types": [
                {
                    "name": "customer",
                    "attributes": [
                        {"name": "id", "type": "string", "required": True},
                    ],
                }
            ],
            "relationship_types": [
                {
                    "name": "purchased",
                    "from_entity_types": ["customer"],
                    "to_entity_types": ["product"],
                }
            ],
        }

        schema = SchemaDefinition.from_dict(data)
        assert schema.name == "custom"
        assert schema.version == "2.0.0"
        assert schema.has_entity_type("customer") is True


class TestSchemaValidator:
    """Tests for SchemaValidator."""

    @pytest.fixture
    def schema(self) -> SchemaDefinition:
        """Create a test schema."""
        return SchemaDefinition(
            name="test",
            entity_types=[
                EntityTypeDefinition(
                    name="person",
                    attributes=[
                        AttributeDefinition(name="name", type=AttributeType.STRING, required=True),
                        AttributeDefinition(name="age", type=AttributeType.INTEGER),
                    ],
                ),
            ],
            relationship_types=[
                RelationshipTypeDefinition(
                    name="knows",
                    from_entity_types=["person"],
                    to_entity_types=["person"],
                ),
            ],
        )

    def test_validate_valid_entity(self, schema: SchemaDefinition) -> None:
        """Test validating a valid entity."""
        validator = SchemaValidator(schema)
        result = validator.validate_entity(
            "person",
            {"name": "John", "age": 30},
        )

        assert result.valid is True
        assert len(result.errors) == 0

    def test_validate_invalid_entity(self, schema: SchemaDefinition) -> None:
        """Test validating an invalid entity."""
        validator = SchemaValidator(schema)
        result = validator.validate_entity(
            "person",
            {"age": 30},  # Missing required 'name'
        )

        assert result.valid is False
        assert len(result.errors) > 0

    def test_validate_unknown_entity_type(self, schema: SchemaDefinition) -> None:
        """Test validating unknown entity type."""
        validator = SchemaValidator(schema)

        # Non-strict: warning only
        result = validator.validate_entity("unknown", {}, strict=False)
        assert result.valid is True
        assert len(result.warnings) > 0

        # Strict: error
        result = validator.validate_entity("unknown", {}, strict=True)
        assert result.valid is False

    def test_validate_valid_relationship(self, schema: SchemaDefinition) -> None:
        """Test validating a valid relationship."""
        validator = SchemaValidator(schema)
        result = validator.validate_relationship(
            "knows",
            "person",
            "person",
        )

        assert result.valid is True

    def test_validate_invalid_relationship_endpoints(self, schema: SchemaDefinition) -> None:
        """Test validating relationship with invalid endpoints."""
        validator = SchemaValidator(schema)
        result = validator.validate_relationship(
            "knows",
            "organization",  # Not allowed
            "person",
        )

        assert result.valid is False


class TestSchemaRegistry:
    """Tests for SchemaRegistry."""

    def test_default_schema(self) -> None:
        """Test that default schema is registered."""
        registry = SchemaRegistry()
        assert "default" in registry.list_schemas()
        assert registry.active_schema.name == "default"

    def test_register_schema(self) -> None:
        """Test registering a custom schema."""
        registry = SchemaRegistry()
        custom = SchemaDefinition(name="custom", version="1.0.0")
        registry.register_schema(custom)

        assert "custom" in registry.list_schemas()
        assert registry.get_schema("custom") is not None

    def test_set_active_schema(self) -> None:
        """Test setting active schema."""
        registry = SchemaRegistry()
        custom = SchemaDefinition(name="custom")
        registry.register_schema(custom)
        registry.set_active_schema("custom")

        assert registry.active_schema.name == "custom"

    def test_load_schema_from_json(self) -> None:
        """Test loading schema from JSON file."""
        schema_data = {
            "name": "json_schema",
            "version": "1.0.0",
            "entity_types": [{"name": "item", "attributes": []}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(schema_data, f)
            f.flush()

            registry = SchemaRegistry()
            schema = registry.load_schema_from_file(f.name)

            assert schema.name == "json_schema"
            assert "json_schema" in registry.list_schemas()

    def test_load_schema_from_dict(self) -> None:
        """Test loading schema from dictionary."""
        registry = SchemaRegistry()
        schema = registry.load_schema_from_dict(
            {
                "name": "dict_schema",
                "entity_types": [],
            }
        )

        assert schema.name == "dict_schema"
        assert "dict_schema" in registry.list_schemas()

    def test_validate_entity_via_registry(self) -> None:
        """Test entity validation through registry."""
        registry = SchemaRegistry()
        result = registry.validate_entity(
            "person",
            {"name": "John"},
        )

        assert isinstance(result, ValidationResult)

    def test_add_entity_type(self) -> None:
        """Test adding entity type to schema."""
        registry = SchemaRegistry()
        new_type = EntityTypeDefinition(
            name="custom_entity",
            attributes=[
                AttributeDefinition(name="id", required=True),
            ],
        )
        registry.add_entity_type(new_type)

        assert registry.active_schema.has_entity_type("custom_entity")

    def test_add_relationship_type(self) -> None:
        """Test adding relationship type to schema."""
        registry = SchemaRegistry()
        new_type = RelationshipTypeDefinition(
            name="custom_rel",
            from_entity_types=["person"],
            to_entity_types=["organization"],
        )
        registry.add_relationship_type(new_type)

        assert registry.active_schema.has_relationship_type("custom_rel")


class TestDefaultSchema:
    """Tests for the default schema."""

    def test_default_schema_entity_types(self) -> None:
        """Test default schema has expected entity types."""
        schema = create_default_schema()

        assert schema.has_entity_type("person")
        assert schema.has_entity_type("organization")
        assert schema.has_entity_type("location")
        assert schema.has_entity_type("product")
        assert schema.has_entity_type("skill")

    def test_default_schema_relationship_types(self) -> None:
        """Test default schema has expected relationship types."""
        schema = create_default_schema()

        assert schema.has_relationship_type("works_at")
        assert schema.has_relationship_type("located_in")
        assert schema.has_relationship_type("knows")
        assert schema.has_relationship_type("has_skill")
        assert schema.has_relationship_type("related_to")

    def test_works_at_relationship_constraints(self) -> None:
        """Test works_at relationship has correct constraints."""
        schema = create_default_schema()
        works_at = schema.get_relationship_type("works_at")

        assert works_at is not None
        assert works_at.allows_from_type("person")
        assert not works_at.allows_from_type("organization")
        assert works_at.allows_to_type("organization")
        assert not works_at.allows_to_type("person")
