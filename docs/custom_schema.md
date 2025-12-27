# Custom Schema Support

HippocampAI supports custom entity and relationship schemas, allowing you to define domain-specific types without modifying code.

## Overview

The schema system provides:
- **Entity type definitions** with typed attributes and constraints
- **Relationship type definitions** with endpoint restrictions
- **Validation** of memory payloads against schemas
- **Backwards-compatible default schema** when none is provided

## Quick Start

```python
from hippocampai.schema import (
    SchemaRegistry,
    SchemaDefinition,
    EntityTypeDefinition,
    RelationshipTypeDefinition,
    AttributeDefinition,
    AttributeType,
)

# Get the global registry (includes default schema)
registry = SchemaRegistry()

# Validate an entity against the default schema
result = registry.validate_entity(
    "person",
    {"name": "John Doe", "email": "john@example.com"}
)
print(f"Valid: {result.valid}")
```

## Schema Definition

### From Python

```python
schema = SchemaDefinition(
    name="crm_schema",
    version="1.0.0",
    description="CRM domain schema",
    entity_types=[
        EntityTypeDefinition(
            name="customer",
            description="A customer entity",
            attributes=[
                AttributeDefinition(
                    name="customer_id",
                    type=AttributeType.STRING,
                    required=True,
                    description="Unique customer identifier"
                ),
                AttributeDefinition(
                    name="name",
                    type=AttributeType.STRING,
                    required=True
                ),
                AttributeDefinition(
                    name="tier",
                    type=AttributeType.STRING,
                    constraints={"enum": ["bronze", "silver", "gold", "platinum"]}
                ),
                AttributeDefinition(
                    name="lifetime_value",
                    type=AttributeType.FLOAT,
                    constraints={"min": 0}
                ),
            ],
        ),
        EntityTypeDefinition(
            name="product",
            attributes=[
                AttributeDefinition(name="sku", type=AttributeType.STRING, required=True),
                AttributeDefinition(name="price", type=AttributeType.FLOAT),
            ],
        ),
    ],
    relationship_types=[
        RelationshipTypeDefinition(
            name="purchased",
            from_entity_types=["customer"],
            to_entity_types=["product"],
            attributes=[
                AttributeDefinition(name="quantity", type=AttributeType.INTEGER),
                AttributeDefinition(name="purchase_date", type=AttributeType.DATETIME),
            ],
        ),
    ],
)

registry.register_schema(schema)
registry.set_active_schema("crm_schema")
```

### From JSON File

Create `schemas/crm.json`:

```json
{
  "name": "crm_schema",
  "version": "1.0.0",
  "entity_types": [
    {
      "name": "customer",
      "attributes": [
        {"name": "customer_id", "type": "string", "required": true},
        {"name": "name", "type": "string", "required": true},
        {"name": "tier", "type": "string", "constraints": {"enum": ["bronze", "silver", "gold"]}}
      ]
    }
  ],
  "relationship_types": [
    {
      "name": "purchased",
      "from_entity_types": ["customer"],
      "to_entity_types": ["product"]
    }
  ]
}
```

Load it:

```python
schema = registry.load_schema_from_file("schemas/crm.json", set_active=True)
```

### From YAML File

Create `schemas/crm.yaml`:

```yaml
name: crm_schema
version: "1.0.0"

entity_types:
  - name: customer
    attributes:
      - name: customer_id
        type: string
        required: true
      - name: name
        type: string
        required: true
      - name: tier
        type: string
        constraints:
          enum: [bronze, silver, gold]

relationship_types:
  - name: purchased
    from_entity_types: [customer]
    to_entity_types: [product]
```

Load it:

```python
schema = registry.load_schema_from_file("schemas/crm.yaml", set_active=True)
```

## Attribute Types

| Type | Python Type | Description |
|------|-------------|-------------|
| `STRING` | `str` | Text values |
| `INTEGER` | `int` | Whole numbers |
| `FLOAT` | `float` | Decimal numbers |
| `BOOLEAN` | `bool` | True/False |
| `DATETIME` | `datetime` | Date and time |
| `DATE` | `date` | Date only |
| `LIST` | `list` | Array of values |
| `DICT` | `dict` | Key-value mapping |
| `ANY` | `Any` | Any type (no validation) |

## Constraints

### Numeric Constraints

```python
AttributeDefinition(
    name="age",
    type=AttributeType.INTEGER,
    constraints={"min": 0, "max": 150}
)
```

### String Constraints

```python
AttributeDefinition(
    name="code",
    type=AttributeType.STRING,
    constraints={"min_length": 3, "max_length": 10}
)
```

### Enum Constraints

```python
AttributeDefinition(
    name="status",
    type=AttributeType.STRING,
    constraints={"enum": ["active", "inactive", "pending"]}
)
```

## Validation

### Entity Validation

```python
result = registry.validate_entity(
    "customer",
    {"customer_id": "C001", "name": "Acme Corp", "tier": "gold"}
)

if result.valid:
    print("Entity is valid")
else:
    for error in result.errors:
        print(f"Error: {error}")
```

### Relationship Validation

```python
result = registry.validate_relationship(
    "purchased",
    from_entity_type="customer",
    to_entity_type="product",
    attributes={"quantity": 5}
)
```

### Strict Mode

By default, unknown entity types produce warnings. Use strict mode to reject them:

```python
# Non-strict (default): unknown types produce warnings
result = registry.validate_entity("unknown_type", {}, strict=False)
# result.valid = True, result.warnings = ["Unknown entity type..."]

# Strict: unknown types produce errors
result = registry.validate_entity("unknown_type", {}, strict=True)
# result.valid = False, result.errors = ["Unknown entity type..."]
```

## Default Schema

When no custom schema is provided, HippocampAI uses a default schema with common entity and relationship types:

### Default Entity Types

| Type | Description | Required Attributes |
|------|-------------|---------------------|
| `person` | A person | `name` |
| `organization` | A company or org | `name` |
| `location` | A place | `name` |
| `event` | An event | `name` |
| `product` | A product | `name` |
| `skill` | A skill or ability | `name` |
| `concept` | An abstract concept | `name` |

### Default Relationship Types

| Type | From | To | Description |
|------|------|-----|-------------|
| `works_at` | person | organization | Employment |
| `located_in` | any | location | Location |
| `knows` | person | person | Personal connection |
| `has_skill` | person | skill | Skill possession |
| `related_to` | any | any | Generic relation |
| `part_of` | any | any | Membership/containment |
| `created` | person, organization | any | Creation |

## Extending the Schema

Add types to an existing schema at runtime:

```python
from hippocampai.schema import EntityTypeDefinition, AttributeDefinition

# Add a new entity type
registry.add_entity_type(
    EntityTypeDefinition(
        name="project",
        attributes=[
            AttributeDefinition(name="name", required=True),
            AttributeDefinition(name="status", constraints={"enum": ["active", "completed"]}),
        ],
    )
)

# Add a new relationship type
registry.add_relationship_type(
    RelationshipTypeDefinition(
        name="assigned_to",
        from_entity_types=["person"],
        to_entity_types=["project"],
    )
)
```

## Exporting Schemas

Export a schema to a file:

```python
registry.export_schema("output/my_schema.yaml", format="yaml")
registry.export_schema("output/my_schema.json", format="json")
```

## Global Registry

Access the global schema registry:

```python
from hippocampai.schema import get_schema_registry, set_schema_registry

# Get global instance
registry = get_schema_registry()

# Replace global instance
custom_registry = SchemaRegistry()
set_schema_registry(custom_registry)
```

## API Reference

### SchemaRegistry

| Method | Description |
|--------|-------------|
| `register_schema(schema)` | Register a schema |
| `set_active_schema(name)` | Set active schema |
| `get_schema(name)` | Get schema by name |
| `list_schemas()` | List all schema names |
| `load_schema_from_file(path)` | Load from JSON/YAML |
| `load_schema_from_dict(data)` | Load from dictionary |
| `validate_entity(type, attrs)` | Validate entity |
| `validate_relationship(...)` | Validate relationship |
| `add_entity_type(type)` | Add entity type |
| `add_relationship_type(type)` | Add relationship type |
| `export_schema(path, format)` | Export to file |

### ValidationResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `valid` | `bool` | Whether validation passed |
| `errors` | `list[str]` | Error messages |
| `warnings` | `list[str]` | Warning messages |

## Common Errors

### "Unknown entity type"

The entity type is not defined in the schema. Either:
1. Add the entity type to your schema
2. Use `strict=False` to allow unknown types (produces warning)

### "Missing required attribute"

A required attribute was not provided. Check the entity type definition.

### "Value must be one of..."

An enum constraint was violated. Use one of the allowed values.

### "Value must be at least/at most..."

A numeric constraint was violated. Adjust the value to be within bounds.
