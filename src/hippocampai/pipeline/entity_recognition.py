"""Entity recognition and knowledge extraction.

This module provides:
- Named Entity Recognition (NER)
- Entity linking and resolution
- Entity profile building
- Relationship extraction
- Entity timeline tracking
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Set, Tuple
from pydantic import BaseModel, Field
from enum import Enum
from collections import defaultdict
import logging
import re

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Types of entities."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    TIME = "time"
    MONEY = "money"
    PRODUCT = "product"
    EVENT = "event"
    SKILL = "skill"
    TOPIC = "topic"
    OTHER = "other"


class RelationType(str, Enum):
    """Types of relationships between entities."""
    WORKS_AT = "works_at"
    LOCATED_IN = "located_in"
    FOUNDED_BY = "founded_by"
    MANAGES = "manages"
    KNOWS = "knows"
    STUDIED_AT = "studied_at"
    PART_OF = "part_of"
    SIMILAR_TO = "similar_to"
    RELATED_TO = "related_to"


class Entity(BaseModel):
    """Represents an extracted entity."""
    text: str = Field(..., description="Entity text as it appears")
    type: EntityType = Field(..., description="Entity type")
    confidence: float = Field(..., ge=0.0, le=1.0)
    entity_id: str = Field(..., description="Unique entity identifier")
    canonical_name: Optional[str] = Field(None, description="Normalized entity name")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    first_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    mention_count: int = Field(default=1)


class EntityRelationship(BaseModel):
    """Represents a relationship between two entities."""
    from_entity_id: str
    to_entity_id: str
    relation_type: RelationType
    confidence: float = Field(..., ge=0.0, le=1.0)
    context: Optional[str] = None
    extracted_from: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EntityProfile(BaseModel):
    """Complete profile of an entity with all known information."""
    entity_id: str
    canonical_name: str
    type: EntityType
    aliases: Set[str] = Field(default_factory=set)
    attributes: Dict[str, Any] = Field(default_factory=dict)
    relationships: List[EntityRelationship] = Field(default_factory=list)
    mentions: List[Dict[str, Any]] = Field(default_factory=list)
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    first_seen: datetime
    last_seen: datetime
    mention_count: int = 0


class EntityRecognizer:
    """Recognizes and tracks entities across memories."""

    def __init__(self, llm=None):
        """Initialize entity recognizer.

        Args:
            llm: Language model for advanced entity recognition
        """
        self.llm = llm

        # Entity storage
        self.entities: Dict[str, EntityProfile] = {}
        self.entity_index: Dict[str, str] = {}  # text -> entity_id mapping

        # Build recognition patterns
        self.entity_patterns = self._build_entity_patterns()
        self.relationship_patterns = self._build_relationship_patterns()

    def _build_entity_patterns(self) -> Dict[EntityType, List[str]]:
        """Build regex patterns for entity recognition."""
        return {
            EntityType.PERSON: [
                r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b',  # First Last
                r'\b((?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+)\b',  # Title Name
            ],
            EntityType.ORGANIZATION: [
                r'\b([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\s+(?:Inc|Corp|LLC|Ltd|Company|University|College|Institute|Foundation))\b',
                r'\b((?:Google|Microsoft|Amazon|Apple|Facebook|Meta|Tesla|Netflix|Uber|Airbnb|Spotify|GitHub|LinkedIn))\b',
            ],
            EntityType.LOCATION: [
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,?\s+(?:California|New York|Texas|Florida|Illinois|Washington|Massachusetts|Colorado|Oregon|Virginia))\b',
                r'\b((?:San Francisco|New York City|Los Angeles|Chicago|Boston|Seattle|Austin|Denver|Portland|Miami|NYC))\b',
                r'\b([A-Z][a-z]+\s+(?:City|State|Country|County|Province))\b',
            ],
            EntityType.DATE: [
                r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',
                r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b',
                r'\b(\d{4})\b',  # Year only
                r'\b((?:yesterday|today|tomorrow|last\s+week|next\s+month))\b',
            ],
            EntityType.SKILL: [
                r'\b(Python|Java|JavaScript|TypeScript|C\+\+|Ruby|Go|Rust|Swift|Kotlin|PHP|SQL|HTML|CSS)\b',
                r'\b(machine\s+learning|artificial\s+intelligence|data\s+science|deep\s+learning|NLP|computer\s+vision)\b',
            ],
            EntityType.PRODUCT: [
                r'\b(iPhone|iPad|MacBook|Android|Windows|Linux|AWS|Azure|GCP)\b',
            ]
        }

    def _build_relationship_patterns(self) -> List[Dict[str, Any]]:
        """Build patterns for relationship extraction."""
        return [
            {
                "pattern": r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:works?|working|employed)\s+(?:at|for)\s+([A-Z][A-Za-z\s&]+)',
                "relation": RelationType.WORKS_AT,
                "from_group": 1,
                "to_group": 2,
                "from_type": EntityType.PERSON,
                "to_type": EntityType.ORGANIZATION
            },
            {
                "pattern": r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:lives?|living|based)\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
                "relation": RelationType.LOCATED_IN,
                "from_group": 1,
                "to_group": 2,
                "from_type": EntityType.PERSON,
                "to_type": EntityType.LOCATION
            },
            {
                "pattern": r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:studied|graduated)\s+(?:at|from)\s+([A-Z][A-Za-z\s]+(?:University|College))',
                "relation": RelationType.STUDIED_AT,
                "from_group": 1,
                "to_group": 2,
                "from_type": EntityType.PERSON,
                "to_type": EntityType.ORGANIZATION
            },
            {
                "pattern": r'\b([A-Z][A-Za-z\s]+)\s+(?:is\s+)?(?:located|based)\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
                "relation": RelationType.LOCATED_IN,
                "from_group": 1,
                "to_group": 2,
                "from_type": EntityType.ORGANIZATION,
                "to_type": EntityType.LOCATION
            }
        ]

    def extract_entities(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Entity]:
        """Extract entities from text.

        Args:
            text: Input text
            context: Optional context (memory_id, user_id, etc.)

        Returns:
            List of extracted entities
        """
        entities = []

        # Pattern-based extraction
        pattern_entities = self._extract_entities_pattern_based(text)
        entities.extend(pattern_entities)

        # LLM-based extraction for better accuracy
        if self.llm and len(entities) < 5:
            llm_entities = self._extract_entities_llm(text)
            entities.extend(llm_entities)

        # Deduplicate and resolve entities
        entities = self._deduplicate_entities(entities)

        # Update entity index and profiles
        for entity in entities:
            self._update_entity_profile(entity, text, context)

        return entities

    def _extract_entities_pattern_based(self, text: str) -> List[Entity]:
        """Extract entities using regex patterns."""
        entities = []

        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)

                for match in matches:
                    entity_text = match.group(1) if match.lastindex >= 1 else match.group(0)
                    entity_text = entity_text.strip()

                    if not entity_text or len(entity_text) < 2:
                        continue

                    # Generate entity ID
                    entity_id = self._generate_entity_id(entity_text, entity_type)

                    entity = Entity(
                        text=entity_text,
                        type=entity_type,
                        confidence=0.80,  # Pattern-based confidence
                        entity_id=entity_id,
                        canonical_name=entity_text,
                        metadata={
                            "extraction_method": "pattern",
                            "pattern": pattern
                        }
                    )
                    entities.append(entity)

        return entities

    def _extract_entities_llm(self, text: str) -> List[Entity]:
        """Extract entities using LLM."""
        if not self.llm:
            return []

        prompt = f"""Extract named entities from this text. For each entity, identify:
1. The entity text
2. Type (person, organization, location, date, skill, product, etc.)
3. Confidence (0.0 to 1.0)

Text: {text}

Return entities in this format:
ENTITY: [entity text] | TYPE: [type] | CONFIDENCE: [0.0-1.0]

Entities:"""

        try:
            response = self.llm.generate(prompt, max_tokens=300)

            entities = []
            for line in response.strip().split('\n'):
                if not line.strip() or not line.startswith('ENTITY:'):
                    continue

                parts = line.split('|')
                if len(parts) < 2:
                    continue

                entity_text = parts[0].replace('ENTITY:', '').strip()
                type_str = parts[1].replace('TYPE:', '').strip().lower() if len(parts) > 1 else 'other'
                confidence_str = parts[2].replace('CONFIDENCE:', '').strip() if len(parts) > 2 else '0.7'

                # Map type
                entity_type = EntityType.OTHER
                for et in EntityType:
                    if et.value in type_str:
                        entity_type = et
                        break

                # Parse confidence
                try:
                    confidence = float(confidence_str)
                except:
                    confidence = 0.7

                entity_id = self._generate_entity_id(entity_text, entity_type)

                entity = Entity(
                    text=entity_text,
                    type=entity_type,
                    confidence=min(max(confidence, 0.0), 1.0),
                    entity_id=entity_id,
                    canonical_name=entity_text,
                    metadata={
                        "extraction_method": "llm"
                    }
                )
                entities.append(entity)

            return entities

        except Exception as e:
            logger.warning(f"LLM entity extraction failed: {e}")
            return []

    def _generate_entity_id(self, text: str, entity_type: EntityType) -> str:
        """Generate unique entity ID."""
        # Normalize text for ID
        normalized = re.sub(r'\s+', '_', text.lower().strip())
        normalized = re.sub(r'[^\w_]', '', normalized)
        return f"{entity_type.value}_{normalized}"

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities."""
        seen = {}
        unique_entities = []

        for entity in entities:
            key = (entity.entity_id, entity.type)
            if key not in seen:
                seen[key] = entity
                unique_entities.append(entity)
            else:
                # Update confidence if higher
                if entity.confidence > seen[key].confidence:
                    seen[key].confidence = entity.confidence

        return unique_entities

    def extract_relationships(
        self,
        text: str,
        entities: Optional[List[Entity]] = None
    ) -> List[EntityRelationship]:
        """Extract relationships between entities.

        Args:
            text: Input text
            entities: Previously extracted entities (optional)

        Returns:
            List of entity relationships
        """
        relationships = []

        # Pattern-based relationship extraction
        for pattern_info in self.relationship_patterns:
            pattern = pattern_info["pattern"]
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                from_text = match.group(pattern_info["from_group"]).strip()
                to_text = match.group(pattern_info["to_group"]).strip()

                from_entity_id = self._generate_entity_id(from_text, pattern_info["from_type"])
                to_entity_id = self._generate_entity_id(to_text, pattern_info["to_type"])

                relationship = EntityRelationship(
                    from_entity_id=from_entity_id,
                    to_entity_id=to_entity_id,
                    relation_type=pattern_info["relation"],
                    confidence=0.85,
                    context=match.group(0),
                    extracted_from=text[:100]
                )
                relationships.append(relationship)

        return relationships

    def _update_entity_profile(
        self,
        entity: Entity,
        source_text: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Update or create entity profile."""
        if entity.entity_id in self.entities:
            # Update existing profile
            profile = self.entities[entity.entity_id]
            profile.last_seen = datetime.now(timezone.utc)
            profile.mention_count += 1

            # Add alias if different
            if entity.text != profile.canonical_name:
                profile.aliases.add(entity.text)

            # Add mention
            mention = {
                "text": entity.text,
                "source": source_text[:200],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "context": context or {}
            }
            profile.mentions.append(mention)

        else:
            # Create new profile
            profile = EntityProfile(
                entity_id=entity.entity_id,
                canonical_name=entity.canonical_name or entity.text,
                type=entity.type,
                aliases={entity.text},
                first_seen=entity.first_seen,
                last_seen=entity.last_seen,
                mention_count=1,
                mentions=[{
                    "text": entity.text,
                    "source": source_text[:200],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "context": context or {}
                }]
            )
            self.entities[entity.entity_id] = profile

        # Update index
        self.entity_index[entity.text.lower()] = entity.entity_id

    def get_entity_profile(self, entity_id: str) -> Optional[EntityProfile]:
        """Get complete profile for an entity."""
        return self.entities.get(entity_id)

    def search_entities(
        self,
        query: str,
        entity_type: Optional[EntityType] = None,
        min_mentions: int = 1
    ) -> List[EntityProfile]:
        """Search for entities.

        Args:
            query: Search query
            entity_type: Filter by entity type
            min_mentions: Minimum number of mentions

        Returns:
            List of matching entity profiles
        """
        results = []
        query_lower = query.lower()

        for profile in self.entities.values():
            # Type filter
            if entity_type and profile.type != entity_type:
                continue

            # Mention filter
            if profile.mention_count < min_mentions:
                continue

            # Text matching
            if (query_lower in profile.canonical_name.lower() or
                any(query_lower in alias.lower() for alias in profile.aliases)):
                results.append(profile)

        # Sort by mention count
        results.sort(key=lambda p: p.mention_count, reverse=True)
        return results

    def get_entity_timeline(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get chronological timeline of entity mentions."""
        profile = self.get_entity_profile(entity_id)
        if not profile:
            return []

        # Sort mentions by timestamp
        timeline = sorted(profile.mentions, key=lambda m: m.get("timestamp", ""))
        return timeline

    def get_related_entities(
        self,
        entity_id: str,
        relation_type: Optional[RelationType] = None
    ) -> List[Tuple[str, RelationType]]:
        """Get entities related to a given entity.

        Args:
            entity_id: Source entity ID
            relation_type: Filter by relationship type

        Returns:
            List of (related_entity_id, relation_type) tuples
        """
        related = []
        profile = self.get_entity_profile(entity_id)

        if not profile:
            return related

        for rel in profile.relationships:
            if relation_type and rel.relation_type != relation_type:
                continue

            if rel.from_entity_id == entity_id:
                related.append((rel.to_entity_id, rel.relation_type))
            elif rel.to_entity_id == entity_id:
                related.append((rel.from_entity_id, rel.relation_type))

        return related
