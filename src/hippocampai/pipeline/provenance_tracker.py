"""Memory provenance and lineage tracking system."""

import logging
from typing import Optional

from hippocampai.adapters.llm_base import BaseLLM
from hippocampai.models.memory import Memory
from hippocampai.models.provenance import (
    Citation,
    MemoryLineage,
    MemorySource,
    ProvenanceChain,
    QualityMetrics,
    TransformationType,
)

logger = logging.getLogger(__name__)


class ProvenanceTracker:
    """
    Tracks memory provenance and lineage throughout the memory lifecycle.

    Features:
    - Source tracking: Where memories come from
    - Lineage tracking: Parent-child relationships
    - Transformation history: How memories evolved
    - Citation management: Link to original sources
    - Quality scoring: Assess memory quality
    """

    def __init__(self, llm: Optional[BaseLLM] = None):
        """
        Initialize provenance tracker.

        Args:
            llm: Optional LLM for quality assessment and citation extraction
        """
        self.llm = llm

    def init_lineage(
        self,
        memory: Memory,
        source: MemorySource = MemorySource.CONVERSATION,
        created_by: Optional[str] = None,
        parent_ids: Optional[list[str]] = None,
        citations: Optional[list[Citation]] = None,
    ) -> MemoryLineage:
        """
        Initialize lineage for a new memory.

        Args:
            memory: Memory object
            source: Source of the memory
            created_by: Who/what created it
            parent_ids: Parent memory IDs if derived
            citations: Initial citations

        Returns:
            MemoryLineage object
        """
        lineage = MemoryLineage(
            memory_id=memory.id,
            source=source,
            created_by=created_by or memory.user_id,
            parent_memory_ids=parent_ids or [],
            citations=citations or [],
        )

        # Add initial transformation
        lineage.add_transformation(
            transformation_type=TransformationType.CREATED,
            description=f"Memory created from {source.value}",
        )

        # Store in memory metadata
        self._attach_lineage_to_memory(memory, lineage)

        return lineage

    def track_merge(
        self,
        merged_memory: Memory,
        parent_memories: list[Memory],
        merge_strategy: str = "auto",
    ) -> MemoryLineage:
        """
        Track a memory merge operation.

        Args:
            merged_memory: New merged memory
            parent_memories: Original memories that were merged
            merge_strategy: Strategy used for merging

        Returns:
            Updated lineage
        """
        parent_ids = [m.id for m in parent_memories]

        lineage = MemoryLineage(
            memory_id=merged_memory.id,
            source=MemorySource.MERGE,
            created_by=merged_memory.user_id,
            parent_memory_ids=parent_ids,
        )

        lineage.add_transformation(
            transformation_type=TransformationType.MERGED,
            parent_ids=parent_ids,
            description=f"Merged {len(parent_memories)} memories using {merge_strategy} strategy",
            metadata={"merge_strategy": merge_strategy, "parent_count": len(parent_memories)},
        )

        # Inherit citations from parents
        for parent in parent_memories:
            parent_lineage = self._extract_lineage_from_memory(parent)
            if parent_lineage and parent_lineage.citations:
                lineage.citations.extend(parent_lineage.citations)

        self._attach_lineage_to_memory(merged_memory, lineage)
        return lineage

    def track_refinement(
        self,
        refined_memory: Memory,
        original_memory: Memory,
        refinement_type: str = "quality",
    ) -> MemoryLineage:
        """
        Track a memory refinement operation.

        Args:
            refined_memory: Refined version
            original_memory: Original memory
            refinement_type: Type of refinement applied

        Returns:
            Updated lineage
        """
        # Get original lineage
        original_lineage = self._extract_lineage_from_memory(original_memory)

        if original_lineage:
            lineage = original_lineage
            lineage.memory_id = refined_memory.id
        else:
            lineage = MemoryLineage(
                memory_id=refined_memory.id,
                source=MemorySource.REFINEMENT,
                parent_memory_ids=[original_memory.id],
            )

        lineage.add_transformation(
            transformation_type=TransformationType.REFINED,
            parent_ids=[original_memory.id],
            description=f"Memory refined: {refinement_type}",
            metadata={"refinement_type": refinement_type},
        )

        self._attach_lineage_to_memory(refined_memory, lineage)
        return lineage

    def track_inference(
        self,
        inferred_memory: Memory,
        source_memories: list[Memory],
        inference_method: str = "llm",
        confidence: float = 0.8,
    ) -> MemoryLineage:
        """
        Track an inferred memory.

        Args:
            inferred_memory: New inferred memory
            source_memories: Memories used for inference
            inference_method: Method used for inference
            confidence: Confidence in the inference

        Returns:
            Lineage for inferred memory
        """
        parent_ids = [m.id for m in source_memories]

        lineage = MemoryLineage(
            memory_id=inferred_memory.id,
            source=MemorySource.INFERENCE,
            parent_memory_ids=parent_ids,
        )

        lineage.add_transformation(
            transformation_type=TransformationType.INFERRED,
            parent_ids=parent_ids,
            description=f"Inferred from {len(source_memories)} memories using {inference_method}",
            metadata={
                "inference_method": inference_method,
                "confidence": confidence,
                "source_count": len(source_memories),
            },
        )

        # Add citations to source memories
        for source_mem in source_memories:
            lineage.add_citation(
                source_type="memory",
                source_id=source_mem.id,
                source_text=source_mem.text[:200],
                confidence=confidence,
            )

        self._attach_lineage_to_memory(inferred_memory, lineage)
        return lineage

    def track_conflict_resolution(
        self,
        resolved_memory: Memory,
        conflicting_memories: list[Memory],
        resolution_strategy: str,
        winner_id: Optional[str] = None,
    ) -> MemoryLineage:
        """
        Track conflict resolution.

        Args:
            resolved_memory: Memory after resolution
            conflicting_memories: Original conflicting memories
            resolution_strategy: Strategy used
            winner_id: ID of winning memory if applicable

        Returns:
            Updated lineage
        """
        parent_ids = [m.id for m in conflicting_memories]

        lineage = MemoryLineage(
            memory_id=resolved_memory.id,
            source=MemorySource.SYSTEM_GENERATED,
            parent_memory_ids=parent_ids,
        )

        lineage.add_transformation(
            transformation_type=TransformationType.CONFLICT_RESOLVED,
            parent_ids=parent_ids,
            description=f"Resolved conflict using {resolution_strategy} strategy",
            metadata={
                "resolution_strategy": resolution_strategy,
                "conflicting_count": len(conflicting_memories),
                "winner_id": winner_id,
            },
        )

        self._attach_lineage_to_memory(resolved_memory, lineage)
        return lineage

    def add_citation(
        self,
        memory: Memory,
        source_type: str,
        source_id: Optional[str] = None,
        source_url: Optional[str] = None,
        source_text: Optional[str] = None,
        confidence: float = 1.0,
    ) -> MemoryLineage:
        """
        Add a citation to a memory.

        Args:
            memory: Memory to add citation to
            source_type: Type of source
            source_id: Source identifier
            source_url: Source URL
            source_text: Excerpt from source
            confidence: Confidence in citation

        Returns:
            Updated lineage
        """
        lineage = self._extract_lineage_from_memory(memory) or MemoryLineage(
            memory_id=memory.id
        )

        lineage.add_citation(
            source_type=source_type,
            source_id=source_id,
            source_url=source_url,
            source_text=source_text,
            confidence=confidence,
        )

        self._attach_lineage_to_memory(memory, lineage)
        return lineage

    def assess_quality(self, memory: Memory, use_llm: bool = True) -> QualityMetrics:
        """
        Assess memory quality.

        Args:
            memory: Memory to assess
            use_llm: Whether to use LLM for assessment

        Returns:
            Quality metrics
        """
        if use_llm and self.llm:
            return self._llm_assess_quality(memory)
        else:
            return self._heuristic_assess_quality(memory)

    def _heuristic_assess_quality(self, memory: Memory) -> QualityMetrics:
        """Simple heuristic-based quality assessment."""
        metrics = QualityMetrics()

        # Specificity: longer, more detailed memories are more specific
        text_len = len(memory.text)
        metrics.specificity = min(1.0, text_len / 200.0)

        # Completeness: based on metadata richness
        metadata_score = len(memory.metadata) / 5.0  # Assume 5 fields is complete
        metrics.completeness = min(1.0, metadata_score)

        # Clarity: penalize very short or very long memories
        if text_len < 10:
            metrics.clarity = 0.3
        elif text_len > 500:
            metrics.clarity = 0.7
        else:
            metrics.clarity = 0.9

        # Verifiability: based on presence of entities/facts
        if memory.entities or memory.facts:
            metrics.verifiability = 0.8
        else:
            metrics.verifiability = 0.5

        # Relevance: based on access count and importance
        access_score = min(1.0, memory.access_count / 10.0)
        importance_score = memory.importance / 10.0
        metrics.relevance = (access_score + importance_score) / 2.0

        metrics.calculate_overall()
        return metrics

    def _llm_assess_quality(self, memory: Memory) -> QualityMetrics:
        """Use LLM to assess memory quality."""
        if not self.llm:
            return self._heuristic_assess_quality(memory)

        prompt = f"""Assess the quality of this memory across multiple dimensions.

Memory: "{memory.text}"
Type: {memory.type}
Metadata: {memory.metadata}

Rate each dimension from 0.0 to 1.0:

1. Specificity: How specific vs vague is this memory?
   - 1.0 = Very specific with concrete details
   - 0.0 = Extremely vague and general

2. Verifiability: Can this memory be verified or fact-checked?
   - 1.0 = Contains verifiable facts/details
   - 0.0 = Purely subjective, no way to verify

3. Completeness: Does it have all necessary information?
   - 1.0 = Complete with full context
   - 0.0 = Missing critical information

4. Clarity: Is it clear and unambiguous?
   - 1.0 = Crystal clear, unambiguous
   - 0.0 = Confusing or contradictory

5. Relevance: Is it likely to be useful/relevant?
   - 1.0 = Highly relevant and useful
   - 0.0 = Trivial or irrelevant

Respond with JSON:
{{
    "specificity": 0.0-1.0,
    "verifiability": 0.0-1.0,
    "completeness": 0.0-1.0,
    "clarity": 0.0-1.0,
    "relevance": 0.0-1.0,
    "explanation": "brief explanation"
}}
"""

        try:
            import json

            response = self.llm.generate(prompt, max_tokens=250, temperature=0.1)

            if "{" in response and "}" in response:
                json_start = response.index("{")
                json_end = response.rindex("}") + 1
                json_str = response[json_start:json_end]
                result = json.loads(json_str)

                metrics = QualityMetrics(
                    specificity=result.get("specificity", 0.5),
                    verifiability=result.get("verifiability", 0.5),
                    completeness=result.get("completeness", 0.5),
                    clarity=result.get("clarity", 0.5),
                    relevance=result.get("relevance", 0.5),
                )
                metrics.calculate_overall()
                return metrics

        except Exception as e:
            logger.warning(f"LLM quality assessment failed: {e}, using heuristics")

        return self._heuristic_assess_quality(memory)

    def extract_citations(self, memory: Memory, context: Optional[str] = None) -> list[Citation]:
        """
        Extract potential citations from memory text or context.

        Args:
            memory: Memory to extract citations from
            context: Optional context (e.g., conversation history)

        Returns:
            List of extracted citations
        """
        if not self.llm:
            return []

        text = memory.text
        if context:
            text = f"Context: {context}\n\nMemory: {memory.text}"

        prompt = f"""Extract potential source citations from this memory.

{text}

Look for:
- References to conversations ("user said", "mentioned")
- Links to documents or URLs
- References to other memories or knowledge
- Source attributions

Return JSON array of citations:
[
    {{
        "source_type": "conversation|document|url|external",
        "source_text": "quoted text or description",
        "confidence": 0.0-1.0
    }}
]

If no citations found, return empty array: []
"""

        try:
            import json

            response = self.llm.generate(prompt, max_tokens=300, temperature=0.1)

            if "[" in response and "]" in response:
                json_start = response.index("[")
                json_end = response.rindex("]") + 1
                json_str = response[json_start:json_end]
                citations_data = json.loads(json_str)

                citations = []
                for cit in citations_data:
                    citation = Citation(
                        source_type=cit.get("source_type", "external"),
                        source_text=cit.get("source_text"),
                        confidence=cit.get("confidence", 0.8),
                    )
                    citations.append(citation)

                return citations

        except Exception as e:
            logger.warning(f"Citation extraction failed: {e}")

        return []

    def build_provenance_chain(
        self, memory: Memory, all_memories: dict[str, Memory]
    ) -> ProvenanceChain:
        """
        Build complete provenance chain for a memory.

        Args:
            memory: Memory to build chain for
            all_memories: Dictionary of all memories (id -> Memory)

        Returns:
            Complete provenance chain
        """
        chain = ProvenanceChain(memory_id=memory.id)

        lineage = self._extract_lineage_from_memory(memory)
        if not lineage:
            # No lineage info, create basic chain
            lineage = MemoryLineage(memory_id=memory.id)

        # Recursively build chain from parents
        visited = set()
        self._build_chain_recursive(lineage, chain, all_memories, visited)

        # Identify root memories
        if chain.chain:
            chain.root_memory_ids = [
                link.memory_id for link in chain.chain if not link.parent_memory_ids
            ]

        return chain

    def _build_chain_recursive(
        self,
        lineage: MemoryLineage,
        chain: ProvenanceChain,
        all_memories: dict[str, Memory],
        visited: set,
    ):
        """Recursively build provenance chain."""
        if lineage.memory_id in visited:
            return

        visited.add(lineage.memory_id)

        # Add parents first (depth-first traversal)
        for parent_id in lineage.parent_memory_ids:
            if parent_id in all_memories and parent_id not in visited:
                parent_memory = all_memories[parent_id]
                parent_lineage = self._extract_lineage_from_memory(parent_memory)
                if parent_lineage:
                    self._build_chain_recursive(parent_lineage, chain, all_memories, visited)

        # Add current lineage
        chain.add_link(lineage)

    def _attach_lineage_to_memory(self, memory: Memory, lineage: MemoryLineage):
        """Attach lineage data to memory metadata."""
        memory.metadata["lineage"] = lineage.model_dump()
        memory.metadata["source"] = lineage.source.value
        memory.metadata["parent_memory_ids"] = lineage.parent_memory_ids
        memory.metadata["is_derived"] = lineage.is_derived()
        memory.metadata["has_citations"] = lineage.has_citations()

    def _extract_lineage_from_memory(self, memory: Memory) -> Optional[MemoryLineage]:
        """Extract lineage from memory metadata."""
        if "lineage" in memory.metadata:
            try:
                return MemoryLineage(**memory.metadata["lineage"])
            except Exception as e:
                logger.warning(f"Failed to parse lineage from metadata: {e}")
        return None

    def record_access(self, memory: Memory, accessor_id: str):
        """Record access to a memory for provenance tracking."""
        lineage = self._extract_lineage_from_memory(memory) or MemoryLineage(
            memory_id=memory.id
        )

        lineage.record_access(accessor_id)
        self._attach_lineage_to_memory(memory, lineage)
