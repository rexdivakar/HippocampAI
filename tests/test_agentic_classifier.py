"""Tests for the agentic memory classifier."""

from unittest.mock import MagicMock, patch

import pytest

from hippocampai.models.memory import MemoryType
from hippocampai.utils.agentic_classifier import (
    MEMORY_TYPE_DEFINITIONS,
    AgenticMemoryClassifier,
    ClassificationConfidence,
    ClassificationResult,
    classify_memory_agentic,
    classify_memory_agentic_with_confidence,
    classify_memory_agentic_with_details,
    clear_agentic_cache,
)


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_basic_result(self):
        """Test creating a basic classification result."""
        result = ClassificationResult(
            memory_type=MemoryType.FACT,
            confidence=0.95,
            confidence_level=ClassificationConfidence.HIGH,
            reasoning="This is a factual statement about identity",
        )
        assert result.memory_type == MemoryType.FACT
        assert result.confidence == 0.95
        assert result.confidence_level == ClassificationConfidence.HIGH
        assert result.alternative_type is None

    def test_result_with_alternative(self):
        """Test result with alternative classification."""
        result = ClassificationResult(
            memory_type=MemoryType.PREFERENCE,
            confidence=0.75,
            confidence_level=ClassificationConfidence.MEDIUM,
            reasoning="Could be preference or fact",
            alternative_type=MemoryType.FACT,
            alternative_confidence=0.6,
        )
        assert result.alternative_type == MemoryType.FACT
        assert result.alternative_confidence == 0.6


class TestAgenticMemoryClassifier:
    """Tests for AgenticMemoryClassifier class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        return llm

    @pytest.fixture
    def classifier_with_mock(self, mock_llm):
        """Create classifier with mock LLM."""
        classifier = AgenticMemoryClassifier(
            llm=mock_llm,
            use_cache=False,
            validate_classifications=False,
        )
        return classifier

    def test_empty_text_returns_context(self, classifier_with_mock):
        """Test that empty text returns CONTEXT type."""
        result = classifier_with_mock.classify_with_details("")
        assert result.memory_type == MemoryType.CONTEXT
        assert result.confidence == 0.3
        assert result.confidence_level == ClassificationConfidence.UNCERTAIN

    def test_whitespace_text_returns_context(self, classifier_with_mock):
        """Test that whitespace-only text returns CONTEXT type."""
        result = classifier_with_mock.classify_with_details("   ")
        assert result.memory_type == MemoryType.CONTEXT

    def test_classify_fact_with_llm(self, classifier_with_mock, mock_llm):
        """Test classifying a fact statement."""
        mock_llm.generate.return_value = '''
        {
            "analysis": "Identity statement with name",
            "primary_type": "fact",
            "primary_confidence": 0.95,
            "reasoning": "This is a personal identity statement",
            "alternative_type": null,
            "alternative_confidence": null
        }
        '''
        
        result = classifier_with_mock.classify_with_details("My name is Alex")
        
        assert result.memory_type == MemoryType.FACT
        assert result.confidence == 0.95
        assert result.confidence_level == ClassificationConfidence.HIGH
        mock_llm.generate.assert_called_once()

    def test_classify_preference_with_llm(self, classifier_with_mock, mock_llm):
        """Test classifying a preference statement."""
        mock_llm.generate.return_value = '''
        {
            "analysis": "Expression of liking",
            "primary_type": "preference",
            "primary_confidence": 0.92,
            "reasoning": "Clear expression of food preference",
            "alternative_type": null,
            "alternative_confidence": null
        }
        '''
        
        result = classifier_with_mock.classify_with_details("I love pizza")
        
        assert result.memory_type == MemoryType.PREFERENCE
        assert result.confidence == 0.92

    def test_classify_goal_with_llm(self, classifier_with_mock, mock_llm):
        """Test classifying a goal statement."""
        mock_llm.generate.return_value = '''
        {
            "analysis": "Future intention with 'want to'",
            "primary_type": "goal",
            "primary_confidence": 0.88,
            "reasoning": "Clear learning goal",
            "alternative_type": null,
            "alternative_confidence": null
        }
        '''
        
        result = classifier_with_mock.classify_with_details("I want to learn Python")
        
        assert result.memory_type == MemoryType.GOAL
        assert result.confidence == 0.88

    def test_classify_habit_with_llm(self, classifier_with_mock, mock_llm):
        """Test classifying a habit statement."""
        mock_llm.generate.return_value = '''
        {
            "analysis": "Regular routine with 'usually'",
            "primary_type": "habit",
            "primary_confidence": 0.91,
            "reasoning": "Morning routine pattern",
            "alternative_type": null,
            "alternative_confidence": null
        }
        '''
        
        result = classifier_with_mock.classify_with_details("I usually wake up at 7am")
        
        assert result.memory_type == MemoryType.HABIT
        assert result.confidence == 0.91

    def test_classify_event_with_llm(self, classifier_with_mock, mock_llm):
        """Test classifying an event statement."""
        mock_llm.generate.return_value = '''
        {
            "analysis": "Past occurrence with temporal marker",
            "primary_type": "event",
            "primary_confidence": 0.89,
            "reasoning": "Specific past meeting",
            "alternative_type": null,
            "alternative_confidence": null
        }
        '''
        
        result = classifier_with_mock.classify_with_details("I met John yesterday")
        
        assert result.memory_type == MemoryType.EVENT
        assert result.confidence == 0.89

    def test_classify_with_alternative(self, classifier_with_mock, mock_llm):
        """Test classification with alternative type."""
        mock_llm.generate.return_value = '''
        {
            "analysis": "Could be preference or fact",
            "primary_type": "preference",
            "primary_confidence": 0.72,
            "reasoning": "Expresses opinion but also states fact",
            "alternative_type": "fact",
            "alternative_confidence": 0.55
        }
        '''
        
        result = classifier_with_mock.classify_with_details("Python is my favorite language")
        
        assert result.memory_type == MemoryType.PREFERENCE
        assert result.alternative_type == MemoryType.FACT
        assert result.alternative_confidence == 0.55

    def test_fallback_on_llm_error(self, classifier_with_mock, mock_llm):
        """Test fallback to pattern-based when LLM fails."""
        mock_llm.generate.side_effect = Exception("LLM error")
        
        result = classifier_with_mock.classify_with_details("My name is Alex")
        
        # Should fall back to pattern-based classification
        assert result.memory_type in list(MemoryType)
        assert "fallback" in result.reasoning.lower() or "pattern" in result.reasoning.lower()

    def test_fallback_on_invalid_json(self, classifier_with_mock, mock_llm):
        """Test fallback when LLM returns invalid JSON."""
        mock_llm.generate.return_value = "This is not valid JSON"
        
        result = classifier_with_mock.classify_with_details("My name is Alex")
        
        # Should fall back to pattern-based classification
        assert result.memory_type in list(MemoryType)

    def test_parse_json_from_markdown(self, classifier_with_mock, mock_llm):
        """Test parsing JSON from markdown code block."""
        mock_llm.generate.return_value = '''
        Here's the classification:
        ```json
        {
            "analysis": "Identity statement",
            "primary_type": "fact",
            "primary_confidence": 0.9,
            "reasoning": "Name statement",
            "alternative_type": null,
            "alternative_confidence": null
        }
        ```
        '''
        
        result = classifier_with_mock.classify_with_details("My name is Alex")
        
        assert result.memory_type == MemoryType.FACT

    def test_confidence_levels(self, classifier_with_mock):
        """Test confidence level mapping."""
        classifier = classifier_with_mock
        
        assert classifier._get_confidence_level(0.95) == ClassificationConfidence.HIGH
        assert classifier._get_confidence_level(0.9) == ClassificationConfidence.HIGH
        assert classifier._get_confidence_level(0.85) == ClassificationConfidence.MEDIUM
        assert classifier._get_confidence_level(0.7) == ClassificationConfidence.MEDIUM
        assert classifier._get_confidence_level(0.6) == ClassificationConfidence.LOW
        assert classifier._get_confidence_level(0.5) == ClassificationConfidence.LOW
        assert classifier._get_confidence_level(0.4) == ClassificationConfidence.UNCERTAIN

    def test_type_string_mapping(self, classifier_with_mock):
        """Test type string to enum mapping."""
        classifier = classifier_with_mock
        
        assert classifier._map_type_string("fact") == MemoryType.FACT
        assert classifier._map_type_string("FACT") == MemoryType.FACT
        assert classifier._map_type_string("preference") == MemoryType.PREFERENCE
        assert classifier._map_type_string("goal") == MemoryType.GOAL
        assert classifier._map_type_string("habit") == MemoryType.HABIT
        assert classifier._map_type_string("event") == MemoryType.EVENT
        assert classifier._map_type_string("context") == MemoryType.CONTEXT
        assert classifier._map_type_string("unknown") == MemoryType.CONTEXT

    def test_classify_simple(self, classifier_with_mock, mock_llm):
        """Test simple classify method."""
        mock_llm.generate.return_value = '''
        {
            "analysis": "Identity",
            "primary_type": "fact",
            "primary_confidence": 0.9,
            "reasoning": "Name",
            "alternative_type": null,
            "alternative_confidence": null
        }
        '''
        
        result = classifier_with_mock.classify("My name is Alex")
        
        assert result == MemoryType.FACT

    def test_classify_with_confidence(self, classifier_with_mock, mock_llm):
        """Test classify_with_confidence method."""
        mock_llm.generate.return_value = '''
        {
            "analysis": "Preference",
            "primary_type": "preference",
            "primary_confidence": 0.85,
            "reasoning": "Like statement",
            "alternative_type": null,
            "alternative_confidence": null
        }
        '''
        
        memory_type, confidence = classifier_with_mock.classify_with_confidence("I love pizza")
        
        assert memory_type == MemoryType.PREFERENCE
        assert confidence == 0.85

    def test_batch_classify(self, classifier_with_mock, mock_llm):
        """Test batch classification."""
        mock_llm.generate.side_effect = [
            '{"analysis": "a", "primary_type": "fact", "primary_confidence": 0.9, "reasoning": "r", "alternative_type": null, "alternative_confidence": null}',
            '{"analysis": "a", "primary_type": "preference", "primary_confidence": 0.85, "reasoning": "r", "alternative_type": null, "alternative_confidence": null}',
        ]
        
        texts = ["My name is Alex", "I love pizza"]
        results = classifier_with_mock.classify_batch(texts)
        
        assert len(results) == 2
        assert results[0].memory_type == MemoryType.FACT
        assert results[1].memory_type == MemoryType.PREFERENCE


class TestCaching:
    """Tests for caching functionality."""

    def test_cache_hit(self):
        """Test that cached results are returned."""
        clear_agentic_cache()
        
        mock_llm = MagicMock()
        mock_llm.generate.return_value = '''
        {
            "analysis": "Identity",
            "primary_type": "fact",
            "primary_confidence": 0.9,
            "reasoning": "Name statement",
            "alternative_type": null,
            "alternative_confidence": null
        }
        '''
        
        classifier = AgenticMemoryClassifier(llm=mock_llm, use_cache=True)
        
        # First call
        result1 = classifier.classify_with_details("My name is Alex")
        # Second call (should use cache)
        result2 = classifier.classify_with_details("My name is Alex")
        
        # LLM should only be called once
        assert mock_llm.generate.call_count == 1
        assert result1.memory_type == result2.memory_type

    def test_cache_disabled(self):
        """Test that caching can be disabled."""
        clear_agentic_cache()
        
        mock_llm = MagicMock()
        mock_llm.generate.return_value = '''
        {
            "analysis": "Identity",
            "primary_type": "fact",
            "primary_confidence": 0.9,
            "reasoning": "Name statement",
            "alternative_type": null,
            "alternative_confidence": null
        }
        '''
        
        classifier = AgenticMemoryClassifier(llm=mock_llm, use_cache=False)
        
        # First call
        classifier.classify_with_details("My name is Alex")
        # Second call (should NOT use cache)
        classifier.classify_with_details("My name is Alex")
        
        # LLM should be called twice
        assert mock_llm.generate.call_count == 2

    def test_clear_cache(self):
        """Test clearing the cache."""
        clear_agentic_cache()
        
        mock_llm = MagicMock()
        mock_llm.generate.return_value = '''
        {
            "analysis": "Identity",
            "primary_type": "fact",
            "primary_confidence": 0.9,
            "reasoning": "Name statement",
            "alternative_type": null,
            "alternative_confidence": null
        }
        '''
        
        classifier = AgenticMemoryClassifier(llm=mock_llm, use_cache=True)
        
        # First call
        classifier.classify_with_details("My name is Alex")
        
        # Clear cache
        clear_agentic_cache()
        
        # Second call (should NOT use cache)
        classifier.classify_with_details("My name is Alex")
        
        # LLM should be called twice
        assert mock_llm.generate.call_count == 2


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @patch('hippocampai.utils.agentic_classifier.get_agentic_classifier')
    def test_classify_memory_agentic(self, mock_get_classifier):
        """Test classify_memory_agentic function."""
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = MemoryType.FACT
        mock_get_classifier.return_value = mock_classifier
        
        result = classify_memory_agentic("My name is Alex")
        
        assert result == MemoryType.FACT
        mock_classifier.classify.assert_called_once_with("My name is Alex", None)

    @patch('hippocampai.utils.agentic_classifier.get_agentic_classifier')
    def test_classify_memory_agentic_with_confidence(self, mock_get_classifier):
        """Test classify_memory_agentic_with_confidence function."""
        mock_classifier = MagicMock()
        mock_classifier.classify_with_confidence.return_value = (MemoryType.PREFERENCE, 0.9)
        mock_get_classifier.return_value = mock_classifier
        
        memory_type, confidence = classify_memory_agentic_with_confidence("I love pizza")
        
        assert memory_type == MemoryType.PREFERENCE
        assert confidence == 0.9

    @patch('hippocampai.utils.agentic_classifier.get_agentic_classifier')
    def test_classify_memory_agentic_with_details(self, mock_get_classifier):
        """Test classify_memory_agentic_with_details function."""
        mock_result = ClassificationResult(
            memory_type=MemoryType.GOAL,
            confidence=0.88,
            confidence_level=ClassificationConfidence.MEDIUM,
            reasoning="Learning goal",
        )
        mock_classifier = MagicMock()
        mock_classifier.classify_with_details.return_value = mock_result
        mock_get_classifier.return_value = mock_classifier
        
        result = classify_memory_agentic_with_details("I want to learn Python")
        
        assert result.memory_type == MemoryType.GOAL
        assert result.confidence == 0.88


class TestMemoryTypeDefinitions:
    """Tests for memory type definitions."""

    def test_definitions_contain_all_types(self):
        """Test that definitions contain all memory types."""
        assert "FACT" in MEMORY_TYPE_DEFINITIONS
        assert "PREFERENCE" in MEMORY_TYPE_DEFINITIONS
        assert "GOAL" in MEMORY_TYPE_DEFINITIONS
        assert "HABIT" in MEMORY_TYPE_DEFINITIONS
        assert "EVENT" in MEMORY_TYPE_DEFINITIONS
        assert "CONTEXT" in MEMORY_TYPE_DEFINITIONS

    def test_definitions_have_examples(self):
        """Test that definitions include examples."""
        # Check for example patterns
        assert "My name is" in MEMORY_TYPE_DEFINITIONS
        assert "I love" in MEMORY_TYPE_DEFINITIONS
        assert "I want to" in MEMORY_TYPE_DEFINITIONS
        assert "usually" in MEMORY_TYPE_DEFINITIONS
        assert "yesterday" in MEMORY_TYPE_DEFINITIONS


class TestValidation:
    """Tests for validation functionality."""

    def test_validation_corrects_classification(self):
        """Test that validation can correct a classification."""
        mock_llm = MagicMock()
        
        # First call returns initial classification
        # Second call (validation) corrects it
        mock_llm.generate.side_effect = [
            '{"analysis": "a", "primary_type": "fact", "primary_confidence": 0.75, "reasoning": "r", "alternative_type": null, "alternative_confidence": null}',
            '{"is_correct": false, "corrected_type": "preference", "explanation": "Actually a preference"}',
        ]
        
        classifier = AgenticMemoryClassifier(
            llm=mock_llm,
            use_cache=False,
            validate_classifications=True,
        )
        
        result = classifier.classify_with_details("I think Python is great")
        
        # Should be corrected to preference
        assert result.memory_type == MemoryType.PREFERENCE
        assert "corrected" in result.reasoning.lower()

    def test_validation_confirms_classification(self):
        """Test that validation confirms correct classification."""
        mock_llm = MagicMock()
        
        mock_llm.generate.side_effect = [
            '{"analysis": "a", "primary_type": "fact", "primary_confidence": 0.75, "reasoning": "r", "alternative_type": null, "alternative_confidence": null}',
            '{"is_correct": true, "corrected_type": null, "explanation": "Correct"}',
        ]
        
        classifier = AgenticMemoryClassifier(
            llm=mock_llm,
            use_cache=False,
            validate_classifications=True,
        )
        
        result = classifier.classify_with_details("My name is Alex")
        
        # Should remain as fact
        assert result.memory_type == MemoryType.FACT
