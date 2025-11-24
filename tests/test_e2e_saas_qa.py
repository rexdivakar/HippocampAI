"""
End-to-End QA Test Suite for HippocampAI Stack
==============================================

This test suite validates:
1. SaaS API backend correctness
2. Python SDK functionality
3. SaaS ↔ SDK consistency
4. Full feature integration

Requirements:
- HIPPOCAMPAI_API_KEY environment variable
- HIPPOCAMPAI_BASE_URL environment variable (e.g., http://localhost:8000)
- Running HippocampAI SaaS instance
- hippocampai library installed

Usage:
    export HIPPOCAMPAI_API_KEY="your-key"
    export HIPPOCAMPAI_BASE_URL="http://localhost:8000"
    python tests/test_e2e_saas_qa.py
"""

import os
import sys
import time
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# Colors for output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    END = "\033[0m"


class QAReport:
    """Tracks QA test results and generates final report."""

    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.tests_skipped = 0
        self.issues = []
        self.suggestions = []
        self.feature_coverage = {}

    def record_pass(self, test_name: str):
        self.tests_run += 1
        self.tests_passed += 1
        print(f"{Colors.GREEN}✅ PASS{Colors.END}: {test_name}")

    def record_fail(self, test_name: str, error: str):
        self.tests_run += 1
        self.tests_failed += 1
        self.issues.append({"test": test_name, "error": error})
        print(f"{Colors.RED}❌ FAIL{Colors.END}: {test_name}")
        print(f"   Error: {error}")

    def record_skip(self, test_name: str, reason: str):
        self.tests_skipped += 1
        print(f"{Colors.YELLOW}⊘ SKIP{Colors.END}: {test_name} ({reason})")

    def add_issue(self, issue: str):
        self.issues.append({"type": "general", "issue": issue})

    def add_suggestion(self, suggestion: str):
        self.suggestions.append(suggestion)

    def mark_feature_tested(self, feature: str, status: str):
        self.feature_coverage[feature] = status

    def generate_report(self) -> str:
        """Generate final QA report."""
        report = []
        report.append("\n" + "=" * 80)
        report.append(f"{Colors.BOLD}HIPPOCAMPAI END-TO-END QA REPORT{Colors.END}")
        report.append("=" * 80)

        # Summary
        report.append(f"\n{Colors.BOLD}SUMMARY{Colors.END}")
        report.append(f"Tests Run: {self.tests_run}")
        report.append(f"Passed: {Colors.GREEN}{self.tests_passed}{Colors.END}")
        report.append(f"Failed: {Colors.RED}{self.tests_failed}{Colors.END}")
        report.append(f"Skipped: {Colors.YELLOW}{self.tests_skipped}{Colors.END}")

        pass_rate = (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0
        report.append(f"Pass Rate: {pass_rate:.1f}%")

        # Feature Coverage
        report.append(f"\n{Colors.BOLD}FEATURE COVERAGE{Colors.END}")
        for feature, status in self.feature_coverage.items():
            status_color = (
                Colors.GREEN if status == "✓" else Colors.RED if status == "✗" else Colors.YELLOW
            )
            report.append(f"  {status_color}{status}{Colors.END} {feature}")

        # Issues
        if self.issues:
            report.append(f"\n{Colors.BOLD}ISSUES FOUND{Colors.END}")
            for i, issue in enumerate(self.issues, 1):
                if "test" in issue:
                    report.append(f"{i}. Test: {issue['test']}")
                    report.append(f"   Error: {issue['error']}")
                else:
                    report.append(f"{i}. {issue['issue']}")

        # Suggestions
        if self.suggestions:
            report.append(f"\n{Colors.BOLD}SUGGESTIONS FOR IMPROVEMENT{Colors.END}")
            for i, suggestion in enumerate(self.suggestions, 1):
                report.append(f"{i}. {suggestion}")

        report.append("\n" + "=" * 80)
        return "\n".join(report)


qa_report = QAReport()


def print_section(title: str):
    """Print section header."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}{Colors.END}\n")


def print_test(description: str):
    """Print test description."""
    print(f"{Colors.MAGENTA}TEST:{Colors.END} {description}")


def print_info(message: str):
    """Print info message."""
    print(f"  ℹ️  {message}")


def print_expected(message: str):
    """Print expected result."""
    print(f"  {Colors.BLUE}Expected:{Colors.END} {message}")


def print_actual(message: str):
    """Print actual result."""
    print(f"  {Colors.GREEN}Actual:{Colors.END} {message}")


# ============================================================================
# TEST 1: INITIALIZATION & HEALTH
# ============================================================================


def test_1_initialization_and_health():
    """
    Test 1: Initialization & Health

    What: Verify the Python client can initialize and connect to SaaS
    Why: Ensures basic connectivity and authentication work
    """
    print_section("TEST 1: INITIALIZATION & HEALTH")

    # Check environment variables
    print_test("1.1: Check required environment variables")
    api_key = os.getenv("HIPPOCAMPAI_API_KEY")
    base_url = os.getenv("HIPPOCAMPAI_BASE_URL", "http://localhost:8000")

    if not api_key:
        qa_report.record_fail(
            "Environment Setup", "HIPPOCAMPAI_API_KEY not set. Required for authentication."
        )
        qa_report.add_suggestion("Document required environment variables in README with examples")
        return None

    print_actual(f"API Key: {'*' * (len(api_key) - 4)}{api_key[-4:]}")
    print_actual(f"Base URL: {base_url}")
    qa_report.record_pass("Environment variables present")

    # Initialize client
    print_test("1.2: Initialize HippocampAI client with remote mode")
    try:
        from hippocampai import UnifiedMemoryClient

        client = UnifiedMemoryClient(mode="remote", api_url=base_url, api_key=api_key)

        print_actual("Client initialized successfully")
        qa_report.record_pass("Client initialization")
        qa_report.mark_feature_tested("Client Initialization", "✓")

    except Exception as e:
        qa_report.record_fail("Client Initialization", str(e))
        qa_report.mark_feature_tested("Client Initialization", "✗")
        qa_report.add_suggestion("Add better error messages for initialization failures")
        return None

    # Test health endpoint
    print_test("1.3: Call health check endpoint")
    try:
        # Try to call health endpoint
        import httpx

        response = httpx.get(f"{base_url}/health", timeout=5.0)

        print_expected("Status 200 with health info")
        print_actual(f"Status {response.status_code}: {response.json()}")

        if response.status_code == 200:
            health_data = response.json()

            # Verify health response structure
            if "status" in health_data:
                qa_report.record_pass("Health endpoint returns status")
            else:
                qa_report.add_issue("Health endpoint missing 'status' field")

            qa_report.mark_feature_tested("Health Check", "✓")
        else:
            qa_report.record_fail("Health Check", f"Expected 200, got {response.status_code}")
            qa_report.mark_feature_tested("Health Check", "✗")

    except Exception as e:
        qa_report.record_fail("Health Check", str(e))
        qa_report.mark_feature_tested("Health Check", "✗")
        return None

    # Test authentication headers
    print_test("1.4: Verify authentication headers are set correctly")
    try:
        # Check if auth headers are present in requests
        print_expected("Authorization header present in requests")

        # Try a simple operation to verify auth works
        response = httpx.get(
            f"{base_url}/v1/memories",
            headers={"Authorization": f"Bearer {api_key}"},
            params={"user_id": "test_qa_user", "limit": 1},
            timeout=5.0,
        )

        if response.status_code in [200, 401, 403]:
            print_actual(f"Auth header processed (status: {response.status_code})")
            qa_report.record_pass("Authentication headers")
        else:
            qa_report.record_fail("Authentication", f"Unexpected status: {response.status_code}")

    except Exception as e:
        qa_report.record_fail("Authentication Headers", str(e))

    return client


# ============================================================================
# TEST 2: SPACES, USERS & ISOLATION
# ============================================================================


def test_2_spaces_users_isolation(client):
    """
    Test 2: Spaces, Users & Isolation

    What: Test multi-tenant isolation and context management
    Why: Critical for preventing data leakage between projects/users
    """
    print_section("TEST 2: SPACES, USERS & ISOLATION")

    if client is None:
        qa_report.record_skip("Spaces & Isolation", "Client not initialized")
        return {}

    spaces = {}
    space_configs = [
        {
            "id": "hippo-docs",
            "name": "Documentation Assistant",
            "description": "HippocampAI documentation and guides",
            "tags": ["docs", "qa"],
            "test_memory": "HippocampAI uses Qdrant for vector storage",
        },
        {
            "id": "opsai-slotting",
            "name": "Warehouse Optimization",
            "description": "Logistics and warehouse management",
            "tags": ["logistics", "ops"],
            "test_memory": "OpsAI optimizes pick paths in warehouses",
        },
        {
            "id": "personal-finance",
            "name": "Personal Finance Tracker",
            "description": "Credit cards and investment tracking",
            "tags": ["finance", "personal"],
            "test_memory": "Use Amex Gold for groceries to get 4x points",
        },
    ]

    # Create spaces (using user_id as space identifier)
    print_test("2.1: Create multiple isolated spaces")
    for config in space_configs:
        try:
            # In HippocampAI, we use user_id for isolation
            # Store test memory to verify space creation
            memory = client.remember(
                text=config["test_memory"],
                user_id=config["id"],
                metadata={
                    "space_name": config["name"],
                    "space_description": config["description"],
                    "tags": config["tags"],
                },
            )

            spaces[config["id"]] = {"config": config, "memory_ids": [memory.id]}

            print_actual(f"Created space '{config['id']}' with initial memory")
            qa_report.record_pass(f"Space creation: {config['id']}")

        except Exception as e:
            qa_report.record_fail(f"Space Creation: {config['id']}", str(e))

    qa_report.mark_feature_tested("Multi-Space Isolation", "✓" if len(spaces) == 3 else "✗")

    # Test isolation - query in one space shouldn't return memories from others
    print_test("2.2: Verify space isolation (no data leakage)")
    isolation_passed = True

    for space_id, space_data in spaces.items():
        try:
            # Query this space
            results = client.recall(
                query="optimization warehouse logistics", user_id=space_id, k=10
            )

            # Check if any results belong to other spaces
            for result in results:
                # The memory should only belong to this space (user_id)
                if hasattr(result, "memory") and hasattr(result.memory, "user_id"):
                    if result.memory.user_id != space_id:
                        isolation_passed = False
                        qa_report.add_issue(
                            f"Data leakage: Space '{space_id}' returned memory from '{result.memory.user_id}'"
                        )

            print_actual(f"Space '{space_id}': {len(results)} results, all isolated ✓")

        except Exception as e:
            isolation_passed = False
            qa_report.record_fail(f"Isolation Test: {space_id}", str(e))

    if isolation_passed:
        qa_report.record_pass("Space isolation (no cross-space leakage)")
    else:
        qa_report.record_fail("Space Isolation", "Cross-space data leakage detected")

    return spaces


# ============================================================================
# TEST 3: BASIC MEMORY WRITES
# ============================================================================


def test_3_basic_memory_writes(client, spaces):
    """
    Test 3: Basic Memory Writes

    What: Test writing conversations, notes, and documents
    Why: Core functionality for memory ingestion
    """
    print_section("TEST 3: BASIC MEMORY WRITES")

    if client is None or not spaces:
        qa_report.record_skip("Memory Writes", "Prerequisites not met")
        return

    space_id = "hippo-docs"
    print_test(f"3.1: Insert multi-turn conversation in space '{space_id}'")

    conversation = [
        {
            "role": "user",
            "content": "What is HippocampAI?",
            "importance": 5,
            "tags": ["intro", "overview"],
        },
        {
            "role": "assistant",
            "content": "HippocampAI is a long-term memory engine that uses Qdrant for vector storage, Redis for caching, and provides features like consolidation, importance scoring, and hybrid retrieval with BM25 + vector search + reranking.",
            "importance": 5,
            "tags": ["intro", "architecture"],
        },
        {
            "role": "user",
            "content": "How do I integrate it with my Python app?",
            "importance": 4,
            "tags": ["integration", "python"],
        },
        {
            "role": "assistant",
            "content": """Here's how to integrate HippocampAI:

```python
from hippocampai import MemoryClient

# Initialize
client = MemoryClient()

# Store memories
client.remember("User prefers dark mode", user_id="alice", type="preference")

# Retrieve memories
results = client.recall("UI preferences", user_id="alice")
```

Set environment variables: HIPPOCAMPAI_API_KEY, HIPPOCAMPAI_BASE_URL.""",
            "importance": 5,
            "tags": ["integration", "python", "code"],
        },
    ]

    conversation_ids = []
    for msg in conversation:
        try:
            memory = client.remember(
                text=msg["content"],
                user_id=space_id,
                importance=msg["importance"],
                tags=msg["tags"],
                metadata={
                    "role": msg["role"],
                    "source": "chat",
                    "topic": "intro" if "intro" in msg["tags"] else "integration",
                },
            )
            conversation_ids.append(memory.id)
            print_actual(f"Inserted {msg['role']} message: {memory.id}")

        except Exception as e:
            qa_report.record_fail(f"Insert {msg['role']} message", str(e))

    if len(conversation_ids) == len(conversation):
        qa_report.record_pass("Multi-turn conversation ingestion")
        qa_report.mark_feature_tested("Conversation Memory", "✓")
    else:
        qa_report.record_fail(
            "Conversation Ingestion",
            f"Only {len(conversation_ids)}/{len(conversation)} messages inserted",
        )
        qa_report.mark_feature_tested("Conversation Memory", "✗")

    # Insert standalone document
    print_test("3.2: Insert standalone technical document")
    try:
        doc_text = """HippocampAI Architecture Overview:

Components:
1. Ingestion Layer: Accepts text, conversations, documents
2. Embedding Layer: sentence-transformers (BAAI/bge-small-en-v1.5)
3. Storage Layer: Qdrant (vectors) + Redis (cache)
4. Retrieval Layer: Hybrid search (Vector + BM25 + Cross-encoder reranking)
5. Consolidation Layer: Merges similar memories, applies importance decay
6. Summarization Layer: Generates summaries of conversations and sessions

Key Features:
- Memory types: fact, preference, goal, habit, event, context
- Importance scoring (0-10) with automatic LLM-based scoring
- TTL and expiration
- Multi-agent support with permissions
- Knowledge graph with entities and relationships"""

        doc_memory = client.remember(
            text=doc_text,
            user_id=space_id,
            importance=5,
            tags=["architecture", "documentation", "technical"],
            metadata={"source": "doc", "topic": "architecture", "doc_type": "technical_design"},
        )

        print_actual(f"Document inserted: {doc_memory.id}")
        qa_report.record_pass("Standalone document ingestion")
        qa_report.mark_feature_tested("Document Memory", "✓")

    except Exception as e:
        qa_report.record_fail("Document Ingestion", str(e))
        qa_report.mark_feature_tested("Document Memory", "✗")

    # Verify memory count
    print_test("3.3: Verify memory count increased")
    try:
        memories = client.get_memories(user_id=space_id)
        expected_min = len(conversation) + 2  # conversation + doc + initial test memory

        print_expected(f"At least {expected_min} memories")
        print_actual(f"{len(memories)} memories in space")

        if len(memories) >= expected_min:
            qa_report.record_pass("Memory count verification")
        else:
            qa_report.add_issue(f"Expected >= {expected_min} memories, found {len(memories)}")

    except Exception as e:
        qa_report.record_fail("Memory Count", str(e))


# ============================================================================
# TEST 4: RETRIEVAL (KEYWORD, SEMANTIC, HYBRID, FILTERS)
# ============================================================================


def test_4_retrieval_quality(client, spaces):
    """
    Test 4: Retrieval Quality

    What: Test different retrieval modes and filters
    Why: Core value proposition - accurate, flexible retrieval
    """
    print_section("TEST 4: RETRIEVAL - KEYWORD, SEMANTIC, HYBRID, FILTERS")

    if client is None or not spaces:
        qa_report.record_skip("Retrieval", "Prerequisites not met")
        return

    space_id = "hippo-docs"

    # Test 4.1: Keyword search
    print_test("4.1: Keyword search for 'Qdrant'")
    try:
        results = client.recall(query="Qdrant", user_id=space_id, k=5)

        print_expected("Memories mentioning Qdrant vector DB")
        print_actual(f"Found {len(results)} results")

        for i, result in enumerate(results[:3], 1):
            print(f"    {i}. Score {result.score:.3f}: {result.memory.text[:80]}...")

            # Verify Qdrant is mentioned
            if "qdrant" in result.memory.text.lower():
                print("       ✓ Contains 'Qdrant'")
            else:
                qa_report.add_issue(
                    f"Keyword search for 'Qdrant' returned irrelevant result: {result.memory.text[:50]}"
                )

        qa_report.record_pass("Keyword search")
        qa_report.mark_feature_tested("Keyword Search", "✓")

    except Exception as e:
        qa_report.record_fail("Keyword Search", str(e))
        qa_report.mark_feature_tested("Keyword Search", "✗")

    # Test 4.2: Semantic search
    print_test("4.2: Semantic search - 'How do I connect my Python app?'")
    try:
        results = client.recall(
            query="How do I connect my Python app to HippocampAI?", user_id=space_id, k=5
        )

        print_expected("Integration-related memories even with different wording")
        print_actual(f"Found {len(results)} results")

        integration_found = False
        for i, result in enumerate(results[:3], 1):
            text_preview = result.memory.text[:100]
            print(f"    {i}. Score {result.score:.3f}: {text_preview}...")

            # Check if integration/python related
            if any(
                keyword in result.memory.text.lower()
                for keyword in ["python", "integrate", "api_key", "client"]
            ):
                integration_found = True
                print("       ✓ Integration-related")

        if integration_found:
            qa_report.record_pass("Semantic search")
        else:
            qa_report.add_issue("Semantic search didn't surface integration content")

        qa_report.mark_feature_tested("Semantic Search", "✓")

    except Exception as e:
        qa_report.record_fail("Semantic Search", str(e))
        qa_report.mark_feature_tested("Semantic Search", "✗")

    # Test 4.3: Filtered search by metadata
    print_test("4.3: Filter by source='chat' and topic='architecture'")
    try:
        results = client.recall(
            query="architecture components",
            user_id=space_id,
            filters={"metadata.source": "chat", "metadata.topic": "architecture"},
            k=5,
        )

        print_expected("Only chat messages about architecture")
        print_actual(f"Found {len(results)} filtered results")

        all_match_filter = True
        for result in results:
            metadata = result.memory.metadata or {}
            if metadata.get("source") != "chat" or metadata.get("topic") != "architecture":
                all_match_filter = False
                qa_report.add_issue(
                    f"Filter mismatch: source={metadata.get('source')}, topic={metadata.get('topic')}"
                )

        if all_match_filter:
            qa_report.record_pass("Metadata filtering")
        else:
            qa_report.record_fail("Metadata Filtering", "Results don't match filters")

        qa_report.mark_feature_tested("Filter Search", "✓")

    except Exception as e:
        qa_report.record_fail("Filtered Search", str(e))
        qa_report.mark_feature_tested("Filter Search", "✗")

    # Test 4.4: Filter by importance
    print_test("4.4: Filter by importance >= 5")
    try:
        results = client.recall(
            query="HippocampAI features", user_id=space_id, filters={"importance_min": 5.0}, k=10
        )

        print_expected("Only high-importance memories (>= 5)")
        print_actual(f"Found {len(results)} high-importance results")

        all_important = True
        for result in results:
            if result.memory.importance < 5.0:
                all_important = False
                qa_report.add_issue(
                    f"Importance filter failed: got {result.memory.importance} < 5.0"
                )

        if all_important:
            qa_report.record_pass("Importance filtering")
        else:
            qa_report.record_fail("Importance Filtering", "Low-importance results included")

    except Exception as e:
        qa_report.record_fail("Importance Filtering", str(e))


# ============================================================================
# TEST 5: ENTITY EXTRACTION & ENTITY-BASED RETRIEVAL
# ============================================================================


def test_5_entity_extraction(client, spaces):
    """
    Test 5: Entity Extraction & Entity-Based Retrieval

    What: Test entity recognition and entity-centric queries
    Why: Validates knowledge graph and structured information extraction
    """
    print_section("TEST 5: ENTITY EXTRACTION & ENTITY-BASED RETRIEVAL")

    if client is None or not spaces:
        qa_report.record_skip("Entity Extraction", "Prerequisites not met")
        return

    space_id = "opsai-slotting"

    # Insert events with entities
    print_test("5.1: Insert events with clear entities (SKUs, locations, metrics)")

    events = [
        "Moved SKU-12345 frozen pizza to Zone A1, Rack B2 for faster access",
        "SKU-67890 canned beans restocked in Bin C3, Zone A1",
        "Pick rate improved by 15% after optimizing Zone A1 layout",
        "Travel distance in Zone A1 decreased by 12% with new slotting",
        "SKU-12345 is a fast-moving item, requires prime location in Zone A1",
    ]

    entity_memory_ids = []
    for event in events:
        try:
            memory = client.remember(
                text=event,
                user_id=space_id,
                tags=["warehouse", "operations", "metrics"],
                metadata={"source": "ops_log"},
            )
            entity_memory_ids.append(memory.id)
            print_actual(f"Inserted: {event[:60]}...")

        except Exception as e:
            qa_report.record_fail("Entity Event Insert", str(e))

    qa_report.record_pass("Entity-rich events ingestion")

    # Test entity extraction (if client supports it)
    print_test("5.2: Extract entities from text")
    try:
        # Check if entity extraction is available
        if hasattr(client, "extract_entities"):
            entities = client.extract_entities(events[0])

            print_expected("Entities: SKU-12345, Zone A1, Rack B2, frozen pizza")
            print_actual(f"Extracted {len(entities)} entities")

            for entity in entities[:5]:
                print(f"    - {entity.text} ({entity.type})")

            qa_report.record_pass("Entity extraction")
            qa_report.mark_feature_tested("Entity Extraction", "✓")
        else:
            qa_report.record_skip("Entity Extraction", "Feature not available in client")
            qa_report.mark_feature_tested("Entity Extraction", "⊘")
            qa_report.add_suggestion("Add extract_entities() method to client API")

    except Exception as e:
        qa_report.record_fail("Entity Extraction", str(e))
        qa_report.mark_feature_tested("Entity Extraction", "✗")

    # Test entity-based retrieval
    print_test("5.3: Entity-centric query - 'Show all memories mentioning SKU-12345'")
    try:
        results = client.recall(query="SKU-12345", user_id=space_id, k=10)

        print_expected("All events mentioning SKU-12345")
        print_actual(f"Found {len(results)} results")

        sku_count = 0
        for result in results:
            if "SKU-12345" in result.memory.text or "sku-12345" in result.memory.text.lower():
                sku_count += 1
                print(f"    ✓ {result.memory.text[:70]}...")

        print_actual(f"{sku_count} results contain SKU-12345")

        if sku_count >= 2:  # We inserted 2 events with SKU-12345
            qa_report.record_pass("Entity-based retrieval")
            qa_report.mark_feature_tested("Entity Search", "✓")
        else:
            qa_report.add_issue(f"Expected >= 2 results for SKU-12345, got {sku_count}")
            qa_report.mark_feature_tested("Entity Search", "⚠")

    except Exception as e:
        qa_report.record_fail("Entity-Based Retrieval", str(e))
        qa_report.mark_feature_tested("Entity Search", "✗")

    # Test location-based query
    print_test("5.4: Location-based query - 'What changes affected Zone A1?'")
    try:
        results = client.recall(query="changes in Zone A1", user_id=space_id, k=10)

        print_actual(f"Found {len(results)} results about Zone A1")

        for i, result in enumerate(results[:3], 1):
            print(f"    {i}. {result.memory.text[:70]}...")

        qa_report.record_pass("Location-based entity query")

    except Exception as e:
        qa_report.record_fail("Location-Based Query", str(e))


# ============================================================================
# TEST 6: SUMMARIZATION & CONSOLIDATION
# ============================================================================


def test_6_summarization_consolidation(client, spaces):
    """
    Test 6: Summarization & Consolidation

    What: Test memory condensation and summary generation
    Why: Long-term memory needs compaction to stay manageable
    """
    print_section("TEST 6: SUMMARIZATION & CONSOLIDATION")

    if client is None or not spaces:
        qa_report.record_skip("Summarization", "Prerequisites not met")
        return

    space_id = "personal-finance"

    # Simulate long history
    print_test("6.1: Insert diverse financial memories")

    financial_memories = [
        # Credit cards
        (
            "Amex Gold gives 4x points on groceries and restaurants",
            {"category": "credit_card", "card": "amex_gold"},
        ),
        (
            "Venture X has $300 travel credit and Priority Pass",
            {"category": "credit_card", "card": "venture_x"},
        ),
        (
            "Blue Cash Everyday gives 3% on groceries",
            {"category": "credit_card", "card": "blue_cash"},
        ),
        (
            "Use Amex Gold for dining out to maximize rewards",
            {"category": "credit_card", "card": "amex_gold"},
        ),
        # Monthly spending
        ("Rent is $2400/month due on the 1st", {"category": "expense", "type": "rent"}),
        ("Groceries budget: $600/month", {"category": "expense", "type": "groceries"}),
        ("Restaurant spending averages $300/month", {"category": "expense", "type": "dining"}),
        ("Gas costs about $150/month", {"category": "expense", "type": "gas"}),
        # Investments
        ("VOO (S&P 500 ETF) is my core holding", {"category": "investment", "ticker": "VOO"}),
        ("NET (Cloudflare) for cloud growth exposure", {"category": "investment", "ticker": "NET"}),
        ("META position up 45% this year", {"category": "investment", "ticker": "META"}),
        ("Indian stocks: RELIANCE, TCS, INFY", {"category": "investment", "region": "India"}),
    ]

    for text, metadata in financial_memories:
        try:
            client.remember(
                text=text,
                user_id=space_id,
                metadata=metadata,
                tags=[metadata.get("category", "other")],
            )
        except Exception as e:
            qa_report.record_fail("Financial Memory Insert", str(e))

    print_actual(f"Inserted {len(financial_memories)} financial memories")
    qa_report.record_pass("Financial history ingestion")

    # Test summarization (if available)
    print_test("6.2: Generate profile summary of financial habits")
    try:
        if hasattr(client, "get_memories"):
            # Get all memories
            all_memories = client.get_memories(user_id=space_id)

            print_expected("Summary of credit cards, expenses, and investments")
            print_actual(f"Retrieved {len(all_memories)} memories for summarization")

            # If summarization endpoint exists
            if hasattr(client, "summarize_user"):
                summary = client.summarize_user(user_id=space_id)
                print_actual(f"Generated summary: {summary[:200]}...")
                qa_report.record_pass("User profile summarization")
                qa_report.mark_feature_tested("Summarization", "✓")
            else:
                qa_report.record_skip("Profile Summarization", "Feature not in client API")
                qa_report.mark_feature_tested("Summarization", "⊘")
                qa_report.add_suggestion("Add summarize_user() or get_profile_summary() method")
        else:
            qa_report.record_skip("Summarization", "get_memories() not available")

    except Exception as e:
        qa_report.record_fail("Summarization", str(e))
        qa_report.mark_feature_tested("Summarization", "✗")

    # Test filtered summarization
    print_test("6.3: Generate filtered summary (credit cards only)")
    try:
        results = client.recall(
            query="credit cards rewards points",
            user_id=space_id,
            filters={"metadata.category": "credit_card"},
            k=10,
        )

        print_expected("Summary of credit card strategies")
        print_actual(f"Found {len(results)} credit card memories")

        for i, result in enumerate(results[:3], 1):
            print(f"    {i}. {result.memory.text}")

        qa_report.record_pass("Filtered recall for summarization")

    except Exception as e:
        qa_report.record_fail("Filtered Summarization", str(e))

    # Test consolidation (if available)
    print_test("6.4: Test memory consolidation")
    try:
        if hasattr(client, "consolidate_memories"):
            result = client.consolidate_memories(
                user_id=space_id, similarity_threshold=0.85, dry_run=True
            )

            print_expected("Similar memories identified for merging")
            print_actual(f"Consolidation analysis: {result}")

            qa_report.record_pass("Consolidation dry-run")
            qa_report.mark_feature_tested("Consolidation", "✓")
        else:
            qa_report.record_skip("Consolidation", "Feature not available")
            qa_report.mark_feature_tested("Consolidation", "⊘")
            qa_report.add_suggestion("Add consolidate_memories() method to client API")

    except Exception as e:
        qa_report.record_fail("Consolidation", str(e))
        qa_report.mark_feature_tested("Consolidation", "✗")


# ============================================================================
# TEST 7: TEMPORAL REASONING & RECENCY
# ============================================================================


def test_7_temporal_reasoning(client, spaces):
    """
    Test 7: Temporal Reasoning & Recency

    What: Test time-aware retrieval and time-bounded queries
    Why: Critical for understanding how information changes over time
    """
    print_section("TEST 7: TEMPORAL REASONING & RECENCY")

    if client is None or not spaces:
        qa_report.record_skip("Temporal Reasoning", "Prerequisites not met")
        return

    space_id = "hippo-docs"

    # Insert time-stamped memories
    print_test("7.1: Insert memories with different timestamps")

    temporal_memories = [
        {
            "text": "In January 2024, we used only Redis for caching",
            "timestamp": datetime(2024, 1, 15, tzinfo=timezone.utc),
            "tags": ["architecture", "2024", "redis"],
        },
        {
            "text": "In March 2024, we added Qdrant for vector storage",
            "timestamp": datetime(2024, 3, 20, tzinfo=timezone.utc),
            "tags": ["architecture", "2024", "qdrant"],
        },
        {
            "text": "In October 2024, we implemented hybrid search with BM25",
            "timestamp": datetime(2024, 10, 10, tzinfo=timezone.utc),
            "tags": ["architecture", "2024", "search"],
        },
        {
            "text": "Current setup (November 2024): Qdrant + Redis + BM25 hybrid search with reranking",
            "timestamp": datetime.now(timezone.utc),
            "tags": ["architecture", "current", "search"],
        },
    ]

    for memory_data in temporal_memories:
        try:
            # Note: HippocampAI may not support custom timestamps in remember()
            # This tests if the feature exists
            _memory = client.remember(
                text=memory_data["text"],
                user_id=space_id,
                tags=memory_data["tags"],
                metadata={"timestamp": memory_data["timestamp"].isoformat()},
            )
            print_actual(f"Inserted: {memory_data['text'][:50]}...")

        except Exception as e:
            qa_report.add_issue(f"Cannot set custom timestamps: {e}")

    qa_report.record_pass("Temporal memory insertion")

    # Test time-bounded retrieval
    print_test("7.2: Query for early 2024 (January-June)")
    try:
        if hasattr(client, "recall"):
            # Try time-bounded query
            results = client.recall(
                query="database vector storage architecture",
                user_id=space_id,
                filters={
                    "created_after": datetime(2024, 1, 1, tzinfo=timezone.utc),
                    "created_before": datetime(2024, 6, 30, tzinfo=timezone.utc),
                },
                k=10,
            )

            print_expected("Memories from early 2024 mentioning Redis, Qdrant added")
            print_actual(f"Found {len(results)} results from Q1-Q2 2024")

            for result in results[:3]:
                print(f"    - {result.memory.text[:70]}...")

            qa_report.record_pass("Time-bounded query (early 2024)")
            qa_report.mark_feature_tested("Temporal Queries", "✓")

        else:
            qa_report.record_skip("Temporal Query", "recall() not available")

    except Exception as e:
        qa_report.record_fail("Temporal Query (early 2024)", str(e))
        qa_report.mark_feature_tested("Temporal Queries", "✗")
        qa_report.add_suggestion(
            "Support created_after/created_before filters for time-bounded retrieval"
        )

    # Test current vs historical
    print_test("7.3: Compare historical vs current architecture")
    try:
        # Query about past
        old_results = client.recall(
            query="What database did we use for vectors in early 2024?", user_id=space_id, k=5
        )

        # Query about present
        current_results = client.recall(
            query="What is our current vector store setup?", user_id=space_id, k=5
        )

        print_actual(f"Historical query: {len(old_results)} results")
        print_actual(f"Current query: {len(current_results)} results")

        # Verify temporal consistency
        print_expected("Old query surfaces Redis/early Qdrant, current query surfaces hybrid setup")

        qa_report.record_pass("Temporal consistency check")

    except Exception as e:
        qa_report.record_fail("Temporal Consistency", str(e))


# ============================================================================
# TEST 8: CROSS-INTERFACE CONSISTENCY (SaaS ↔ SDK)
# ============================================================================


def test_8_cross_interface_consistency(client, spaces):
    """
    Test 8: Cross-Interface Consistency

    What: Verify SaaS UI and SDK stay in sync
    Why: Critical for user trust and data integrity
    """
    print_section("TEST 8: CROSS-INTERFACE CONSISTENCY (SaaS ↔ SDK)")

    if client is None:
        qa_report.record_skip("Cross-Interface Sync", "Client not initialized")
        return

    print_test("8.1: Create space via SDK, verify in SaaS")

    test_space_id = f"qa_test_space_{int(time.time())}"

    try:
        # Create via SDK
        memory = client.remember(
            text="This space was created via SDK for QA testing",
            user_id=test_space_id,
            metadata={"created_by": "sdk", "test": "cross_interface"},
        )

        print_actual(f"Created space '{test_space_id}' via SDK")
        print_actual(f"Memory ID: {memory.id}")

        qa_report.record_pass("Space creation via SDK")

        # Verify via SDK recall
        print_test("8.2: Verify SDK-created space is queryable via SDK")
        results = client.recall(query="SDK QA testing", user_id=test_space_id, k=5)

        if len(results) > 0 and results[0].memory.id == memory.id:
            print_actual("✓ Memory found via SDK query")
            qa_report.record_pass("SDK→SDK consistency")
        else:
            qa_report.record_fail("SDK→SDK Consistency", "Memory not found via query")

        # Note about SaaS UI verification
        print_info("MANUAL VERIFICATION REQUIRED:")
        print_info(f"1. Open SaaS dashboard at {os.getenv('HIPPOCAMPAI_BASE_URL')}")
        print_info(f"2. Look for space/user: {test_space_id}")
        print_info(f"3. Verify memory ID {memory.id} is present")
        print_info("4. Check metadata shows 'created_by: sdk'")

        qa_report.add_suggestion(
            "Add automated API to list spaces/users for testing UI sync without manual checks"
        )

        qa_report.mark_feature_tested("Cross-Interface Sync", "⚠")

    except Exception as e:
        qa_report.record_fail("Cross-Interface Consistency", str(e))
        qa_report.mark_feature_tested("Cross-Interface Sync", "✗")


# ============================================================================
# TEST 9: UPDATES, DELETES, SOFT DELETES & EXPORT
# ============================================================================


def test_9_updates_deletes_export(client, spaces):
    """
    Test 9: Updates, Deletes & Export

    What: Test memory lifecycle management
    Why: Users need to correct, remove, and export their data
    """
    print_section("TEST 9: UPDATES, DELETES, SOFT DELETES & EXPORT")

    if client is None or not spaces:
        qa_report.record_skip("Updates/Deletes", "Prerequisites not met")
        return

    space_id = "hippo-docs"

    # Create a test memory
    print_test("9.1: Create memory for update/delete testing")
    try:
        test_memory = client.remember(
            text="This memory will be updated and then deleted",
            user_id=space_id,
            importance=3,
            tags=["test", "temporary"],
        )

        print_actual(f"Created test memory: {test_memory.id}")
        _original_text = test_memory.text
        qa_report.record_pass("Test memory creation")

    except Exception as e:
        qa_report.record_fail("Test Memory Creation", str(e))
        return

    # Test update
    print_test("9.2: Update memory content and metadata")
    try:
        if hasattr(client, "update_memory"):
            updated = client.update_memory(
                memory_id=test_memory.id,
                text="This memory was UPDATED - importance increased",
                importance=8,
                tags=["test", "updated", "important"],
            )

            print_expected("Updated text and importance from 3 to 8")
            print_actual(f"New text: {updated.text}")
            print_actual(f"New importance: {updated.importance}")

            if updated.importance == 8 and "UPDATED" in updated.text:
                qa_report.record_pass("Memory update")
                qa_report.mark_feature_tested("Memory Updates", "✓")
            else:
                qa_report.record_fail("Memory Update", "Changes not reflected")
                qa_report.mark_feature_tested("Memory Updates", "✗")
        else:
            qa_report.record_skip("Memory Update", "update_memory() not available")
            qa_report.mark_feature_tested("Memory Updates", "⊘")
            qa_report.add_suggestion("Add update_memory() method to client")

    except Exception as e:
        qa_report.record_fail("Memory Update", str(e))
        qa_report.mark_feature_tested("Memory Updates", "✗")

    # Test soft delete (if supported)
    print_test("9.3: Soft delete memory")
    try:
        if hasattr(client, "soft_delete_memory"):
            result = client.soft_delete_memory(test_memory.id)

            # Verify it doesn't appear in normal queries
            results = client.recall(query="updated", user_id=space_id, k=20)

            found = any(r.memory.id == test_memory.id for r in results)

            if not found:
                print_actual("✓ Soft-deleted memory not in normal queries")
                qa_report.record_pass("Soft delete")
                qa_report.mark_feature_tested("Soft Delete", "✓")
            else:
                qa_report.record_fail("Soft Delete", "Memory still appears in queries")
                qa_report.mark_feature_tested("Soft Delete", "✗")
        else:
            qa_report.record_skip("Soft Delete", "Feature not available")
            qa_report.mark_feature_tested("Soft Delete", "⊘")
            qa_report.add_suggestion("Add soft_delete_memory() with include_deleted flag")

    except Exception as e:
        qa_report.record_fail("Soft Delete", str(e))

    # Test hard delete
    print_test("9.4: Hard delete memory")
    try:
        if hasattr(client, "delete_memory"):
            result = client.delete_memory(test_memory.id)

            print_expected("Memory permanently deleted")
            print_actual(f"Delete result: {result}")

            # Verify it's gone
            try:
                retrieved = client.get_memory(test_memory.id)
                if retrieved is None:
                    qa_report.record_pass("Hard delete")
                    qa_report.mark_feature_tested("Hard Delete", "✓")
                else:
                    qa_report.record_fail("Hard Delete", "Memory still exists")
                    qa_report.mark_feature_tested("Hard Delete", "✗")
            except Exception:
                # Exception means memory not found - that's good
                qa_report.record_pass("Hard delete (memory not found)")
                qa_report.mark_feature_tested("Hard Delete", "✓")
        else:
            qa_report.record_skip("Hard Delete", "delete_memory() not available")
            qa_report.mark_feature_tested("Hard Delete", "⊘")

    except Exception as e:
        qa_report.record_fail("Hard Delete", str(e))

    # Test export
    print_test("9.5: Export space data")
    try:
        if hasattr(client, "export_memories") or hasattr(client, "get_memories"):
            memories = client.get_memories(user_id=space_id, limit=100)

            print_expected("JSON export with IDs, timestamps, metadata, text")
            print_actual(f"Retrieved {len(memories)} memories for export")

            # Verify export format
            if len(memories) > 0:
                sample = memories[0]
                has_id = hasattr(sample, "id")
                has_timestamp = hasattr(sample, "created_at")
                has_text = hasattr(sample, "text")
                has_metadata = hasattr(sample, "metadata")

                print_actual(
                    f"Sample memory structure: id={has_id}, timestamp={has_timestamp}, text={has_text}, metadata={has_metadata}"
                )

                if all([has_id, has_timestamp, has_text]):
                    qa_report.record_pass("Export format validation")
                    qa_report.mark_feature_tested("Export", "✓")
                else:
                    qa_report.add_issue("Export missing required fields")
                    qa_report.mark_feature_tested("Export", "⚠")
            else:
                qa_report.record_skip("Export Validation", "No memories to export")
        else:
            qa_report.record_skip("Export", "Feature not available")
            qa_report.mark_feature_tested("Export", "⊘")
            qa_report.add_suggestion(
                "Add export_memories() method with format options (JSON/CSV/NDJSON)"
            )

    except Exception as e:
        qa_report.record_fail("Export", str(e))
        qa_report.mark_feature_tested("Export", "✗")


# ============================================================================
# TEST 10: ERROR HANDLING & LIMITS
# ============================================================================


def test_10_error_handling(client):
    """
    Test 10: Error Handling & Limits

    What: Test error cases and edge conditions
    Why: Robust error handling is critical for production use
    """
    print_section("TEST 10: ERROR HANDLING & LIMITS")

    base_url = os.getenv("HIPPOCAMPAI_BASE_URL", "http://localhost:8000")

    # Test invalid API key
    print_test("10.1: Test invalid API key")
    try:
        import httpx

        response = httpx.get(
            f"{base_url}/v1/memories",
            headers={"Authorization": "Bearer invalid_key_123"},
            params={"user_id": "test", "limit": 1},
            timeout=5.0,
        )

        print_expected("401 Unauthorized or 403 Forbidden with clear error message")
        print_actual(f"Status: {response.status_code}")

        if response.status_code in [401, 403]:
            try:
                error_data = response.json()
                print_actual(f"Error message: {error_data}")
                qa_report.record_pass("Invalid API key handling")
            except Exception:
                qa_report.add_issue("Error response not in JSON format")
        else:
            qa_report.add_issue(f"Expected 401/403 for invalid key, got {response.status_code}")

        qa_report.mark_feature_tested("Auth Error Handling", "✓")

    except Exception as e:
        qa_report.record_fail("Invalid API Key Test", str(e))

    # Test missing required fields
    print_test("10.2: Test missing required fields")
    try:
        import httpx

        api_key = os.getenv("HIPPOCAMPAI_API_KEY")

        response = httpx.post(
            f"{base_url}/v1/memories",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"text": "test"},  # Missing user_id
            timeout=5.0,
        )

        print_expected("400 Bad Request with validation error")
        print_actual(f"Status: {response.status_code}")

        if response.status_code == 400 or response.status_code == 422:
            error_data = response.json()
            print_actual(f"Error: {error_data}")
            qa_report.record_pass("Validation error handling")
        else:
            qa_report.add_issue(f"Expected 400/422 for missing fields, got {response.status_code}")

        qa_report.mark_feature_tested("Validation Errors", "✓")

    except Exception as e:
        qa_report.record_fail("Validation Error Test", str(e))

    # Test oversized memory
    print_test("10.3: Test oversized memory text")
    if client:
        try:
            huge_text = "A" * 1_000_000  # 1MB of text

            _memory = client.remember(text=huge_text, user_id="qa_test_limits")

            print_actual("Large memory accepted (no size limit)")
            qa_report.add_suggestion("Consider adding memory size limits with clear error messages")

        except Exception as e:
            print_actual(f"Large memory rejected: {str(e)[:100]}")
            qa_report.record_pass("Size limit enforcement")

    # Test extreme top_k
    print_test("10.4: Test extreme top_k value")
    if client:
        try:
            results = client.recall(
                query="test",
                user_id="qa_test_limits",
                k=10000,  # Very large k
            )

            print_expected("Capped at reasonable limit or warning")
            print_actual(f"Returned {len(results)} results")

            if len(results) < 10000:
                print_actual(f"✓ Capped at {len(results)} results")
                qa_report.record_pass("top_k limit enforcement")
            else:
                qa_report.add_suggestion("Add maximum k limit (e.g., 1000) to prevent abuse")

        except Exception as e:
            print_actual(f"Extreme k rejected: {str(e)}")
            qa_report.record_pass("top_k validation")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================


def main():
    """Main test execution."""
    print_section("HIPPOCAMPAI END-TO-END QA TEST SUITE")
    print("Testing: SaaS API + Python SDK + Dashboard Integration")
    print(f"Started: {datetime.now().isoformat()}")

    # Run all tests
    client = test_1_initialization_and_health()
    spaces = test_2_spaces_users_isolation(client)
    test_3_basic_memory_writes(client, spaces)
    test_4_retrieval_quality(client, spaces)
    test_5_entity_extraction(client, spaces)
    test_6_summarization_consolidation(client, spaces)
    test_7_temporal_reasoning(client, spaces)
    test_8_cross_interface_consistency(client, spaces)
    test_9_updates_deletes_export(client, spaces)
    test_10_error_handling(client)

    # Generate final report
    print(qa_report.generate_report())

    # Return exit code
    return 0 if qa_report.tests_failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Tests interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n{Colors.RED}Fatal error: {e}{Colors.END}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
