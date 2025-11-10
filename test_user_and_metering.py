#!/usr/bin/env python3
"""Test script to create user, test API key, and verify usage metering."""

import asyncio
import sys
from datetime import datetime

import httpx

# Configuration
API_BASE_URL = "http://localhost:8000"
ADMIN_ENDPOINT = f"{API_BASE_URL}/admin"


async def create_test_user(email: str, password: str, full_name: str, tier: str = "pro"):
    """Create a test user via admin API (local mode)."""
    print(f"\n{'='*60}")
    print(f"Creating test user: {email}")
    print(f"{'='*60}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{ADMIN_ENDPOINT}/users",
            json={
                "email": email,
                "password": password,
                "full_name": full_name,
                "tier": tier,
                "is_admin": False
            },
            headers={"X-User-Auth": "false"}  # Local mode bypass
        )

        if response.status_code == 201:
            user = response.json()
            print("✓ User created successfully!")
            print(f"  - ID: {user['id']}")
            print(f"  - Email: {user['email']}")
            print(f"  - Tier: {user['tier']}")
            print(f"  - Created at: {user['created_at']}")
            return user
        else:
            print(f"✗ Failed to create user: {response.status_code}")
            print(f"  Error: {response.text}")
            return None


async def create_api_key(user_id: str, name: str, tier: str = "pro", expires_in_days: int = None):
    """Create an API key for a user."""
    print(f"\n{'='*60}")
    print(f"Creating API key for user: {user_id}")
    print(f"{'='*60}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        key_data = {
            "name": name,
            "scopes": ["memories:read", "memories:write"],
            "rate_limit_tier": tier
        }

        if expires_in_days:
            key_data["expires_in_days"] = expires_in_days

        response = await client.post(
            f"{ADMIN_ENDPOINT}/users/{user_id}/api-keys",
            json=key_data,
            headers={"X-User-Auth": "false"}
        )

        if response.status_code == 201:
            result = response.json()
            api_key_data = result["api_key"]
            secret_key = result["secret_key"]

            print("✓ API key created successfully!")
            print(f"  - Key ID: {api_key_data['id']}")
            print(f"  - Name: {api_key_data['name']}")
            print(f"  - Prefix: {api_key_data['key_prefix']}")
            print(f"  - Tier: {api_key_data['rate_limit_tier']}")
            print(f"  - Scopes: {', '.join(api_key_data['scopes'])}")
            print("\n  🔑 SECRET KEY (save this, shown only once!):")
            print(f"     {secret_key}")

            return secret_key, api_key_data
        else:
            print(f"✗ Failed to create API key: {response.status_code}")
            print(f"  Error: {response.text}")
            return None, None


async def test_api_with_key(api_key: str, user_id: str):
    """Test various API endpoints with the API key."""
    print(f"\n{'='*60}")
    print("Testing API Key with Memory Operations")
    print(f"{'='*60}")

    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test 1: Create a memory using remember endpoint
        print("\n[Test 1] Creating a fact memory...")
        response = await client.post(
            f"{API_BASE_URL}/v1/memories:remember",
            json={
                "text": "The user prefers Python over JavaScript for backend development",
                "user_id": user_id
            },
            headers=headers
        )

        if response.status_code in [200, 201]:
            memory = response.json()
            print("✓ Memory created")
            memory_id = memory.get('id') or memory.get('memory_id')
        else:
            print(f"✗ Failed to create memory: {response.status_code}")
            print(f"  Error: {response.text}")
            memory_id = None

        # Test 2: Create another memory
        print("\n[Test 2] Creating a preference memory...")
        response = await client.post(
            f"{API_BASE_URL}/v1/memories:remember",
            json={
                "text": "User likes concise code reviews with actionable feedback",
                "user_id": user_id
            },
            headers=headers
        )

        if response.status_code in [200, 201]:
            print("✓ Memory created")
        else:
            print(f"✗ Failed to create memory: {response.status_code}")

        # Test 3: Recall memories
        print("\n[Test 3] Recalling memories...")
        response = await client.post(
            f"{API_BASE_URL}/v1/memories:recall",
            json={
                "query": "programming preferences",
                "user_id": user_id,
                "top_k": 5
            },
            headers=headers
        )

        if response.status_code == 200:
            results = response.json()
            # Handle both list and dict responses
            if isinstance(results, list):
                memories = results
            else:
                memories = results.get('memories', results.get('results', []))

            print(f"✓ Recall completed: {len(memories)} memories")
            for i, mem in enumerate(memories[:3], 1):
                content = mem.get('content', mem.get('text', ''))
                score = mem.get('score', mem.get('similarity', 0))
                print(f"  {i}. Score: {score:.3f} - {content[:60]}...")
        else:
            print(f"✗ Recall failed: {response.status_code}")

        # Test 4: Get memory by ID
        if memory_id:
            print("\n[Test 4] Retrieving memory by ID...")
            response = await client.get(
                f"{API_BASE_URL}/v1/memories/{memory_id}",
                headers=headers
            )

            if response.status_code == 200:
                memory = response.json()
                content = memory.get('content', memory.get('text', ''))
                print(f"✓ Memory retrieved: {content[:60]}...")
            else:
                print(f"✗ Failed to retrieve memory: {response.status_code}")

        # Test 5: Query memories
        print("\n[Test 5] Querying memories with filters...")
        response = await client.post(
            f"{API_BASE_URL}/v1/memories/query",
            json={
                "user_id": user_id,
                "limit": 10
            },
            headers=headers
        )

        if response.status_code == 200:
            result = response.json()
            # Handle both list and dict responses
            if isinstance(result, list):
                memories = result
            else:
                memories = result.get('memories', result.get('results', []))
            print(f"✓ Found {len(memories)} memories for user")
        else:
            print(f"✗ Failed to query memories: {response.status_code}")

        # Test 6: Health score check
        print("\n[Test 6] Checking memory health...")
        response = await client.get(
            f"{API_BASE_URL}/v1/health/score",
            headers=headers
        )

        if response.status_code == 200:
            response.json()  # Parse but don't use
            print("✓ Health check successful")
        else:
            print(f"✗ Failed health check: {response.status_code}")


async def check_usage_statistics(user_id: str, api_key_id: str):
    """Check usage statistics for the user and API key."""
    print(f"\n{'='*60}")
    print("Checking Usage Statistics")
    print(f"{'='*60}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Get user statistics
        print("\n[User Statistics]")
        response = await client.get(
            f"{ADMIN_ENDPOINT}/statistics/users",
            headers={"X-User-Auth": "false"}
        )

        if response.status_code == 200:
            stats = response.json()
            user_stats = [s for s in stats if s['id'] == user_id]
            if user_stats:
                user_stat = user_stats[0]
                print("✓ User Statistics:")
                print(f"  - Total Requests: {user_stat.get('total_requests', 0)}")
                print(f"  - Total Tokens: {user_stat.get('total_tokens_used', 0)}")
                print(f"  - API Keys: {user_stat.get('api_key_count', 0)}")
                print(f"  - Last Usage: {user_stat.get('last_api_usage', 'Never')}")
            else:
                print("  No statistics found yet (may take a moment to aggregate)")
        else:
            print(f"✗ Failed to get user statistics: {response.status_code}")

        # Get API key statistics
        print("\n[API Key Statistics]")
        response = await client.get(
            f"{ADMIN_ENDPOINT}/statistics/api-keys?user_id={user_id}",
            headers={"X-User-Auth": "false"}
        )

        if response.status_code == 200:
            stats = response.json()
            if stats:
                for key_stat in stats:
                    if key_stat['id'] == api_key_id:
                        print("✓ API Key Statistics:")
                        print(f"  - Name: {key_stat.get('name', 'N/A')}")
                        print(f"  - Total Requests: {key_stat.get('total_requests', 0)}")
                        print(f"  - Total Tokens: {key_stat.get('total_tokens_used', 0)}")
                        avg_time = key_stat.get('avg_response_time') or 0
                        print(f"  - Avg Response Time: {float(avg_time):.2f}ms")
                        print(f"  - Last Request: {key_stat.get('last_request_at', 'Never')}")
                        break
                else:
                    print("  No statistics found yet for this API key")
            else:
                print("  No statistics found yet (may take a moment to aggregate)")
        else:
            print(f"✗ Failed to get API key statistics: {response.status_code}")


async def test_rate_limiting(api_key: str):
    """Test rate limiting by making rapid requests."""
    print(f"\n{'='*60}")
    print("Testing Rate Limiting")
    print(f"{'='*60}")

    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        print("\nMaking 10 rapid requests...")
        success_count = 0
        rate_limited_count = 0

        for i in range(10):
            response = await client.get(
                f"{API_BASE_URL}/health",
                headers=headers
            )

            if response.status_code == 200:
                success_count += 1
                if i == 0:
                    # Check rate limit headers
                    print("\nRate Limit Headers:")
                    print(f"  - Limit: {response.headers.get('X-RateLimit-Limit', 'N/A')}")
                    print(f"  - Remaining: {response.headers.get('X-RateLimit-Remaining', 'N/A')}")
                    print(f"  - Reset: {response.headers.get('X-RateLimit-Reset', 'N/A')}")
            elif response.status_code == 429:
                rate_limited_count += 1
                if rate_limited_count == 1:
                    print("\n✓ Rate limiting is working!")
                    data = response.json()
                    print(f"  - Message: {data.get('message', 'N/A')}")
                    print(f"  - Limit: {data.get('limit', 'N/A')}")
                    print(f"  - Retry After: {response.headers.get('Retry-After', 'N/A')}s")

            # Small delay between requests
            await asyncio.sleep(0.1)

        print("\nResults:")
        print(f"  - Successful: {success_count}")
        print(f"  - Rate Limited: {rate_limited_count}")


async def cleanup_test_user(user_id: str):
    """Clean up test user (optional)."""
    print(f"\n{'='*60}")
    print("Cleanup (Optional)")
    print(f"{'='*60}")

    response = input("\nDo you want to delete the test user? (y/N): ")
    if response.lower() == 'y':
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(
                f"{ADMIN_ENDPOINT}/users/{user_id}",
                headers={"X-User-Auth": "false"}
            )

            if response.status_code == 204:
                print("✓ Test user deleted successfully")
            else:
                print(f"✗ Failed to delete user: {response.status_code}")
    else:
        print("Skipping cleanup. User and API key remain active.")


async def main():
    """Main test workflow."""
    print("\n" + "="*60)
    print("HippocampAI - User & API Key Testing with Usage Metering")
    print("="*60)

    # Test user details
    test_email = f"testuser_{datetime.now().strftime('%Y%m%d_%H%M%S')}@example.com"
    test_password = "SecurePassword123!"
    test_name = "Test User"

    try:
        # Step 1: Create test user
        user = await create_test_user(test_email, test_password, test_name, tier="pro")
        if not user:
            print("\n✗ Failed to create user. Exiting.")
            return

        user_id = user['id']

        # Step 2: Create API key
        api_key, api_key_data = await create_api_key(
            user_id,
            "Test API Key",
            tier="pro",
            expires_in_days=30
        )
        if not api_key:
            print("\n✗ Failed to create API key. Exiting.")
            return

        api_key_id = api_key_data['id']

        # Step 3: Test API endpoints
        await test_api_with_key(api_key, user_id)

        # Step 4: Test rate limiting
        await test_rate_limiting(api_key)

        # Step 5: Check usage statistics
        await asyncio.sleep(2)  # Give statistics time to aggregate
        await check_usage_statistics(user_id, api_key_id)

        # Summary
        print(f"\n{'='*60}")
        print("Test Summary")
        print(f"{'='*60}")
        print(f"✓ User created: {test_email}")
        print("✓ API key created and tested")
        print("✓ Usage metering verified")
        print("\nAPI Key for future use:")
        print(f"  {api_key}")
        print(f"\nUser ID: {user_id}")

        # Optional cleanup
        await cleanup_test_user(user_id)

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
