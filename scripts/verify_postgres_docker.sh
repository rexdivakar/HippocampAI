#!/bin/bash
# Verify PostgreSQL database schema using docker exec (no local psql required)

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}HippocampAI PostgreSQL Schema Verification (Docker)${NC}"
echo "======================================================="

# Check if PostgreSQL container is running
echo -e "\n${YELLOW}Checking PostgreSQL container...${NC}"
if ! docker ps | grep -q hippocampai-postgres; then
    echo -e "${RED}✗ PostgreSQL container not running${NC}"
    echo "Start it with: docker-compose up -d postgres"
    exit 1
fi

echo -e "${GREEN}✓ PostgreSQL container is running${NC}"

# Expected tables
EXPECTED_TABLES=(
    "users"
    "api_keys"
    "api_key_usage"
    "organizations"
    "rate_limit_buckets"
    "sessions"
    "audit_log"
)

# Check tables
echo -e "\n${YELLOW}Checking tables...${NC}"
MISSING_TABLES=()

for table in "${EXPECTED_TABLES[@]}"; do
    COUNT=$(docker exec hippocampai-postgres psql -U hippocampai -d hippocampai -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_name = '$table';" 2>/dev/null | xargs)

    if [ "$COUNT" -eq 1 ]; then
        echo -e "  ${GREEN}✓${NC} $table"
    else
        echo -e "  ${RED}✗${NC} $table (missing)"
        MISSING_TABLES+=("$table")
    fi
done

# Check views
echo -e "\n${YELLOW}Checking views...${NC}"
EXPECTED_VIEWS=(
    "user_statistics"
    "api_key_statistics"
)

for view in "${EXPECTED_VIEWS[@]}"; do
    COUNT=$(docker exec hippocampai-postgres psql -U hippocampai -d hippocampai -t -c "SELECT COUNT(*) FROM information_schema.views WHERE table_schema = 'public' AND table_name = '$view';" 2>/dev/null | xargs)

    if [ "$COUNT" -eq 1 ]; then
        echo -e "  ${GREEN}✓${NC} $view"
    else
        echo -e "  ${RED}✗${NC} $view (missing)"
    fi
done

# Check functions
echo -e "\n${YELLOW}Checking functions...${NC}"
EXPECTED_FUNCTIONS=(
    "update_updated_at_column"
    "create_default_admin"
)

for func in "${EXPECTED_FUNCTIONS[@]}"; do
    COUNT=$(docker exec hippocampai-postgres psql -U hippocampai -d hippocampai -t -c "SELECT COUNT(*) FROM pg_proc WHERE proname = '$func';" 2>/dev/null | xargs)

    if [ "$COUNT" -ge 1 ]; then
        echo -e "  ${GREEN}✓${NC} $func"
    else
        echo -e "  ${RED}✗${NC} $func (missing)"
    fi
done

# Check admin user
echo -e "\n${YELLOW}Checking default admin user...${NC}"
ADMIN_COUNT=$(docker exec hippocampai-postgres psql -U hippocampai -d hippocampai -t -c "SELECT COUNT(*) FROM users WHERE email = 'admin@hippocampai.com' AND is_admin = true;" 2>/dev/null | xargs)

if [ "$ADMIN_COUNT" -eq 1 ]; then
    echo -e "  ${GREEN}✓${NC} Admin user exists (admin@hippocampai.com)"
else
    echo -e "  ${RED}✗${NC} Admin user not found"
fi

# Count records
echo -e "\n${YELLOW}Database statistics...${NC}"
USER_COUNT=$(docker exec hippocampai-postgres psql -U hippocampai -d hippocampai -t -c "SELECT COUNT(*) FROM users;" 2>/dev/null | xargs)
API_KEY_COUNT=$(docker exec hippocampai-postgres psql -U hippocampai -d hippocampai -t -c "SELECT COUNT(*) FROM api_keys;" 2>/dev/null | xargs)

echo -e "  Users: ${GREEN}$USER_COUNT${NC}"
echo -e "  API Keys: ${GREEN}$API_KEY_COUNT${NC}"

# Final verdict
echo -e "\n${GREEN}========================================${NC}"
if [ ${#MISSING_TABLES[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ Schema verification passed!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "\nYour database is ready to use!"
    echo -e "\n${YELLOW}Quick Start:${NC}"
    echo -e "  1. Access admin UI: http://localhost:3001"
    echo -e "  2. Login: ${GREEN}admin@hippocampai.com${NC} / ${YELLOW}admin123${NC}"
    echo -e "  3. ${RED}IMPORTANT:${NC} Change the default admin password!"
    exit 0
else
    echo -e "${RED}✗ Schema verification failed!${NC}"
    echo -e "${RED}========================================${NC}"
    echo -e "\nMissing tables: ${MISSING_TABLES[*]}"
    echo -e "\nTo fix, recreate the database:"
    echo -e "  ${YELLOW}docker-compose stop postgres${NC}"
    echo -e "  ${YELLOW}docker-compose rm -f postgres${NC}"
    echo -e "  ${YELLOW}docker volume rm hippocampai_postgres_data${NC}"
    echo -e "  ${YELLOW}docker-compose up -d postgres${NC}"
    exit 1
fi
