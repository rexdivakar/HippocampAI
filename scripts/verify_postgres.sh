#!/bin/bash
# Verify PostgreSQL database schema for HippocampAI authentication

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}HippocampAI PostgreSQL Schema Verification${NC}"
echo "==========================================="

# Configuration from environment or defaults
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-hippocampai}"
POSTGRES_USER="${POSTGRES_USER:-hippocampai}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-hippocampai_secret}"

# Check if PostgreSQL is running
echo -e "\n${YELLOW}Checking PostgreSQL connection...${NC}"
if ! PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c '\q' 2>/dev/null; then
    echo -e "${RED}✗ Cannot connect to PostgreSQL${NC}"
    echo "Make sure PostgreSQL is running:"
    echo "  docker-compose up -d postgres"
    exit 1
fi

echo -e "${GREEN}✓ Connected to PostgreSQL${NC}"

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
    COUNT=$(PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_name = '$table';" 2>/dev/null | xargs)

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
    COUNT=$(PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT COUNT(*) FROM information_schema.views WHERE table_schema = 'public' AND table_name = '$view';" 2>/dev/null | xargs)

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
    COUNT=$(PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT COUNT(*) FROM pg_proc WHERE proname = '$func';" 2>/dev/null | xargs)

    if [ "$COUNT" -ge 1 ]; then
        echo -e "  ${GREEN}✓${NC} $func"
    else
        echo -e "  ${RED}✗${NC} $func (missing)"
    fi
done

# Check admin user
echo -e "\n${YELLOW}Checking default admin user...${NC}"
ADMIN_COUNT=$(PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT COUNT(*) FROM users WHERE email = 'admin@hippocampai.com' AND is_admin = true;" 2>/dev/null | xargs)

if [ "$ADMIN_COUNT" -eq 1 ]; then
    echo -e "  ${GREEN}✓${NC} Admin user exists (admin@hippocampai.com)"
else
    echo -e "  ${RED}✗${NC} Admin user not found"
fi

# Count records
echo -e "\n${YELLOW}Database statistics...${NC}"
USER_COUNT=$(PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT COUNT(*) FROM users;" 2>/dev/null | xargs)
API_KEY_COUNT=$(PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT COUNT(*) FROM api_keys;" 2>/dev/null | xargs)

echo -e "  Users: ${GREEN}$USER_COUNT${NC}"
echo -e "  API Keys: ${GREEN}$API_KEY_COUNT${NC}"

# Final verdict
echo -e "\n${GREEN}========================================${NC}"
if [ ${#MISSING_TABLES[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ Schema verification passed!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "\nYour database is ready to use!"
    exit 0
else
    echo -e "${RED}✗ Schema verification failed!${NC}"
    echo -e "${RED}========================================${NC}"
    echo -e "\nMissing tables: ${MISSING_TABLES[*]}"
    echo -e "\nRun the initialization script:"
    echo -e "  ${YELLOW}./scripts/init_postgres.sh${NC}"
    exit 1
fi
