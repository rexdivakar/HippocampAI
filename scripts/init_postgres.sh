#!/bin/bash
# Initialize or reset PostgreSQL database schema for HippocampAI authentication

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}HippocampAI PostgreSQL Schema Initialization${NC}"
echo "=============================================="

# Configuration from environment or defaults
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-hippocampai}"
POSTGRES_USER="${POSTGRES_USER:-hippocampai}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-hippocampai_secret}"

# Check if PostgreSQL is running
echo -e "\n${YELLOW}Checking PostgreSQL connection...${NC}"
if ! PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c '\q' 2>/dev/null; then
    echo -e "${RED}ERROR: Cannot connect to PostgreSQL${NC}"
    echo "Make sure PostgreSQL is running:"
    echo "  docker-compose up -d postgres"
    exit 1
fi

echo -e "${GREEN}✓ Connected to PostgreSQL${NC}"

# Get schema file path
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCHEMA_FILE="$SCRIPT_DIR/../src/hippocampai/auth/schema.sql"

if [ ! -f "$SCHEMA_FILE" ]; then
    echo -e "${RED}ERROR: Schema file not found: $SCHEMA_FILE${NC}"
    exit 1
fi

# Ask for confirmation if tables already exist
echo -e "\n${YELLOW}Checking existing tables...${NC}"
EXISTING_TABLES=$(PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_name IN ('users', 'api_keys', 'sessions');" 2>/dev/null | xargs)

if [ "$EXISTING_TABLES" -gt 0 ]; then
    echo -e "${YELLOW}Found $EXISTING_TABLES existing authentication tables${NC}"
    read -p "Do you want to DROP existing tables and recreate them? (yes/no): " -r
    echo
    if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
        echo -e "${YELLOW}Skipping schema initialization${NC}"
        exit 0
    fi

    echo -e "${YELLOW}Dropping existing tables...${NC}"
    PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" <<EOF
DROP TABLE IF EXISTS audit_log CASCADE;
DROP TABLE IF EXISTS sessions CASCADE;
DROP TABLE IF EXISTS rate_limit_buckets CASCADE;
DROP TABLE IF EXISTS api_key_usage CASCADE;
DROP TABLE IF EXISTS api_key_usage_2024_01 CASCADE;
DROP TABLE IF EXISTS api_key_usage_2024_02 CASCADE;
DROP TABLE IF EXISTS api_keys CASCADE;
DROP TABLE IF EXISTS users CASCADE;
DROP TABLE IF EXISTS organizations CASCADE;
DROP VIEW IF EXISTS user_statistics CASCADE;
DROP VIEW IF EXISTS api_key_statistics CASCADE;
DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE;
DROP FUNCTION IF EXISTS create_default_admin() CASCADE;
EOF
    echo -e "${GREEN}✓ Existing tables dropped${NC}"
fi

# Run schema initialization
echo -e "\n${YELLOW}Running schema initialization...${NC}"
PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f "$SCHEMA_FILE"

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ Schema initialized successfully!${NC}"

    # Verify tables were created
    echo -e "\n${YELLOW}Verifying tables...${NC}"
    TABLES=$(PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;")

    echo -e "${GREEN}Tables created:${NC}"
    echo "$TABLES" | while read -r table; do
        if [ ! -z "$table" ]; then
            echo "  - $table"
        fi
    done

    # Check default admin user
    echo -e "\n${YELLOW}Checking default admin user...${NC}"
    ADMIN_COUNT=$(PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT COUNT(*) FROM users WHERE email = 'admin@hippocampai.com';" | xargs)

    if [ "$ADMIN_COUNT" -eq 1 ]; then
        echo -e "${GREEN}✓ Default admin user created${NC}"
        echo -e "  Email: ${GREEN}admin@hippocampai.com${NC}"
        echo -e "  Password: ${YELLOW}admin123${NC} (CHANGE THIS IN PRODUCTION!)"
    else
        echo -e "${RED}WARNING: Default admin user not found${NC}"
    fi

    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}Schema initialization complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "\nNext steps:"
    echo "  1. Access admin UI: http://localhost:3001"
    echo "  2. Login with admin@hippocampai.com / admin123"
    echo "  3. Change the default admin password!"
    echo "  4. Create users and API keys"

else
    echo -e "\n${RED}ERROR: Schema initialization failed${NC}"
    exit 1
fi
