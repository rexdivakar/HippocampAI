-- Migration: Add metadata columns to users table
-- Run this after initial schema setup to add user metadata tracking

-- Add metadata columns to users table
ALTER TABLE users ADD COLUMN IF NOT EXISTS country VARCHAR(2);  -- ISO country code
ALTER TABLE users ADD COLUMN IF NOT EXISTS region VARCHAR(100);  -- State/Province
ALTER TABLE users ADD COLUMN IF NOT EXISTS city VARCHAR(100);
ALTER TABLE users ADD COLUMN IF NOT EXISTS timezone VARCHAR(50);
ALTER TABLE users ADD COLUMN IF NOT EXISTS signup_ip INET;  -- IP address on signup
ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login_ip INET;  -- IP address on last login
ALTER TABLE users ADD COLUMN IF NOT EXISTS user_agent TEXT;  -- Browser/client info
ALTER TABLE users ADD COLUMN IF NOT EXISTS referrer VARCHAR(500);  -- How they found us
ALTER TABLE users ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb;  -- Additional custom metadata

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_users_country ON users(country) WHERE country IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_users_signup_ip ON users(signup_ip) WHERE signup_ip IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_users_last_login_ip ON users(last_login_ip) WHERE last_login_ip IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_users_metadata ON users USING GIN(metadata) WHERE metadata IS NOT NULL;

-- Add metadata to sessions table
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS country VARCHAR(2);
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS city VARCHAR(100);

-- Update existing admin user with metadata
UPDATE users
SET
    country = 'US',
    city = 'San Francisco',
    timezone = 'America/Los_Angeles',
    signup_ip = '127.0.0.1'::inet,
    metadata = '{"source": "default_admin"}'::jsonb
WHERE email = 'admin@hippocampai.com' AND metadata IS NULL;
