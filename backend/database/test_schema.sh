#!/bin/bash
# Test script for database schema validation

echo "Testing LLM Arena Analytics Database Schema"
echo "============================================"
echo ""

# Check if database exists, if not create it
echo "1. Creating database if it doesn't exist..."
psql -U postgres -c "SELECT 1 FROM pg_database WHERE datname='llm_arena'" | grep -q 1 || psql -U postgres -c "CREATE DATABASE llm_arena;"

echo ""
echo "2. Running schema.sql..."
psql -U postgres -d llm_arena -f backend/database/schema.sql

echo ""
echo "3. Listing all tables (should show 5 tables):"
psql -U postgres -d llm_arena -c "\dt"

echo ""
echo "4. Showing models table structure:"
psql -U postgres -d llm_arena -c "\d models"

echo ""
echo "5. Verifying foreign keys:"
psql -U postgres -d llm_arena -c "
SELECT
    tc.table_name,
    kcu.column_name,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name
FROM information_schema.table_constraints AS tc
JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
JOIN information_schema.constraint_column_usage AS ccu
    ON ccu.constraint_name = tc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY'
    AND ccu.table_name = 'models';
"

echo ""
echo "6. Verifying indexes on date columns:"
psql -U postgres -d llm_arena -c "
SELECT
    tablename,
    indexname
FROM pg_indexes
WHERE schemaname = 'public'
    AND (indexname LIKE '%date%' OR indexname LIKE '%_at%')
ORDER BY tablename, indexname;
"

echo ""
echo "7. Verifying TIMESTAMPTZ usage:"
psql -U postgres -d llm_arena -c "
SELECT
    table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_schema = 'public'
    AND data_type = 'timestamp with time zone'
ORDER BY table_name, column_name;
"

echo ""
echo "Schema validation complete!"

