#!/bin/bash
# Database migration script
# Runs schema updates and migrations

set -e

DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_NAME=${DB_NAME:-llm_arena_analytics}
DB_USER=${DB_USER:-postgres}

echo "Running database migrations..."

# Run schema
PGPASSWORD=$DB_PASSWORD psql \
    -h $DB_HOST \
    -p $DB_PORT \
    -U $DB_USER \
    -d $DB_NAME \
    -f ../backend/database/schema.sql

echo "Migrations completed successfully"

