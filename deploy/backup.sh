#!/bin/bash
# Automated backup script for LLM Arena Analytics database
# Can be run as a cron job for regular backups

set -e

# Configuration
BACKUP_DIR=${BACKUP_DIR:-./backups}
RETENTION_DAYS=${RETENTION_DAYS:-30}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Database connection (from environment or arguments)
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_NAME=${DB_NAME:-llm_arena_analytics}
DB_USER=${DB_USER:-postgres}

mkdir -p $BACKUP_DIR

echo "Starting database backup..."
echo "Database: $DB_NAME"
echo "Backup file: $BACKUP_DIR/backup_${DB_NAME}_${TIMESTAMP}.sql"

# Create backup
PGPASSWORD=$DB_PASSWORD pg_dump \
    -h $DB_HOST \
    -p $DB_PORT \
    -U $DB_USER \
    -d $DB_NAME \
    -F c \
    -f $BACKUP_DIR/backup_${DB_NAME}_${TIMESTAMP}.dump

# Compress backup
gzip $BACKUP_DIR/backup_${DB_NAME}_${TIMESTAMP}.dump

echo "Backup completed: $BACKUP_DIR/backup_${DB_NAME}_${TIMESTAMP}.dump.gz"

# Clean up old backups
find $BACKUP_DIR -name "backup_*.dump.gz" -mtime +$RETENTION_DAYS -delete

echo "Old backups cleaned up (retention: $RETENTION_DAYS days)"

