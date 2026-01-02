# Quick Start Deployment Guide

## Local Development with Docker

```bash
# Build and start all services
docker-compose up --build

# Access services:
# - Frontend: http://localhost:8501
# - Backend API: http://localhost:8000
# - API Docs: http://localhost:8000/docs

# Stop services
docker-compose down
```

## AWS Deployment

### Prerequisites
```bash
# Install AWS CLI
aws --version

# Configure credentials
aws configure
```

### Deploy
```bash
cd deploy
./aws_setup.sh
```

### Post-Deployment
1. SSH into EC2 instance
2. Clone your repository
3. Update `.env` with RDS credentials
4. Run `docker-compose up -d`
5. Set up SSL: `certbot --nginx`

## GCP Deployment

### Prerequisites
```bash
# Install gcloud CLI
gcloud --version

# Login and set project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### Deploy
```bash
cd deploy
export GCP_PROJECT_ID=your-project-id
./gcp_setup.sh
```

### Post-Deployment
1. Run database migrations
2. Access services via provided URLs
3. Configure custom domain (optional)

## Database Setup

### Initial Schema
```bash
# Local
psql -U postgres -d llm_arena_analytics -f backend/database/schema.sql

# Or use migration script
./deploy/migrate.sh
```

### Backups
```bash
# Manual backup
./deploy/backup.sh

# Automated (add to crontab)
0 2 * * * /path/to/deploy/backup.sh
```

## Environment Variables

Create `.env` file:
```env
DB_HOST=your-db-host
DB_PORT=5432
DB_NAME=llm_arena_analytics
DB_USER=postgres
DB_PASSWORD=your-password
```

## Troubleshooting

### Database Connection
```bash
# Test connection
psql -h DB_HOST -U DB_USER -d DB_NAME

# Check from container
docker-compose exec backend python -c "from database.db_manager import DatabaseManager; db = DatabaseManager(); print(len(db.get_models()), 'models')"
```

### Container Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Health Checks
```bash
# Backend
curl http://localhost:8000/health

# Frontend
curl http://localhost:8501/_stcore/health
```

