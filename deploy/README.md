# Deployment Guide

This guide covers deploying LLM Arena Analytics to AWS and GCP.

## Prerequisites

### AWS Deployment
- AWS CLI installed and configured
- EC2 key pair created
- AWS account with appropriate permissions

### GCP Deployment
- gcloud CLI installed and configured
- GCP project created
- Billing enabled

## AWS EC2 + RDS Deployment

### Quick Start

```bash
cd deploy
chmod +x aws_setup.sh
./aws_setup.sh
```

### Manual Steps

1. **Create Security Group**
   ```bash
   aws ec2 create-security-group --group-name llm-arena-sg --description "LLM Arena Analytics"
   ```

2. **Create RDS Instance**
   ```bash
   aws rds create-db-instance \
     --db-instance-identifier llm-arena-db \
     --db-instance-class db.t3.micro \
     --engine postgres \
     --master-username postgres \
     --master-user-password YOUR_PASSWORD \
     --allocated-storage 20
   ```

3. **Launch EC2 Instance**
   - Use Amazon Linux 2023 AMI
   - Instance type: t3.medium (or larger)
   - Attach security group
   - Use your key pair

4. **SSH into Instance**
   ```bash
   ssh -i your-key.pem ec2-user@YOUR_IP
   ```

5. **Install Docker**
   ```bash
   sudo yum update -y
   sudo yum install -y docker git
   sudo systemctl start docker
   sudo usermod -a -G docker ec2-user
   ```

6. **Install Docker Compose**
   ```bash
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

7. **Clone Repository**
   ```bash
   git clone https://github.com/yourusername/llm-arena-analytics.git
   cd llm-arena-analytics
   ```

7. **Configure Environment**
   ```bash
   cd llm-arena-analytics
   cp .env.example .env
   # Edit .env with your RDS endpoint and credentials
   ```

8. **Run Application**
   ```bash
   docker-compose up -d
   ```

9. **Set up SSL (Let's Encrypt)**
   ```bash
   sudo yum install -y certbot python3-certbot-nginx
   sudo certbot --nginx -d your-domain.com
   ```

### Cost Optimization

- Use Spot Instances for non-production
- Enable RDS automated backups
- Use CloudWatch for monitoring
- Set up auto-scaling based on CPU

### Monitoring

- CloudWatch dashboards
- RDS performance insights
- EC2 CloudWatch metrics
- Application logs in CloudWatch Logs

## GCP Cloud Run + Cloud SQL Deployment

### Quick Start

```bash
cd deploy
chmod +x gcp_setup.sh
export GCP_PROJECT_ID=your-project-id
./gcp_setup.sh
```

### Manual Steps

1. **Enable APIs**
   ```bash
   gcloud services enable cloudbuild.googleapis.com run.googleapis.com sqladmin.googleapis.com
   ```

2. **Create Cloud SQL Instance**
   ```bash
   gcloud sql instances create llm-arena-db \
     --database-version=POSTGRES_15 \
     --tier=db-f1-micro \
     --region=us-central1
   ```

3. **Create Database**
   ```bash
   gcloud sql databases create llm_arena_analytics --instance=llm-arena-db
   ```

4. **Build Docker Images**
   ```bash
   # Backend
   cd backend
   gcloud builds submit --tag gcr.io/PROJECT_ID/llm-arena-backend
   
   # Frontend
   cd ../frontend
   gcloud builds submit --tag gcr.io/PROJECT_ID/llm-arena-frontend
   ```

5. **Deploy to Cloud Run**
   ```bash
   # Backend
   gcloud run deploy llm-arena-backend \
     --image gcr.io/PROJECT_ID/llm-arena-backend \
     --platform managed \
     --region us-central1 \
     --add-cloudsql-instances=PROJECT_ID:REGION:INSTANCE_NAME
   
   # Frontend
   gcloud run deploy llm-arena-frontend \
     --image gcr.io/PROJECT_ID/llm-arena-frontend \
     --platform managed \
     --region us-central1
   ```

6. **Set up VPC Connector**
   ```bash
   gcloud compute networks vpc-access connectors create llm-arena-connector \
     --region=us-central1 \
     --network=default \
     --range=10.8.0.0/28
   ```

### Cost Optimization

- Use Cloud Run (pay per request)
- Use Cloud SQL with automatic backups
- Enable Cloud Run min instances = 0 for cost savings
- Use Cloud SQL read replicas for scaling

### Monitoring

- Cloud Run metrics in Console
- Cloud SQL insights
- Cloud Logging for application logs
- Error Reporting for exceptions

## Database Migrations

### Run Migrations

```bash
cd deploy
chmod +x migrate.sh
./migrate.sh
```

Or manually:
```bash
psql -h DB_HOST -U postgres -d llm_arena_analytics -f backend/database/schema.sql
```

## Automated Backups

### Set up Cron Job

```bash
# Add to crontab (runs daily at 2 AM)
0 2 * * * /path/to/deploy/backup.sh >> /var/log/llm-arena-backup.log 2>&1
```

### AWS RDS Backups

RDS automatically creates daily backups. Retention period is configurable (default: 7 days).

### GCP Cloud SQL Backups

Cloud SQL automatically creates daily backups. Configure in Cloud Console.

## Environment Variables

### Required Variables

```env
DB_HOST=your-db-host
DB_PORT=5432
DB_NAME=llm_arena_analytics
DB_USER=postgres
DB_PASSWORD=your-secure-password
```

### Optional Variables

```env
API_BASE_URL=http://backend:8000
REDDIT_CLIENT_ID=your-reddit-client-id
REDDIT_CLIENT_SECRET=your-reddit-client-secret
```

## Security Best Practices

1. **Use Secrets Manager**
   - AWS: AWS Secrets Manager
   - GCP: Secret Manager

2. **Enable SSL/TLS**
   - Use Let's Encrypt for free SSL
   - Configure HTTPS in nginx

3. **Restrict Database Access**
   - Use security groups/firewall rules
   - Whitelist only necessary IPs

4. **Regular Updates**
   - Keep Docker images updated
   - Update system packages regularly

## Troubleshooting

### Database Connection Issues

1. Check security group/firewall rules
2. Verify database credentials
3. Test connection: `psql -h HOST -U USER -d DB_NAME`

### Container Issues

1. Check logs: `docker-compose logs`
2. Verify environment variables
3. Check health endpoints: `/health`

### Performance Issues

1. Monitor CloudWatch/Cloud Monitoring
2. Check database connection pool
3. Review query performance
4. Scale up resources if needed

## Cost Estimates

### AWS (Monthly)
- EC2 t3.medium: ~$30
- RDS db.t3.micro: ~$15
- Data transfer: ~$5-10
- **Total: ~$50-60/month**

### GCP (Monthly)
- Cloud Run (pay-per-use): ~$10-20
- Cloud SQL db-f1-micro: ~$10
- Storage: ~$5
- **Total: ~$25-35/month**

## Support

For issues or questions:
1. Check application logs
2. Review CloudWatch/Cloud Monitoring
3. Check database connection status
4. Verify environment variables

