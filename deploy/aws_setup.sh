#!/bin/bash
# AWS EC2 + RDS Deployment Script for LLM Arena Analytics
# This script sets up the application on AWS EC2 with RDS PostgreSQL

set -e

echo "=========================================="
echo "AWS Deployment Setup"
echo "=========================================="
echo ""

# Configuration
INSTANCE_TYPE=${INSTANCE_TYPE:-t3.medium}
REGION=${AWS_REGION:-us-east-1}
KEY_NAME=${AWS_KEY_NAME:-llm-arena-key}
SECURITY_GROUP_NAME=${SECURITY_GROUP_NAME:-llm-arena-sg}
DB_INSTANCE_CLASS=${DB_INSTANCE_CLASS:-db.t3.micro}
DB_NAME=${DB_NAME:-llm_arena_analytics}
DB_USER=${DB_USER:-postgres}
DB_PASSWORD=${DB_PASSWORD:-$(openssl rand -base64 32)}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI not installed. Install from https://aws.amazon.com/cli/${NC}"
    exit 1
fi

# Check if AWS credentials are configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}Error: AWS credentials not configured. Run 'aws configure'${NC}"
    exit 1
fi

echo -e "${GREEN}Step 1: Creating Security Group...${NC}"
# Create security group
SG_ID=$(aws ec2 create-security-group \
    --group-name $SECURITY_GROUP_NAME \
    --description "Security group for LLM Arena Analytics" \
    --region $REGION \
    --query 'GroupId' \
    --output text 2>/dev/null || \
    aws ec2 describe-security-groups \
    --group-names $SECURITY_GROUP_NAME \
    --region $REGION \
    --query 'SecurityGroups[0].GroupId' \
    --output text)

# Add inbound rules
aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0 \
    --region $REGION 2>/dev/null || true

aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0 \
    --region $REGION 2>/dev/null || true

aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0 \
    --region $REGION 2>/dev/null || true

aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 8000 \
    --cidr 0.0.0.0/0 \
    --region $REGION 2>/dev/null || true

aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 8501 \
    --cidr 0.0.0.0/0 \
    --region $REGION 2>/dev/null || true

echo -e "${GREEN}Security Group ID: $SG_ID${NC}"
echo ""

echo -e "${GREEN}Step 2: Creating RDS PostgreSQL Instance...${NC}"
# Create RDS instance
DB_INSTANCE_ID="llm-arena-db"

# Check if instance already exists
if aws rds describe-db-instances --db-instance-identifier $DB_INSTANCE_ID --region $REGION &>/dev/null; then
    echo -e "${YELLOW}RDS instance already exists${NC}"
    DB_ENDPOINT=$(aws rds describe-db-instances \
        --db-instance-identifier $DB_INSTANCE_ID \
        --region $REGION \
        --query 'DBInstances[0].Endpoint.Address' \
        --output text)
else
    aws rds create-db-instance \
        --db-instance-identifier $DB_INSTANCE_ID \
        --db-instance-class $DB_INSTANCE_CLASS \
        --engine postgres \
        --engine-version 15.4 \
        --master-username $DB_USER \
        --master-user-password $DB_PASSWORD \
        --allocated-storage 20 \
        --storage-type gp2 \
        --vpc-security-group-ids $SG_ID \
        --backup-retention-period 7 \
        --region $REGION \
        --publicly-accessible \
        --no-multi-az

    echo -e "${YELLOW}Waiting for RDS instance to be available (this may take 10-15 minutes)...${NC}"
    aws rds wait db-instance-available \
        --db-instance-identifier $DB_INSTANCE_ID \
        --region $REGION

    DB_ENDPOINT=$(aws rds describe-db-instances \
        --db-instance-identifier $DB_INSTANCE_ID \
        --region $REGION \
        --query 'DBInstances[0].Endpoint.Address' \
        --output text)
fi

echo -e "${GREEN}RDS Endpoint: $DB_ENDPOINT${NC}"
echo ""

echo -e "${GREEN}Step 3: Launching EC2 Instance...${NC}"
# Get latest Amazon Linux 2023 AMI
AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=al2023-ami-2023*" "Name=architecture,Values=x86_64" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text \
    --region $REGION)

# Create user data script
USER_DATA=$(cat <<EOF
#!/bin/bash
yum update -y
yum install -y docker git

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-\$(uname -s)-\$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Start Docker
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install certbot for SSL
yum install -y certbot python3-certbot-nginx

# Clone repository (update with your repo URL)
cd /home/ec2-user
# git clone https://github.com/yourusername/llm-arena-analytics.git
# cd llm-arena-analytics

# Create .env file
cat > .env <<ENVEOF
DB_HOST=$DB_ENDPOINT
DB_PORT=5432
DB_NAME=$DB_NAME
DB_USER=$DB_USER
DB_PASSWORD=$DB_PASSWORD
ENVEOF

# Run docker-compose
# docker-compose up -d
EOF
)

# Launch EC2 instance
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SG_ID \
    --user-data "$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=llm-arena-analytics}]" \
    --region $REGION \
    --query 'Instances[0].InstanceId' \
    --output text)

echo -e "${GREEN}EC2 Instance ID: $INSTANCE_ID${NC}"
echo -e "${YELLOW}Waiting for instance to be running...${NC}"
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo -e "${GREEN}Public IP: $PUBLIC_IP${NC}"
echo ""

echo -e "${GREEN}Step 4: Setting up CloudWatch Monitoring...${NC}"
# Create CloudWatch alarm for high CPU
aws cloudwatch put-metric-alarm \
    --alarm-name llm-arena-high-cpu \
    --alarm-description "Alert when CPU exceeds 80%" \
    --metric-name CPUUtilization \
    --namespace AWS/EC2 \
    --statistic Average \
    --period 300 \
    --threshold 80 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2 \
    --dimensions Name=InstanceId,Value=$INSTANCE_ID \
    --region $REGION 2>/dev/null || echo "CloudWatch alarm creation skipped"

echo ""
echo "=========================================="
echo -e "${GREEN}Deployment Summary${NC}"
echo "=========================================="
echo "EC2 Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo "RDS Endpoint: $DB_ENDPOINT"
echo "Database Name: $DB_NAME"
echo "Database User: $DB_USER"
echo "Database Password: $DB_PASSWORD"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. SSH into instance: ssh -i your-key.pem ec2-user@$PUBLIC_IP"
echo "2. Clone your repository"
echo "3. Update .env file with database credentials"
echo "4. Run: docker-compose up -d"
echo "5. Set up SSL with Let's Encrypt: certbot --nginx"
echo ""
echo -e "${RED}IMPORTANT: Save the database password securely!${NC}"
echo "=========================================="

