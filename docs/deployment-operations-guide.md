# ColPali-BAML Deployment & Operations Guide

> **Operational Reference Document**: Step-by-step deployment procedures, monitoring setup, and operational runbooks for the ColPali-BAML Vision Processing Engine.

## Table of Contents
1. [Environment Overview](#environment-overview)
2. [Local Development Deployment](#local-development-deployment)
3. [AWS Lambda Deployment](#aws-lambda-deployment)
4. [Monitoring & Alerting Setup](#monitoring--alerting-setup)
5. [Backup & Recovery Procedures](#backup--recovery-procedures)
6. [Scaling & Capacity Planning](#scaling--capacity-planning)
7. [Incident Response Procedures](#incident-response-procedures)
8. [Operational Runbooks](#operational-runbooks)

---

## Environment Overview

### Deployment Targets

| Environment | Container | Purpose | Memory | Ports |
|------------|-----------|---------|--------|-------|
| Development | `Dockerfile.dev` | Local development with hot-reload | 16GB+ | 8000, 5000 |
| Jupyter | `Dockerfile.jupyter` | Interactive analysis & experimentation | 16GB+ | 8888 |
| Lambda | `Dockerfile.lambda` | Production serverless deployment | 10GB | N/A |

### Infrastructure Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    ColPali-BAML Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   API GW /   │    │   Lambda     │    │   Qdrant     │      │
│  │   ALB        │───▶│   Container  │───▶│   Vector DB  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                              │                                   │
│                              ▼                                   │
│                      ┌──────────────┐                           │
│                      │  CloudWatch  │                           │
│                      │  Logs/Metrics│                           │
│                      └──────────────┘                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Local Development Deployment

### Prerequisites

- Docker Desktop 4.0+ with 16GB+ memory allocation
- Docker Compose v2.0+
- 50GB+ free disk space (for models and data)
- Git

### Step 1: Clone and Configure

```bash
# Clone repository
git clone https://github.com/your-org/colpali-baml.git
cd colpali-baml

# Create environment file
cat > .env << 'EOF'
QDRANT_URL=http://qdrant:6333
BAML_ENV=development
LOG_LEVEL=DEBUG
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
EOF
```

### Step 2: Build Development Containers

```bash
# Build all containers
docker-compose build

# Verify builds
docker images | grep colpali
```

**Expected Output:**
```
colpali-baml_colpali-engine   latest   abc123   <size>
colpali-baml_jupyter          latest   def456   <size>
```

### Step 3: Start Services

```bash
# Start all services
docker-compose up -d

# Verify services are healthy
docker-compose ps

# Check service logs
docker-compose logs -f colpali-engine
```

**Expected Service Status:**
```
NAME              STATUS                   PORTS
colpali-qdrant    Up (healthy)            0.0.0.0:6333-6334->6333-6334/tcp
colpali-dev       Up (healthy)            0.0.0.0:8000->8000/tcp, 0.0.0.0:5000->5000/tcp
colpali-jupyter   Up (healthy)            0.0.0.0:8888->8888/tcp
```

### Step 4: Verify Health

```bash
# Check Qdrant health
curl http://localhost:6333/health

# Check API health
curl http://localhost:8000/health

# Check Jupyter
open http://localhost:8888
```

### Step 5: Run Tests

```bash
# Execute in development container
docker-compose exec colpali-engine bash

# Inside container
PYTHONPATH=. pytest tests/ -v --tb=short
```

### Stopping Services

```bash
# Stop all services (preserve data)
docker-compose stop

# Stop and remove containers (preserve volumes)
docker-compose down

# Complete cleanup (removes volumes)
docker-compose down -v
```

---

## AWS Lambda Deployment

### Prerequisites

- AWS CLI v2 configured with appropriate permissions
- ECR repository created
- IAM role with Lambda execution permissions
- (Optional) Qdrant Cloud instance or self-hosted Qdrant

### Step 1: Build Lambda Container

```bash
# Build Lambda-optimized image
docker build -f Dockerfile.lambda -t colpali-baml-lambda:latest .

# Verify image size (<10GB required for Lambda)
docker images colpali-baml-lambda:latest --format "{{.Size}}"
```

### Step 2: Push to ECR

```bash
# Configure variables
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO=colpali-baml-lambda

# Authenticate Docker to ECR
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Create ECR repository (if not exists)
aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION || true

# Tag and push image
docker tag colpali-baml-lambda:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest
```

### Step 3: Create Lambda Function

```bash
# Create Lambda function from container
aws lambda create-function \
  --function-name colpali-baml-processor \
  --package-type Image \
  --code ImageUri=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest \
  --role arn:aws:iam::$AWS_ACCOUNT_ID:role/LambdaExecutionRole \
  --memory-size 10240 \
  --timeout 300 \
  --environment "Variables={QDRANT_HOST=your-qdrant-host,QDRANT_PORT=6333}" \
  --region $AWS_REGION
```

### Step 4: Configure API Gateway

```bash
# Create REST API
API_ID=$(aws apigateway create-rest-api \
  --name "ColPali-BAML-API" \
  --description "ColPali-BAML Document Processing API" \
  --query 'id' --output text)

# Get root resource ID
ROOT_ID=$(aws apigateway get-resources \
  --rest-api-id $API_ID \
  --query 'items[0].id' --output text)

# Create /process resource
RESOURCE_ID=$(aws apigateway create-resource \
  --rest-api-id $API_ID \
  --parent-id $ROOT_ID \
  --path-part "process" \
  --query 'id' --output text)

# Create POST method
aws apigateway put-method \
  --rest-api-id $API_ID \
  --resource-id $RESOURCE_ID \
  --http-method POST \
  --authorization-type NONE

# Integrate with Lambda
aws apigateway put-integration \
  --rest-api-id $API_ID \
  --resource-id $RESOURCE_ID \
  --http-method POST \
  --type AWS_PROXY \
  --integration-http-method POST \
  --uri "arn:aws:apigateway:$AWS_REGION:lambda:path/2015-03-31/functions/arn:aws:lambda:$AWS_REGION:$AWS_ACCOUNT_ID:function:colpali-baml-processor/invocations"

# Deploy API
aws apigateway create-deployment \
  --rest-api-id $API_ID \
  --stage-name prod

echo "API Endpoint: https://$API_ID.execute-api.$AWS_REGION.amazonaws.com/prod/process"
```

### Step 5: Configure Provisioned Concurrency (Optional)

For production workloads, configure provisioned concurrency to eliminate cold starts:

```bash
# Publish a version
VERSION=$(aws lambda publish-version \
  --function-name colpali-baml-processor \
  --query 'Version' --output text)

# Configure provisioned concurrency
aws lambda put-provisioned-concurrency-config \
  --function-name colpali-baml-processor \
  --qualifier $VERSION \
  --provisioned-concurrent-executions 2
```

### Step 6: Update Lambda Function

```bash
# Build and push new image
docker build -f Dockerfile.lambda -t colpali-baml-lambda:latest .
docker tag colpali-baml-lambda:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest

# Update Lambda function
aws lambda update-function-code \
  --function-name colpali-baml-processor \
  --image-uri $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest
```

---

## Monitoring & Alerting Setup

### CloudWatch Dashboard Configuration

Create a comprehensive CloudWatch dashboard for monitoring:

```bash
# Create dashboard
aws cloudwatch put-dashboard \
  --dashboard-name "ColPali-BAML-Operations" \
  --dashboard-body file://cloudwatch-dashboard.json
```

**cloudwatch-dashboard.json:**
```json
{
  "widgets": [
    {
      "type": "metric",
      "x": 0,
      "y": 0,
      "width": 12,
      "height": 6,
      "properties": {
        "title": "Lambda Invocations & Errors",
        "metrics": [
          ["AWS/Lambda", "Invocations", "FunctionName", "colpali-baml-processor"],
          [".", "Errors", ".", "."],
          [".", "Throttles", ".", "."]
        ],
        "period": 60,
        "stat": "Sum",
        "region": "us-east-1"
      }
    },
    {
      "type": "metric",
      "x": 12,
      "y": 0,
      "width": 12,
      "height": 6,
      "properties": {
        "title": "Lambda Duration",
        "metrics": [
          ["AWS/Lambda", "Duration", "FunctionName", "colpali-baml-processor", {"stat": "Average"}],
          ["...", {"stat": "p99"}],
          ["...", {"stat": "Maximum"}]
        ],
        "period": 60,
        "region": "us-east-1"
      }
    },
    {
      "type": "metric",
      "x": 0,
      "y": 6,
      "width": 12,
      "height": 6,
      "properties": {
        "title": "Memory Usage",
        "metrics": [
          ["ColPali/Processing", "MemoryUsage", "Environment", "production"]
        ],
        "period": 60,
        "stat": "Average",
        "region": "us-east-1"
      }
    },
    {
      "type": "metric",
      "x": 12,
      "y": 6,
      "width": 12,
      "height": 6,
      "properties": {
        "title": "Processing Accuracy",
        "metrics": [
          ["ColPali/Processing", "ExtractionAccuracy", "Environment", "production"]
        ],
        "period": 300,
        "stat": "Average",
        "region": "us-east-1"
      }
    }
  ]
}
```

### CloudWatch Alarms

Configure critical alerts:

```bash
# High Error Rate Alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "ColPali-HighErrorRate" \
  --alarm-description "Lambda error rate exceeds 5%" \
  --metric-name Errors \
  --namespace AWS/Lambda \
  --dimensions Name=FunctionName,Value=colpali-baml-processor \
  --statistic Sum \
  --period 300 \
  --threshold 5 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2 \
  --alarm-actions arn:aws:sns:us-east-1:$AWS_ACCOUNT_ID:ops-alerts

# High Latency Alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "ColPali-HighLatency" \
  --alarm-description "P99 latency exceeds 30 seconds" \
  --metric-name Duration \
  --namespace AWS/Lambda \
  --dimensions Name=FunctionName,Value=colpali-baml-processor \
  --extended-statistic p99 \
  --period 300 \
  --threshold 30000 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2 \
  --alarm-actions arn:aws:sns:us-east-1:$AWS_ACCOUNT_ID:ops-alerts

# Memory Usage Alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "ColPali-HighMemory" \
  --alarm-description "Memory usage exceeds 80%" \
  --metric-name MemoryUsage \
  --namespace ColPali/Processing \
  --dimensions Name=Environment,Value=production \
  --statistic Average \
  --period 300 \
  --threshold 8589934592 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2 \
  --alarm-actions arn:aws:sns:us-east-1:$AWS_ACCOUNT_ID:ops-alerts
```

### Log Insights Queries

Useful CloudWatch Log Insights queries:

**Error Analysis:**
```sql
fields @timestamp, @message, correlation_id, error
| filter @message like /ERROR/
| sort @timestamp desc
| limit 100
```

**Latency Distribution:**
```sql
fields @timestamp, processing_time_seconds
| filter processing_time_seconds > 0
| stats avg(processing_time_seconds) as avg_latency,
        pct(processing_time_seconds, 50) as p50,
        pct(processing_time_seconds, 95) as p95,
        pct(processing_time_seconds, 99) as p99
  by bin(1h)
```

**Cold Start Analysis:**
```sql
fields @timestamp, cold_start, @duration
| filter cold_start = true
| stats count() as cold_starts, avg(@duration) as avg_cold_start_duration
  by bin(1h)
```

---

## Backup & Recovery Procedures

### Qdrant Vector Data Backup

#### Automated Backup Script

```bash
#!/bin/bash
# backup-qdrant.sh

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/qdrant"
QDRANT_HOST="${QDRANT_HOST:-localhost}"
QDRANT_PORT="${QDRANT_PORT:-6333}"

# Create backup directory
mkdir -p $BACKUP_DIR

# Create collection snapshot
for collection in $(curl -s "http://$QDRANT_HOST:$QDRANT_PORT/collections" | jq -r '.result.collections[].name'); do
  echo "Backing up collection: $collection"

  # Create snapshot
  SNAPSHOT_NAME=$(curl -s -X POST "http://$QDRANT_HOST:$QDRANT_PORT/collections/$collection/snapshots" | jq -r '.result.name')

  # Download snapshot
  curl -o "$BACKUP_DIR/${collection}_${BACKUP_DATE}.snapshot" \
    "http://$QDRANT_HOST:$QDRANT_PORT/collections/$collection/snapshots/$SNAPSHOT_NAME"

  echo "Saved: $BACKUP_DIR/${collection}_${BACKUP_DATE}.snapshot"
done

# Upload to S3 (optional)
if [ -n "$S3_BACKUP_BUCKET" ]; then
  aws s3 sync $BACKUP_DIR s3://$S3_BACKUP_BUCKET/qdrant-backups/
  echo "Uploaded to S3: s3://$S3_BACKUP_BUCKET/qdrant-backups/"
fi

# Cleanup old local backups (keep 7 days)
find $BACKUP_DIR -name "*.snapshot" -mtime +7 -delete

echo "Backup completed successfully"
```

#### Recovery Procedure

```bash
#!/bin/bash
# restore-qdrant.sh

SNAPSHOT_FILE=$1
COLLECTION_NAME=$2
QDRANT_HOST="${QDRANT_HOST:-localhost}"
QDRANT_PORT="${QDRANT_PORT:-6333}"

if [ -z "$SNAPSHOT_FILE" ] || [ -z "$COLLECTION_NAME" ]; then
  echo "Usage: restore-qdrant.sh <snapshot_file> <collection_name>"
  exit 1
fi

# Download from S3 if needed
if [[ "$SNAPSHOT_FILE" == s3://* ]]; then
  LOCAL_FILE="/tmp/$(basename $SNAPSHOT_FILE)"
  aws s3 cp $SNAPSHOT_FILE $LOCAL_FILE
  SNAPSHOT_FILE=$LOCAL_FILE
fi

# Upload snapshot to Qdrant
curl -X POST "http://$QDRANT_HOST:$QDRANT_PORT/collections/$COLLECTION_NAME/snapshots/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "snapshot=@$SNAPSHOT_FILE"

echo "Restored collection: $COLLECTION_NAME from $SNAPSHOT_FILE"
```

### Configuration Backup

```bash
# Backup all configuration files
tar -czf config-backup-$(date +%Y%m%d).tar.gz \
  docker-compose.yml \
  .env \
  requirements/ \
  baml_src/

# Store in S3
aws s3 cp config-backup-$(date +%Y%m%d).tar.gz \
  s3://$S3_BACKUP_BUCKET/config-backups/
```

---

## Scaling & Capacity Planning

### Lambda Scaling Configuration

| Document Type | Memory Requirement | Timeout | Concurrency |
|--------------|-------------------|---------|-------------|
| Single-page PDF | 4GB | 60s | 50 |
| Multi-page PDF (2-5) | 6GB | 120s | 25 |
| Complex Layout | 8GB | 180s | 10 |
| Batch Processing | 10GB | 300s | 5 |

### Auto-Scaling Guidelines

```bash
# Configure reserved concurrency
aws lambda put-function-concurrency \
  --function-name colpali-baml-processor \
  --reserved-concurrent-executions 50

# For high-throughput workloads, use Application Auto Scaling
aws application-autoscaling register-scalable-target \
  --service-namespace lambda \
  --resource-id function:colpali-baml-processor:prod \
  --scalable-dimension lambda:function:ProvisionedConcurrency \
  --min-capacity 2 \
  --max-capacity 20

aws application-autoscaling put-scaling-policy \
  --service-namespace lambda \
  --resource-id function:colpali-baml-processor:prod \
  --scalable-dimension lambda:function:ProvisionedConcurrency \
  --policy-name ColPali-TargetTracking \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 0.7,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "LambdaProvisionedConcurrencyUtilization"
    },
    "ScaleInCooldown": 300,
    "ScaleOutCooldown": 60
  }'
```

### Qdrant Scaling

For high-volume deployments, consider Qdrant Cloud or self-hosted cluster:

```yaml
# qdrant-cluster.yaml (Kubernetes)
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant-cluster
spec:
  serviceName: qdrant
  replicas: 3
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:v1.7.3
        ports:
        - containerPort: 6333
        - containerPort: 6334
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        volumeMounts:
        - name: storage
          mountPath: /qdrant/storage
  volumeClaimTemplates:
  - metadata:
      name: storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
```

### Capacity Planning Formula

```
Required Concurrency = (Peak Requests/min × Avg Processing Time) / 60

Example:
- Peak: 100 requests/minute
- Avg Processing: 15 seconds
- Required: (100 × 15) / 60 = 25 concurrent executions
```

---

## Incident Response Procedures

### Severity Levels

| Level | Description | Response Time | Escalation |
|-------|-------------|---------------|------------|
| SEV-1 | Complete outage | 15 min | On-call + Manager |
| SEV-2 | Degraded performance (>50% errors) | 30 min | On-call |
| SEV-3 | Minor issues (<10% errors) | 2 hours | Next business day |
| SEV-4 | Cosmetic/low impact | 1 week | Backlog |

### Incident Response Runbook

#### SEV-1: Complete Outage

1. **Acknowledge** (within 15 minutes)
   ```bash
   # Check Lambda function status
   aws lambda get-function --function-name colpali-baml-processor

   # Check recent invocations
   aws lambda list-function-event-invoke-configs \
     --function-name colpali-baml-processor
   ```

2. **Diagnose**
   ```bash
   # Check CloudWatch logs for errors
   aws logs filter-log-events \
     --log-group-name /aws/lambda/colpali-baml-processor \
     --start-time $(date -d '30 minutes ago' +%s000) \
     --filter-pattern "ERROR"

   # Check memory/timeout issues
   aws logs filter-log-events \
     --log-group-name /aws/lambda/colpali-baml-processor \
     --filter-pattern "Task timed out"
   ```

3. **Mitigate**
   ```bash
   # Increase memory if OOM
   aws lambda update-function-configuration \
     --function-name colpali-baml-processor \
     --memory-size 10240

   # Increase timeout if timing out
   aws lambda update-function-configuration \
     --function-name colpali-baml-processor \
     --timeout 300
   ```

4. **Verify Recovery**
   ```bash
   # Test with health check
   aws lambda invoke \
     --function-name colpali-baml-processor \
     --payload '{"healthCheck": true}' \
     response.json

   cat response.json
   ```

5. **Post-Incident Review** (within 48 hours)
   - Document timeline
   - Identify root cause
   - Create action items for prevention

#### SEV-2: High Error Rate

1. **Check Error Patterns**
   ```sql
   -- CloudWatch Log Insights
   fields @timestamp, @message, error, correlation_id
   | filter @message like /ERROR/
   | stats count() by error
   | sort count desc
   | limit 20
   ```

2. **Common Causes & Fixes**

   | Error Type | Cause | Fix |
   |------------|-------|-----|
   | `RuntimeError: CUDA out of memory` | Model too large | Reduce batch size |
   | `ConnectionError: Qdrant` | Network/service issue | Check Qdrant health |
   | `ValidationError` | Invalid input | Check client input |
   | `TimeoutError` | Processing too slow | Increase timeout |

3. **Temporary Mitigations**
   ```bash
   # Enable circuit breaker
   aws lambda update-function-configuration \
     --function-name colpali-baml-processor \
     --environment "Variables={CIRCUIT_BREAKER_ENABLED=true,ERROR_THRESHOLD=10}"
   ```

---

## Operational Runbooks

### Daily Operations Checklist

```markdown
## Daily Operations - ColPali-BAML

### Morning Checks (9:00 AM)
- [ ] Review CloudWatch dashboard for overnight issues
- [ ] Check Lambda error rate (target: <1%)
- [ ] Verify Qdrant health status
- [ ] Review any triggered alarms

### Commands:
```bash
# Check overnight errors
aws logs filter-log-events \
  --log-group-name /aws/lambda/colpali-baml-processor \
  --start-time $(date -d 'yesterday 6pm' +%s000) \
  --end-time $(date -d 'today 9am' +%s000) \
  --filter-pattern "ERROR" | jq '.events | length'

# Check Lambda metrics (last 24h)
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Errors \
  --dimensions Name=FunctionName,Value=colpali-baml-processor \
  --start-time $(date -d '24 hours ago' --iso-8601=seconds) \
  --end-time $(date --iso-8601=seconds) \
  --period 3600 \
  --statistics Sum
```
```

### Weekly Maintenance Tasks

```bash
#!/bin/bash
# weekly-maintenance.sh

echo "=== ColPali-BAML Weekly Maintenance ==="
echo "Date: $(date)"

# 1. Qdrant optimization
echo "1. Optimizing Qdrant collections..."
for collection in $(curl -s "http://$QDRANT_HOST:6333/collections" | jq -r '.result.collections[].name'); do
  curl -X POST "http://$QDRANT_HOST:6333/collections/$collection/index" \
    -H "Content-Type: application/json" \
    -d '{"field_name": "document_id", "field_schema": "keyword"}'
  echo "   Optimized: $collection"
done

# 2. Clean up old snapshots
echo "2. Cleaning old Qdrant snapshots..."
find /backups/qdrant -name "*.snapshot" -mtime +30 -delete

# 3. Update Lambda function (if new image available)
echo "3. Checking for Lambda updates..."
LATEST_DIGEST=$(aws ecr describe-images \
  --repository-name colpali-baml-lambda \
  --query 'sort_by(imageDetails,& imagePushedAt)[-1].imageDigest' \
  --output text)

CURRENT_DIGEST=$(aws lambda get-function \
  --function-name colpali-baml-processor \
  --query 'Code.ImageUri' \
  --output text | cut -d'@' -f2)

if [ "$LATEST_DIGEST" != "$CURRENT_DIGEST" ]; then
  echo "   New image available. Update manually if needed."
else
  echo "   Lambda is up to date."
fi

# 4. Generate weekly report
echo "4. Generating weekly metrics report..."
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Invocations \
  --dimensions Name=FunctionName,Value=colpali-baml-processor \
  --start-time $(date -d '7 days ago' --iso-8601=seconds) \
  --end-time $(date --iso-8601=seconds) \
  --period 604800 \
  --statistics Sum > /tmp/weekly-invocations.json

echo "   Total invocations: $(cat /tmp/weekly-invocations.json | jq '.Datapoints[0].Sum')"

echo "=== Maintenance Complete ==="
```

### Deployment Checklist

```markdown
## Deployment Checklist - ColPali-BAML

### Pre-Deployment
- [ ] All tests passing (`pytest tests/ -v`)
- [ ] Docker build successful (`docker build -f Dockerfile.lambda .`)
- [ ] Image size under 10GB
- [ ] CHANGELOG updated
- [ ] Version bumped in code

### Deployment Steps
- [ ] Push image to ECR
- [ ] Update Lambda function code
- [ ] Wait for function update to complete
- [ ] Run smoke test
- [ ] Monitor error rate for 15 minutes
- [ ] Update documentation if needed

### Post-Deployment Verification
- [ ] Health check returns 200
- [ ] Process test document successfully
- [ ] CloudWatch metrics normal
- [ ] No new error patterns

### Rollback Procedure (if needed)
```bash
# Get previous image digest
PREVIOUS_DIGEST=$(aws ecr describe-images \
  --repository-name colpali-baml-lambda \
  --query 'sort_by(imageDetails,& imagePushedAt)[-2].imageDigest' \
  --output text)

# Rollback
aws lambda update-function-code \
  --function-name colpali-baml-processor \
  --image-uri $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/colpali-baml-lambda@$PREVIOUS_DIGEST
```
```

---

## Appendix

### Environment Variables Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `QDRANT_HOST` | Qdrant server hostname | `localhost` | Yes |
| `QDRANT_PORT` | Qdrant HTTP port | `6333` | Yes |
| `OPENAI_API_KEY` | OpenAI API key | - | For GPT models |
| `ANTHROPIC_API_KEY` | Anthropic API key | - | For Claude models |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `OMP_NUM_THREADS` | OpenMP threads | `4` | No |
| `COLPALI_MODEL` | ColPali model name | `vidore/colqwen2-v0.1` | No |

### Useful Commands Quick Reference

```bash
# Lambda
aws lambda invoke --function-name colpali-baml-processor --payload '{"healthCheck":true}' out.json
aws lambda get-function --function-name colpali-baml-processor
aws lambda list-versions-by-function --function-name colpali-baml-processor

# CloudWatch
aws logs tail /aws/lambda/colpali-baml-processor --follow
aws cloudwatch describe-alarms --alarm-names "ColPali-HighErrorRate"

# ECR
aws ecr describe-images --repository-name colpali-baml-lambda
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_URL

# Docker Compose (Local)
docker-compose logs -f colpali-engine
docker-compose exec colpali-engine pytest tests/ -v
docker-compose restart colpali-engine
```

---

*This operations guide is maintained by the ColPali-BAML team. For updates or questions, contact the on-call engineering team.*
