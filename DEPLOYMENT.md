# Self Soul AGI System - Docker Deployment Guide

This guide provides comprehensive instructions for deploying the Self Soul AGI System using Docker and Docker Compose.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [System Requirements](#system-requirements)
3. [Quick Start](#quick-start)
4. [Detailed Deployment Steps](#detailed-deployment-steps)
5. [Configuration](#configuration)
6. [Data Persistence](#data-persistence)
7. [Monitoring and Logs](#monitoring-and-logs)
8. [Troubleshooting](#troubleshooting)
9. [Production Deployment](#production-deployment)
10. [Backup and Recovery](#backup-and-recovery)

## Prerequisites

### Required Software
- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher
- **Git**: For cloning the repository (optional)

### Hardware Requirements
- **Minimum**: 4 CPU cores, 8GB RAM, 20GB free disk space
- **Recommended**: 8+ CPU cores, 16GB+ RAM, 50GB+ free disk space
- **GPU** (optional): NVIDIA GPU with CUDA support for accelerated model inference

### Network Requirements
- Ports 5175 (frontend), 8000 (API), 8766 (real-time streaming) must be available
- Internet connection for downloading Docker images and models (first run)

## System Architecture

The Self Soul AGI System is deployed as a multi-container application:

| Service | Container Name | Ports | Description |
|---------|----------------|-------|-------------|
| Backend | `self-soul-backend` | 8000, 8766, 8001-8027 | Python FastAPI application with 27 model services |
| Frontend | `self-soul-frontend` | 80 (mapped to 5175) | Vue.js SPA served by Nginx |

## Quick Start

Follow these steps for a basic deployment:

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd Self-Soul-B
   ```

2. **Create environment configuration**:
   ```bash
   cp .env.example .env
   # Edit .env file with your configuration (optional for basic deployment)
   ```

3. **Build and start the containers**:
   ```bash
   docker-compose up -d
   ```

4. **Verify the deployment**:
   ```bash
   docker-compose ps
   ```

5. **Access the application**:
   - Frontend: http://localhost:5175
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## Detailed Deployment Steps

### Step 1: Environment Setup

Create a `.env` file in the project root (optional but recommended for production):

```bash
# Copy example environment file
cp .env.example .env

# Edit the .env file with your preferred editor
nano .env
```

Example `.env` content:
```env
# Application Environment
ENVIRONMENT=production
LOG_LEVEL=INFO

# Database Configuration
DATABASE_URL=sqlite:///./data/self_soul.db

# File Storage
UPLOAD_DIR=/app/uploads
MODEL_CACHE_DIR=/app/models

# Security (generate your own secrets)
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here

# External Services (if used)
# OPENAI_API_KEY=your-openai-api-key
# HUGGINGFACE_TOKEN=your-hf-token
```

### Step 2: Build the Docker Images

Build the images for both backend and frontend:

```bash
# Build all services
docker-compose build

# Or build specific services
docker-compose build backend
docker-compose build frontend
```

### Step 3: Start the Services

```bash
# Start all services in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Step 4: Initialize the System

After starting the containers, the backend will automatically:
- Create necessary directories
- Initialize the database
- Start all model services

Wait for the initialization to complete (check logs for confirmation).

### Step 5: Verify Deployment

Check if all services are running:

```bash
# List containers
docker-compose ps

# Check backend health
curl http://localhost:8000/health

# Check frontend
curl -I http://localhost:5175
```

## Configuration

### Port Configuration

By default, the following ports are exposed:

| Service | Container Port | Host Port |
|---------|----------------|-----------|
| Frontend | 80 | 5175 |
| Main API | 8000 | 8000 |
| Real-time Stream | 8766 | 8766 |
| Model Services | 8001-8027 | 8001-8027 |

To change port mappings, edit `docker-compose.yml`:

```yaml
services:
  frontend:
    ports:
      - "8080:80"  # Change host port from 5175 to 8080
```

### Resource Limits

Adjust resource allocation in `docker-compose.yml`:

```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

### GPU Support (Optional)

To enable GPU acceleration for PyTorch models:

1. Install NVIDIA Container Toolkit:
   ```bash
   # Follow NVIDIA's instructions for your distribution
   ```

2. Uncomment GPU configuration in `docker-compose.yml`:
   ```yaml
   services:
     backend:
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: 1
                 capabilities: [gpu]
   ```

3. Rebuild and restart:
   ```bash
   docker-compose up -d --build
   ```

## Data Persistence

The following directories are persisted via Docker volumes:

| Directory | Purpose | Volume Mount |
|-----------|---------|--------------|
| `./data` | Database and knowledge files | Host: `./data` → Container: `/app/data` |
| `./uploads` | User uploads | Host: `./uploads` → Container: `/app/uploads` |
| `./logs` | Application logs | Host: `./logs` → Container: `/app/logs` |
| `./models` | Model cache (optional) | Host: `./models` → Container: `/app/models` |

### Backup Strategy

1. **Regular backups**:
   ```bash
   # Create backup of all persisted data
   tar -czf backup-$(date +%Y%m%d-%H%M%S).tar.gz data/ uploads/ logs/
   ```

2. **Database backup** (SQLite):
   ```bash
   # Copy the database file
   cp data/self_soul.db data/backup/self_soul-$(date +%Y%m%d).db
   ```

## Monitoring and Logs

### Viewing Logs

```bash
# All services logs
docker-compose logs

# Follow logs in real-time
docker-compose logs -f

# Service-specific logs
docker-compose logs backend
docker-compose logs frontend

# View last 100 lines
docker-compose logs --tail=100
```

### Health Monitoring

```bash
# Check container health status
docker-compose ps

# Manual health check
curl http://localhost:8000/health

# Detailed system status
curl http://localhost:8000/api/system/status
```

### Performance Monitoring

```bash
# Container resource usage
docker stats

# Backend metrics
curl http://localhost:8000/api/metrics
```

## Troubleshooting

### Common Issues

1. **Port conflicts**:
   ```bash
   # Check which process is using a port
   sudo netstat -tulpn | grep :5175
   
   # Change port in docker-compose.yml
   ```

2. **Container fails to start**:
   ```bash
   # Check logs
   docker-compose logs backend
   
   # Restart with clean build
   docker-compose down -v
   docker-compose up -d --build
   ```

3. **Out of memory**:
   ```bash
   # Increase memory limits
   # Edit docker-compose.yml and increase memory reservation
   ```

4. **Database errors**:
   ```bash
   # Reset database (warning: loses data)
   docker-compose exec backend python -c "import os; os.remove('/app/data/self_soul.db')"
   docker-compose restart backend
   ```

### Debugging Commands

```bash
# Enter backend container
docker-compose exec backend bash

# Check Python dependencies
docker-compose exec backend pip list

# View application directory
docker-compose exec backend ls -la /app

# Test API endpoints
docker-compose exec backend curl http://localhost:8000/health
```

## Production Deployment

### Security Considerations

1. **Use HTTPS**:
   - Configure Nginx with SSL certificates
   - Use Let's Encrypt for free certificates
   - Update `nginx.docker.conf` to redirect HTTP to HTTPS

2. **Secure secrets**:
   - Never commit `.env` file to version control
   - Use Docker secrets or external secret management
   - Rotate secrets regularly

3. **Network security**:
   - Use Docker internal networks
   - Restrict exposed ports
   - Configure firewall rules

4. **Regular updates**:
   ```bash
   # Update base images
   docker-compose pull
   docker-compose up -d
   ```

### High Availability (Optional)

For production deployments requiring high availability:

1. **Use external database**:
   - Replace SQLite with PostgreSQL
   - Uncomment PostgreSQL service in `docker-compose.yml`
   - Update `DATABASE_URL` in `.env`

2. **Load balancing**:
   - Deploy multiple backend instances
   - Use Nginx as load balancer

3. **Container orchestration**:
   - Consider Kubernetes for production scaling
   - Use Docker Swarm for simpler orchestration

### Scaling

```bash
# Scale backend service (requires load balancer)
docker-compose up -d --scale backend=3
```

## Backup and Recovery

### Automated Backup Script

Create `backup.sh`:

```bash
#!/bin/bash
BACKUP_DIR="/path/to/backups"
DATE=$(date +%Y%m%d-%H%M%S)

# Stop services
docker-compose stop

# Create backup
tar -czf "$BACKUP_DIR/backup-$DATE.tar.gz" data/ uploads/ logs/

# Start services
docker-compose start

# Remove old backups (keep 7 days)
find "$BACKUP_DIR" -name "backup-*.tar.gz" -mtime +7 -delete
```

### Recovery Process

1. **Stop services**:
   ```bash
   docker-compose down
   ```

2. **Restore backup**:
   ```bash
   tar -xzf backup-20250101-120000.tar.gz
   ```

3. **Restart services**:
   ```bash
   docker-compose up -d
   ```

## Maintenance

### Regular Maintenance Tasks

1. **Update dependencies**:
   ```bash
   # Update Python packages
   docker-compose exec backend pip install -r requirements.txt --upgrade
   
   # Update Node.js packages
   docker-compose exec frontend npm update
   ```

2. **Clean up unused resources**:
   ```bash
   # Remove unused containers, networks, images
   docker system prune -f
   
   # Remove unused volumes
   docker volume prune -f
   ```

3. **Log rotation**:
   ```bash
   # Configure log rotation in nginx.docker.conf
   # and Docker daemon settings
   ```

### Performance Optimization

1. **Enable caching**:
   - Uncomment Redis service in `docker-compose.yml`
   - Configure backend to use Redis cache

2. **Model optimization**:
   - Pre-download frequently used models
   - Enable model quantization for smaller memory footprint

## Support and Resources

- **Documentation**: See `README.md` and `HELP.md`
- **Issue Tracking**: GitHub Issues
- **Community**: [Add community links]

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.

---

*Last Updated: $(date)*