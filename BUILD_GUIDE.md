# Build Guide

Complete guide for building, running, and deploying VeriFact.

## Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- Git

## Local Development

### 1. Clone the Repository

```bash
git clone https://github.com/AdetyaJamwal04/Fake_News_Detection.git
cd Fake_News_Detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Set Environment Variables

```bash
cp .env.example .env
# Edit .env and add your TAVILY_API_KEY
```

### 5. Run the Application

```bash
python app_flask.py
```

Access at: http://localhost:5000

## Docker Build

### Build Image

```bash
docker build -t fake-news-detector .
```

### Run Container

```bash
docker run -d \
  --name fake-news-detector \
  -p 5000:5000 \
  -e TAVILY_API_KEY=your_key \
  --restart unless-stopped \
  fake-news-detector
```

### Using Docker Compose

```bash
docker-compose up -d
```

## AWS EC2 Deployment

### 1. Launch EC2 Instance

- **AMI:** Ubuntu 22.04 LTS
- **Instance Type:** t3.medium (2 vCPU, 4GB RAM minimum)
- **Storage:** 30GB+ (Docker images are large)
- **Security Group:** Allow ports 22 (SSH), 5000 (App)

### 2. Install Docker on EC2

```bash
sudo apt update
sudo apt install -y docker.io
sudo usermod -aG docker ubuntu
```

### 3. Pull and Run

```bash
docker pull adetyajamwal/fake-news-detector:latest
docker run -d --name fake-news-detector -p 5000:5000 \
  -e TAVILY_API_KEY=your_key \
  --restart unless-stopped \
  adetyajamwal/fake-news-detector:latest
```

### 4. Set Up Elastic IP (Recommended)

By default, EC2 public IPs change when you stop/start the instance. To get a static IP:

1. Go to **AWS Console → EC2 → Elastic IPs**
2. Click **Allocate Elastic IP address**
3. Click **Allocate**
4. Select the new IP → **Actions → Associate Elastic IP address**
5. Choose your instance and click **Associate**
6. Update the `EC2_HOST` secret in GitHub with this new static IP

### Instance Restart Behavior

The container uses `--restart unless-stopped`, which means:

| Scenario | Container Status |
|----------|------------------|
| Instance reboot | ✅ Auto-starts |
| Instance stop → start | ✅ Auto-starts |
| Container crashes | ✅ Auto-restarts |
| Manual `docker stop` before shutdown | ❌ Won't auto-start |

**Note:** If you don't have an Elastic IP, the public IP will change after stop/start. You'll need to update:
- Your browser bookmarks
- The `EC2_HOST` GitHub secret

## CI/CD with GitHub Actions

The project uses automated deployment via `.github/workflows/deploy.yml`:

1. Push to `main` branch triggers the workflow
2. Tests run first
3. Docker image is built and pushed to DockerHub
4. EC2 instance pulls and runs the new image

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `DOCKERHUB_USERNAME` | DockerHub username |
| `DOCKERHUB_TOKEN` | DockerHub access token |
| `EC2_HOST` | EC2 public IP |
| `EC2_SSH_KEY` | Private key for SSH |
| `TAVILY_API_KEY` | Tavily search API key |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TAVILY_API_KEY` | Yes | API key for Tavily search |
| `PORT` | No | Server port (default: 5000) |
| `DEBUG` | No | Enable debug mode (default: false) |

## Troubleshooting

### Disk Space Issues

The Docker image is ~4-5GB. If deployment fails with "no space left":

```bash
docker system prune -a --volumes -f
df -h
```

### Container Logs

```bash
docker logs fake-news-detector --tail 100
```

### Health Check

```bash
curl http://localhost:5000/api/health
```

## Running Tests

```bash
pytest tests/ -v
```

Individual test files:
```bash
pytest tests/test_source_scorer.py -v
pytest tests/test_query_generator.py -v
pytest tests/test_verdict_engine.py -v
```
