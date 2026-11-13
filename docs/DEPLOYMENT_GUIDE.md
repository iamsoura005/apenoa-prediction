# Deployment & Infrastructure Guide

## 1. Quickstart (Local Development)

### Requirements
- Python 3.10+ (Verified on Python 3.13)
- SQLite 3.35+ (Included standard with Python)

### Setup & Launch
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the Enterprise FastAPI application
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
Access the application at `http://localhost:8000`.

---

## 2. Docker Deployment

### Build and Run with Docker
```bash
# Build the container image
docker build -t apneasphere:latest .

# Run container exposing port 8000
docker run -d -p 8000:8000 --name apneasphere_app -v $(pwd)/data:/app/data apneasphere:latest
```

---

## 3. Docker Compose (Production Environment)

To deploy with volume persistence for the SQLite database:
```bash
docker compose up -d
```
Check health:
```bash
docker compose ps
curl http://localhost:8000/api/health
```

---

## 4. Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///data/apneasphere.db` | SQLAlchemy connection string (SQLite or PostgreSQL) |
| `PORT` | `8000` | Port for Uvicorn server |
| `ENVIRONMENT` | `production` | Deployment mode flag |
