# Production Deployment & Hosting Guide
**Platform:** ApneaSphere Enterprise Biomedical Platform  
**Backend:** Hugging Face Spaces (Docker Space, Port 7860)  
**Frontend:** Vercel Static Hosting  
**Local Training Engine:** Windows PC with NVIDIA GeForce RTX 4050 6GB Laptop GPU  

---

## 1. System Deployment Architecture

```
  [ Client Browser / Clinician ]
               │
               ▼
  [ Vercel Edge CDN Frontend ]  (index.html, styles.css, app.js)
               │
               ▼ HTTPS REST API (CORS enabled)
  [ Hugging Face Space Backend ] (FastAPI + PyTorch 1D-CNN + SQLite + ReportLab PDF)
               │ (Port 7860)
         ┌─────┴──────────────┐
         ▼                    ▼
   [ Apnea1DCNN ]     [ SQLite History ]
   (Checkpoint: .pt)  (apneasphere.db)
```

---

## 2. Deploying Backend to Hugging Face Spaces

1. **Create Space on Hugging Face:**
   - Space Name: `apneasphere-backend`
   - License: MIT
   - SDK: **Docker**
   - Visibility: Public

2. **Hugging Face Docker Specification:**
   The repository includes a ready-to-deploy `Dockerfile`:
   - Listens on port `7860`.
   - Runs as non-root user `user` (UID 1000).
   - Serves FastAPI with PyTorch checkpoint `ml/models/apnea_cnn_production.pt`.
   - Healthcheck configured at `http://localhost:7860/health`.

3. **Deploy using Git:**
   ```bash
   git remote add space https://huggingface.co/spaces/<your-hf-username>/apneasphere-backend
   git push space main
   ```

---

## 3. Deploying Frontend to Vercel

1. **Deploy using Vercel CLI:**
   From the `frontend/` directory:
   ```bash
   cd frontend
   vercel --prod
   ```

2. **Environment Variable Configuration:**
   If backend is hosted separately on Hugging Face Spaces:
   - In Vercel Project Settings $\to$ Environment Variables:
     - Key: `VITE_API_BASE_URL` or set `window.API_BASE_URL` in `app.js`
     - Value: `https://<your-hf-space>.hf.space`

---

## 4. Local Execution

### Backend Server
```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
API documentation available at: `http://localhost:8000/docs`

### GPU Training
```bash
python ml/train.py --device cuda --mixed-precision --epochs 15 --batch-size 32
```
