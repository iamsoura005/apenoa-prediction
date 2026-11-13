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

## 2. 100% Free Backend Hosting Options

### Option A: Hugging Face Spaces (Gradio SDK — 100% Free Forever)
> **Why Gradio SDK?** Hugging Face recently required credit card verification for Docker Spaces to prevent crypto-mining abuse. However, **Gradio Spaces are 100% FREE without any payment verification or credit card required** (providing 2 vCPU and 16 GB RAM).
> 
> We have configured `app.py` in the root repository to mount the complete FastAPI backend. Hugging Face hosts the Gradio Space for free while exposing all REST API endpoints (`/api/predict`, `/health`, etc.) to your Vercel frontend.

1. **Create Space on Hugging Face:**
   - Space Name: `apneasphere-backend`
   - License: `mit`
   - SDK: Select **Gradio** (Free CPU tier: 2 vCPU, 16 GB RAM)
   - Visibility: Public

2. **Deploy using Git:**
   ```bash
   git remote add space https://huggingface.co/spaces/<your-hf-username>/apneasphere-backend
   git push space main
   ```
   All endpoints will be live at `https://<your-hf-username>-apneasphere-backend.hf.space`.

---

### Option B: Render.com (Free Web Service Tier)
The repository includes a ready-to-deploy `render.yaml` blueprint:
1. Go to [https://dashboard.render.com](https://dashboard.render.com).
2. Click **New +** $\to$ **Web Service**.
3. Connect your GitHub repository: `https://github.com/iamsoura005/apenoa-prediction`.
4. Render automatically configures:
   - **Environment:** Python
   - **Plan:** Free
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
5. Click **Create Web Service**. Your free backend URL will be `https://apneasphere-backend.onrender.com`.

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

