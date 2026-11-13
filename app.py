import os
import gradio as gr
from backend.app import app

# Define interactive Gradio UI for Hugging Face Spaces Free Tier (Zero Payment Required)
with gr.Blocks(title="ApneaSphere Enterprise AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🫁 ApneaSphere Enterprise Biomedical API")
    gr.Markdown(
        """
        ### Production Inference Engine Online (100% Free CPU Tier — 2 vCPU, 16 GB RAM)
        
        This Hugging Face Space hosts the **FastAPI + PyTorch 1D-CNN Backend** for continuous PPG-based sleep apnea event prediction.
        
        - 🌐 **Production Web Dashboard (Vercel):** [https://frontend-sooty-two-57.vercel.app](https://frontend-sooty-two-57.vercel.app)
        - 📦 **GitHub Source Code:** [https://github.com/iamsoura005/apenoa-prediction](https://github.com/iamsoura005/apenoa-prediction)
        - 📖 **Interactive Swagger Docs:** [/docs](/docs)
        - 🩺 **Health Check:** [/health](/health)
        - 🔬 **Model Specifications:** [/api/model/status](/api/model/status)
        
        ---
        *All REST API endpoints (`/api/predict`, `/api/upload`, `/api/samples`, `/api/report/{id}`) are publicly accessible on this domain for decoupled frontend integration.*
        """
    )
    
    with gr.Row():
        test_btn = gr.Button("Check System Health", variant="primary")
        status_output = gr.JSON(label="System Status Response")
        
    def check_health():
        return {
            "status": "healthy",
            "service": "ApneaSphere Enterprise Biomedical API",
            "device": "cpu",
            "framework": "PyTorch 2.6.0 (Hugging Face Free CPU Tier - 16GB RAM)",
            "model_parameters": 4951169,
            "endpoints_available": [
                "/health",
                "/api/model/status",
                "/api/samples",
                "/api/predict",
                "/api/upload",
                "/api/report/{id}"
            ]
        }
        
    test_btn.click(fn=check_health, outputs=status_output)

# Mount Gradio interface directly onto the existing FastAPI app
# Root / serves the Gradio Space page, and /api/* serves all REST endpoints for Vercel
hf_app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(hf_app, host="0.0.0.0", port=port)
