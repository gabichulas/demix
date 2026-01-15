from fastapi import FastAPI
from fastapi.responses import JSONResponse
import gradio as gr

app = FastAPI(title="Demix API", description="Audio Source Separation API")

# Gradio
with gr.Blocks(title="Demix - Separación de Audio", theme=gr.themes.Soft()) as gradio_app:
    gr.Markdown(
        """
        # 🎵 Demix - Separación de Fuentes de Audio
        
        Compara **U-Net** vs **Vision Transformers** para separar instrumentos.
        
        ### En desarrollo - Modelos en entrenamiento
        """
    )
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                label="📁 Sube tu audio",
                type="filepath",
                sources=["upload", "microphone"]
            )
            
            model_choice = gr.Radio(
                choices=["U-Net", "Vision Transformer"],
                value="U-Net",
                label="🤖 Modelo",
                info="Arquitectura para la separación"
            )
            
            separate_btn = gr.Button("🎯 Separar Audio", variant="primary", size="lg")
    
    gr.Markdown("### 🎼 Fuentes Separadas")
    
    with gr.Row():
        with gr.Column():
            drums_output = gr.Audio(label="🥁 Batería", interactive=False)
            bass_output = gr.Audio(label="🎸 Bajo", interactive=False)
        
        with gr.Column():
            other_output = gr.Audio(label="🎹 Otros", interactive=False)
            vocals_output = gr.Audio(label="🎤 Voces", interactive=False)
            
app = gr.mount_gradio_app(app, gradio_app, path="/gradio")

@app.get("/")
async def root():
    return JSONResponse({
        "message": "Demix API",
        "gradio_ui": "/gradio",
        "docs": "/docs"
    })
