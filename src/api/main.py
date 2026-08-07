from fastapi import FastAPI
from fastapi.responses import JSONResponse
import gradio as gr

app = FastAPI(title="Demix API", description="Audio Source Separation API")

def handle_separation(audio_filepath, model_choice):
    if not audio_filepath:
        return None, None, None, None
    print(f"[Mock Interface] Recibido archivo '{audio_filepath}' con modelo '{model_choice}'")
    # Mock return: las 4 fuentes separadas (drums, bass, other, vocals)
    # Una vez que los modelos estén entrenados, aquí se invocará el pipeline de inferencia real
    return None, None, None, None

# Gradio Interface
with gr.Blocks(title="Demix - Separación de Audio", theme=gr.themes.Soft()) as gradio_app:
    gr.Markdown(
        """
        # 🎵 Demix - Separación de Fuentes de Audio
        
        Compara **U-Net** vs **Vision Transformers (ViT)** para separar instrumentos en canciones.
        
        ### 🧪 Interfaz de Pruebas y Mockup
        Sube una pista de audio y selecciona la arquitectura del modelo para ejecutar la separación.
        """
    )
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                label="📁 Sube tu audio",
                type="filepath",
                sources=["upload"]
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
    
    separate_btn.click(
        fn=handle_separation,
        inputs=[audio_input, model_choice],
        outputs=[drums_output, bass_output, other_output, vocals_output]
    )

app = gr.mount_gradio_app(app, gradio_app, path="/gradio")

@app.get("/")
async def root():
    return JSONResponse({
        "message": "Demix API",
        "gradio_ui": "/gradio",
        "docs": "/docs"
    })

if __name__ == "__main__":
    gradio_app.launch(server_name="0.0.0.0", server_port=7860, share=False)


