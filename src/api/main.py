from fastapi import FastAPI
from fastapi.responses import JSONResponse
import gradio as gr
import boto3
import uuid

app = FastAPI(title="Demix API", description="Audio Source Separation API")

s3_client = boto3.client('s3', endpoint_url="http://localstack:4566")

def handle_separation(audio_filepath, model_choice):
    job_id = str(uuid.uuid4())
    key = f"inputs/{job_id}/original.mp3"
    s3_client.upload_file(audio_filepath, 'demix-raw-audio', key)
    print(f"File '{audio_filepath}' uploaded and added to S3 bucket succesfully.")
    return None, None, None, None

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

