from fastapi import FastAPI
from fastapi.responses import JSONResponse
import gradio as gr
from gradio import themes
import boto3
import uuid
import json

app = FastAPI(title="Demix API", description="Audio Source Separation API")

s3_client = boto3.client('s3', endpoint_url="http://localstack:4566")
sqs_client = boto3.client('sqs', endpoint_url="http://localstack:4566")

def handle_separation(audio_filepath, model_choice):
    job_id = str(uuid.uuid4())
    key = f"{job_id}/original.mp3"
    s3_client.upload_file(audio_filepath, 'demix-raw-audio', key)
    print(f"File '{audio_filepath}' uploaded and added to S3 bucket succesfully.")
    message = {
        "job_id": job_id,
        "s3_key": key,
        "model": model_choice,
        "bucket": 'demix-raw-audio'
    }
    message = json.dumps(message)
    sqs_client.send_message(QueueUrl = 'http://localstack:4566/000000000000/demix-processing-queue', MessageBody=message)
    #TODO: polling until processing is complete, download stems from s3 and return them
    return None, None, None, None

# Gradio
with gr.Blocks(title="Demix - Separación de Audio", theme=themes.Soft()) as gradio_app:
    gr.Markdown(
        """
        # 🎵 Demix - Separación de Fuentes de Audio
        
        Compara **U-Net** vs **Vision Transformers** para separar instrumentos.
        
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
    
@app.get("/status/{job_id}")
async def work_status(job_id: str):
    stems = ['drums.mp3', 'bass.mp3', 'other.mp3', 'vocals.mp3']
    for stem in stems:
        try:  
            key = f"{job_id}/{stem}"
            s3_client.get_object(Bucket="demix-output-audio", Key=key,)
        except s3_client.exceptions.NoSuchKey:
            return JSONResponse({
                "status": "processing"
            })
        except Exception as e:
            return {"status": "error", "message": str(e)}
    return JSONResponse({
        "status": "complete",
        "stems": {
            "drums": f"{job_id}/drums.mp3",
            "bass": f"{job_id}/bass.mp3",
            "other": f"{job_id}/other.mp3",
            "vocals": f"{job_id}/vocals.mp3"
        }
    })
        

