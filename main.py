from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
import json
import shutil 

import whisper 

app = FastAPI(
    title="AI Meeting Transcript Summarizer Backend",
    description="Backend for summarizing meeting transcripts using Gemini API and handling audio uploads with local Whisper."
)

# Configure CORS to allow requests from index.html
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")

if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
    print("WARNING: Please replace 'YOUR_GEMINI_API_KEY' in main.py with your actual Gemini API key.")
    print("         Alternatively, set the GEMINI_API_KEY environment variable.")

genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel('gemini-2.0-flash') 


print("INFO: Loading local Whisper model. This may take a moment and download files on first run.")
try:
    whisper_model = whisper.load_model("base") # change "base" to "small", "medium"
    print("INFO: Local Whisper model loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load local Whisper model: {e}")
    print("Please ensure 'openai-whisper' is installed and ffmpeg is in your system PATH.")

    exit(1) # Exit if Whisper model cannot be loaded


# Define request body model for transcript
class TranscriptRequest(BaseModel):
    transcript: str

@app.post("/summarize")
async def summarize_transcript(request: TranscriptRequest):
    """
    Summarizes a meeting transcript using the Gemini API,
    extracting summary, objections, and action items.
    """
    if not request.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript cannot be empty.")

    prompt = f"""
        Analyze the following meeting transcript and provide the following:
        1. A concise summary of the main discussion points.
        2. Any client objections or pain points raised, along with their resolutions or proposed solutions. If no explicit resolution, state that.
        3. A list of clear action items or follow-ups, specifying who is responsible if mentioned, and the task itself.

        Format the output as a JSON object with the following structure:
        {{
          "summary": "string",
          "objections": [
            {{
              "point": "string",
              "resolution": "string"
            }}
          ],
          "actionItems": [
            {{
              "task": "string",
              "responsible": "string"
            }}
          ]
        }}

        Meeting Transcript:
        "{request.transcript}"
    """

    try:
        response = model_gemini.generate_content( 
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "summary": {"type": "STRING"},
                        "objections": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "point": {"type": "STRING"},
                                    "resolution": {"type": "STRING"}
                                }
                            }
                        },
                        "actionItems": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "task": {"type": "STRING"},
                                    "responsible": {"type": "STRING"}
                                }
                            }
                        }
                    }
                }
            )
        )

        parsed_data = json.loads(response.text)
        return parsed_data

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing transcript: {e}")

@app.post("/upload-audio")
async def upload_audio_for_transcription(audio_file: UploadFile = File(...)):
    """
    Handles audio file upload and transcribes it using the local Whisper model.
    """
    if not audio_file.content_type.startswith("audio/") and \
       not audio_file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only audio or video files are allowed.")

    temp_file_path = f"temp_{audio_file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        print(f"INFO: Transcribing file '{audio_file.filename}' using local Whisper...")
        result = whisper_model.transcribe(temp_file_path)
        transcribed_text = result["text"]
        print(f"INFO: Transcription complete for '{audio_file.filename}'.")

        return {"transcribed_text": transcribed_text}

    except Exception as e:
        print(f"Error processing audio file with local Whisper: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to transcribe audio locally: {e}. Ensure ffmpeg is installed and in your system PATH, and the audio file is valid.")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>FastAPI Backend</title>
        </head>
        <body>
            <h1>FastAPI Meeting Summarizer Backend is Running!</h1>
            <p>Access the frontend by opening 'index.html' in your browser.</p>
            <p>API endpoints: <code>/summarize</code> (POST), <code>/upload-audio</code> (POST)</p>
        </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
