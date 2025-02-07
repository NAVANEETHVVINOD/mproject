import logging
import json
import sys
import os
import subprocess
import tempfile
import re
import asyncio
from pathlib import Path
from datetime import datetime
from sanic import Sanic, response
from sanic.exceptions import MethodNotAllowed
import torch
from transformers import pipeline
from langdetect import detect, LangDetectException
import aiohttp
import librosa
import numpy as np

# Fix Unicode encoding for Windows console
os.environ["PYTHONIOENCODING"] = "utf-8"

class Config:
    GEMINI_API_KEY = "AIzaSyAw4IGA20aRGlyfppjHcr1KcHECvTCP2vE"
    HOST = "0.0.0.0"
    PORT = 8080
    DEBUG = True
    CORS_ORIGINS = ["http://localhost:8000", "http://127.0.0.1:8000"]
    MAX_AUDIO_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_AUDIO_TYPES = {'audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/webm'}
    TARGET_SAMPLE_RATE = 16000
    BASE_DIR = Path(__file__).parent
    FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"
    MODELS_DIR = BASE_DIR / "models"
    CHUNK_SIZE = 30000  # Larger sample for better language detection

    for dir in [BASE_DIR / "uploads", BASE_DIR / "logs", MODELS_DIR]:
        dir.mkdir(exist_ok=True, parents=True)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Config.BASE_DIR / "logs/app.log", encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()
app = Sanic("MedicalTranscription")

# Disable Sanic's MOTD to prevent encoding issues
app.config.MOTD = False

# ================= CORS Handling =================
@app.middleware("request")
async def cors_middleware(request):
    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": request.headers.get("origin", ""),
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600",
            "Access-Control-Allow-Credentials": "true"
        }
        return response.text("", headers=headers)

@app.middleware("response")
async def add_cors_headers(request, response):
    origin = request.headers.get("origin")
    if origin in Config.CORS_ORIGINS:
        response.headers.update({
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
            "Vary": "Origin"
        })

# ================= FFmpeg Audio Processing =================
def convert_audio(input_path, output_path):
    """Convert audio to WAV format using FFmpeg"""
    try:
        cmd = [
            Config.FFMPEG_PATH,
            '-y',  # Overwrite output
            '-i', input_path,
            '-ar', str(Config.TARGET_SAMPLE_RATE),
            '-ac', '1',
            '-f', 'wav',
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            raise RuntimeError(f"Audio conversion failed: {result.stderr}")
            
        return True
    except Exception as e:
        logger.error(f"FFmpeg processing error: {str(e)}")
        raise

# ================= ML Models =================
models = {
    'whisper_en': None,
    'whisper_ml': None,
    'translator': None
}

def load_models():
    try:
        logger.info("Initializing ML models...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Whisper models
        models['whisper_en'] = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device=device,
            generate_kwargs={"language": "en"}
        )
        
        models['whisper_ml'] = pipeline(
            "automatic-speech-recognition",
            model="kavyamanohar/whisper-small-malayalam",
            device=device,
            generate_kwargs={"language": "ml"}
        )
        
        # Correct translation model name
        models['translator'] = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-ml-en",
            device=device
        )
        return True
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        return False

# ================= Language Detection =================
def detect_language_from_audio(audio, is_recorded=False):
    """Improved language detection ensuring English is prioritized."""
    try:
        sample_text = models["whisper_en"](
            audio[:Config.CHUNK_SIZE] if len(audio) > Config.CHUNK_SIZE else audio
        )["text"]

        # Prioritize English detection
        lang = detect(sample_text)
        if lang == "en":
            return "en"

        # Check for English word dominance in recorded speech
        if is_recorded:
            english_word_pattern = re.compile(r'\b[a-zA-Z]+\b')
            english_words = len(english_word_pattern.findall(sample_text))
            non_english_chars = len(re.findall(r'[^\x00-\x7F]', sample_text))

            if english_words > non_english_chars:
                return "en"

        # If English is not confidently detected, default to Malayalam
        return "ml"
    except LangDetectException:
        return "ml"  # Fallback to Malayalam if detection fails
    except Exception as e:
        logger.error(f"Language detection failed: {str(e)}")
        return "en"  # Default to English if there's an error

# ================= Gemini API Helpers =================
async def call_gemini_api(prompt, retry=1):
    """Make API call to Gemini with retry for failures."""
    try:
        async with aiohttp.ClientSession() as session:
            response = await session.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={Config.GEMINI_API_KEY}",
                headers={'Content-Type': 'application/json'},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.3, "topP": 0.8, "topK": 40}
                }
            )

            if response.status != 200:
                logger.error(f"Gemini API error: {response.status}")
                if retry > 0:
                    logger.info("Retrying Gemini API call...")
                    return await call_gemini_api(prompt, retry - 1)
                return None

            result = await response.json()
            if 'candidates' not in result or not result['candidates']:
                logger.error("No valid candidates in Gemini response")
                return None

            return result['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        logger.error(f"Gemini API call failed: {str(e)}")
        if retry > 0:
            logger.info("Retrying Gemini API call...")
            return await call_gemini_api(prompt, retry - 1)
        return None

def process_gemini_response(response_text):
    """Process and validate Gemini API response to ensure it's valid JSON."""
    try:
        if not response_text or not response_text.strip():
            raise ValueError("Empty response from Gemini API")

        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in response")

        json_str = json_match.group()
        return json.loads(json_str)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON response: {response_text}")
        return {"error": "Invalid JSON from Gemini API"}
    except Exception as e:
        logger.error(f"Failed to process Gemini response: {str(e)}")
        return {"error": str(e)}

# ================= API Endpoint =================
@app.route("/asr", methods=["POST", "OPTIONS"])
async def transcribe(request):
    try:
        if "audio" not in request.files:
            return response.json({"error": "No audio file provided"}, status=400)
            
        audio_file = request.files["audio"][0]
        
        if len(audio_file.body) > Config.MAX_AUDIO_SIZE:
            return response.json({"error": "File size exceeds 10MB limit"}, status=413)
            
        if audio_file.type not in Config.ALLOWED_AUDIO_TYPES:
            return response.json({"error": "Unsupported file type"}, status=415)

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, "input")
            output_path = os.path.join(tmp_dir, "output.wav")
            
            with open(input_path, "wb") as f:
                f.write(audio_file.body)

            try:
                convert_audio(input_path, output_path)
            except Exception as e:
                return response.json({"error": str(e)}, status=400)

            try:
                audio, sr = librosa.load(output_path, sr=Config.TARGET_SAMPLE_RATE)
            except Exception as e:
                logger.error(f"Error loading audio: {str(e)}")
                return response.json({"error": "Invalid audio file"}, status=400)

            # Language detection
            is_recorded = "recorded" in request.headers.get("audio-source", "")
            lang = detect_language_from_audio(audio, is_recorded)
            logger.info(f"Detected language: {lang}")

            # Transcription
            try:
                if lang == "ml":
                    transcription = models["whisper_ml"](audio)["text"]
                else:
                    transcription = models["whisper_en"](audio)["text"]
            except Exception as e:
                logger.error(f"Transcription failed: {str(e)}")
                return response.json({"error": "Transcription failed"}, status=500)

            # Translation only if detected language is Malayalam
            translation = transcription
            if lang == "ml":
                try:
                    translation = models["translator"](transcription)[0]["translation_text"]
                except Exception as e:
                    logger.error(f"Translation failed: {str(e)}")

            # Medical processing
            try:
                emr_data = await extract_emr(translation)
                suggestions = await generate_suggestions(emr_data)
            except Exception as e:
                logger.error(f"Medical processing failed: {str(e)}")
                return response.json({"error": "Medical analysis failed"}, status=500)

            return response.json({
                "status": "success",
                "transcription": transcription,
                "translation": translation,
                "emr": emr_data,
                "suggestions": suggestions
            })

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return response.json({"error": "Internal server error"}, status=500)

# ================= Medical Processing =================
async def extract_emr(text):
    try:
        prompt = f"""Extract medical information from this text and return JSON with: disease, allergy, timeline, medical_history.
        Return ONLY valid JSON. Text: {text}"""

        response_text = await call_gemini_api(prompt)
        if not response_text:
            raise ValueError("Empty response from Gemini API")

        return process_gemini_response(response_text)
    except Exception as e:
        logger.error(f"EMR extraction failed: {str(e)}")
        return {"error": "Medical data extraction failed"}

async def generate_suggestions(emr_data):
    try:
        prompt = f"""Suggest medical treatments based on this data. Return JSON with: treatment_suggestion, medicine_suggestion.
        Return ONLY valid JSON. Patient Data: {json.dumps(emr_data)}"""

        response_text = await call_gemini_api(prompt)
        if not response_text:
            raise ValueError("Empty response from Gemini API")

        return process_gemini_response(response_text)
    except Exception as e:
        logger.error(f"Suggestion generation failed: {str(e)}")
        return {"error": "Could not generate suggestions"}

# ================= Startup =================
@app.before_server_start
async def setup_resources(app, loop):
    if not load_models():
        logger.critical("Failed to initialize ML models")
        sys.exit(1)

@app.exception(MethodNotAllowed)
def handle_method_not_allowed(request, exception):
    return response.text(
        "",
        status=405,
        headers={"Allow": "POST, OPTIONS"}
    )

if __name__ == "__main__":
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        access_log=True,
        single_process=True
    )