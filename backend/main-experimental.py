import logging
import json
import os
from sanic import Sanic
from sanic import response
from sanic.request import Request
from sanic.response import JSONResponse
from transformers import pipeline
from langdetect import detect

# Configure logging for better error tracking
logging.basicConfig(level=logging.ERROR)

app = Sanic(__name__)

# Load machine learning models 
try:
    pipe_en = pipeline("automatic-speech-recognition", model="openai/whisper-small")
    pipe_ml = pipeline("automatic-speech-recognition", model="kavyamanohar/whisper-small-malayalam")
    pipe_translate = pipeline("translation", model="Helsinki-NLP/opus-mt-ml-en")
    text_generator = pipeline("text-generation", model="microsoft/Phi-3.5-mini-instruct", trust_remote_code=True) 

except Exception as e:
    logging.error(f"Error loading models: {e}")
    raise

# Function to extract EMR data 
def extract_emr(text):
    prompt = f"""
    You are a helpful medical assistant. 
    Please extract and structure relevant Electronic Medical Record (EMR) information from the following text. 
    If the text seems like gibberish, try to guess what it means in the context of medical information.
    Focus on:
    - Disease: Any mentioned illnesses or health conditions.
    - Allergy: Any mentioned allergies.
    - Timeline: Any dates, durations, or time-related information about the medical context.
    - Medical History: Any past medical conditions or treatments.

    Text: {text}

    Output:
    ```json
    {{
    "disease": "...", 
    "allergy": "...",
    "timeline": "...",
    "medical_history": "..." 
    }}
    ```
    """
    try:
        response = text_generator(prompt, max_length=256)  # Adjust max_length if needed
        extracted_data = response[0]['generated_text'] 
        return json.loads(extracted_data)

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logging.error(f"Error parsing text generation output: {e}. Output: {response}")
        return {
            "disease": "N/A (Error parsing response)",
            "allergy": "N/A (Error parsing response)",
            "timeline": "N/A (Error parsing response)",
            "medical_history": "N/A (Error parsing response)"
        }

# Function to generate suggestions 
def generate_suggestions(emr_data):
    emr_data_str = json.dumps(emr_data)
    prompt = f"""
    You are a helpful medical assistant. Based on the following EMR data, 
    suggest potential treatments and medications. If you cannot make a suggestion, respond with "N/A".

    EMR Data: 
    {emr_data_str}

    Output:
    ```json
    {{
        "treatment_suggestion": "...",
        "medicine_suggestion": "..."
    }}
    ```
    """
    try:
        response = text_generator(prompt, max_length=256)  # Adjust max_length if needed
        suggestions_data = response[0]['generated_text']
        return json.loads(suggestions_data)

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logging.error(f"Error parsing text generation output: {e}. Output: {response}")
        return {
            "treatment_suggestion": "N/A (Error parsing response)",
            "medicine_suggestion": "N/A (Error parsing response)"
        }
# Function to generate suggestions
def generate_suggestions(emr_data):
    emr_data_str = json.dumps(emr_data)
    prompt = f"""
    You are a helpful medical assistant. Based on the following EMR data, 
    suggest potential treatments and medications. If you cannot make a suggestion, respond with "N/A".

    EMR Data: 
    {emr_data_str}

    Output:
    ```json
    {{
        "treatment_suggestion": "...",
        "medicine_suggestion": "..."
    }}
    ```
    """
    
    try:
        response = text_generator(prompt, max_length=256)  # Adjust max_length as needed
        suggestions_data = response[0]['generated_text']
        return json.loads(suggestions_data) 

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logging.error(f"Error parsing text generation output: {e}. Output: {response}")
        return {
            "treatment_suggestion": "N/A (Error parsing response)",
            "medicine_suggestion": "N/A (Error parsing response)"
        }

@app.route("/")
async def test(request):
    return response.json({"test": True})

@app.route('/asr', methods=['POST'])
async def transcribe(request: Request) -> JSONResponse:
    audio_data = request.body
    if not audio_data:
        return JSONResponse({"error": "No audio data provided."}, status=400)

    try:
        # 1. Language Detection
        try:
            text_snippet = pipe_en(audio_data)["text"][:200]
            detected_lang = detect(text_snippet)

            if detected_lang != 'en':
                detected_lang = 'ml'

            print(f"Detected Language: {detected_lang}")

        except Exception as e:
            print(f"Language detection failed: {str(e)} - Defaulting to Malayalam")
            detected_lang = 'ml'

        # 2. Choose Speech Recognition Pipeline
        pipe = pipe_ml if detected_lang == 'ml' else pipe_en

        # 3. Perform Speech Recognition
        transcription = pipe(audio_data)["text"]
        print(f"Full Transcription: {transcription}")

        # 4. Translate to English if necessary
        if detected_lang == 'ml':
            translation_output = pipe_translate(transcription)
            english_text = translation_output[0]['translation_text']
            print(f"English Translation: {english_text}")
        else:
            english_text = transcription

        # 5. Extract EMR data 
        emr_data = extract_emr(english_text)
        print(f"Extracted EMR Data: {emr_data}")

        # 6. Generate suggestions from EMR data
        suggestions = generate_suggestions(emr_data)
        print(f"Generated Suggestions: {suggestions}")

        # 7. Prepare and send the response
        response_data = {
            "transcription": transcription,
            "translation": english_text,
            "emr_data": emr_data,
            "suggestions": suggestions
        }
        return JSONResponse(response_data)

    except Exception as e:
        return JSONResponse({"error": f"Error: {str(e)}"}, status=500)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
