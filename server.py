# server.py
import os
import logging
import tempfile
import subprocess
import requests
import shutil
import re
import imageio_ffmpeg 
from flask import Flask, request, jsonify
from gradio_client import Client, handle_file 

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", 10000))
HF_SPACE_URL = "selva58/voice-detector-api" 

def extract_video_id(url):
    """Extracts YouTube ID from various URL formats."""
    pattern = r"(?:v=|\/shorts\/|\/embed\/|youtu.be\/|\/v\/|watch\?v=|\&v=)([^#\&\?]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def query_huggingface(audio_file_path):
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return {"error": "Server configuration error: HF_TOKEN is missing"}
    try:
        client = Client(HF_SPACE_URL, token=hf_token)
        result = client.predict(
            audio_path=handle_file(audio_file_path),
            api_name="/analyze_audio"
        )
        return result
    except Exception as e:
        logger.error(f"Space request error: {e}")
        return {"error": f"Failed to connect to AI Space: {str(e)}"}

@app.route("/")
def home():
    return jsonify({
        "status": "AI Voice Detector API Running", 
        "mode": "RapidAPI Secure Extraction"
    })

@app.route("/analyze", methods=["POST"])
def analyze_audio():
    if not os.environ.get("HF_TOKEN"):
        return jsonify({"error": "HF_TOKEN not set"}), 500
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    fd, temp_file_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    try:
        file.save(temp_file_path)
        wav_file = temp_file_path + "_converted.wav"
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [ffmpeg_path, "-y", "-i", temp_file_path, "-ac", "1", "-ar", "16000", wav_file],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        result = query_huggingface(wav_file)
        return jsonify(result)
    finally:
        if os.path.exists(temp_file_path): os.remove(temp_file_path)
        if 'wav_file' in locals() and os.path.exists(wav_file): os.remove(wav_file)

import io

@app.route("/analyze_url", methods=["POST"])
def analyze_url():
    if not os.environ.get("HF_TOKEN"):
        return jsonify({"error": "HF_TOKEN missing"}), 500

    data = request.get_json()
    url = data.get("url")
    rapid_api_key = os.environ.get("RAPID_API_KEY") 

    try:
        api_url = "https://youtube-mp310.p.rapidapi.com/download/mp3"
        headers = {"x-rapidapi-key": rapid_api_key, "x-rapidapi-host": "youtube-mp310.p.rapidapi.com"}

        # 1. Get the Link
        response = requests.get(api_url, headers=headers, params={"url": url}, timeout=30)
        download_url = response.json().get("downloadUrl")

        if not download_url:
            return jsonify({"error": "Bridge link failed"}), 500

        # 🔥 THE DIRECT PIPE FIX: 
        # We don't download. We point Hugging Face directly to the Bridge URL.
        # This lets Hugging Face's high-speed network handle the IncompleteRead.
        logger.info("Redirecting stream to Hugging Face...")
        
        client = Client(HF_SPACE_URL, token=os.environ.get("HF_TOKEN"))
        
        # We use the URL itself as the handle_file source
        result = client.predict(
            audio_path=download_url, # Pass the URL directly!
            api_name="/analyze_audio"
        )
        
        return jsonify(result)

    except Exception as e:
        logger.error(f"Stream Error: {str(e)}")
        return jsonify({"error": "AI Space is busy. Please try again."}), 500
    



    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)