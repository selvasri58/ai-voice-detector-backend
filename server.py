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
    temp_dir = None
    if not os.environ.get("HF_TOKEN"):
        return jsonify({"error": "HF_TOKEN missing"}), 500

    data = request.get_json()
    url = data.get("url")
    rapid_api_key = os.environ.get("RAPID_API_KEY") 

    try:
        temp_dir = tempfile.mkdtemp()
        raw_path = os.path.join(temp_dir, "raw_audio.mp3")
        final_wav_path = os.path.join(temp_dir, "final_audio.wav")
        
        # 1. Get the Link from your current YouTube MP3 API
        api_url = "https://youtube-mp310.p.rapidapi.com/download/mp3"
        headers = {
            "x-rapidapi-key": rapid_api_key,
            "x-rapidapi-host": "youtube-mp310.p.rapidapi.com"
        }

        logger.info(f"Requesting link for: {url}")
        response = requests.get(api_url, headers=headers, params={"url": url}, timeout=30)
        download_url = response.json().get("downloadUrl")

        if not download_url:
            return jsonify({"error": "Bridge did not provide a link"}), 500

        # 2. THE PERSISTENT PIPE: Force the download to finish
        logger.info("Starting Persistent Pipe download...")
        
        # We use a specific 'Accept-Encoding' to prevent the server from cutting us off
        download_headers = {"Accept-Encoding": "identity"}
        
        # Using a raw stream to bypass the 'IncompleteRead' logic of the requests library
        with requests.get(download_url, headers=download_headers, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(raw_path, "wb") as f:
                # We use a very large buffer to 'catch' the data before the connection drops
                for chunk in r.iter_content(chunk_size=1024 * 512): # 512KB chunks
                    if chunk:
                        f.write(chunk)
        
        # Check if we got enough data (at least 50KB)
        if os.path.getsize(raw_path) < 50000:
             return jsonify({"error": "File was too small/incomplete"}), 500

        # 3. Convert to WAV for AI processing
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [ffmpeg_path, "-y", "-i", raw_path, "-ac", "1", "-ar", "16000", final_wav_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )

        # 4. Analyze
        result = query_huggingface(final_wav_path)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Pipe Error: {str(e)}")
        # If it's the specific 177k error, we tell the user to try once more 
        # because the file is now 'cached' on the bridge server
        return jsonify({"error": "Connection reset. Please try the same link again."}), 500
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info("Auto-cleanup complete.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)