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
        
        api_url = "https://youtube-mp310.p.rapidapi.com/download/mp3"
        headers = {"x-rapidapi-key": rapid_api_key, "x-rapidapi-host": "youtube-mp310.p.rapidapi.com"}

        # 1. Get the Link
        response = requests.get(api_url, headers=headers, params={"url": url}, timeout=30)
        download_url = response.json().get("downloadUrl")

        if not download_url:
            return jsonify({"error": "Bridge link generation failed"}), 500

        # 2. THE STITCHER: Download in two separate segments to bypass the 394KB cutoff
        logger.info("Starting Segmented Stitching...")
        
        with open(raw_path, "wb") as f:
            # Segment 1: 0 to 300KB
            headers_1 = {"Range": "bytes=0-300000"}
            r1 = requests.get(download_url, headers=headers_1, timeout=30)
            if r1.status_code in [200, 206]:
                f.write(r1.content)
                logger.info("Segment 1 (0-300KB) saved.")
            
            # Segment 2: 300KB to the end
            headers_2 = {"Range": "bytes=300001-"}
            r2 = requests.get(download_url, headers=headers_2, timeout=30)
            if r2.status_code in [200, 206]:
                f.write(r2.content)
                logger.info("Segment 2 (300KB+) saved.")

        # Check if we actually got a usable file size
        if os.path.getsize(raw_path) < 10000:
             return jsonify({"error": "Segment stitching failed"}), 500

        # 3. Convert and Analyze
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [ffmpeg_path, "-y", "-i", raw_path, "-ac", "1", "-ar", "16000", final_wav_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )

        result = query_huggingface(final_wav_path)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Stitcher Error: {str(e)}")
        return jsonify({"error": "Network limit reached. Try a shorter video."}), 500
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info("Auto-cleanup complete.")
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)