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

@app.route("/analyze_url", methods=["POST"])
def analyze_url():
    if not os.environ.get("HF_TOKEN"):
        return jsonify({"error": "HF_TOKEN missing"}), 500

    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "URL missing"}), 400

    url = data["url"]
    rapid_api_key = os.environ.get("RAPID_API_KEY") 
    
    if not rapid_api_key:
        return jsonify({"error": "RAPID_API_KEY missing in Render settings"}), 500

    # 1. Handle YouTube vs Instagram/Others
    is_youtube = "youtube.com" in url or "youtu.be" in url
    
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, "cloud_audio.wav")

    try:
        # Configuration for the specific API from your screenshot
        headers = {
            "X-RapidAPI-Key": rapid_api_key,
            "X-RapidAPI-Host": "social-media-video-downloader.p.rapidapi.com"
        }

        if is_youtube:
            video_id = extract_video_id(url)
            if not video_id:
                return jsonify({"error": "Invalid YouTube URL"}), 400
            
            # Endpoint from your screenshot
            api_url = "https://social-media-video-downloader.p.rapidapi.com/smvd/get/all"
            querystring = {"videoId": video_id}
            response = requests.get(api_url, headers=headers, params=querystring)
        else:
            # For Instagram/Other social links, we use the standard URL parameter
            api_url = "https://social-media-video-downloader.p.rapidapi.com/smvd/get/all"
            querystring = {"url": url}
            response = requests.get(api_url, headers=headers, params=querystring)

        if response.status_code != 200:
            logger.error(f"RapidAPI failed: {response.text}")
            return jsonify({"error": "Extraction service error"}), 500

        result_data = response.json()
        audio_url = None
        
        # Parsing logic for 'medias' or 'links'
        links = result_data.get("links", result_data.get("medias", []))
        
        # Look for best audio
        for link in links:
            if link.get("type") == "audio" or link.get("extension") == "mp3":
                audio_url = link.get("url")
                break
        
        # Fallback to video if no audio-only link is provided
        if not audio_url and links:
            audio_url = links[0].get("url")

        if not audio_url:
            return jsonify({"error": "Could not find a downloadable stream"}), 500

        # Download to Render Cloud
        with requests.get(audio_url, stream=True) as r:
            r.raise_for_status()
            with open(temp_file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Handoff to Hugging Face AI
        result = query_huggingface(temp_file_path)
        return jsonify(result)

    except Exception as e:
        logger.error(f"System Error: {str(e)}")
        return jsonify({"error": "Failed to process link"}), 500
    finally:
        # AUTO-DELETE
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info("Auto-cleanup complete.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)