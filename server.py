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

    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, "cloud_audio.wav")

    try:
        api_url = "https://social-media-video-downloader.p.rapidapi.com/youtube/v3/video/details"
        video_id = extract_video_id(url)
        if not video_id:
            return jsonify({"error": "Could not extract YouTube ID"}), 400

        headers = {
            "X-RapidAPI-Key": rapid_api_key,
            "X-RapidAPI-Host": "social-media-video-downloader.p.rapidapi.com"
        }

        querystring = {
            "videoId": video_id,
            "renderableFormats": "720p"
        }

        response = requests.get(api_url, headers=headers, params=querystring)
        
        if response.status_code != 200:
            logger.error(f"RapidAPI failed ({response.status_code}): {response.text}")
            return jsonify({"error": f"API Error {response.status_code}"}), 500

        result_data = response.json()
        audio_url = None
        contents = result_data.get("contents", [])
        
        if contents:
            audios = contents[0].get("audios", [])
            if audios:
                audio_url = audios[0].get("url")
            else:
                videos = contents[0].get("videos", [])
                if videos:
                    audio_url = videos[0].get("url")

        if not audio_url:
            return jsonify({"error": "No download link found"}), 500

        # 🔥 FIX: Use custom headers to bypass 403 Forbidden Error
        download_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Connection": "keep-alive",
            "Referer": "https://www.youtube.com/"
        }

        # Download the actual file from the extracted link
        with requests.get(audio_url, stream=True, headers=download_headers, timeout=30, allow_redirects=True) as r:
            r.raise_for_status()
            with open(temp_file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        # Send for AI Prediction
        result = query_huggingface(temp_file_path)
        return jsonify(result)

    except Exception as e:
        logger.error(f"System Error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info("Auto-cleanup complete.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)