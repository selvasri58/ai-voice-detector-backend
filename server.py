# server.py
import os
import logging
import tempfile
import subprocess
import requests
import shutil
import imageio_ffmpeg 
from flask import Flask, request, jsonify
from gradio_client import Client, handle_file 

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", 10000))
HF_SPACE_URL = "selva58/voice-detector-api" 

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
        "mode": "Secure Cloud Extraction"
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
        # Automatic Cleanup
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
    logger.info(f"Processing URL: {url}")
    
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, "cloud_audio.wav")

    # 🔥 LIST OF VERIFIED V10 MIRRORS
    mirrors = [
        "https://cobalt.lucas.sh/api/json",
        "https://cobalt.peris.me/api/json",
        "https://api.cobalt.tools/api/json" 
    ]

    try:
        audio_url = None
        
        # Try each bridge until one works
        for mirror in mirrors:
            try:
                logger.info(f"Trying bridge: {mirror}")
                payload = {"url": url, "downloadMode": "audio", "audioFormat": "wav"}
                headers = {"Accept": "application/json", "Content-Type": "application/json"}
                
                response = requests.post(mirror, json=payload, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    audio_url = response.json().get("url")
                    if audio_url:
                        logger.info(f"Success with {mirror}!")
                        break
                else:
                    logger.warning(f"Mirror {mirror} returned status {response.status_code}")
            except Exception as e:
                logger.warning(f"Mirror {mirror} failed: {str(e)}")
                continue

        if not audio_url:
            return jsonify({"error": "All extraction bridges are currently busy. Try again in 1 minute."}), 503

        # Download the file to Render
        with requests.get(audio_url, stream=True) as r:
            r.raise_for_status()
            with open(temp_file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Send to Hugging Face
        result = query_huggingface(temp_file_path)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Final System Error: {str(e)}")
        return jsonify({"error": "Failed to extract audio from link"}), 500
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info("Auto-cleanup complete: Local files deleted.")
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)