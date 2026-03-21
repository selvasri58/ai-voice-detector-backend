# server.py
import os
import logging
import tempfile
import subprocess
import shutil
import requests
import yt_dlp
import time
import imageio_ffmpeg 

from flask import Flask, request, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", 10000))
HF_API_URL = "https://api-inference.huggingface.co/models/selva58/ai-voice-detector"

def query_huggingface(audio_bytes):
    # Fetch token securely at runtime
    hf_token = os.environ.get("HF_TOKEN")
    
    if not hf_token:
        return {"error": "Server configuration error: HF_TOKEN is missing"}

    headers = {"Authorization": f"Bearer {hf_token}"}
    
    for attempt in range(5):
        try:
            response = requests.post(
                HF_API_URL,
                headers=headers,
                data=audio_bytes,
                timeout=60
            )
            
            try:
                result = response.json()
            except ValueError:
                return {"error": "Invalid API response from Hugging Face"}

            if isinstance(result, dict) and "error" in result:
                if "loading" in result["error"].lower():
                    logger.info(f"Model loading... retry {attempt+1}")
                    time.sleep(5)
                    continue
            return result
        except Exception as e:
            logger.error(f"HF request error: {e}")
            time.sleep(3)
    return {"error": "Model failed after retries"}
@app.route("/")
def home():
    token = os.environ.get("HF_TOKEN")
    
    # Grab all the keys (names) of the environment variables Render is providing
    # We only grab the names, NOT the secret values, to keep it secure!
    env_keys = list(os.environ.keys())
    logger.info(f"RENDER PROVIDED THESE KEYS: {env_keys}")
    
    if token:
        logger.info(f"Token loaded successfully! Starts with: {token[:5]}") 
        return jsonify({
            "status": "AI Voice Detector API Running", 
            "token_status": "Loaded correctly!"
        })
    else:
        logger.error("🚨 ERROR: TOKEN IS MISSING!")
        return jsonify({
            "status": "AI Voice Detector API Running", 
            "token_status": "MISSING!",
            "render_is_seeing_these_variables": env_keys
        })

@app.route("/analyze", methods=["POST"])
def analyze_audio():
    if not os.environ.get("HF_TOKEN"):
        return jsonify({"error": "HF_TOKEN not set in environment variables"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_ext = os.path.splitext(file.filename)[1].lower() or '.wav'

    fd, temp_file_path = tempfile.mkstemp(suffix=file_ext)
    os.close(fd)

    try:
        file.save(temp_file_path)
        wav_file = temp_file_path + "_converted.wav"

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        subprocess.run(
            [ffmpeg_path, "-y", "-i", temp_file_path, "-ac", "1", "-ar", "16000", wav_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        with open(wav_file, "rb") as f:
            audio_bytes = f.read()

        result = query_huggingface(audio_bytes)
        return jsonify(result)

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Audio conversion failed: {e}"}), 500
    except Exception as e:
        logger.error(f"Analyze error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if 'wav_file' in locals() and os.path.exists(wav_file):
            os.remove(wav_file)


@app.route("/analyze_url", methods=["POST"])
def analyze_url():
    if not os.environ.get("HF_TOKEN"):
        return jsonify({"error": "HF_TOKEN not set in environment variables"}), 500

    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "URL missing"}), 400

    url = data["url"]
    temp_dir = tempfile.mkdtemp()

    try:
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe() 
        
        ydl_opts = {
            "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
            "format": "bestaudio/best",
            "ffmpeg_location": ffmpeg_path, 
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }]
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)

        audio_file = None
        for f in os.listdir(temp_dir):
            if f.endswith(".wav"):
                audio_file = os.path.join(temp_dir, f)

        if audio_file is None:
            return jsonify({"error": "Audio extraction failed"}), 500

        with open(audio_file, "rb") as f:
            audio_bytes = f.read()

        result = query_huggingface(audio_bytes)
        return jsonify(result)

    except yt_dlp.utils.DownloadError:
        return jsonify({"error": "Failed to download video."}), 500
    except Exception as e:
        logger.error(f"URL analysis error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)