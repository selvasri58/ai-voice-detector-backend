# server.py
import os
import logging
import tempfile
import subprocess
import shutil
import yt_dlp
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
    token = os.environ.get("HF_TOKEN")
    
    if token:
        logger.info(f"Token loaded successfully! Starts with: {token[:5]}") 
        return jsonify({
            "status": "AI Voice Detector API Running", 
            "connection": "Connected to Hugging Face Space",
            "token_status": "Loaded correctly!"
        })
    else:
        logger.error("🚨 ERROR: TOKEN IS MISSING!")
        return jsonify({
            "status": "AI Voice Detector API Running", 
            "token_status": "MISSING!"
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

        result = query_huggingface(wav_file)
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
            "format": "best", 
            "ffmpeg_location": ffmpeg_path, 
            "cookiefile": "cookies.txt", 
            "nocheckcertificate": True,
            "quiet": True,
            "no_warnings": True,
            # 🔥 FIX: Disguise as Safari and Smart TV to bypass the bot check without breaking Shorts formats
            "extractor_args": {"youtube": {"player_client": ["web_safari", "tv"]}},
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

        result = query_huggingface(audio_file)
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