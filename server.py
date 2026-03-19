import os
import logging
import tempfile
import subprocess
import shutil
import requests
import yt_dlp
import time

from flask import Flask, request, jsonify

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", 10000))

# Hugging Face API (FIXED URL)
HF_API_URL = "https://api-inference.huggingface.co/models/selva58/ai-voice-detector"
HF_TOKEN = os.environ.get("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


# -------- HELPER FUNCTION (IMPORTANT) --------
def query_huggingface(audio_bytes):
    for attempt in range(5):
        try:
            response = requests.post(
                HF_API_URL,
                headers=headers,
                data=audio_bytes,
                timeout=60
            )

            result = response.json()

            # Handle model loading
            if "error" in result:
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
    return jsonify({"status": "AI Voice Detector API Running"})


# -------- AUDIO FILE ANALYSIS --------
@app.route("/analyze", methods=["POST"])
def analyze_audio():

    try:
        if not HF_TOKEN:
            return jsonify({"error": "HF_TOKEN not set"}), 500

        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        file_ext = os.path.splitext(file.filename)[1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file_path = temp_file.name
            file.save(temp_file_path)

        # Convert to WAV 16kHz mono
        wav_file = temp_file_path.replace(file_ext, ".wav")

        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_file_path, "-ac", "1", "-ar", "16000", wav_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        # ✅ ADD THIS CHECK
        if not os.path.exists(wav_file):
            logger.error("FFmpeg failed to create WAV file")
            return jsonify({"error": "Audio conversion failed"}), 500

        os.remove(temp_file_path)

        with open(wav_file, "rb") as f:
            audio_bytes = f.read()

        os.remove(wav_file)

        # 🔥 Use retry function
        result = query_huggingface(audio_bytes)

        logger.info(f"Model response: {result}")

        return jsonify(result)

    except Exception as e:
        logger.error(f"Analyze error: {e}")
        return jsonify({"error": "Processing failed"}), 500


# -------- URL ANALYSIS --------
@app.route("/analyze_url", methods=["POST"])
def analyze_url():

    try:
        if not HF_TOKEN:
            return jsonify({"error": "HF_TOKEN not set"}), 500

        data = request.get_json()

        if not data or "url" not in data:
            return jsonify({"error": "URL missing"}), 400

        url = data["url"]

        temp_dir = tempfile.mkdtemp()

        ydl_opts = {
            "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
            "format": "bestaudio/best",
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

        shutil.rmtree(temp_dir)

        # 🔥 Use retry function
        result = query_huggingface(audio_bytes)

        return jsonify(result)

    except Exception as e:
        logger.error(f"URL analysis error: {e}")
        return jsonify({"error": "URL processing failed"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)