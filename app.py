"""Web UI: python app.py  ->  http://127.0.0.1:5000"""
import os
import tempfile
import threading

from flask import Flask, jsonify, request, send_from_directory

from stt import Transcriber

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
transcriber = Transcriber()


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/routes")
def routes():
    return jsonify(transcriber.config["routes"])


@app.get("/api/status")
def status():
    return jsonify(ready=transcriber.ready, loading=transcriber.loading, error=transcriber.warm_error)


@app.post("/api/transcribe")
def transcribe():
    f = request.files.get("audio")
    if f is None:
        return jsonify(error="No audio was sent."), 400
    language = request.form.get("language", "auto")
    if language not in ("auto", "en", "hi", "ta"):
        return jsonify(error="Unknown language."), 400
    suffix = os.path.splitext(f.filename or "")[1] or ".webm"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        f.save(path)
        translate = request.form.get("translate") == "1"
        return jsonify(transcriber.transcribe_file(path, language, translate=translate))
    except (ValueError, RuntimeError) as e:
        return jsonify(error=str(e)), 400
    except Exception as e:
        app.logger.exception("transcription failed")
        return jsonify(error=f"{type(e).__name__}: {e}"), 500
    finally:
        os.remove(path)


if __name__ == "__main__":
    if transcriber.config.get("preload", True):
        threading.Thread(target=transcriber.warm_up, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, threaded=True)
