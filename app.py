from flask import Flask, render_template, request, jsonify
import os
import speech_recognition as sr
from pydub import AudioSegment
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Helper: Convert any audio format to WAV
def convert_to_wav(file_path):
    if not file_path.endswith(".wav"):
        sound = AudioSegment.from_file(file_path)
        wav_path = os.path.splitext(file_path)[0] + ".wav"
        sound.export(wav_path, format="wav")
        return wav_path
    return file_path

# Route: Homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route: Handle file upload and transcription
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['audio_file']
    language = request.form.get('language', 'en-IN')

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Convert to WAV
    wav_path = convert_to_wav(file_path)

    try:
        # Using OpenAI Whisper for transcription
        with open(wav_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        text = transcript.text

        # Fallback to Google API if OpenAI fails
        if not text:
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data, language=language)

        return jsonify({'transcription': text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(wav_path)  # Clean up after processing

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
