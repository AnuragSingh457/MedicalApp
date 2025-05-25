from flask import Flask, request, jsonify
import os
import re
from dotenv import load_dotenv
import openai
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
import time
import threading
import functools
import gc
import requests
from flask import send_from_directory

# Initialize Flask App
app = Flask(
    __name__,
    static_folder='medical-chatbot-frontend/build',  # 👈 this is the correct path
    static_url_path=''  # 👈 important for serving index.html from /
)

CORS(app, resources={r"/*": {"origins": "*"}})

# Global variables for models
diagnosis_model = None
diagnosis_vectorizer = None
diagnosis_label_encoder = None
disease_dict = None
model_loading_lock = threading.Lock()
models_ready = threading.Event()

# NLTK Download
nltk.download('punkt', quiet=True)

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def memoize(func):
    cache = {}
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper

def load_models_in_background():
    global diagnosis_model, diagnosis_vectorizer, diagnosis_label_encoder, disease_dict

    try:
        print("Loading diagnosis models...")
        diagnosis_model = tf.keras.models.load_model("lstm_disease_model_clean.keras", compile=False)
        with open("tfidf_vectorizer.pkl", "rb") as file:
            diagnosis_vectorizer = pickle.load(file)
        with open("label_encoder.pkl", "rb") as file:
            diagnosis_label_encoder = pickle.load(file)
        disease_df = pd.read_csv("final_processed_dataset.csv")
        disease_dict = disease_df.groupby("Disease").agg(lambda x: list(set(x.dropna()))).to_dict(orient="index")
        print("Diagnosis models loaded")
    except Exception as e:
        print(f"Diagnosis loading error: {e}")

    try:
        print("OpenAI API is being used for medical chatbot.")
    except Exception as e:
        print(f"OpenAI setup error: {e}")

    models_ready.set()

model_thread = threading.Thread(target=load_models_in_background)
model_thread.daemon = True
model_thread.start()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(filepath):
    class_labels = {
        0: 'BA- cellulitis', 1: 'BA-impetigo', 2: 'Bulging_Eyes', 3: 'Cataracts',
        4: 'Crossed_Eyes', 5: 'FU-athlete-foot', 6: 'FU-nail-fungus', 7: 'FU-ringworm',
        8: 'Glaucoma', 9: 'PA-cutaneous-larva-migrans', 10: 'Uveitis',
        11: 'VI-chickenpox', 12: 'VI-shingles'
    }

    try:
        model = tf.keras.models.load_model("disease_detection_model.keras", compile=False)
        img = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction)
        del model
        gc.collect()
        return class_labels.get(predicted_class, "Unknown condition")
    except Exception as e:
        print(f"process_image error: {e}")
        return f"Error: {e}"


load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def generate_medical_response(user_input):
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "mistralai/mistral-7b-instruct",  # or "openai/gpt-3.5-turbo"
            "messages": [
                {"role": "system", "content": "You are a helpful medical assistant. Keep answers clear and safe."},
                {"role": "user", "content": user_input}
            ]
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"OpenRouter Error: {response.status_code} - {response.text}"

    except Exception as e:
        print(f"OpenRouter generate_medical_response error: {e}")
        return "Error generating response via OpenRouter."

def diagnose_disease(user_input):
    if not models_ready.is_set() or not diagnosis_model:
        return {"error": "Diagnosis initializing."}

    try:
        tokens = word_tokenize(user_input.lower())
        symptoms_text = " ".join(tokens)
        input_vector = diagnosis_vectorizer.transform([symptoms_text]).toarray()
        prediction = diagnosis_model.predict(input_vector, verbose=0)
        top_indices = np.argsort(prediction[0])[-3:][::-1]
        top_diseases = diagnosis_label_encoder.inverse_transform(top_indices)
        top_probabilities = prediction[0][top_indices]
        top_predictions = [(d, p) for d, p in zip(top_diseases, top_probabilities) if p > 0.4]
        if not top_predictions:
            top_predictions = [(top_diseases[0], top_probabilities[0])]
        predicted_disease = top_predictions[0][0]
        disease_info = disease_dict.get(predicted_disease, {})
        return {
            "disease": predicted_disease,
            "probability": float(top_predictions[0][1]),
            "description": disease_info.get("Description", ["N/A"])[0],
            "medicine": disease_info.get("Medicine", ["Consult doctor"])[0],
            "workout": disease_info.get("Workout", ["Exercise regularly"])[0],
            "precautions": disease_info.get("Precautions", ["Stay healthy"])
        }
    except Exception as e:
        print(f"diagnose_disease error: {e}")
        return {"error": f"Diagnosis error: {e}"}

@app.route("/api/medical-chat", methods=["POST"])
def medical_chat():
    if not models_ready.is_set():
        return jsonify({"response": "Initializing models..."}), 202

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        user_input = data.get("message", "").strip()
        if not user_input:
            return jsonify({"error": "Empty message"}), 400

        mode = data.get("mode", "diagnosis")
        if mode == "diagnosis" and diagnosis_model:
            response = diagnose_disease(user_input)
            return jsonify(response)
        else:
            bot_response = generate_medical_response(user_input)
            return jsonify({"response": bot_response})
    except Exception as e:
        print(f"medical_chat error: {e}")
        return jsonify({"error": "Server error"}), 500

@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        predicted_symptom = process_image(filepath)
        return jsonify({'symptom': predicted_symptom})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    models_loaded = {
        'diagnosis': diagnosis_model is not None,
        'models_ready': models_ready.is_set()
    }
    return jsonify({'status': 'running', 'models_loaded': models_loaded})

@app.route('/api/loading-status', methods=['GET'])
def loading_status():
    return jsonify({
        'models_loaded': models_ready.is_set(),
        'message': "Models ready" if models_ready.is_set() else "Models are still loading"
    })

import json

@app.route('/api/symptom-checker', methods=['POST'])
def symptom_checker():
    if not models_ready.is_set():
        return jsonify({'error': 'Models are still loading. Please wait.'}), 503

    try:
        data = request.get_json()
        symptoms = data.get('symptoms', [])
        severity = data.get('severity', '')
        duration = data.get('duration', '')

        if not symptoms:
            return jsonify({'error': 'No symptoms provided'}), 400

        symptom_text = ', '.join(symptoms)
        prompt = (
            f"A patient reports the following symptoms: {symptom_text}. "
            f"The severity is {severity}, and the duration is {duration}. "
            "Provide the following medical response in **raw JSON only** (no explanation), with the keys: "
            "`disease`, `description`, `medicine`, `precautions`."
        )

        ai_response = generate_medical_response(prompt)

        # Remove markdown code blocks or extra text (if any)
        cleaned = re.search(r'\{.*\}', ai_response, re.DOTALL)
        if cleaned:
            ai_response = cleaned.group(0)

        try:
            structured_response = json.loads(ai_response)
            return jsonify(structured_response)
        except json.JSONDecodeError:
            print("JSON parse failed. Returning raw text.")
            return jsonify({'raw_response': ai_response})

    except Exception as e:
        print(f"symptom_checker error: {e}")
        return jsonify({'error': 'Server error occurred'}), 500
    

# Serve React build files from Flask
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    root_dir = os.path.join(os.getcwd(), 'medical-chatbot-frontend', 'build')
    if path != "" and os.path.exists(os.path.join(root_dir, path)):
        return send_from_directory(root_dir, path)
    else:
        return send_from_directory(root_dir, 'index.html')    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
