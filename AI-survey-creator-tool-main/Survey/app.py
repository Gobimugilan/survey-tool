from flask import Flask, request, render_template, render_template_string, jsonify, Response, redirect, url_for, flash, session, make_response, send_file
import time
from transformers import pipeline
from deep_translator import GoogleTranslator
from datetime import datetime, timezone
import sqlite3
import json
import csv
import io
import requests
from dotenv import load_dotenv
import os
import spacy
from collections import Counter
import uuid
from werkzeug.utils import secure_filename
import qrcode
import io
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
import geocoder
from dateutil.parser import isoparse
from google.cloud import vision
from twilio.twiml.voice_response import VoiceResponse, Gather

# --- Imports for Authentication ---
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from flask_bcrypt import Bcrypt

load_dotenv()
app = Flask(__name__)
SUPPORTED_LANGUAGES = [
    ('en', 'English'),
    ('hi', 'Hindi (हिन्दी)'),
    ('ta', 'Tamil (தமிழ்)'),
    ('te', 'Telugu (తెలుగు)'),
    ('kn', 'Kannada (ಕನ್ನಡ)'),
    ('ml', 'Malayalam (മലയാളം)'),
    ('bn', 'Bengali (বাংলা)'),
    ('mr', 'Marathi (मराठी)'),
    ('gu', 'Gujarati (ગુજરાતી)'),
    ('pa', 'Punjabi (ਪੰਜਾਬੀ)'),
    ('ur', 'Urdu (اردو)'),
    ('es', 'Spanish (Español)'),
    ('fr', 'French (Français)'),
    ('de', 'German (Deutsch)'),
    ('zh-cn', 'Chinese (中文)'),
    ('ja', 'Japanese (日本語)'),
    ('pt', 'Portuguese (Português)'),
    ('ru', 'Russian (Русский)'),
    ('ar', 'Arabic (العربية)'),
    ('ko', 'Korean (한국어)'),
    ('it', 'Italian (Italiano)'),
    ('id', 'Indonesian (Bahasa Indonesia)'),
    ('tr', 'Turkish (Türkçe)'),
    ('nl', 'Dutch (Nederlands)'),
    ('vi', 'Vietnamese (Tiếng Việt)')
]
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_default_secret_key_for_development')
MIN_SECONDS_PER_QUESTION = 2 # Minimum reasonable time to answer one question
# Create a static/uploads directory if it doesn't exist
if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads')

bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- AI Model Setup ---
from flask import Flask, request, render_template_string, jsonify, Response, redirect, url_for, flash, session, make_response, send_file
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from deep_translator import GoogleTranslator
from datetime import datetime, timezone
import sqlite3
import json
import csv
import io
import requests
from dotenv import load_dotenv
import os
import spacy
from collections import Counter
import uuid
from werkzeug.utils import secure_filename
import qrcode
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
import geocoder
from dateutil.parser import isoparse
from google.cloud import vision
from twilio.twiml.voice_response import VoiceResponse, Gather
import threading
import torch
import time

# --- Imports for Authentication ---
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt

load_dotenv()
app = Flask(__name__)
SUPPORTED_LANGUAGES = [
    ('en', 'English'), ('hi', 'Hindi (हिन्दी)'), ('ta', 'Tamil (தமிழ்)'),
    ('te', 'Telugu (తెలుగు)'), ('kn', 'Kannada (ಕನ್ನಡ)'), ('ml', 'Malayalam (മലയാളം)'),
    ('bn', 'Bengali (বাংলা)'), ('mr', 'Marathi (మరాఠీ)'), ('gu', 'Gujarati (ગુજરાતી)'),
    ('pa', 'Punjabi (ਪੰਜਾਬੀ)'), ('ur', 'Urdu (اردو)'), ('es', 'Spanish (Español)'),
    ('fr', 'French (Français)'), ('de', 'German (Deutsch)'), ('zh-cn', 'Chinese (中文)'),
    ('ja', 'Japanese (日本語)'), ('pt', 'Portuguese (Português)'), ('ru', 'Russian (Русский)'),
    ('ar', 'Arabic (العربية)'), ('ko', 'Korean (한국어)'), ('it', 'Italian (Italiano)'),
    ('id', 'Indonesian (Bahasa Indonesia)'), ('tr', 'Turkish (Türkçe)'),
    ('nl', 'Dutch (Nederlands)'), ('vi', 'Vietnamese (Tiếng Việt)')
]
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_default_secret_key_for_development')
MIN_SECONDS_PER_QUESTION = 2 # Minimum reasonable time to answer one question

if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads')

bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- AI Model Setup ---
print("Loading AI models...")
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    nlp = spacy.load("en_core_web_sm")
    print("AI models loaded successfully.")
except (OSError, ImportError) as e:
    print(f"Error loading AI models: {e}")
    sentiment_analyzer = None
    nlp = None

# --- Custom Model and Caching Setup ---
dataset_lock = threading.Lock()
DATASET_FILE = "survey_dataset.json"
custom_model = None
custom_tokenizer = None

def load_custom_model():
    global custom_model, custom_tokenizer
    base_model_name = "google/gemma-2b-it"
    adapter_path = "ai-survey-finetuned-model/final"
    print("Loading custom fine-tuned model...")
    if not os.path.exists(adapter_path):
        print(f"⚠️  Custom model not found at '{adapter_path}'. Will use API fallback.")
        return
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        custom_model = PeftModel.from_pretrained(base_model, adapter_path)
        custom_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        print("✅ Custom model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Could not load custom model: {e}")
        custom_model = None

def generate_with_custom_model(prompt, num_questions=3):
    if not custom_model or not custom_tokenizer: return None
    instruction = f"### INSTRUCTION\nGenerate a {num_questions}-question survey about {prompt}.\n\n### RESPONSE\n"
    pipe = pipeline("text-generation", model=custom_model, tokenizer=custom_tokenizer, max_new_tokens=500)
    result = pipe(instruction)
    try:
        response_text = result[0]['generated_text'].split("### RESPONSE")[1].strip()
        return json.loads(response_text)
    except Exception as e:
        print(f"Error parsing response from custom model: {e}")
        return None

def get_questions_from_dataset(prompt_text):
    try:
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        normalized_prompt = prompt_text.strip().lower()
        for entry in dataset:
            if entry.get("prompt", "").strip().lower() == normalized_prompt:
                print(f"CACHE HIT: Found prompt in local dataset.")
                return entry.get("completion")
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return None

def add_questions_to_dataset(prompt_text, completion_json_str):
    with dataset_lock:
        try:
            with open(DATASET_FILE, "r+", encoding="utf-8") as f:
                dataset = json.load(f)
                dataset.append({"prompt": prompt_text, "completion": completion_json_str})
                f.seek(0)
                json.dump(dataset, f, indent=2, ensure_ascii=False)
        except (FileNotFoundError, json.JSONDecodeError):
            with open(DATASET_FILE, "w", encoding="utf-8") as f:
                json.dump([{"prompt": prompt_text, "completion": completion_json_str}], f, indent=2, ensure_ascii=False)
    print(f"CACHE WRITE: Added prompt to local dataset.")

# --- User Model for Flask-Login ---
class User(UserMixin):
    def __init__(self, id, username, password):
        self.id, self.username, self.password = id, username, password

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    user_row = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    if user_row:
        return User(id=user_row['id'], username=user_row['username'], password=user_row['password_hash'])
    return None

def generate_ai_questions(prompt, num_questions=3, max_retries=3):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key: return [{"type": "OPEN", "text": "Error: API key not configured.", "options": []}]
# --- AI-Powered Generation ---
def generate_ai_questions(prompt, num_questions=3, max_retries=3):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key: return [{"type": "OPEN", "text": "Error: API key not configured.", "options": []}]
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    schema = {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {"type": {"type": "STRING"}, "text": {"type": "STRING"}, "options": {"type": "ARRAY", "items": {"type": "STRING"}}}, "required": ["type", "text", "options"]}}
    instruction = f"Generate {num_questions} unique and diverse survey questions about: '{prompt}'."
    payload = {"contents": [{"parts": [{"text": instruction}]}], "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema, "temperature": 1.0}}
    for attempt in range(max_retries):
        print(f"Generating questions... (Attempt {attempt + 1}/{max_retries})")
        try:
            response = requests.post(api_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            if not result.get('candidates'):
                print(f"Warning: API returned no candidates. Retrying...")
                time.sleep(1)
                continue
            json_string = result['candidates'][0]['content']['parts'][0]['text']
            questions = json.loads(json_string)
            question_texts = [q['text'].strip().lower() for q in questions]
            if len(set(question_texts)) == len(question_texts):
                print("✅ Successfully generated a set of unique questions.")
                return questions
            else:
                print(f"Warning: Duplicate questions detected. Retrying...")
        except Exception as e:
            print(f"An error occurred on attempt {attempt + 1}: {e}. Retrying...")
            time.sleep(1)
    print(f"❌ Failed to get unique questions after {max_retries} attempts. Returning fallback.")
    fallback_questions = [{"type": "OPEN", "text": f"What is your primary opinion on {prompt}?", "options": []}, {"type": "OPEN", "text": f"Can you describe a specific challenge related to {prompt}?", "options": []}, {"type": "OPEN", "text": f"How do you think the situation regarding {prompt} could be improved?", "options": []}]
    return fallback_questions[:num_questions]

# --- Helper Functions ---
def translate_text(text, target_language):
    if not text or target_language == 'en': return text
    try: return GoogleTranslator(source='auto', target=target_language).translate(text)
    except Exception as e: print(f"Translation error: {e}"); return text

def analyze_themes(responses):
    if not nlp: return {'positive': [], 'negative': []}
    positive_keywords, negative_keywords = [], []
    for response in responses:
        text, sentiment = response['answer'], response['sentiment']['label']
        doc = nlp(text)
        keywords = [token.lemma_.lower() for token in doc if token.pos_ in ['NOUN', 'ADJ'] and not token.is_stop and not token.is_punct]
        if sentiment == 'POSITIVE': positive_keywords.extend(keywords)
        elif sentiment == 'NEGATIVE': negative_keywords.extend(keywords)
    return {'positive': Counter(positive_keywords).most_common(5), 'negative': Counter(negative_keywords).most_common(5)}

# --- NEW: Cloud-based OCR Function ---
def ocr_image_with_google_vision(image_content):
    """Uses Google Cloud Vision API to extract text from image bytes."""
    try:
        # This automatically finds your gcp_credentials.json file
        # because we set the environment variable in the .env file.
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=image_content)

        # Use document_text_detection, which is best for dense text like a paper form.
        response = client.document_text_detection(image=image)

        if response.error.message:
            # If Google returns an error (e.g., bad API key, API not enabled), raise it.
            raise Exception(f"{response.error.message}")

        return response.full_text_annotation.text
        
    except Exception as e:
        # This will catch any other errors (e.g., library not installed, network issues).
        print(f"Google Cloud Vision API Error: {e}")
        print("Please ensure 'google-cloud-vision' is installed and your credentials are set correctly.")
        return None

    # --- NEW: Helper function for Twilio SMS ---
def send_sms_notification(recipient_number, survey_url):
    """Sends an SMS with the survey link using Twilio."""
    account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
    auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
    from_number = os.environ.get('TWILIO_PHONE_NUMBER')

    if not all([account_sid, auth_token, from_number]):
        print("Twilio credentials not fully configured in .env file.")
        return (False, "SMS service is not configured on the server.")

    client = Client(account_sid, auth_token)
    message_body = f"You've been invited to take a survey. Please click the link to participate: {survey_url}"

    try:
        message = client.messages.create(
            to=recipient_number,
            from_=from_number,
            body=message_body
        )
        return (True, message.sid)
    except TwilioRestException as e:
        print(f"Twilio Error: {e}")
        return (False, str(e))
    except Exception as e:
        print(f"General Error sending SMS: {e}")
        return (False, "An unexpected error occurred.")

        # --- NEW: Helper function for Twilio WhatsApp ---
def send_whatsapp_notification(recipient_number, survey_url):
    """Sends a WhatsApp message with the survey link using Twilio."""
    account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
    auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
    from_number = os.environ.get('TWILIO_WHATSAPP_NUMBER')

    if not all([account_sid, auth_token, from_number]):
        print("Twilio WhatsApp credentials not fully configured.")
        return (False, "WhatsApp service is not configured.")

    client = Client(account_sid, auth_token)
    message_body = f"You have been invited to take a survey. Please click the link to participate: {survey_url}"

    try:
        message = client.messages.create(
            # The 'whatsapp:' prefix is required by Twilio
            to=f'whatsapp:{recipient_number}',
            from_=f'whatsapp:{from_number}',
            body=message_body
        )
        return (True, message.sid)
    except TwilioRestException as e:
        print(f"Twilio WhatsApp Error: {e}")
        return (False, str(e))


# --- NEW: Route to generate QR code for a survey ---
@app.route('/survey/<uuid:survey_uuid>/qr')
@login_required
def generate_qr_code(survey_uuid):
    """Generates a QR code image for the survey link."""
    conn = get_db_connection()
    # Ensure the current user owns this survey before generating QR
    survey = conn.execute('SELECT user_id FROM surveys WHERE share_uuid = ?', (str(survey_uuid),)).fetchone()
    conn.close()
    if not survey or survey['user_id'] != current_user.id:
        return "Not Found or Access Denied", 404

    survey_url = url_for('take_survey', survey_uuid=survey_uuid, _external=True)
    qr_img = qrcode.make(survey_url)
    
    buf = io.BytesIO()
    qr_img.save(buf, format='PNG')
    buf.seek(0)

    return send_file(buf, mimetype='image/png')


# --- NEW: API Route for sending SMS invitations ---
@app.route('/api/survey/<int:survey_id>/send/sms', methods=['POST'])
@login_required
def send_sms_invites(survey_id):
    """API endpoint to send survey links via SMS."""
    conn = get_db_connection()
    survey = conn.execute('SELECT share_uuid, user_id FROM surveys WHERE id = ?', (survey_id,)).fetchone()
    if not survey or survey['user_id'] != current_user.id:
        conn.close()
        return jsonify({'error': 'Survey not found or permission denied'}), 403

    data = request.json
    recipients_str = data.get('recipients', '')
    # Basic validation and cleaning of phone numbers
    recipients = [num.strip() for num in recipients_str.split(',') if num.strip()]

    if not recipients:
        return jsonify({'error': 'No recipients provided'}), 400

    survey_url = url_for('take_survey', survey_uuid=survey['share_uuid'], _external=True)
    success_count = 0
    failed_count = 0

    for number in recipients:
        success, message = send_sms_notification(number, survey_url)
        if success:
            success_count += 1
            status, error_msg = 'Sent', None
        else:
            failed_count += 1
            status, error_msg = 'Failed', message
        
        # Log the attempt
        conn.execute(
            'INSERT INTO delivery_log (survey_id, channel, recipient, status, error_message) VALUES (?, ?, ?, ?, ?)',
            (survey_id, 'SMS', number, status, error_msg)
        )
    
    conn.commit()
    conn.close()
    
    return jsonify({
        'message': f'Processing complete. Sent: {success_count}, Failed: {failed_count}.',
        'success_count': success_count,
        'failed_count': failed_count
    })

    # --- NEW: API Route for sending WhatsApp invitations ---
@app.route('/api/survey/<int:survey_id>/send/whatsapp', methods=['POST'])
@login_required
def send_whatsapp_invites(survey_id):
    conn = get_db_connection()
    survey = conn.execute('SELECT share_uuid, user_id FROM surveys WHERE id = ?', (survey_id,)).fetchone()
    if not survey or survey['user_id'] != current_user.id:
        conn.close()
        return jsonify({'error': 'Survey not found or permission denied'}), 403

    recipients = [num.strip() for num in request.json.get('recipients', '').split(',') if num.strip()]
    if not recipients:
        return jsonify({'error': 'No recipients provided'}), 400

    survey_url = url_for('take_survey', survey_uuid=survey['share_uuid'], _external=True)
    success_count, failed_count = 0, 0

    for number in recipients:
        success, message = send_whatsapp_notification(number, survey_url)
        status, error_msg = ('Sent', None) if success else ('Failed', message)
        if success: success_count += 1
        else: failed_count += 1
        
        # Log the attempt to the database
        conn.execute(
            'INSERT INTO delivery_log (survey_id, channel, recipient, status, error_message) VALUES (?, ?, ?, ?, ?)',
            (survey_id, 'WhatsApp', number, status, error_msg)
        )
    
    conn.commit()
    conn.close()
    
    return jsonify({'message': f'WhatsApp processing complete. Sent: {success_count}, Failed: {failed_count}.'})

# --- Database Functions ---
def get_db_connection():
    conn = sqlite3.connect("survey_data.db")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # Create tables if they don't exist (original logic)
    c.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS surveys (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, title TEXT NOT NULL, share_uuid TEXT NOT NULL UNIQUE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, is_active INTEGER DEFAULT 0, password_hash TEXT, theme_color TEXT, logo_url TEXT, FOREIGN KEY (user_id) REFERENCES users(id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS questions (id INTEGER PRIMARY KEY, survey_id INTEGER NOT NULL, position INTEGER NOT NULL, text TEXT NOT NULL, type TEXT NOT NULL, options TEXT, logic_rules TEXT, correct_answer TEXT, FOREIGN KEY (survey_id) REFERENCES surveys(id) ON DELETE CASCADE)''')
    c.execute('''CREATE TABLE IF NOT EXISTS respondent_sessions (id INTEGER PRIMARY KEY, survey_id INTEGER NOT NULL, start_time TIMESTAMP, end_time TIMESTAMP, ip_address TEXT, user_agent TEXT, geo_location TEXT, access_mode TEXT, quality_flags TEXT, FOREIGN KEY (survey_id) REFERENCES surveys(id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS responses (id INTEGER PRIMARY KEY, respondent_session_id INTEGER NOT NULL, question_id INTEGER NOT NULL, answer TEXT NOT NULL, FOREIGN KEY (respondent_session_id) REFERENCES respondent_sessions(id) ON DELETE CASCADE)''')
    c.execute('''CREATE TABLE IF NOT EXISTS delivery_log (id INTEGER PRIMARY KEY, survey_id INTEGER, channel TEXT, recipient TEXT, status TEXT, sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, error_message TEXT, FOREIGN KEY (survey_id) REFERENCES surveys(id))''')

    # --- NEW: Logic to add missing columns to existing tables ---
    # This is a simple form of database migration.
    
    # Check for submission_type in the surveys table
    c.execute("PRAGMA table_info(surveys)")
    columns = [row['name'] for row in c.fetchall()]
    
    if 'submission_type' not in columns:
        print("DATABASE UPDATE: Adding 'submission_type' column to 'surveys' table...")
        # Add the column with a default value for all existing surveys
        c.execute("ALTER TABLE surveys ADD COLUMN submission_type TEXT DEFAULT 'DIGITAL'")
        print("Update complete.")
    if 'default_lang' not in columns:
        print("DATABASE UPDATE: Adding 'default_lang' column to 'surveys' table...")
        c.execute("ALTER TABLE surveys ADD COLUMN default_lang TEXT DEFAULT 'en'")
        print("Update complete.")
    conn.commit()
    conn.close()

# Make sure this line is outside the function, at the bottom of your script before the 'if __name__ ...' block
init_db()


# --- HTML TEMPLATES ---
LOGIN_TEMPLATE = '''
<!DOCTYPE html><html lang="en"><head><title>Login - AI Survey Tool</title><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet"><style>:root{--bg-dark:#111827;--bg-card:#1F2937;--primary:#3B82F6;--primary-hover:#2563EB;--text-light:#F9FAFB;--text-dark:#9CA3AF;--border-color:#374151;--danger:#EF4444}body{font-family:'Inter',sans-serif;background-color:var(--bg-dark);color:var(--text-light);display:flex;align-items:center;justify-content:center;height:100vh;margin:0}.container{background:var(--bg-card);padding:40px;border-radius:12px;box-shadow:0 10px 25px rgba(0,0,0,0.3);width:100%;max-width:400px;text-align:center;border:1px solid var(--border-color)}h1{color:var(--text-light);margin-bottom:10px;font-size:1.8rem}p.subtitle{color:var(--text-dark);margin-top:0;margin-bottom:30px}input{width:100%;padding:12px;margin-bottom:15px;border:1px solid var(--border-color);background-color:#374151;color:var(--text-light);border-radius:8px;box-sizing:border-box;font-size:16px}input:focus{outline:none;border-color:var(--primary)}button{width:100%;padding:12px;background:var(--primary);color:white;border:none;border-radius:8px;cursor:pointer;font-size:16px;font-weight:600;transition:background-color 0.2s ease}button:hover{background:var(--primary-hover)}.flash{padding:15px;margin-bottom:20px;border:1px solid var(--danger);border-radius:8px;color:var(--text-light);background-color:rgba(239,68,68,0.2)}p{margin-top:20px;color:var(--text-dark)}a{color:var(--primary);text-decoration:none;font-weight:600}a:hover{text-decoration:underline}</style></head><body><div class="container"><h1>Welcome Back</h1><p class="subtitle">Log in to manage your surveys</p>{% with messages = get_flashed_messages() %}{% if messages %}<div class="flash">{{ messages[0] }}</div>{% endif %}{% endwith %}<form method="POST" action="/login"><input type="text" name="username" placeholder="Username" required><input type="password" name="password" placeholder="Password" required><button type="submit">Login</button></form><p>Don't have an account? <a href="{{ url_for('register') }}">Register here</a></p></div></body></html>
'''
REGISTER_TEMPLATE = '''
<!DOCTYPE html><html lang="en"><head><title>Register - AI Survey Tool</title><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet"><style>:root{--bg-dark:#111827;--bg-card:#1F2937;--primary:#3B82F6;--primary-hover:#2563EB;--text-light:#F9FAFB;--text-dark:#9CA3AF;--border-color:#374151;--danger:#EF4444}body{font-family:'Inter',sans-serif;background-color:var(--bg-dark);color:var(--text-light);display:flex;align-items:center;justify-content:center;height:100vh;margin:0}.container{background:var(--bg-card);padding:40px;border-radius:12px;box-shadow:0 10px 25px rgba(0,0,0,0.3);width:100%;max-width:400px;text-align:center;border:1px solid var(--border-color)}h1{color:var(--text-light);margin-bottom:10px;font-size:1.8rem}p.subtitle{color:var(--text-dark);margin-top:0;margin-bottom:30px}input{width:100%;padding:12px;margin-bottom:15px;border:1px solid var(--border-color);background-color:#374151;color:var(--text-light);border-radius:8px;box-sizing:border-box;font-size:16px}input:focus{outline:none;border-color:var(--primary)}button{width:100%;padding:12px;background:var(--primary);color:white;border:none;border-radius:8px;cursor:pointer;font-size:16px;font-weight:600;transition:background-color 0.2s ease}button:hover{background:var(--primary-hover)}.flash{padding:15px;margin-bottom:20px;border:1px solid var(--danger);border-radius:8px;color:var(--text-light);background-color:rgba(239,68,68,0.2)}p{margin-top:20px;color:var(--text-dark)}a{color:var(--primary);text-decoration:none;font-weight:600}a:hover{text-decoration:underline}</style></head><body><div class="container"><h1>Create Account</h1><p class="subtitle">Get started with your own AI-powered surveys</p>{% with messages = get_flashed_messages() %}{% if messages %}<div class="flash">{{ messages[0] }}</div>{% endif %}{% endwith %}<form method="POST" action="/register"><input type="text" name="username" placeholder="Username" required><input type="password" name="password" placeholder="Password" required><button type="submit">Register</button></form><p>Already have an account? <a href="{{ url_for('login') }}">Login here</a></p></div></body></html>
'''
EDIT_SURVEY_TEMPLATE = '''
{% block content %}
<script src="https://cdn.jsdelivr.net/npm/sortablejs@latest/Sortable.min.js"></script>

<style>
    .question-card-container { display: flex; align-items: center; gap: 15px; }
    .drag-handle { cursor: grab; color: var(--text-dark); }
    .question-info { flex-grow: 1; }
    .action-toolbar { display: flex; align-items: center; }
    .icon-btn { background: none; border: none; color: var(--text-dark); cursor: pointer; padding: 8px; border-radius: 50%; display: flex; }
    .icon-btn:hover { background-color: var(--bg-light); color: var(--text-light); }
    .icon-btn.delete:hover { background-color: var(--danger); color: white; }
    .question-editor-form { display: none; margin-top: 20px; padding-top: 20px; border-top: 1px solid var(--border-color); }
    .options-editor { font-family: monospace; font-size: 0.9em; background-color: #111827; }
    .modal-backdrop { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); z-index: 100; display: none; }
    .modal-content { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: var(--bg-card); padding: 30px; border-radius: 12px; z-index: 101; width: 90%; max-width: 600px; }
    .logic-rule-builder { display: grid; grid-template-columns: auto 1fr auto 1fr auto 1fr; gap: 10px; align-items: center; background: var(--bg-dark); padding: 15px; border-radius: 8px; margin-bottom: 10px; }
</style>

<div class="page-header" style="display:flex; justify-content: space-between; align-items: center;">
    <div>
        <h2 style="margin-bottom: 5px;">Survey Editor</h2>
        <p style="margin:0; color:var(--text-dark);">{{ survey.title }}</p>
    </div>
    <div>
        <button id="add-question-btn">+ Add Question</button>
        <a href="{{ url_for('manage_survey_page', survey_id=survey.id) }}">
            <button class="btn-secondary">Manage & Share</button>
        </a>
    </div>
</div>

<div id="questions-list">
    {% for q in questions %}
    <div class="card question-card" data-question-id="{{ q.id }}" data-question-type="{{ q.type }}" data-logic-rules='{{ q.logic_rules or "[]" }}'>
        <div class="question-card-container">
            <div class="drag-handle" title="Drag to reorder">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"></polyline><polyline points="9 18 3 12 9 6"></polyline></svg>
            </div>
            <div class="question-info">
                <p style="font-size: 0.9em; color: var(--text-dark); margin:0;">{{ loop.index }}. [{{ q.type }}]</p>
                <p class="question-text" style="font-size: 1.2rem; color:var(--text-light); margin: 5px 0 0 0;">{{ q.text }}</p>
            </div>
            <div class="action-toolbar">
                {% if q.type in ['MCQ', 'DROPDOWN', 'LIKERT'] %}
                    <button class="icon-btn btn-logic" title="Logic"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.9 4.1L22 8.2l-4.1 4.1"/><path d="M2 12.2h20"/><path d="M6.1 20.1L2 16l4.1-4.1"/><path d="M22 12.2H2"/></svg></button>
                {% endif %}
                <button class="icon-btn btn-edit" title="Edit"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg></button>
                <button class="icon-btn btn-delete delete" title="Delete"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg></button>
            </div>
        </div>
        <div class="question-editor-form">
            <label>Question Text</label>
            <input type="text" class="edit-q-text" value="{{ q.text }}">
            <div style="display:grid; grid-template-columns: 1fr 2fr; gap: 20px;">
                <div>
                    <label>Question Type</label>
                    <select class="edit-q-type">
                        <option value="OPEN" {{ 'selected' if q.type == 'OPEN' else '' }}>Open Text</option>
                        <option value="MCQ" {{ 'selected' if q.type == 'MCQ' else '' }}>Multiple Choice</option>
                        <option value="CHECKBOX" {{ 'selected' if q.type == 'CHECKBOX' else '' }}>Checkbox</option>
                        <option value="DROPDOWN" {{ 'selected' if q.type == 'DROPDOWN' else '' }}>Dropdown</option>
                        <option value="STARS" {{ 'selected' if q.type == 'STARS' else '' }}>Star Rating</option>
                        <option value="DATE" {{ 'selected' if q.type == 'DATE' else '' }}>Date Picker</option>
                        <option value="LIKERT" {{ 'selected' if q.type == 'LIKERT' else '' }}>Likert Scale</option>
                    </select>
                </div>
                <div>
                    <label>Options (one per line)</label>
                    <textarea class="edit-q-options options-editor" rows="4">{{ q.options.replace(',', '\n') if q.options else '' }}</textarea>
                </div>
            </div>
            
            <div style="margin-top: 15px;">
                <label>Correct Answer (Optional, for Quizzes)</label>
                <p style="font-size: 0.8em; color: var(--text-dark); margin: -5px 0 10px 0;">For MCQ/Dropdown, this must exactly match one of the options above.</p>
                <input type="text" class="edit-q-correct-answer" value="{{ q.correct_answer or '' }}" placeholder="Enter the exact correct option text">
            </div>
            <button class="btn-save">Save Changes</button> <button class="btn-cancel-edit btn-secondary">Cancel</button>
        </div>
    </div>
    {% endfor %}
</div>

<div class="modal-backdrop" id="logic-modal-backdrop">
    <div class="modal-content">
        <h3 id="logic-modal-title">Conditional Logic</h3>
        <p>Create rules to show other questions based on the answer to this one.</p>
        <div id="logic-rules-container"></div>
        <div style="display:flex; gap: 10px; margin-top:20px;">
            <button id="logic-modal-save" class="btn-save">Save Logic</button>
            <button id="logic-modal-close" class="btn-secondary">Close</button>
        </div>
    </div>
</div>

<script>
    document.addEventListener('DOMContentLoaded', () => {
        const questionsList = document.getElementById('questions-list');
        const surveyId = {{ survey.id }};
        if (!questionsList) return;

        Sortable.create(questionsList, {
            animation: 150,
            handle: '.drag-handle',
            onEnd: async (evt) => {
                const newOrder = Array.from(questionsList.querySelectorAll('.question-card')).map(q => q.dataset.questionId);
                await fetch(`/api/survey/${surveyId}/reorder`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ order: newOrder })
                });
            }
        });

        const logicModalBackdrop = document.getElementById('logic-modal-backdrop');
        const logicModalTitle = document.getElementById('logic-modal-title');
        const logicRulesContainer = document.getElementById('logic-rules-container');
        let activeQuestionCardForLogic = null;

        function openLogicModal(questionCard) {
            activeQuestionCardForLogic = questionCard;
            const questionText = questionCard.querySelector('.question-text').textContent;
            logicModalTitle.textContent = `Logic for: "${questionText.trim()}"`;
            const existingRules = JSON.parse(questionCard.dataset.logicRules);
            renderRuleBuilders(existingRules);
            logicModalBackdrop.style.display = 'block';
        }

        function renderRuleBuilders(rules) {
            logicRulesContainer.innerHTML = '';
            const rule = rules[0] || {};
            const allQuestions = Array.from(document.querySelectorAll('.question-card')).map((card, index) => ({
                id: card.dataset.questionId,
                text: `${index + 1}. ${card.querySelector('.question-text').textContent.trim()}`
            }));
            let targetQuestionOptions = allQuestions
                .filter(q => q.id !== activeQuestionCardForLogic.dataset.questionId)
                .map(q => `<option value="${q.id}" ${rule.target_question_id == q.id ? 'selected' : ''}>${q.text}</option>`)
                .join('');
            const builderHTML = `
                <div class="logic-rule-builder">
                    <span>If answer</span>
                    <select class="logic-condition"><option value="is">is</option></select>
                    <span>to this question</span>
                    <input type="text" class="logic-value" placeholder="e.g., Yes" value="${rule.value || ''}">
                    <span>then show</span>
                    <select class="logic-target"><option value="">-- select question --</option>${targetQuestionOptions}</select>
                </div>
            `;
            logicRulesContainer.innerHTML = builderHTML;
        }

        async function saveLogic() {
            if (!activeQuestionCardForLogic) return;
            const questionId = activeQuestionCardForLogic.dataset.questionId;
            const value = logicRulesContainer.querySelector('.logic-value').value;
            const targetId = logicRulesContainer.querySelector('.logic-target').value;
            const newRules = (value && targetId) ? [{ condition: 'is', value: value, target_question_id: parseInt(targetId) }] : [];
            const response = await fetch(`/api/question/${questionId}/logic`, {
                method: 'PUT',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ rules: newRules })
            });
            if (response.ok) {
                activeQuestionCardForLogic.dataset.logicRules = JSON.stringify(newRules);
                logicModalBackdrop.style.display = 'none';
            } else {
                alert('Failed to save logic.');
            }
        }

        document.body.addEventListener('click', async (e) => {
            const addBtn = e.target.closest('#add-question-btn');
            const questionCard = e.target.closest('.question-card');
            
            if (addBtn) {
                const response = await fetch(`/api/survey/${surveyId}/questions`, { method: 'POST' });
                if (response.ok) { window.location.reload(); }
            }

            if (!questionCard) return;

            if (e.target.closest('.btn-edit')) {
                questionCard.querySelector('.question-editor-form').style.display = 'block';
            }
            if (e.target.closest('.btn-cancel-edit')) {
                questionCard.querySelector('.question-editor-form').style.display = 'none';
            }
            if (e.target.closest('.btn-logic')) {
                openLogicModal(questionCard);
            }
            if (e.target.closest('.btn-delete')) {
                if (confirm('Are you sure you want to delete this question?')) {
                    const questionId = questionCard.dataset.questionId;
                    const response = await fetch(`/api/question/${questionId}`, { method: 'DELETE' });
                    if (response.ok) { questionCard.remove(); }
                }
            }
            if (e.target.closest('.btn-save')) {
                const questionId = questionCard.dataset.questionId;
                const editorForm = questionCard.querySelector('.question-editor-form');
                const payload = {
                text: editorForm.querySelector('.edit-q-text').value,
                type: editorForm.querySelector('.edit-q-type').value,
                options: editorForm.querySelector('.edit-q-options').value.split('\\n').join(','), // Use a double backslash here
                correct_answer: editorForm.querySelector('.edit-q-correct-answer').value.trim() // # NEW: Send correct answer
            };
                const response = await fetch(`/api/question/${questionId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (response.ok) { window.location.reload(); }
                else { alert('Failed to save changes.'); }
            }
        });

        if (logicModalBackdrop) {
            document.getElementById('logic-modal-close').addEventListener('click', () => logicModalBackdrop.style.display = 'none');
            document.getElementById('logic-modal-save').addEventListener('click', saveLogic);
        }
    });
</script>
{% endblock %}
'''
SHARED_APP_TEMPLATE = '''
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>{{ page_title }} - AI Survey Tool</title><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"><script src="https://cdn.jsdelivr.net/npm/chart.js"></script><style>:root{--bg-dark:#111827;--bg-card:#1F2937;--bg-light:#374151;--primary:#3B82F6;--primary-hover:#2563EB;--secondary:#10B981;--text-light:#F9FAFB;--text-dark:#9CA3AF;--text-headings:#ffffff;--border-color:#374151;--shadow-light:rgba(0,0,0,0.3);--sentiment-positive:rgba(16,185,129,0.1);--sentiment-negative:rgba(239,68,68,0.1);--sentiment-pos-text:#10B981;--sentiment-neg-text:#EF4444}*{box-sizing:border-box}body{font-family:'Inter',sans-serif;margin:0;background-color:var(--bg-dark);color:var(--text-light);display:flex;min-height:100vh}.sidebar{width:260px;background-color:var(--bg-card);padding:25px;display:flex;flex-direction:column;border-right:1px solid var(--border-color);position:fixed;height:100%}.sidebar-header{margin-bottom:40px}.sidebar-header h1{font-size:1.5rem;color:var(--text-headings);margin:0;display:flex;align-items:center;gap:10px}.sidebar-header span{font-size:2rem}.sidebar nav{flex-grow:1}.sidebar nav a{display:flex;align-items:center;gap:12px;padding:12px;margin-bottom:8px;text-decoration:none;color:var(--text-dark);border-radius:8px;transition:all .2s ease;font-weight:500}.sidebar nav a:hover{background-color:var(--bg-light);color:var(--text-light)}.sidebar nav a.active{background-color:var(--primary);color:var(--text-light);font-weight:600}.sidebar nav a svg{width:20px;height:20px}.user-profile{margin-top:auto;padding-top:20px;border-top:1px solid var(--border-color)}.user-profile-info{display:flex;align-items:center;gap:10px}.user-profile-avatar{width:40px;height:40px;border-radius:50%;background-color:var(--primary);display:flex;align-items:center;justify-content:center;font-weight:600;color:var(--text-light)}.user-profile-name{font-weight:600;color:var(--text-light)}.user-profile-logout{font-size:.9em;color:var(--text-dark);text-decoration:none}.user-profile-logout:hover{color:var(--primary)}.main-content{margin-left:260px;flex-grow:1;padding:40px}.page-header{margin-bottom:30px}.page-header h2{font-size:2rem;margin:0;color:var(--text-headings)}.card{background-color:var(--bg-card);border-radius:12px;padding:25px;margin-bottom:30px;border:1px solid var(--border-color)}label,.question-text{display:block;font-weight:500;margin-bottom:10px;color:var(--text-dark)}input[type=text],input[type=number],input[type=password],select,textarea{width:100%;padding:12px;margin-bottom:15px;border:1px solid var(--border-color);background-color:var(--bg-light);color:var(--text-light);border-radius:8px;font-size:16px;transition:all .2s ease}input:focus,select:focus,textarea:focus{outline:none;border-color:var(--primary);background-color:#1F2937}button{padding:12px 24px;background:var(--primary);color:#fff;border:none;border-radius:8px;cursor:pointer;font-size:16px;font-weight:600;transition:background-color .2s ease}button:hover{background:var(--primary-hover)}button.btn-secondary{background-color:var(--bg-light)}button.btn-secondary:hover{background-color:#4B5563}.loader{border:5px solid var(--bg-light);border-top:5px solid var(--primary);border-radius:50%;width:40px;height:40px;animation:spin 1s linear infinite;margin:20px auto;display:none}@keyframes spin{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}.mcq-option{display:block;margin-bottom:10px;background-color:var(--bg-light);padding:12px;border-radius:8px;cursor:pointer;transition:all .2s ease}.mcq-option:hover{background-color:#4B5563}.mcq-option input{margin-right:10px}</style></head><body><aside class="sidebar"><div class="sidebar-header"><h1><span>🧠</span> AI Surveys</h1></div><nav><a href="{{ url_for('dashboard') }}" class="{{ 'active' if 'dashboard' in request.path else '' }}"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M3.75 6A2.25 2.25 0 016 3.75h2.25A2.25 2.25 0 0110.5 6v2.25a2.25 2.25 0 01-2.25 2.25H6a2.25 2.25 0 01-2.25-2.25V6zM3.75 15.75A2.25 2.25 0 016 13.5h2.25a2.25 2.25 0 012.25 2.25V18a2.25 2.25 0 01-2.25 2.25H6A2.25 2.25 0 013.75 18v-2.25zM13.5 6a2.25 2.25 0 012.25-2.25H18A2.25 2.25 0 0120.25 6v2.25A2.25 2.25 0 0118 10.5h-2.25a2.25 2.25 0 01-2.25-2.25V6zM13.5 15.75a2.25 2.25 0 012.25-2.25H18a2.25 2.25 0 012.25 2.25V18A2.25 2.25 0 0118 20.25h-2.25A2.25 2.25 0 0113.5 18v-2.25z" /></svg> Dashboard</a><a href="{{ url_for('create_survey_page') }}" class="{{ 'active' if 'create' in request.path else '' }}"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v6m3-3H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z" /></svg> Create Survey</a><a href="{{ url_for('results_dashboard') }}" class="{{ 'active' if 'results' in request.path else '' }}"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" /></svg> View Results</a></nav><div class="user-profile"><div class="user-profile-info"><div class="user-profile-avatar">{{ current_user.username[0]|upper }}</div><div><div class="user-profile-name">{{ current_user.username }}</div><a href="/logout" class="user-profile-logout">Logout</a></div></div></div></aside><main class="main-content">{{ content|safe }}</main></body></html>
'''
DASHBOARD_TEMPLATE = '''
{% block content %}
<div class="page-header" style="display:flex; justify-content: space-between; align-items: center;">
    <h2>My Surveys Dashboard</h2>
    <a href="{{ url_for('create_survey_page') }}"><button>+ New Survey</button></a>
</div>
{% with messages = get_flashed_messages(with_categories=true) %}
    {% if messages %}
        {% for category, message in messages %}
            <div class="card" style="border-left: 5px solid var(--{{ 'secondary' if category == 'success' else 'danger' }}); padding: 15px;">{{ message }}</div>
        {% endfor %}
    {% endif %}
{% endwith %}
{% if surveys %}
    {% for survey in surveys %}
    <div class="card">
        <div style="display:flex; justify-content: space-between; align-items: start;">
            <div>
                <h3 style="margin:0 0 10px 0;">{{ survey.title }}</h3>
                <p style="margin:0; font-size: 0.9em; color: var(--text-dark);">Created: {{ survey.created_at.strftime('%d %b %Y, %I:%M %p') }}</p>
                <p style="margin:5px 0 0 0; font-size: 0.9em;">Status: 
                    <span style="font-weight: bold; color: {{ 'var(--secondary)' if survey.is_active else 'var(--danger)' }};">
                        {{ 'Active / Collecting Responses' if survey.is_active else 'Draft / Inactive' }}
                    </span>
                </p>
            </div>
            <div>
                <a href="{{ url_for('edit_survey_page', survey_id=survey.id) }}"><button class="btn-secondary">Edit Survey</button></a>
                <a href="{{ url_for('manage_survey_page', survey_id=survey.id) }}"><button>Manage & Share</button></a>
                <a href="{{ url_for('results_dashboard', survey_id=survey.id) }}"><button>View Results</button></a>
            </div>
        </div>
    </div>
    {% endfor %}
{% else %}
    <div class="card" style="text-align:center; padding: 50px;">
        <h3>You haven't created any surveys yet.</h3>
        <p>Click the button below to create your first AI-powered survey!</p>
        <a href="{{ url_for('create_survey_page') }}"><button>+ New Survey</button></a>
    </div>
{% endif %}
{% endblock %}
'''
CREATE_SURVEY_TEMPLATE = '''
{% block content %}
    <style>
        /* Style for the new topic tags */
        .topic-tags-container {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 25px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
        }
        .topic-tag {
            background-color: var(--bg-light);
            padding: 6px 12px;
            border-radius: 15px;
            font-size: 0.9em;
            cursor: pointer;
            transition: all 0.2s ease;
            border: 2px solid transparent;
        }
        .topic-tag:hover { background-color: #4B5563; }
        .topic-tag:active { cursor: grabbing; background-color: var(--primary); }
        
        /* Style for the drop target when dragging over it */
        #prompt.drag-over {
            border-color: var(--primary);
            box-shadow: 0 0 15px rgba(59, 130, 246, 0.4);
        }
    </style>

    <div class="page-header"><h2>Create a New Survey</h2></div>
    <div class="card">
        <form id="survey-form">
            <div style="display: grid; grid-template-columns: 1fr 150px 180px; gap: 20px; align-items: flex-end;">
                <div style="flex-grow: 1; position: relative;">
                    <label for="prompt">What is your survey about?</label>
                    <input type="text" id="prompt" name="prompt" placeholder="Click or drag a topic below, or type your own" required>
                    <button type="button" id="mic-btn-prompt" class="mic-btn" data-target="prompt" style="position: absolute; right: 10px; top: 38px; background: none; border: none; padding: 5px; cursor: pointer; color: var(--text-dark);">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line></svg>
                    </button>
                </div>
                <div>
                    <label for="num_questions"># of Questions</label>
                    <input type="number" id="num_questions" name="num_questions" value="3" min="1" max="10">
                </div>
                <div>
                    <label for="lang">Default Language</label>
                    <select id="lang" name="lang">
                        {% for code, name in languages %}
                            <option value="{{ code }}">{{ name }}</option>
                        {% endfor %}
                    </select>
                </div>
            </div>

            <div class="topic-tags-container">
                <span style="color: var(--text-dark);">e.g.,</span>
                {% for tag in topic_tags %}
                    <div class="topic-tag" draggable="true">{{ tag }}</div>
                {% endfor %}
            </div>

            <div style="display: flex; margin-top: 25px; gap: 10px;">
                <button type="submit">✨ Generate & Edit Survey</button>
            </div>
        </form>
    </div>

    <div id="loader" class="loader"></div>

<script>
document.addEventListener('DOMContentLoaded', () => {
    const surveyForm = document.getElementById('survey-form');
    const loader = document.getElementById('loader');
    
    // --- Click and Drag-and-Drop Logic ---
    const promptInput = document.getElementById('prompt');
    const topicTags = document.querySelectorAll('.topic-tag');

    topicTags.forEach(tag => {
        tag.addEventListener('click', () => {
            promptInput.value = tag.textContent;
        });

        tag.addEventListener('dragstart', (e) => {
            e.dataTransfer.setData('text/plain', tag.textContent);
        });
    });

    promptInput.addEventListener('dragover', (e) => {
        e.preventDefault();
        promptInput.classList.add('drag-over');
    });

    promptInput.addEventListener('dragleave', () => {
        promptInput.classList.remove('drag-over');
    });

    promptInput.addEventListener('drop', (e) => {
        e.preventDefault();
        promptInput.classList.remove('drag-over');
        const draggedText = e.dataTransfer.getData('text/plain');
        promptInput.value = draggedText;
    });

    // --- Voice-to-Text Logic (New) ---
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const micBtn = document.getElementById('mic-btn-prompt');

    if (SpeechRecognition && micBtn) {
        let recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        let isListening = false;

        micBtn.addEventListener('click', () => {
            const targetInput = document.getElementById(micBtn.dataset.target);
            if (!targetInput) return;

            if (isListening) {
                recognition.stop();
                return;
            }
            
            targetInput.value = '';
            recognition.start();

            recognition.onstart = () => {
                isListening = true;
                micBtn.style.color = 'var(--primary)';
                targetInput.placeholder = "Listening...";
            };

            recognition.onend = () => {
                isListening = false;
                micBtn.style.color = 'var(--text-dark)';
                targetInput.placeholder = "Click or drag a topic below, or type your own";
            };

            recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                targetInput.value = transcript;
            };
            
            recognition.onerror = (event) => {
                console.error("Speech recognition error:", event.error);
                isListening = false;
                micBtn.style.color = 'var(--text-dark)';
                targetInput.placeholder = "Voice input failed. Please type.";
            };
        });
    } else {
        if (micBtn) micBtn.style.display = 'none';
    }
    
    // --- Original Form Submission Logic ---
    surveyForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        loader.style.display = 'block';
        const formData = new FormData(surveyForm);
        const data = Object.fromEntries(formData.entries());
        const response = await fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const result = await response.json();
        if (result.redirect_url) {
            window.location.href = result.redirect_url;
        } else {
            loader.style.display = 'none';
            alert('An error occurred while creating the survey.');
        }
    });
});
</script>
{% endblock %}
'''
MANAGE_SURVEY_TEMPLATE = '''
{% block content %}
<div class="page-header" style="display:flex; justify-content: space-between; align-items: center;">
    <h2>Manage Survey: "{{ survey.title }}"</h2>
    <a href="{{ url_for('edit_survey_page', survey_id=survey.id) }}"><button class="btn-secondary">Back to Editor</button>
</div>

{% with messages = get_flashed_messages(with_categories=true) %}
    {% if messages %}
        {% for category, message in messages %}
            <div class="card" style="border-left: 5px solid var(--{{ 'secondary' if category == 'success' else 'danger' }}); padding: 15px;">{{ message }}</div>
        {% endfor %}
    {% endif %}
{% endwith %}

<div class="card">
    <h3>Distribution Channels</h3>
    <p>Use these methods to share your survey and collect responses.</p>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 30px; margin-top: 20px;">

        <div>
            <h4>Web Link</h4>
            <p style="font-size:0.9em; color:var(--text-dark); min-height: 40px;">The most direct way to share. Copy the link and send it anywhere.</p>
            <div style="display:flex; gap:10px; align-items:center; background-color: var(--bg-dark); padding:10px; border-radius:8px;">
                <input type="text" id="share-link" value="{{ share_link }}" readonly style="margin:0;">
                <button id="copy-btn" class="btn-secondary">Copy</button>
            </div>
        </div>

        <div>
            <h4>QR Code</h4>
            <p style="font-size:0.9em; color:var(--text-dark); min-height: 40px;">Perfect for posters, flyers, and in-person events.</p>
            <div style="display: flex; gap: 20px; align-items: center;">
                <img src="{{ url_for('generate_qr_code', survey_uuid=survey.share_uuid) }}" alt="Survey QR Code" style="width:100px; height:100px; border-radius:8px; background:white; padding:5px;">
                <a href="{{ url_for('generate_qr_code', survey_uuid=survey.share_uuid) }}" download="survey_{{ survey.id }}_qr.png">
                    <button class="btn-secondary" style="width:100%;">Download</button>
                </a>
            </div>
        </div>
    </div>
</div>

<div class="card">
    <h3>Advanced Distribution</h3>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 30px; margin-top: 20px;">
        
        <div>
            <h4>Send via SMS</h4>
            <p style="font-size:0.9em; color:var(--text-dark);">Enter phone numbers (with country code, comma-separated) to send an invitation link. (Requires Twilio setup)</p>
            <form id="sms-form" style="margin-top:10px;">
                <textarea id="sms-recipients" placeholder="+919876543210, +14155552671" rows="3"></textarea>
                <button type="submit" style="margin-top: 10px;">Send SMS Invites</button>
                <div id="sms-status" style="font-size: 0.9em; margin-top: 10px; color: var(--text-dark);"></div>
            </form>
        </div>

        <div>
            <h4>Send via WhatsApp</h4>
            <p style="font-size:0.9em; color:var(--text-dark);">Enter WhatsApp numbers (with country code, e.g., +91) to send an invite. (Recipients must join your Twilio Sandbox first)</p>
            <form id="whatsapp-form" style="margin-top:10px;">
                <textarea id="whatsapp-recipients" placeholder="+919876543210, +14155552671" rows="3"></textarea>
                <button type="submit" style="margin-top: 10px;">Send WhatsApp Invites</button>
                <div id="whatsapp-status" style="font-size: 0.9em; margin-top: 10px; color: var(--text-dark);"></div>
            </form>
        </div>

    </div>
</div>

<div class="card">
    <h3>Appearance</h3>
    <p style="font-size:0.9em; color:var(--text-dark); margin-top: 0;">Customize the look of your public survey page.</p>
    <form method="POST" action="{{ url_for('set_survey_branding', survey_id=survey.id) }}" enctype="multipart/form-data" style="margin-top:20px;">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; align-items: start;">
            <div>
                <label for="theme_color">Theme Color</label>
                <input type="color" id="theme_color" name="theme_color" value="{{ survey.theme_color or '#3B82F6' }}" style="padding: 2px; height: 45px; width: 100%;">
            </div>
        </div>
        <button type="submit" style="margin-top: 25px;">Save Branding</button>
    </form>
</div>

<div class="card">
    <h3>Collector Settings</h3>
    <p style="font-size:0.9em; color:var(--text-dark); margin-top: 0;">Control how and when you collect responses.</p>
    <div style="margin-top:20px; display: grid; grid-template-columns: 1fr 1fr; gap: 30px; align-items: start;">
        
       <div>
    <label>Survey Status</label>
    <div style="display: flex; align-items: center; gap: 15px; margin-top: 10px; padding-bottom: 30px; border-bottom: 1px solid var(--border-color); margin-bottom: 20px;">
        {% if survey.is_active %}
            <span style="color: var(--secondary); font-weight: bold;">Active</span>
            <form action="{{ url_for('toggle_survey_status', survey_id=survey.id) }}" method="POST" style="margin:0;">
                <button type="submit" class="btn-secondary">Deactivate</button>
            </form>
        {% else %}
            <span style="color: var(--danger); font-weight: bold;">Inactive / Draft</span>
            <form action="{{ url_for('toggle_survey_status', survey_id=survey.id) }}" method="POST" style="margin:0;">
                <button type="submit" class="btn-secondary">Activate</button>
            </form>
        {% endif %}
    </div>

    <label>Submission Mode</label>
    <p style="font-size:0.9em; color:var(--text-dark); margin: -5px 0 10px 0;">Choose how respondents will submit their answers.</p>
    <form action="{{ url_for('set_submission_type', survey_id=survey.id) }}" method="POST" style="display:flex; align-items:center;">
        <select name="submission_type" style="margin:0; width: auto; flex-grow: 1;">
            <option value="DIGITAL" {{ 'selected' if survey.submission_type == 'DIGITAL' or not survey.submission_type else '' }}>Digital Form (Default)</option>
            <option value="PAPER" {{ 'selected' if survey.submission_type == 'PAPER' else '' }}>Paper Form Upload (OCR)</option>
        </select>
        <button type="submit" style="margin-left: 10px;" class="btn-secondary">Save Mode</button>
    </form>
</div>

        <div>
            <label>Password Protection</label>
            {% if survey.password_hash %}
                <p style="font-size: 0.9em; color: var(--secondary); margin: 10px 0;">This survey is currently password protected.</p>
                <form action="{{ url_for('set_survey_password', survey_id=survey.id, remove=1) }}" method="POST" style="margin:0;">
                    <button type="submit" class="btn-secondary">Remove Password</button>
                </form>
            {% else %}
                 <form action="{{ url_for('set_survey_password', survey_id=survey.id) }}" method="POST" style="margin-top:10px; display: flex; gap: 10px;">
                    <input type="password" name="password" placeholder="Enter new password" style="margin:0;" required>
                    <button type="submit">Set Password</button>
                </form>
            {% endif %}
        </div>

    </div>
</div>

<script>
// --- Copy Button for Sharable Link ---
document.getElementById('copy-btn').addEventListener('click', () => {
    const linkInput = document.getElementById('share-link');
    navigator.clipboard.writeText(linkInput.value).then(() => {
        const btn = document.getElementById('copy-btn');
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
    });
});

// --- NEW: SMS Form Submission Logic ---
document.getElementById('sms-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const recipients = document.getElementById('sms-recipients').value;
    const statusDiv = document.getElementById('sms-status');
    const sendButton = e.target.querySelector('button');
    
    if (!recipients) {
        statusDiv.textContent = 'Please enter at least one phone number.';
        statusDiv.style.color = 'var(--danger)';
        return;
    }

    statusDiv.textContent = 'Sending...';
    statusDiv.style.color = 'var(--text-dark)';
    sendButton.disabled = true;

    try {
        const response = await fetch(`/api/survey/{{ survey.id }}/send/sms`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ recipients: recipients })
        });

        const result = await response.json();

        if (response.ok) {
            statusDiv.textContent = result.message;
            statusDiv.style.color = 'var(--secondary)';
            document.getElementById('sms-recipients').value = ''; // Clear on success
        } else {
            statusDiv.textContent = 'Error: ' + (result.error || 'Failed to send.');
            statusDiv.style.color = 'var(--danger)';
        }
    } catch (error) {
        statusDiv.textContent = 'A network error occurred. Please try again.';
        statusDiv.style.color = 'var(--danger)';
    } finally {
        sendButton.disabled = false;
    }
});
// --- NEW: WhatsApp Form Submission Logic ---
document.getElementById('whatsapp-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const recipients = document.getElementById('whatsapp-recipients').value;
    const statusDiv = document.getElementById('whatsapp-status');
    const sendButton = e.target.querySelector('button');
    
    if (!recipients) {
        statusDiv.textContent = 'Please enter at least one phone number.';
        statusDiv.style.color = 'var(--danger)';
        return;
    }

    statusDiv.textContent = 'Sending...';
    statusDiv.style.color = 'var(--text-dark)';
    sendButton.disabled = true;

    try {
        const response = await fetch(`/api/survey/{{ survey.id }}/send/whatsapp`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ recipients: recipients })
        });
        const result = await response.json();
        if (response.ok) {
            statusDiv.textContent = result.message;
            statusDiv.style.color = 'var(--secondary)';
            document.getElementById('whatsapp-recipients').value = ''; // Clear on success
        } else {
            statusDiv.textContent = 'Error: ' + (result.error || 'Failed to send.');
            statusDiv.style.color = 'var(--danger)';
        }
    } catch (error) {
        statusDiv.textContent = 'A network error occurred. Please try again.';
        statusDiv.style.color = 'var(--danger)';
    } finally {
        sendButton.disabled = false;
    }
});
</script>
{% endblock %}
'''
SURVEY_TAKER_TEMPLATE = '''
<!DOCTYPE html><html lang="en"><head>
    <title>Survey: {{ survey.title }}</title><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root{--bg-dark:#111827;--bg-card:#1F2937;--primary:{{ survey.theme_color or '#3B82F6' }};--primary-hover:#2563EB;--text-light:#F9FAFB;--text-dark:#9CA3AF;--border-color:#374151;--secondary:#10B981}
        body{font-family:'Inter',sans-serif;background-color:var(--bg-dark);color:var(--text-light);display:flex;justify-content:center;padding:40px;margin:0}
        .container{width:100%;max-width:800px}
        .survey-logo { text-align: center; margin-bottom: 20px; } .survey-logo img { max-height: 70px; max-width: 200px; }
        h1{color:var(--text-light);margin-bottom:10px;font-size:2.2rem; text-align:center;}
        p.subtitle{color:var(--text-dark);margin:0 auto 30px auto;text-align:center;max-width:600px}
        input[type=text],input[type=password],textarea,select{width:100%;padding:12px;margin-bottom:15px;border:1px solid var(--border-color);background-color:#374151;color:var(--text-light);border-radius:8px;box-sizing:border-box;font-size:16px}
        input:focus,textarea:focus,select:focus{outline:none;border-color:var(--primary)}
        button{width:100%;padding:15px;background:var(--primary);color:#fff;border:none;border-radius:8px;cursor:pointer;font-size:18px;font-weight:600;transition:background-color .2s ease;margin-top:20px}
        button:hover{background:var(--primary-hover)}
        .card{background:var(--bg-card);padding:25px;border-radius:12px;box-shadow:0 10px 25px rgba(0,0,0,.3);border:1px solid var(--border-color);margin-bottom:20px}
        .question-text{font-size:1.2rem;font-weight:600;margin-bottom:20px}
        .mcq-option{display:block;margin-bottom:10px;background-color:#374151;padding:12px;border-radius:8px;cursor:pointer;transition:all .2s ease}
        .mcq-option:hover{background-color:#4B5563}.mcq-option input{margin-right:15px;transform:scale(1.2)}
        .thank-you h2{color:var(--secondary)}
        .star-rating{display:flex;flex-direction:row-reverse;justify-content:flex-end;gap:5px;}
        .star-rating input{display:none} .star-rating label{font-size:2rem;color:var(--bg-light);cursor:pointer;transition:color .2s}
        .star-rating input:checked~label,.star-rating label:hover,.star-rating label:hover~label{color:#F59E0B}
    </style>
</head>
<body>
<div class="container">
    <div style="width: 100%; text-align: right; margin-bottom: 20px;">
        <form method="GET" id="lang-form" style="display: inline-block;">
            <select name="lang" id="lang-select" onchange="this.form.submit()" style="background-color: var(--bg-card); color: var(--text-light); border: 1px solid var(--border-color); border-radius: 8px; padding: 8px; font-family: 'Inter', sans-serif; font-size: 14px;">
                {% for code, name in supported_languages %}
                    <option value="{{ code }}" {% if code == selected_lang %}selected{% endif %}>{{ name }}</option>
                {% endfor %}
            </select>
        </form>
    </div>
    {% if survey.logo_url %}<div class="survey-logo"><img src="{{ url_for('static', filename='uploads/' + survey.logo_url) }}" alt="Survey Logo"></div>{% endif %}
    <h1>{{ survey.title }}</h1>
    <p class="subtitle">Please answer the following questions.</p>
    <form id="survey-form" method="POST" action="{{ url_for('submit_survey', survey_uuid=survey.share_uuid) }}">
        <input type="hidden" name="start_time_iso" id="start_time_iso">
        <input type="hidden" name="access_mode" value="{{ access_mode or 'direct' }}">
        
        {% for question in questions %}
            {% set q_loop = loop %}
            <div class="card question-container" data-question-id="{{ question.id }}">
                <p class="question-text">{{ q_loop.index }}. {{ question.text }}</p>
                <input type="hidden" name="q_id_{{ q_loop.index0 }}" value="{{ question.id }}">
                <input type="hidden" name="q_type_{{ q_loop.index0 }}" value="{{ question.type }}">
                {% if question.type == 'MCQ' or question.type == 'LIKERT' %}{% for option in question.options.split(',') %}<label class="mcq-option"><input type="radio" name="answer_{{ q_loop.index0 }}" value="{{ option | trim }}" required> {{ option | trim }}</label>{% endfor %}{% elif question.type == 'CHECKBOX' %}{% for option in question.options.split(',') %}{% set opt_loop = loop %}<label class="mcq-option"><input type="checkbox" name="answer_{{ q_loop.index0 }}_{{ opt_loop.index0 }}" value="{{ option | trim }}"> {{ option | trim }}</label>{% endfor %}{% elif question.type == 'DROPDOWN' %}<select name="lang" id="lang" onchange="this.form.submit()" ... > {% for code, name in supported_languages %} <option value="{{ code }}" {% if code == selected_lang %}selected{% endif %}>{{ name }}</option> {% endfor %} </select>{% elif question.type == 'STARS' %}<div class="star-rating">{% for i in range(5, 0, -1) %}<input type="radio" id="star_{{ q_loop.index0 }}_{{i}}" name="answer_{{ q_loop.index0 }}" value="{{i}}" required/><label for="star_{{ q_loop.index0 }}_{{i}}" title="{{i}} stars">&#9733;</label>{% endfor %}</div>{% elif question.type == 'DATE' %}<input type="date" name="answer_{{ q_loop.index0 }}" required style="padding: 12px;">{% else %}<textarea name="answer_{{ q_loop.index0 }}" placeholder="Your Answer" rows="4" required></textarea>{% endif %}
            </div>
        {% endfor %}
        <button type="submit">Submit Survey</button>
    </form>
</div>
<script>
document.addEventListener('DOMContentLoaded', () => {
    const startTime = new Date();
    const startTimeInput = document.getElementById('start_time_iso');
    if (startTimeInput) {
        startTimeInput.value = startTime.toISOString();
    }
});
</script>
</body></html>
'''
SURVEY_MESSAGE_TEMPLATE = '''
<!DOCTYPE html><html lang="en"><head><title>{{ title }}</title><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet"><style>:root{--bg-dark:#111827;--bg-card:#1F2937;--primary:#3B82F6;--text-light:#F9FAFB;--text-dark:#9CA3AF;--danger:#EF4444}body{font-family:'Inter',sans-serif;background-color:var(--bg-dark);color:var(--text-light);display:flex;align-items:center;justify-content:center;height:100vh;margin:0}.container{background:var(--bg-card);padding:40px;border-radius:12px;box-shadow:0 10px 25px rgba(0,0,0,.3);width:100%;max-width:500px;text-align:center;border:1px solid var(--border-color)}h1{color:var(--text-light);margin-bottom:10px}p{color:var(--text-dark);line-height:1.6}form{margin-top:20px}input{width:100%;padding:12px;margin-bottom:15px;border:1px solid var(--border-color);background-color:#374151;color:var(--text-light);border-radius:8px;box-sizing:border-box;font-size:16px}input:focus{outline:none;border-color:var(--primary)}button{width:100%;padding:12px;background:var(--primary);color:#fff;border:none;border-radius:8px;cursor:pointer;font-size:16px;font-weight:600;transition:background-color .2s ease}button:hover{background:var(--primary-hover)}.flash{padding:15px;margin-bottom:20px;border:1px solid var(--danger);border-radius:8px;color:var(--text-light);background-color:rgba(239,68,68,.2)}</style></head><body><div class="container"><h1>{{ title }}</h1><p>{{ message }}</p>{{ form_html|safe }}</div></body></html>
'''
RESULTS_TEMPLATE = '''
{% block content %}
<style>
    /* Tab Navigation Styles */
    .results-nav { display: flex; border-bottom: 2px solid var(--border-color); margin-bottom: 20px; }
    .results-nav a { padding: 10px 20px; text-decoration: none; color: var(--text-dark); font-weight: 600; border-bottom: 2px solid transparent; cursor: pointer; }
    .results-nav a.active { color: var(--primary); border-bottom-color: var(--primary); }
    
    /* View Panes */
    .results-view { display: none; }
    .results-view.active { display: block; }
    
    /* Individual Response Styles (from before) */
    .response-card { background-color: var(--bg-dark); padding: 20px; border-radius: 8px; margin-bottom: 20px; }
    .paradata-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; padding-bottom: 15px; border-bottom: 1px solid var(--border-color); margin-bottom: 15px; }
    .paradata-item { font-size: 0.9em; }
    .paradata-item strong { color: var(--text-dark); display: block; }
    .quality-flag { background-color: var(--danger); color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.8em; font-weight: bold; }
    .answer-item { margin-top: 10px; }
    
    /* Question Summary Styles */
    .question-result { background: var(--bg-dark); padding: 20px; border-radius: 8px; margin-top: 20px; }
    .open-response { background:var(--bg-light); border-radius:8px; padding:15px; margin-top:10px; }
    .sentiment.POSITIVE { color: var(--sentiment-pos-text); font-weight: bold; }
    .sentiment.NEGATIVE { color: var(--sentiment-neg-text); font-weight: bold; }
</style>

<div class="page-header"><h2>Survey Results Dashboard</h2></div>

<div class="card">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <form>
            <label for="survey_id">Filter by Survey:</label>
            <select name="survey_id" id="survey_id" onchange="this.form.submit()">
                <option value="">-- Select a Survey --</option>
                {% for survey in distinct_surveys %}
                <option value="{{ survey.id }}" {% if survey.id == selected_survey_id %}selected{% endif %}>{{ survey.title }}</option>
                {% endfor %}
            </select>
        </form>
        {% if selected_survey_id %}
        <a href="{{ url_for('export_csv', survey_id=selected_survey_id) }}"><button class="btn-secondary">Export to CSV</button></a>
        {% endif %}
    </div>
</div>

{% if selected_survey_id %}
    <div class="results-nav">
        <a id="show-individual-btn" class="nav-tab active">Individual Responses</a>
        <a id="show-summary-btn" class="nav-tab">Question Summary</a>
    </div>

    <div id="individual-view" class="results-view active">
        {% if not individual_responses %}
            <div class="card" style="text-align:center; padding: 40px;"><p>No responses found for the selected survey.</p></div>
        {% else %}
            {% for resp in individual_responses %}
                <div class="response-card">
                     <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 15px;">
                        <h4 style="margin:0;">Response #{{ loop.index }}</h4>
                        <div>
                            {% if resp.quality_flags %}
                                {% for flag in resp.quality_flags.split(',') %}
                                    <span class="quality-flag" title="This response was flagged for review.">🚩 {{ flag }}</span>
                                {% endfor %}
                            {% endif %}
                        </div>
                    </div>
                    <div class="paradata-grid">
                        <div class="paradata-item"><strong>Submitted On</strong> {{ resp.end_time.strftime('%d %b %Y, %I:%M %p') if resp.end_time }}</div>
                        <div class="paradata-item"><strong>Time Taken</strong> {{ "%.1f"|format((resp.end_time - resp.start_time).total_seconds()) }} seconds</div>
                        <div class="paradata-item"><strong>Location</strong> {{ resp.geo_location }}</div>
                        <div class="paradata-item"><strong>Access Mode</strong> {{ resp.access_mode|upper }}</div>
                        <div class="paradata-item"><strong>Device/Browser</strong> {{ resp.user_agent|truncate(50) }}</div>
                    </div>
                    <div class="answers-section">
    {% for answer in resp.answers %}
    <div class="answer-item">
        <strong style="color: var(--text-dark);">{{ answer.text }}</strong>
        
        {# --- MODIFIED BLOCK START: Logic to show correct/incorrect marks --- #}
        <div style="display: flex; align-items: center; gap: 10px; margin: 5px 0 0 0; padding-left: 15px; border-left: 2px solid var(--border-color);">
            <p style="margin:0;">{{ answer.answer }}</p>
            
            {% if answer.correct_answer %} {# Only show marks if a correct answer is set #}
                {% if answer.answer == answer.correct_answer %}
                    <span style="color: var(--secondary); font-weight: bold;" title="Correct">✅</span>
                {% else %}
                    <span style="color: var(--danger); font-weight: bold;" title="Incorrect. Correct answer was: {{answer.correct_answer}}">❌</span>
                {% endif %}
            {% endif %}
        </div>
        {# --- MODIFIED BLOCK END --- #}

    </div>
    {% endfor %}
</div>
                </div>
            {% endfor %}
        {% endif %}
    </div>

    <div id="summary-view" class="results-view">
        {% if not results %}
            <div class="card" style="text-align:center; padding: 40px;"><p>No results found to generate a summary.</p></div>
        {% else %}
            {% for survey_title, questions in results.items() %}
            <div class="card survey-group">
                <h3>{{ survey_title }}</h3>
                {% for question_text, data in questions.items() %}
                <div class="question-result">
                    <h4>{{ question_text }}</h4>
                    {% if data.chart_id and data.chart_data.labels %}
                        <div style="max-height: 300px; margin: 20px 0;"><canvas id="{{ data.chart_id }}"></canvas></div>
                    {% elif data.type == 'OPEN' %}
                        {% for response in data.responses %}
                        <div class="open-response">
                            <p style="margin:0;">"{{ response.answer }}"</p>
                            <small style="color:var(--text-dark);">Sentiment: <span class="sentiment {{ response.sentiment.label }}">{{ response.sentiment.label }} ({{ "%.0f"|format(response.sentiment.score * 100) }}%)</span></small>
                        </div>
                        {% endfor %}
                    {% else %}
                        <p style="color:var(--text-dark);">No responses recorded for this question yet.</p>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
            {% endfor %}
        {% endif %}
    </div>
{% endif %}

<script>
    // Tab switching logic
    document.addEventListener('DOMContentLoaded', () => {
        const individualBtn = document.getElementById('show-individual-btn');
        const summaryBtn = document.getElementById('show-summary-btn');
        const individualView = document.getElementById('individual-view');
        const summaryView = document.getElementById('summary-view');
        const navTabs = document.querySelectorAll('.nav-tab');

        function switchView(viewToShow) {
            // Hide all views
            document.querySelectorAll('.results-view').forEach(v => v.style.display = 'none');
            // Deactivate all tabs
            navTabs.forEach(t => t.classList.remove('active'));

            if (viewToShow === 'summary') {
                summaryView.style.display = 'block';
                summaryBtn.classList.add('active');
            } else {
                individualView.style.display = 'block';
                individualBtn.classList.add('active');
            }
        }

        if (individualBtn) {
            individualBtn.addEventListener('click', (e) => {
                e.preventDefault();
                switchView('individual');
            });
        }
        if (summaryBtn) {
            summaryBtn.addEventListener('click', (e) => {
                e.preventDefault();
                switchView('summary');
                // We need to initialize charts only when the view is visible
                initializeCharts();
            });
        }
    });

    // Chart.js logic
    function initializeCharts() {
        const chartData = {{ all_chart_data|tojson }};
        if (!chartData || window.chartsInitialized) return;

        Chart.defaults.color = '#9CA3AF';
        Object.entries(chartData).forEach(([chartId, data]) => {
            const ctx = document.getElementById(chartId);
            if (!ctx) return;
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: data.labels,
                    datasets: [{ label: 'Response Count', data: data.values, backgroundColor: ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#6366F1', '#8B5CF6'], borderRadius: 4 }]
                },
                options: { 
                    responsive: true, maintainAspectRatio: false,
                    scales: { y: { beginAtZero: true, ticks: { color: '#9CA3AF', stepSize: 1 }, grid: { color: '#374151' } }, x: { ticks: { color: '#9CA3AF' }, grid: { display: false } } },
                    plugins: { legend: { display: false } }
                }
            });
        });
        window.chartsInitialized = true; // Prevents re-initializing
    }
</script>
{% endblock %}
'''
OCR_UPLOAD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Upload Survey Form: {{ survey.title }}</title>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root{--bg-dark:#111827;--bg-card:#1F2937;--primary:{{ survey.theme_color or '#3B82F6' }};--primary-hover:#2563EB;--text-light:#F9FAFB;--text-dark:#9CA3AF;--border-color:#374151}
        body{font-family:'Inter',sans-serif;background-color:var(--bg-dark);color:var(--text-light);display:flex;justify-content:center;padding:40px;margin:0}
        .container{width:100%;max-width:800px;text-align:center}
        h1{color:var(--text-light);margin-bottom:10px;font-size:2.2rem}
        p.subtitle{color:var(--text-dark);margin:0 auto 30px auto;max-width:600px;line-height:1.6}
        
        /* NEW styles for the simple upload button */
        .upload-wrapper {
            background-color: var(--bg-card);
            padding: 40px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .upload-btn {
            background-color: var(--bg-light);
            color: var(--text-light);
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            display: inline-block; /* Makes the label act like a button */
            transition: background-color 0.2s ease;
        }
        .upload-btn:hover {
            background-color: #4B5563;
        }
        #file-name {
            margin-top: 20px;
            color: var(--text-dark);
            font-style: italic;
            min-height: 20px;
        }
        input[type="file"] {
            display: none; /* The actual file input is hidden */
        }
        
        button[type="submit"]{padding:15px;background:var(--primary);color:#fff;border:none;border-radius:8px;cursor:pointer;font-size:18px;font-weight:600;width:100%;max-width:400px}
        button[type="submit"]:hover{background:var(--primary-hover)}
        .loader {display:none; margin: 20px auto; border: 5px solid #374151; border-top: 5px solid #3B82F6; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite;}
        @keyframes spin {0% {transform: rotate(0deg);} 100% {transform: rotate(360deg);}}
    </style>
</head>
<body>
<div class="container">
    <h1>{{ survey.title }}</h1>
    <form id="ocr-form" method="POST" action="{{ url_for('submit_ocr_survey', survey_uuid=survey.share_uuid) }}" enctype="multipart/form-data">
        <div class="upload-wrapper">
            <p class="subtitle">Please upload a clear image of your completed form (PNG or JPG).</p>
            
            <label for="survey-image" class="upload-btn">
                Choose Image
            </label>
            <input type="file" id="survey-image" name="survey_image" accept="image/png, image/jpeg" required>
            
            <div id="file-name">No file selected</div>
        </div>
        
        <div id="loader" class="loader"></div>
        <button type="submit" id="submit-btn">Process and Submit Answers</button>
    </form>
</div>
<script>
    const uploadInput = document.getElementById('survey-image');
    const fileNameDisplay = document.getElementById('file-name');
    const form = document.getElementById('ocr-form');
    const loader = document.getElementById('loader');
    const submitBtn = document.getElementById('submit-btn');

    uploadInput.addEventListener('change', () => {
        if (uploadInput.files.length > 0) {
            fileNameDisplay.textContent = 'Selected: ' + uploadInput.files[0].name;
        } else {
            fileNameDisplay.textContent = 'No file selected';
        }
    });

    form.addEventListener('submit', () => {
        loader.style.display = 'block';
        submitBtn.disabled = true;
        submitBtn.textContent = 'Processing...';
    });
</script>
</body>
</html>
'''

# --- Flask Routes ---

# (Routes for register, login, logout are unchanged and assumed to be here)

@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        conn = get_db_connection()
        user_exists = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        if user_exists:
            flash('Username already exists. Please choose a different one.')
            conn.close()
            return redirect(url_for('register'))
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        conn.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, hashed_password))
        conn.commit()
        conn.close()
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template_string(REGISTER_TEMPLATE)

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        conn = get_db_connection()
        user_row = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        if user_row and bcrypt.check_password_hash(user_row['password_hash'], password):
            user = User(id=user_row['id'], username=user_row['username'], password=user_row['password_hash'])
            login_user(user, remember=True)
            return redirect(url_for('dashboard'))
        else:
            flash('Login Unsuccessful. Please check username and password')
    return render_template_string(LOGIN_TEMPLATE)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route("/")
@login_required
def dashboard():
    conn = get_db_connection()
    surveys_rows = conn.execute('SELECT * FROM surveys WHERE user_id = ? ORDER BY created_at DESC', (current_user.id,)).fetchall()
    conn.close()
    surveys = []
    for row in surveys_rows:
        survey = dict(row)
        if isinstance(survey.get('created_at'), str):
            survey['created_at'] = datetime.strptime(survey['created_at'], '%Y-%m-%d %H:%M:%S')
        surveys.append(survey)
    page_content = render_template_string(DASHBOARD_TEMPLATE, surveys=surveys)
    return render_template_string(SHARED_APP_TEMPLATE, page_title="Dashboard", content=page_content)

@app.route("/create")
@login_required
def create_survey_page():
    topic_tags = ["Health", "Entertainment", "Technology", "Food", "Travel", "Education", "Work Life"]
    # Pass the central language list to the template
    page_content = render_template_string(CREATE_SURVEY_TEMPLATE, 
                                          topic_tags=topic_tags,
                                          languages=SUPPORTED_LANGUAGES) 
    return render_template_string(SHARED_APP_TEMPLATE, page_title="Create Survey", content=page_content)

@app.route("/generate", methods=["POST"])
@login_required
def generate():
    data = request.json
    prompt = data.get("prompt", "New Survey")
    num_questions = int(data.get("num_questions", 3))
    # Get the default language selected by the creator
    default_lang = data.get("lang", "en") 
    
    conn = get_db_connection()
    cursor = conn.cursor()
    share_uuid = str(uuid.uuid4())
    # Add the default_lang to the INSERT statement
    cursor.execute('INSERT INTO surveys (user_id, title, share_uuid, is_active, default_lang) VALUES (?, ?, ?, 0, ?)', 
                   (current_user.id, prompt, share_uuid, default_lang))
    survey_id = cursor.lastrowid
    
    # Generate and save questions in English (base language)
    questions_raw = generate_ai_questions(prompt, num_questions)
    for i, q in enumerate(questions_raw):
        cursor.execute('INSERT INTO questions (survey_id, position, text, type, options) VALUES (?, ?, ?, ?, ?)',
                       (survey_id, i, q['text'], q['type'], ",".join(q['options'])))
    
    conn.commit()
    conn.close()
    return jsonify({"redirect_url": url_for('edit_survey_page', survey_id=survey_id)})

@app.route('/survey/<int:survey_id>/edit')
@login_required
def edit_survey_page(survey_id):
    conn = get_db_connection()
    survey = conn.execute('SELECT * FROM surveys WHERE id = ? AND user_id = ?', (survey_id, current_user.id)).fetchone()

    if not survey:
        flash("Survey not found or you don't have permission to edit it.", "danger")
        return redirect(url_for('dashboard'))

    # 1. Fetch the original English questions
    original_questions = conn.execute('SELECT * FROM questions WHERE survey_id = ? ORDER BY position', (survey_id,)).fetchall()
    conn.close()

    # 2. Get the default language for this survey
    default_lang = survey['default_lang']

    # 3. Translate the questions if the default language is not English
    translated_questions = []
    if default_lang != 'en':
        for q in original_questions:
            q_dict = dict(q) # Make the row mutable
            q_dict['text'] = translate_text(q['text'], default_lang)
            if q['options']:
                options_list = q['options'].split(',')
                translated_options = [translate_text(opt, default_lang) for opt in options_list]
                q_dict['options'] = ",".join(translated_options)
            translated_questions.append(q_dict)
    else:
        # If the language is English, just use the original questions
        translated_questions = original_questions

    # 4. Pass the TRANSLATED questions to the template
    page_content = render_template_string(EDIT_SURVEY_TEMPLATE, survey=survey, questions=translated_questions)
    return render_template_string(SHARED_APP_TEMPLATE, page_title="Edit Survey", content=page_content)

@app.route('/manage/<int:survey_id>')
@login_required
def manage_survey_page(survey_id):
    conn = get_db_connection()
    survey = conn.execute('SELECT * FROM surveys WHERE id = ? AND user_id = ?', (survey_id, current_user.id)).fetchone()
    conn.close()
    if not survey: return "Survey not found or you don't have permission.", 404
    share_link = url_for('take_survey', survey_uuid=survey['share_uuid'], _external=True)
    page_content = render_template_string(MANAGE_SURVEY_TEMPLATE, survey=survey, share_link=share_link)
    return render_template_string(SHARED_APP_TEMPLATE, page_title="Manage Survey", content=page_content)

@app.route('/manage/<int:survey_id>/branding', methods=['POST'])
@login_required
def set_survey_branding(survey_id):
    conn = get_db_connection()
    survey = conn.execute('SELECT id FROM surveys WHERE id = ? AND user_id = ?', (survey_id, current_user.id)).fetchone()
    if not survey:
        flash("Survey not found or permission denied.", "danger")
        return redirect(url_for('dashboard'))
    theme_color = request.form.get('theme_color')
    conn.execute('UPDATE surveys SET theme_color = ? WHERE id = ?', (theme_color, survey_id))
    if 'logo' in request.files:
        logo_file = request.files['logo']
        if logo_file.filename != '':
            filename = secure_filename(logo_file.filename)
            unique_filename = str(uuid.uuid4()) + "_" + filename
            logo_file.save(os.path.join('static/uploads', unique_filename))
            conn.execute('UPDATE surveys SET logo_url = ? WHERE id = ?', (unique_filename, survey_id))
    conn.commit()
    conn.close()
    flash("Branding settings saved successfully!", "success")
    return redirect(url_for('manage_survey_page', survey_id=survey_id))

@app.route('/manage/<int:survey_id>/toggle', methods=['POST'])
@login_required
def toggle_survey_status(survey_id):
    conn = get_db_connection()
    survey = conn.execute('SELECT is_active FROM surveys WHERE id = ? AND user_id = ?', (survey_id, current_user.id)).fetchone()
    if survey:
        new_status = 0 if survey['is_active'] else 1
        conn.execute('UPDATE surveys SET is_active = ? WHERE id = ?', (new_status, survey_id))
        conn.commit()
        flash(f"Survey has been {'deactivated' if new_status == 0 else 'activated'}.", 'success')
    conn.close()
    return redirect(url_for('manage_survey_page', survey_id=survey_id))


@app.route('/manage/<int:survey_id>/password', methods=['POST'])
@login_required
def set_survey_password(survey_id):
    conn = get_db_connection()
    if request.args.get('remove'):
        conn.execute('UPDATE surveys SET password_hash = NULL WHERE id = ? AND user_id = ?', (survey_id, current_user.id))
        flash("Password removed successfully.", 'success')
    else:
        password = request.form.get('password')
        if password:
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            conn.execute('UPDATE surveys SET password_hash = ? WHERE id = ? AND user_id = ?', (hashed_password, survey_id, current_user.id))
            flash("Password set successfully.", 'success')
        else:
            flash("Password cannot be empty.", 'danger')
    conn.commit()
    conn.close()
    return redirect(url_for('manage_survey_page', survey_id=survey_id))

@app.route('/survey/<int:survey_id>/set_submission_type', methods=['POST'])
@login_required
def set_submission_type(survey_id):
    conn = get_db_connection()
    # Security check: Ensure the current user owns this survey
    survey = conn.execute('SELECT user_id FROM surveys WHERE id = ?', (survey_id,)).fetchone()

    if survey is None or survey['user_id'] != current_user.id:
        flash('Survey not found or permission denied.', 'danger')
        conn.close()
        return redirect(url_for('dashboard'))

    submission_type = request.form.get('submission_type')
    
    # Update the database with the new submission type
    if submission_type in ['DIGITAL', 'PAPER']:
        conn.execute('UPDATE surveys SET submission_type = ? WHERE id = ?', (submission_type, survey_id))
        conn.commit()
        flash(f'Submission mode has been updated to {submission_type.capitalize()}.', 'success')
    else:
        flash('Invalid submission type selected.', 'danger')
        
    conn.close()
    # Redirect back to the same manage page
    return redirect(url_for('manage_survey_page', survey_id=survey_id))

# --- Public Respondent Routes ---

@app.route('/s/<uuid:survey_uuid>', methods=['GET', 'POST'])
def take_survey(survey_uuid):
    conn = get_db_connection()
    survey = conn.execute('SELECT * FROM surveys WHERE share_uuid = ?', (str(survey_uuid),)).fetchone()

    if not survey:
        conn.close()
        return render_template_string(SURVEY_MESSAGE_TEMPLATE, title="Not Found", message="The survey you are looking for does not exist.")
    
    if not survey['is_active']:
        conn.close()
        return render_template_string(SURVEY_MESSAGE_TEMPLATE, title="Survey Closed", message="This survey is not currently accepting responses.")

    cookie_name = f'submitted_{survey["id"]}'
    if request.cookies.get(cookie_name):
        conn.close()
        return render_template_string(SURVEY_MESSAGE_TEMPLATE, title="Already Submitted", message="Thank you, but our records show you have already completed this survey.")

    if survey['password_hash']:
        if session.get('authenticated_survey_uuid') != str(survey_uuid):
            if request.method == 'POST':
                submitted_password = request.form.get('password')
                if bcrypt.check_password_hash(survey['password_hash'], submitted_password):
                    session['authenticated_survey_uuid'] = str(survey_uuid)
                    return redirect(url_for('take_survey', survey_uuid=survey_uuid))
                else:
                    flash('Incorrect password. Please try again.')
            
            form_html = '''<form method="POST"><input type="password" name="password" placeholder="Enter Password" required><button type="submit">Continue</button></form>'''
            return render_template_string(SURVEY_MESSAGE_TEMPLATE, title="Password Protected", message="This survey requires a password to continue.", form_html=form_html)

    # --- THIS IS THE MODIFIED LOGIC FOR LANGUAGE HANDLING ---
    # 1. Get the language from the URL (e.g., ?lang=hi). If not present, use the survey's default language.
    selected_lang = request.args.get('lang', survey['default_lang'])
    
    original_questions = conn.execute('SELECT id, text, type, options, logic_rules FROM questions WHERE survey_id = ? ORDER BY position', (survey['id'],)).fetchall()
    conn.close()

    # 2. Translate questions to the 'selected_lang'
    translated_questions = []
    for q in original_questions:
        q_dict = dict(q)
        # Always translate from the original English text stored in the DB
        q_dict['text'] = translate_text(q['text'], selected_lang)
        if q['options']:
            options_list = q['options'].split(',')
            translated_options = [translate_text(opt.strip(), selected_lang) for opt in options_list]
            q_dict['options'] = ",".join(translated_options)
        translated_questions.append(q_dict)

    # 3. Pass the full language list and the currently selected language to the template
    return render_template_string(
        SURVEY_TAKER_TEMPLATE, 
        survey=survey, 
        questions=translated_questions, 
        selected_lang=selected_lang,
        supported_languages=SUPPORTED_LANGUAGES
    )

@app.route('/survey/<uuid:survey_uuid>/submit', methods=['POST'])
def submit_survey(survey_uuid):
    conn = get_db_connection()
    survey = conn.execute('SELECT id, is_active FROM surveys WHERE share_uuid = ?', (str(survey_uuid),)).fetchone()
    if not survey or not survey['is_active']:
        return "This survey is closed.", 403
# ... (the code to detect flags remains here) ...

    if len(answers) >= 4:
        for i in range(len(answers) - 3):
            if answers[i] and answers[i] == answers[i+1] == answers[i+2] == answers[i+3]:
                quality_flags.append("selecting the same answer repeatedly")
                break
    
    # --- ADD THIS IF-STATEMENT ---
    if quality_flags:
        print(f"✅ Quality Alert: Submission for survey ID {survey['id']} was flagged for: {', '.join(quality_flags)}")
    # --- END OF NEW CODE ---
            
    flags_string = ",".join(quality_flags) if quality_flags else None
    
    # ... (the rest of the function continues) ...
    quality_flags = []
    end_time = datetime.now(timezone.utc)
    start_time = isoparse(request.form.get('start_time_iso'))
    duration = (end_time - start_time).total_seconds()
    num_questions = conn.execute('SELECT COUNT(id) FROM questions WHERE survey_id = ?', (survey['id'],)).fetchone()[0]
    min_duration_threshold = num_questions * MIN_SECONDS_PER_QUESTION
    
    if duration < min_duration_threshold:
        quality_flags.append("very rapid completion")

    answers = []
    idx = 0
    while f'q_id_{idx}' in request.form:
        q_type = request.form.get(f'q_type_{idx}')
        if q_type in ['MCQ', 'LIKERT', 'STARS', 'DROPDOWN']:
            answers.append(request.form.get(f'answer_{idx}'))
        idx += 1
    
    if len(answers) >= 4:
        for i in range(len(answers) - 3):
            if answers[i] and answers[i] == answers[i+1] == answers[i+2] == answers[i+3]:
                quality_flags.append("selecting the same answer repeatedly")
                break
                
    flags_string = ",".join(quality_flags) if quality_flags else None
    
    ip_address = request.remote_addr
    g = geocoder.ip('me' if ip_address in ('127.0.0.1', '::1') else ip_address)
    geo_location = f"{g.city}, {g.country}" if g.ok else "Unknown"
    
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO respondent_sessions (survey_id, start_time, end_time, ip_address, geo_location, quality_flags) VALUES (?, ?, ?, ?, ?, ?)',
        (survey['id'], start_time, end_time, ip_address, geo_location, flags_string)
    )
    session_id = cursor.lastrowid

    idx = 0
    while f'q_id_{idx}' in request.form:
        answer = ", ".join(request.form.getlist(f'answer_{idx}')) if request.form.get(f'q_type_{idx}') == 'CHECKBOX' else request.form.get(f'answer_{idx}')
        if answer:
            conn.execute('INSERT INTO responses (respondent_session_id, question_id, answer) VALUES (?, ?, ?)',
                         (session_id, request.form.get(f'q_id_{idx}'), answer))
        idx += 1
    
    conn.commit()
    conn.close()

    if quality_flags:
        reasons = " and ".join(quality_flags)
        message_to_respondent = f"Your response has been submitted but was flagged for review due to {reasons}."
    else:
        message_to_respondent = "Your response has been submitted."

    resp = make_response(render_template('survey_message.html', title="Thank You!", message=message_to_respondent))
    resp.set_cookie(f'submitted_{survey["id"]}', 'true', max_age=90*24*60*60)
    return resp

    # --- 1. Capture Paradata ---
    end_time = datetime.now(timezone.utc)
    start_time_iso = request.form.get('start_time_iso')
    start_time = isoparse(start_time_iso) if start_time_iso else end_time
    ip_address = request.remote_addr
    user_agent = request.headers.get('User-Agent', '')
    access_mode = request.form.get('access_mode', 'direct')
    
    # ### NEW, SIMPLER GEOLOCATION LOGIC ###
    geo_location = "Unknown"
    if ip_address:
        # Use 'me' for local testing, or the actual ip_address for production
        ip_to_check = 'me' if ip_address in ('127.0.0.1', '::1') else ip_address
        g = geocoder.ip(ip_to_check)
        if g.ok:
            geo_location = f"{g.city}, {g.country}"
    # ### END OF NEW LOGIC ###

    # --- 2. Apply Quality Flags ---
    flags = []
    num_questions = len([k for k in request.form if k.startswith('q_id_')])
    duration_seconds = (end_time - start_time).total_seconds()
    if duration_seconds < (num_questions * 2):
        flags.append('SPEEDING')
    answers_for_flagging = [v for k, v in request.form.items() if k.startswith('answer_')]
    if len(answers_for_flagging) > 2 and len(set(answers_for_flagging)) == 1:
        flags.append('STRAIGHT_LINING')
    quality_flags_str = ",".join(flags) if flags else None

    # --- 3. Save Session and Responses ---
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO respondent_sessions (survey_id, start_time, end_time, ip_address, user_agent, geo_location, access_mode, quality_flags)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (survey['id'], start_time, end_time, ip_address, user_agent, geo_location, access_mode, quality_flags_str))
    session_id = cursor.lastrowid

    # Corrected loop to save all answer types
    idx = 0
    while f'q_id_{idx}' in request.form:
        question_id = request.form.get(f'q_id_{idx}')
        question_type = request.form.get(f'q_type_{idx}')
        answer = None

        if question_type == 'CHECKBOX':
            checkbox_answers = [v for k, v in request.form.items() if k.startswith(f'answer_{idx}_')]
            if checkbox_answers:
                answer = ", ".join(checkbox_answers)
        else:
            answer = request.form.get(f'answer_{idx}')

        if question_id and answer:
            conn.execute(
                'INSERT INTO responses (respondent_session_id, question_id, answer) VALUES (?, ?, ?)',
                (session_id, int(question_id), answer)
            )
        idx += 1
    
    conn.commit()
    conn.close()

    resp = make_response(render_template_string(SURVEY_MESSAGE_TEMPLATE, title="Thank You!", message="Your response has been successfully submitted."))
    resp.set_cookie(f'submitted_{survey["id"]}', 'true', max_age=90*24*60*60)
    return resp

# --- Results & Export Routes ---

@app.route("/results")
@login_required
def results_dashboard():
    selected_survey_id = request.args.get('survey_id', type=int)
    conn = get_db_connection()
    distinct_surveys = conn.execute('SELECT id, title FROM surveys WHERE user_id = ? ORDER BY created_at DESC', (current_user.id,)).fetchall()
    
    summary_results = {}
    individual_responses = []
    all_chart_data = {}

    if selected_survey_id:
        selected_survey = conn.execute('SELECT title FROM surveys WHERE id = ?', (selected_survey_id,)).fetchone()
        
        # --- PART 1: Logic for Individual Responses View ---
        sessions = conn.execute('SELECT * FROM respondent_sessions WHERE survey_id = ? ORDER BY end_time DESC', (selected_survey_id,)).fetchall()
        for session in sessions:
            session_data = dict(session)
            
            # ### FIX: CONVERT DATE STRINGS TO DATETIME OBJECTS ###
            if session_data.get('start_time'):
                session_data['start_time'] = isoparse(session_data['start_time'])
            if session_data.get('end_time'):
                session_data['end_time'] = isoparse(session_data['end_time'])
            # ### END OF FIX ###

            session_data['answers'] = conn.execute('''
            SELECT q.text, r.answer, q.correct_answer
            FROM responses r
            JOIN questions q ON r.question_id = q.id
            WHERE r.respondent_session_id = ?
            ORDER BY q.position
            ''', (session['id'],)).fetchall()
            individual_responses.append(session_data)
            
        # --- PART 2: Logic for Question Summary View ---
        if selected_survey:
            questions = conn.execute('SELECT * FROM questions WHERE survey_id = ? ORDER BY position', (selected_survey_id,)).fetchall()
            summary_results[selected_survey['title']] = {}
            for q in questions:
                responses = conn.execute('SELECT answer FROM responses WHERE question_id = ?', (q['id'],)).fetchall()
                data = {'type': q['type'], 'responses': []}
                
                if q['type'] == 'OPEN': 
                    if sentiment_analyzer:
                        for r in responses:
                            try: sentiment = sentiment_analyzer(r['answer'])[0]
                            except Exception: sentiment = {'label': 'NEUTRAL', 'score': 0.5}
                            data['responses'].append({'answer': r['answer'], 'sentiment': sentiment})
                        data['themes'] = analyze_themes(data['responses'])
                elif q['type'] in ['MCQ', 'LIKERT', 'DROPDOWN', 'STARS']:
                    answer_counts = Counter([r['answer'] for r in responses])
                    chart_id = f"chart-{selected_survey_id}-{q['id']}"
                    chart_payload = {'labels': list(answer_counts.keys()), 'values': list(answer_counts.values())}
                    data['chart_id'] = chart_id
                    data['chart_data'] = chart_payload
                    all_chart_data[chart_id] = chart_payload
                elif q['type'] == 'CHECKBOX':
                    all_options = [opt.strip() for r in responses for opt in r['answer'].split(',')]
                    answer_counts = Counter(all_options)
                    chart_id = f"chart-{selected_survey_id}-{q['id']}"
                    chart_payload = {'labels': list(answer_counts.keys()), 'values': list(answer_counts.values())}
                    data['chart_id'] = chart_id
                    data['chart_data'] = chart_payload
                    all_chart_data[chart_id] = chart_payload
                else:
                    data['responses'] = [{'answer': r['answer']} for r in responses]
                summary_results[selected_survey['title']][q['text']] = data

    conn.close()
    
    page_content = render_template_string(
        RESULTS_TEMPLATE, 
        distinct_surveys=distinct_surveys, 
        selected_survey_id=selected_survey_id,
        individual_responses=individual_responses,
        results=summary_results,
        all_chart_data=all_chart_data
    )
    return render_template_string(SHARED_APP_TEMPLATE, page_title="Survey Results", content=page_content)

@app.route("/export/csv")
@login_required
def export_csv():
    survey_id = request.args.get('survey_id', type=int)
    if not survey_id:
        return "Survey ID is required.", 400

    conn = get_db_connection()
    # Security check to ensure the user owns the survey
    survey = conn.execute('SELECT title FROM surveys WHERE id = ? AND user_id = ?', (survey_id, current_user.id)).fetchone()
    if not survey:
        conn.close()
        return "Survey not found or you do not have permission to export it.", 403

    # This is the corrected SQL query
    query = '''
        SELECT 
            s.title, 
            q.text, 
            r.answer, 
            rs.end_time 
        FROM responses r
        JOIN questions q ON r.question_id = q.id
        JOIN surveys s ON q.survey_id = s.id
        JOIN respondent_sessions rs ON r.respondent_session_id = rs.id
        WHERE s.id = ?
        ORDER BY rs.end_time
    '''
    
    cursor = conn.execute(query, (survey_id,))
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # This is the corrected header for the CSV file
    writer.writerow(['Survey Title', 'Question', 'Answer', 'Timestamp'])
    
    for row in cursor:
        writer.writerow(row)
        
    conn.close()
    output.seek(0)
    
    # Create a unique filename for the download
    filename = secure_filename(f"{survey['title']}_responses.csv")
    
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={filename}"}
    )

# --- API Routes for Editor ---

@app.route('/api/survey/<int:survey_id>/questions', methods=['POST'])
@login_required
def add_question(survey_id):
    conn = get_db_connection()
    max_pos_row = conn.execute('SELECT MAX(position) FROM questions WHERE survey_id = ?', (survey_id,)).fetchone()
    max_pos = max_pos_row[0] if max_pos_row and max_pos_row[0] is not None else -1
    new_pos = max_pos + 1

    cursor = conn.execute('INSERT INTO questions (survey_id, position, text, type, options) VALUES (?, ?, ?, ?, ?)',
                          (survey_id, new_pos, 'New Question (Click to Edit)', 'OPEN', ''))
    new_question_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'question_id': new_question_id})


@app.route('/api/question/<int:question_id>', methods=['PUT', 'DELETE'])
@login_required
def update_delete_question(question_id):
    conn = get_db_connection()
    if request.method == 'PUT':
        # This block is now correctly indented with 4 spaces
        data = request.json
        correct_answer = data.get('correct_answer') or None 
        conn.execute(
            'UPDATE questions SET text = ?, type = ?, options = ?, correct_answer = ? WHERE id = ?',
            (data['text'], data['type'], data['options'], correct_answer, question_id)
        )
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Question updated.'})

    elif request.method == 'DELETE':
        # This block is also correctly indented
        conn.execute('DELETE FROM questions WHERE id = ?', (question_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Question deleted.'})


@app.route('/api/survey/<int:survey_id>/reorder', methods=['POST'])
@login_required
def reorder_questions(survey_id):
    order_data = request.json.get('order', []) 
    conn = get_db_connection()
    for new_position, question_id in enumerate(order_data):
        conn.execute('UPDATE questions SET position = ? WHERE id = ? AND survey_id = ?', 
                     (new_position, question_id, survey_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Order saved.'})

@app.route('/api/question/<int:question_id>/logic', methods=['PUT'])
@login_required
def save_question_logic(question_id):
    data = request.json
    logic_rules_json = json.dumps(data.get('rules'))
    conn = get_db_connection()
    conn.execute('UPDATE questions SET logic_rules = ? WHERE id = ?', (logic_rules_json, question_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Logic rules saved.'})

@app.route('/s/<uuid:survey_uuid>/ocr_submit', methods=['POST'])    
def submit_ocr_survey(survey_uuid):
    conn = get_db_connection()
    survey = conn.execute('SELECT id, is_active FROM surveys WHERE share_uuid = ?', (str(survey_uuid),)).fetchone()

    if not survey or not survey['is_active']:
        conn.close()
        return render_template_string(SURVEY_MESSAGE_TEMPLATE, title="Error", message="This survey is inactive or not found.")

    if 'survey_image' not in request.files or request.files['survey_image'].filename == '':
        flash("No file selected for uploading.")
        return redirect(request.url)

    file = request.files['survey_image']
    image_content = file.read() # Read the image content into memory

    extracted_text = ocr_image_with_google_vision(image_content)

    if extracted_text is None:
        return render_template_string(SURVEY_MESSAGE_TEMPLATE, title="Error", message="Could not process the image due to an OCR service error.")

    # Basic parsing: assume one answer per line
    answers = [line.strip() for line in extracted_text.split('\n') if line.strip()]
    questions = conn.execute('SELECT id FROM questions WHERE survey_id = ? ORDER BY position', (survey['id'],)).fetchall()

    if len(answers) < len(questions):
         return render_template_string(SURVEY_MESSAGE_TEMPLATE, title="Processing Error", message=f"OCR could only detect {len(answers)} answers, but the survey has {len(questions)} questions. Please try with a clearer image.")

    # Save session and responses
    session_cursor = conn.cursor()
    session_cursor.execute('INSERT INTO respondent_sessions (survey_id, start_time, end_time, access_mode) VALUES (?, ?, ?, ?)',
                          (survey['id'], datetime.now(timezone.utc), datetime.now(timezone.utc), 'OCR'))
    session_id = session_cursor.lastrowid

    for i, question in enumerate(questions):
        if i < len(answers):
            conn.execute('INSERT INTO responses (respondent_session_id, question_id, answer) VALUES (?, ?, ?)',
                         (session_id, question['id'], answers[i]))
    
    conn.commit()
    conn.close()

    # --- MODIFIED: Create a response object and set the cookie ---
    # Render the specific "Thank You" message for OCR submissions
    response_html = render_template_string(SURVEY_MESSAGE_TEMPLATE, title="Thank You!", message="Your paper survey has been processed and submitted successfully.")

    # Create a response object from the HTML
    resp = make_response(response_html)

    # Set a cookie that is unique to this survey and lasts for 90 days
    cookie_name = f'submitted_{survey["id"]}'
    resp.set_cookie(cookie_name, 'true', max_age=90*24*60*60) # max_age is in seconds

    return resp
    # --- END MODIFICATION ---

    # --- NEW: IVR Routes ---

@app.route("/ivr/welcome", methods=['POST'])
def ivr_welcome():
    """This is the first route Twilio calls when someone dials the number."""
    response = VoiceResponse()

    # Use <Gather> to read a message and collect keypad digits
    gather = Gather(num_digits=5, action='/ivr/handle-survey-id', method='POST')
    gather.say('Welcome to the AI Survey Tool. Please enter the five digit survey I D to begin.')
    response.append(gather)

    # If the user doesn't enter anything, repeat the welcome message
    response.redirect('/ivr/welcome')

    return str(response)


@app.route("/ivr/handle-survey-id", methods=['POST'])
def ivr_handle_survey_id():
    """This route is called after the user enters the survey ID."""
    response = VoiceResponse()
    survey_id = request.form.get('Digits')

    if survey_id:
        conn = get_db_connection()
        survey = conn.execute('SELECT id, title FROM surveys WHERE id = ?', (survey_id,)).fetchone()
        conn.close()

        if survey:
            # For this simple example, we just confirm the choice and hang up.
            response.say(f"Thank you. You have selected the survey titled: {survey['title']}. This is a demonstration. Goodbye.")
            response.hangup()
        else:
            response.say("Sorry, a survey with that I D was not found. Please hang up and try again.")
            response.hangup()
    else:
        # If no digits were entered, send them back to the start
        response.redirect('/ivr/welcome')

    return str(response)

# --- END OF NEW IVR ROUTES ---
# --- Application Startup ---
# --- Application Startup ---
# --- Application Startup ---
init_db()
load_custom_model() # <-- PASTE IT HERE

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)