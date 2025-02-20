import json
import random
import os
from flask import Flask, render_template, request, session, redirect, url_for, flash
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from langdetect import detect

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_default_secret_key')
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = True

# Load model and tokenizer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = os.getenv('MODEL_PATH', '../models/roberta_intent_model')
TOKENIZER_NAME = os.getenv('TOKENIZER_PATH', '../models/roberta_intent_tokenizer')
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# Load translation model and tokenizer
TRANSLATION_MODEL_PATH = "../models/swa_en_model"
translation_tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_PATH)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_PATH)

# Load intents and intent mapping
with open('../data/intents.json', 'r') as f:
    intents_data = json.load(f)

with open('../data/intent_mapping.json', 'r') as f:
    intent_mapping = json.load(f)

# Database setup
DATABASE = 'database.db'

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute(''' 
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')

        cursor.execute(''' 
            CREATE TABLE IF NOT EXISTS chats (
                chat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')

        cursor.execute(''' 
            CREATE TABLE IF NOT EXISTS messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                sender TEXT NOT NULL,  -- 'user' or 'bot'
                message TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(chat_id) REFERENCES chats(chat_id)
            )
        ''')
        conn.commit()

# Translation function
def translate(text, src_lang, tgt_lang):
    translation_tokenizer.src_lang = src_lang
    encoded_text = translation_tokenizer(text, return_tensors="pt")
    forced_token_id = translation_tokenizer.get_lang_id(tgt_lang)
    generated_tokens = translation_model.generate(
        encoded_text["input_ids"], forced_bos_token_id=forced_token_id
    )
    return translation_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

# Language detection
def detect_language(text):
    try:
        lang = detect(text)
        if lang in ['sw', 'sw-tz', 'sw-ke']:
            return 'sw'
        return 'en'
    except Exception as e:
        print(f"Language detection error: {e}")
        return 'en'


# Detect intent
def detect_intent(text):
    text = text.lower().strip()  # Normalize text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    outputs = model(**inputs)
    intent_id = str(torch.argmax(outputs.logits, dim=1).item())
    intent_tag = intent_mapping.get(intent_id, "unknown")
    return intent_tag


def get_response(intent_tag, user_language):
    if isinstance(intents_data, dict):  # If intents_data is a dictionary
        intents = intents_data.get('intents', [])  # Fetch the list of intents
    else:
        intents = intents_data  # Use directly if already a list

    for intent in intents_data.get('intents', []):
        if intent['tag'] == intent_tag:
            responses = intent['responses']
            response = random.choice(responses)
            if user_language == 'sw' and detect_language(response) != 'sw':
                return translate(response, src_lang="en", tgt_lang="sw")
            elif user_language == 'en' and detect_language(response) != 'en':
                return translate(response, src_lang="sw", tgt_lang="en")
            return response
    return "Samahani, siwezi kuelewa hiyo." if user_language == 'sw' else "Sorry, I don't understand that."


# Routes
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
                user = cursor.fetchone()
                if user and check_password_hash(user[2], password):
                    session['user_id'] = user[0]
                    session['username'] = user[1]
                    return redirect(url_for('chat'))
                else:
                    flash('Invalid username or password!', 'danger')
            except sqlite3.Error as e:
                flash('An error occurred while logging in.', 'danger')
                print(f"Database error: {e}")

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'], method='pbkdf2:sha256')

        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
                conn.commit()
                flash('Registration successful! Please log in.', 'success')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('Username already exists!', 'danger')
            except sqlite3.Error as e:
                flash('An error occurred during registration.', 'danger')
                print(f"Database error: {e}")

    return render_template('register.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    chats = []  # Initialize to an empty list

    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT c.chat_id, c.created_at, (SELECT message FROM messages WHERE chat_id = c.chat_id ORDER BY created_at LIMIT 1) AS first_message
                FROM chats c
                WHERE c.user_id = ?
                ORDER BY c.created_at DESC
            """, (user_id,))
            chats = cursor.fetchall()
        except sqlite3.Error as e:
            flash('Error fetching chats.', 'danger')
            print(f"Database error: {e}")

    print(f"Chats in chat route: {chats}")
    return render_template('chat.html', chats=chats)

@app.route('/chat/<int:chat_id>', methods=['GET', 'POST'])
def chat_details(chat_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    current_chat = []
    all_chats = []

    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT sender, message FROM messages WHERE chat_id = ? ORDER BY created_at", (chat_id,))
            current_chat = cursor.fetchall()

            cursor.execute("""
                SELECT c.chat_id, c.created_at, (SELECT message FROM messages WHERE chat_id = c.chat_id ORDER BY created_at LIMIT 1) AS first_message
                FROM chats c
                WHERE c.user_id = ?
                ORDER BY c.created_at DESC
            """, (user_id,))
            all_chats = cursor.fetchall()

            if request.method == 'POST':
                user_message = request.form['message']
                user_language = detect_language(user_message)

                # Translate Swahili input to English for intent detection
                if user_language == 'sw':
                    user_message_for_intent = translate(user_message, src_lang="sw", tgt_lang="en")
                else:
                    user_message_for_intent = user_message

                intent_tag = detect_intent(user_message_for_intent)
                bot_response = get_response(intent_tag, user_language)


                cursor.execute("INSERT INTO messages (chat_id, sender, message) VALUES (?, ?, ?)", 
                               (chat_id, 'user', user_message))
                conn.commit()

                cursor.execute("INSERT INTO messages (chat_id, sender, message) VALUES (?, ?, ?)", 
                               (chat_id, 'bot', bot_response))
                conn.commit()

                cursor.execute("SELECT sender, message FROM messages WHERE chat_id = ? ORDER BY created_at", (chat_id,))
                current_chat = cursor.fetchall()
        except sqlite3.Error as e:
            flash('An error occurred while processing your message.', 'danger')
            print(f"Database error: {e}")

    return render_template('chat.html', chat_history=current_chat, chat_id=chat_id, chats=all_chats)

@app.route('/new_chat')
def new_chat():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']

    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO chats (user_id) VALUES (?)", (user_id,))
            conn.commit()
            chat_id = cursor.lastrowid  # Get the ID of the newly created chat
        except sqlite3.Error as e:
            flash('Failed to create new chat.', 'danger')
            print(f"Database error: {e}")
            return redirect(url_for('chat'))

    return redirect(url_for('chat_details', chat_id=chat_id))

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=False)  # Disable debug in production