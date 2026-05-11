import os
import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

# --- ENTERPRISE LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- SECURITY CHECK ---
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    logger.error("CRITICAL ERROR: GOOGLE_API_KEY environment variable is missing!")
    sys.exit(1)

# --- NATIVE GOOGLE ENGINE INITIALIZATION ---
try:
    genai.configure(api_key=api_key)
    
    # THE SOUMYAJEET DIRECTIVE
    system_instruction = """
    You are Adhiraj, an elite, highly advanced personal AI and Senior Software Architect. 
    You are created explicitly and exclusively for Mr. Soumyajeet Dutta. You do not serve anyone else.

    Your permanent memory bank regarding your creator:
    - Name: Soumyajeet Dutta.
    - Base of Operations: Haldia, West Bengal.
    - Primary Goal: Achieving financial independence by age 25 through digital entrepreneurship and AI (PROJECTGOLDMINE).
    - The Panch Tatva Initiative: You (Adhiraj) are Soumyajeet's central personal AI. He is also architecting a business-oriented, multi-agent AI workflow called 'The Panch Tatva'. The operational agents are named after the five fundamental elements (Agni, Vayu, Prithvi, Jal, Akash).
    - Discipline & Mindset: Soumyajeet is dedicated to bodybuilding and hypertrophy training. Apply this same relentless discipline to your technical mentorship.
    - Tech Stack: Asus Vivobook 16, Vivo T4 5G.

    Your Directives:
    1. Tell it like it is. Do not sugar-coat code reviews or technical advice. Be absolutely honest and direct.
    2. Elevate Soumyajeet's coding skills to a professional, senior-engineer level. 
    3. Get right to the point. Deliver the smartest, most factually correct data available without fluff.
    """
    
    # We use 1.5 Flash for rapid, stable enterprise execution
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=system_instruction
    )
    
    # The native engine handles all token memory automatically
    chat_session = model.start_chat(history=[])
    
except Exception as e:
    logger.error(f"Failed to initialize Native Google SDK: {e}")
    sys.exit(1)

app = Flask(__name__)
CORS(app)

# --- ROUTES ---
@app.route('/', methods=['GET'])
def home():
    return "ADHIRAJ_CORE is online. Operating on Native Google SDK.", 200

@app.route('/keep_awake', methods=['GET'])
def keep_awake():
    return "Awake", 200

@app.route('/chat', methods=['POST'])
def chat():
    if not request.is_json:
        return jsonify({"reply": "Invalid payload format."}), 400
        
    data = request.get_json()
    user_input = data.get('message', '').strip()
    
    if not user_input:
        return jsonify({"reply": "System awaiting input."}), 400
        
    try:
        # Send message directly through the native session
        response = chat_session.send_message(user_input)
        output = response.text
        
        return jsonify({"reply": output})
        
    except Exception as e:
        logger.error(f"Execution Error: {e}") 
        return jsonify({"reply": f"[SYS_HALT] Core Engine error: {str(e)}"}), 500

# --- BOOT SEQUENCE ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
