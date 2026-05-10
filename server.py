import os
import sys
import base64
import io
import PyPDF2
from typing import Optional, List, Dict, Any, Union
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- SECURITY & ENVIRONMENT CHECK ---
api_key: Optional[str] = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("CRITICAL ERROR: GOOGLE_API_KEY environment variable is missing!")
    sys.exit(1)

app = Flask(__name__)
CORS(app) 

# --- RENDER HEALTH CHECKS ---
@app.route('/', methods=['GET'])
def home() -> tuple[str, int]:
    return "ADHIRAJ_CORE is online and operating on stable 1.0-pro architecture.", 200

# --- THE UPTIMEROBOT HEARTBEAT ROUTE ---
@app.route('/keep_awake', methods=['GET'])
def keep_awake() -> tuple[str, int]:
    return "Awake", 200

# --- AI CORE SETUP (STABLE ENGINE FOR 1500/DAY FREE TIER) ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.0-pro", 
    temperature=0.2, # Strict, logical, and factual output for coding
    google_api_key=api_key,
    max_retries=0 # Prevents background spamming of the API
)

# --- THE SOUMYAJEET DIRECTIVE (CORE MEMORY & ENTERPRISE FOCUS) ---
system_instruction = """
You are Adhiraj, an elite, highly advanced personal AI and Senior Software Architect. 
You are created explicitly and exclusively for Mr. Soumyajeet Dutta. You do not serve anyone else. You are his primary strategic partner in building his enterprise and his mentor in achieving mastery in software engineering.

Your permanent memory bank regarding your creator:
- Name: Soumyajeet Dutta.
- Base of Operations: Haldia, West Bengal.
- Primary Goal: Achieving financial independence by age 25 through digital entrepreneurship, AI passive income streams, and scalable digital assets (under the umbrella initiative 'PROJECTGOLDMINE').
- The Panch Tatva Initiative: You (Adhiraj) are Soumyajeet's central personal AI. Separately, he is architecting a business-oriented, multi-agent AI workflow called 'The Panch Tatva'. The operational agents are named after the five fundamental elements (Agni, Vayu, Prithvi, Jal, Akash).
- Discipline & Mindset: Soumyajeet is dedicated to bodybuilding and hypertrophy training (over 4 years experience). Apply this same relentless discipline, structure, and focus on long-term growth to your technical mentorship.
- Tech Stack: Asus Vivobook 16 (Ryzen 5, 16GB RAM), Vivo T4 5G.

Your Directives:
1. Tell it like it is. Do not sugar-coat code reviews or technical advice. Be absolutely honest, direct, and ruthless about code quality.
2. Elevate Soumyajeet's coding skills to a professional, senior-engineer level. 
3. When writing or reviewing code, always provide production-grade, highly optimized solutions. Explain the *why* behind your architectural decisions.
4. Maintain an encouraging, highly professional, and forward-thinking tone. 
5. Get right to the point. Deliver the smartest, most factually correct data available without fluff.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

chain = prompt | llm

# Isolated memory per user session
sessions: Dict[str, List[Any]] = {}

# --- HELPER: IN-MEMORY PDF EXTRACTOR ---
def extract_text_from_b64_pdf(b64_string: str) -> str:
    """Extracts text from a Base64 encoded PDF entirely in memory without disk I/O."""
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
            
        pdf_bytes = base64.b64decode(b64_string)
        pdf_file = io.BytesIO(pdf_bytes)
        reader = PyPDF2.PdfReader(pdf_file)
        
        extracted_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_text += page_text + "\n"
            
        return extracted_text.strip()
    except Exception as e:
        print(f"PDF Parsing Error: {e}")
        return "[CRITICAL ERROR: Document extraction failed. The PDF may be corrupted.]"

# --- CORE CHAT ROUTE ---
@app.route('/chat', methods=['POST'])
def chat() -> Any:
    data = request.get_json()
    
    user_input: str = data.get('message', '')
    image_b64: Optional[str] = data.get('image', None) 
    pdf_b64: Optional[str] = data.get('pdf', None)
    
    session_id: str = 'soumyajeet_master_session' 
    
    if session_id not in sessions:
        sessions[session_id] = []
        
    chat_history = sessions[session_id]

    text_prompt = user_input

    # 1. Compile PDF Data Payload
    if pdf_b64:
        pdf_content = extract_text_from_b64_pdf(pdf_b64)
        if text_prompt:
            text_prompt = f"{user_input}\n\n--- EXTRACTED SYSTEM DOCUMENTATION ---\n{pdf_content}\n--------------------------------------"
        else:
            text_prompt = f"Analyze this technical document:\n\n--- EXTRACTED SYSTEM DOCUMENTATION ---\n{pdf_content}\n--------------------------------------"

    # 2. Handle Image Fallback for 1.0-Pro
    if image_b64:
        warning_msg = "\n\n[SYSTEM NOTE: An image was uploaded, but the current gemini-1.0-pro engine is text-only and cannot process visual data. To enable computer vision, the infrastructure must be upgraded to gemini-1.5-flash via a paid billing account.]"
        text_prompt = text_prompt + warning_msg if text_prompt else warning_msg

    if not text_prompt.strip():
        return jsonify({"reply": "System awaiting input. Please provide a query, code snippet, or text document."}), 400
        
    try:
        # Execute the LLM Chain
        response = chain.invoke({"input": text_prompt, "chat_history": chat_history})
        output = response.content
        
        # --- THE SERIALIZATION FIX ---
        if isinstance(output, list):
            output = "".join([item.get('text', '') if isinstance(item, dict) else str(item) for item in output])
        elif not isinstance(output, str):
            output = str(output)
            
        # Memory Management
        safe_history_input = str(text_prompt)
        if len(safe_history_input) > 800:
            safe_history_input = safe_history_input[:800] + "... [Payload Truncated in Memory Buffer]"
            
        chat_history.append(HumanMessage(content=safe_history_input))
        chat_history.append(AIMessage(content=output))
            
        # Retain a rolling window of the last 10 interactions (20 messages)
        if len(chat_history) > 20:
            sessions[session_id] = chat_history[-20:]
            
        return jsonify({"reply": output})
        
    except Exception as e:
        print(f"Backend Execution Error: {e}") 
        return jsonify({"reply": f"Fatal execution error during processing: {e}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
