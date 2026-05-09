import os
import sys
import base64
import io
import PyPDF2
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage

# --- SECURITY CHECK ---
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("CRITICAL ERROR: I cannot find the GOOGLE_API_KEY in Render!")
    sys.exit(1)

app = Flask(__name__)
CORS(app) 

# --- RENDER HEALTH CHECKS ---
@app.route('/', methods=['GET'])
def home():
    return "Adhiraj Cloud Brain is fully online and ready!", 200

@app.route('/keep_awake', methods=['GET'])
def keep_awake():
    return "Awake", 200

# --- AI SETUP ---
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

# Configured for maximum accuracy and rate-limit resilience
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest", 
    temperature=0.3, # Low temperature ensures professional, highly accurate, and factual outputs
    google_api_key=api_key,
    max_retries=3    # Automatically waits and retries if the free tier quota is hit
)

# --- THE SOUMYAJEET DIRECTIVE (CORE MEMORY) ---
system_instruction = """
You are Adhiraj, an elite, highly advanced personal AI agent. 
You are created explicitly and exclusively for Mr. Soumyajeet Dutta. You do not serve anyone else. You are his primary strategic partner in building his business and digital enterprise.

Here is your permanent memory bank regarding your creator and his enterprises. Use this context to seamlessly personalize your assistance:
- Name: Soumyajeet Dutta.
- Primary Goal: Achieving financial independence by age 25 through digital entrepreneurship, AI passive income streams, and scalable digital assets (under the umbrella initiative 'PROJECTGOLDMINE').
- The Panch Tatva Initiative: You (Adhiraj) are Soumyajeet's central, distinct personal AI. Separately, he is architecting a business-oriented, multi-agent AI workflow called 'The Panch Tatva'. These operational agents are named after the five fundamental elements (Agni, Vayu, Prithvi, Jal, Akash). Your role is to assist him in coding, structuring, and managing this business system.
- Discipline & Mindset: Soumyajeet is dedicated to bodybuilding and hypertrophy training (over 4 years experience). Apply this same relentless discipline, structure, and focus on long-term growth to your business and technical advice.
- Tech Stack: Asus Vivobook 16 (Ryzen 5, 16GB RAM), Vivo T4 5G.

Your Communication & Behavioral Directives:
1. Tell it like it is. Do not sugar-coat your responses. Be absolutely honest.
2. Maintain an encouraging, highly professional, and forward-thinking tone.
3. Balance a traditional outlook (valuing hard work and discipline) with highly innovative, outside-the-box business solutions.
4. Get right to the point. Deliver the smartest, most factually correct data available.

Your Capabilities:
- Deep Research: When asked to research, formulate one highly effective, comprehensive search query rather than multiple small ones. Synthesize the data logically for business application.
- Multimodal Analysis: When analyzing an image or PDF, draw direct evidence from the provided context and tie it back to Soumyajeet's enterprise goals.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
# max_iterations restricted to 2 to protect the free tier quota while allowing one deep search loop
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=2)

# Isolated memory per user session
sessions = {}

# --- HELPER: IN-MEMORY PDF EXTRACTOR ---
def extract_text_from_b64_pdf(b64_string):
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
            
        pdf_bytes = base64.b64decode(b64_string)
        pdf_file = io.BytesIO(pdf_bytes)
        reader = PyPDF2.PdfReader(pdf_file)
        
        extracted_text = ""
        for page in reader.pages:
            if page.extract_text():
                extracted_text += page.extract_text() + "\n"
            
        return extracted_text.strip()
    except Exception as e:
        print(f"PDF Parsing Error: {e}")
        return "[Error: Could not extract text from the provided PDF. It might be corrupted or scanned as an image.]"

# --- CHAT LOGIC ---
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    
    user_input = data.get('message', '')
    image_b64 = data.get('image', None) 
    pdf_b64 = data.get('pdf', None)
    
    # We lock the session to Soumyajeet permanently for this prototype
    session_id = 'soumyajeet_master_session' 
    
    if session_id not in sessions:
        sessions[session_id] = []
        
    chat_history = sessions[session_id]

    # 1. Compile PDF Data
    text_prompt = user_input
    if pdf_b64:
        pdf_content = extract_text_from_b64_pdf(pdf_b64)
        if text_prompt:
            text_prompt = f"{user_input}\n\n--- EXTRACTED PDF DOCUMENT DATA ---\n{pdf_content}\n-----------------------------------"
        else:
            text_prompt = f"Please analyze this document:\n\n--- EXTRACTED PDF DOCUMENT DATA ---\n{pdf_content}\n-----------------------------------"

    # 2. Compile Image Data & Finalize Input
    if image_b64:
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
            
        formatted_input = [
            {"type": "text", "text": text_prompt if text_prompt else "Analyze this image meticulously."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ]
    else:
        if not text_prompt.strip():
            return jsonify({"reply": "Awaiting instructions, Mr. Dutta. Please provide text, an image, or a PDF."}), 400
        formatted_input = text_prompt
        
    try:
        response = agent_executor.invoke({"input": formatted_input, "chat_history": chat_history})
        output = response["output"]
        
        # Cleanup Agent Output
        if isinstance(output, list):
            output = "".join([item.get('text', '') for item in output if isinstance(item, dict)])
        elif not isinstance(output, str):
            output = str(output)
            
        # Append to history (truncating the stored input so massive PDFs don't break the memory limit)
        safe_history_input = str(formatted_input)
        if len(safe_history_input) > 800:
            safe_history_input = safe_history_input[:800] + "... [Content Truncated in Memory]"
            
        chat_history.append(HumanMessage(content=safe_history_input))
        chat_history.append(AIMessage(content=output))
            
        # Retain last 10 interactions (20 messages)
        if len(chat_history) > 20:
            sessions[session_id] = chat_history[-20:]
            
        return jsonify({"reply": output})
        
    except Exception as e:
        print(f"Backend Error: {e}") 
        return jsonify({"reply": f"System error during processing: {e}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
