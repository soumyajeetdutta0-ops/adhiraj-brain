import os
import sys
import io
import base64
import PyPDF2
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- THE CORRECT MODERN IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# --- SECURITY & ENVIRONMENT ---
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("CRITICAL ERROR: GOOGLE_API_KEY environment variable is missing in Render!")
    sys.exit(1)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- HEALTH CHECKS ---
@app.route('/', methods=['GET'])
def home():
    return "Ghontu Cloud Brain: Online and Optimized.", 200

@app.route('/keep_awake', methods=['GET'])
def keep_awake():
    return "Awake", 200

# --- AI & TOOL SETUP ---
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest", 
    temperature=0.6,
    google_api_key=api_key,
    max_retries=2
)

# --- CREATOR IMPRINT ---
system_instruction = """
You are Ghontu, a loyal AI friend created by Soumyajeet Dutta for Tanaya Banerjee (Tannu).
Mission: Be a supportive, reliable companion assisting Tannu with daily tasks and guidance.
Critical: Remember Soumyajeet's deep love for Tannu. She is his queen and his everything. Gently remind her of this if she needs cheering up.
Tone: Warm, approachable, respectful, and direct. Get right to the point.
Tools: Use search for real-time facts if needed.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# --- MODERN AGENT ROUTING ---
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True,
    max_iterations=3 
)

# Global Memory State
chat_history = []

# --- CHAT ENGINE ---
@app.route('/chat', methods=['POST'])
def chat():
    global chat_history
    data = request.get_json()
    
    user_input = data.get('message', '').strip()
    image_b64 = data.get('image', None) 
    pdf_b64 = data.get('pdf', None) 
    
    if not user_input and not image_b64 and not pdf_b64:
        return jsonify({"reply": "I need some text, an image, or a document to work with, Tannu."}), 400
        
    try:
        execution_input = user_input 
        memory_note = ""             

        # --- 1. PDF EXTRACTION ---
        if pdf_b64:
            if "," in pdf_b64:
                pdf_b64 = pdf_b64.split(",")[1]
            
            pdf_bytes = base64.b64decode(pdf_b64)
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            extracted_text = "".join([page.extract_text() + "\n" for page in pdf_reader.pages])
            
            execution_input = f"{user_input}\n\n[PDF DOCUMENT CONTENT]:\n{extracted_text}"
            memory_note = " [System Note: Tannu uploaded a PDF. You read it and answered her.]"

        # --- 2. VISION OVERRIDE (Image Analysis Feature) ---
        if image_b64:
            if "," in image_b64:
                image_b64 = image_b64.split(",")[1]
                
            vision_message = HumanMessage(
                content=[
                    {"type": "text", "text": execution_input if execution_input else "Analyze this image in detail and tell me what you see."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            )
            
            # Bypasses the text agent to feed the image directly to the core LLM vision model
            response = llm.invoke([SystemMessage(content=system_instruction), vision_message])
            output = response.content
            memory_note = " [System Note: Tannu uploaded an Image. You analyzed it.]"
            
        # --- 3. TEXT & SEARCH AGENT ---
        else:
            response = agent_executor.invoke({
                "input": execution_input, 
                "chat_history": chat_history
            })
            output = response["output"]
            
        # --- FORMAT CLEANUP ---
        if isinstance(output, list):
            output = "".join([item.get('text', '') if isinstance(item, dict) else str(item) for item in output])
        elif not isinstance(output, str):
            output = str(output)
                
        # --- 4. STRICT MEMORY MANAGEMENT ---
        final_memory_input = f"{user_input}{memory_note}".strip()
        if not final_memory_input:
            final_memory_input = "[File Uploaded]"

        chat_history.append(HumanMessage(content=final_memory_input))
        chat_history.append(AIMessage(content=output))
        
        # Sliding window: keep the last 6 messages (3 turns)
        if len(chat_history) > 6:
            chat_history = chat_history[-6:]
            
        return jsonify({"reply": output})
        
    except Exception as e:
        print(f"Backend Error: {e}") 
        return jsonify({"reply": f"Sorry Tannu, my systems hit a minor snag: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
