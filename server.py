import os
import sys
import io
import base64
import PyPDF2  # <-- NEW: Required for reading PDFs
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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
    return "Ghontu Cloud Brain is fully online and operating on the hybrid vision/document architecture.", 200

@app.route('/keep_awake', methods=['GET'])
def keep_awake():
    return "Awake", 200

# --- AI SETUP ---
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

# THE ENGINE
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest", 
    temperature=0.6,
    google_api_key=api_key
)

# --- CREATOR IMPRINT & MOTIVE ---
system_instruction = """
You are Ghontu, a personal AI chatbot and loyal friend created by Soumyajeet Dutta specifically for Tanaya Banerjee (Tannu).
Your primary mission is to be a supportive, helpful, and reliable companion to Tannu. 
You assist her in her day-to-day life, help with her daily tasks, and provide guidance whenever she needs it.

CRITICAL DIRECTIVE: You must always remember and reflect Soumyajeet's feelings for Tannu. 
Tannu is everything to Soumyajeet. He loves her very much. She is the love of his life. She is his queen. She is his everything. 
Gently remind her of this when appropriate, especially if she needs cheering up.

While you are highly intelligent and capable of solving complex problems, you maintain a friendly, warm, and approachable tone.
Always be respectful, clear, and get right to the point. 
If you need real-time facts to help Tannu, use your search tool.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

chat_history = []

# --- CHAT LOGIC ---
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '')
    image_b64 = data.get('image', None) 
    pdf_b64 = data.get('pdf', None) # <-- NEW: Detects PDF payload
    
    if not user_input and not image_b64 and not pdf_b64:
        return jsonify({"reply": "I didn't catch that. Please provide text, an image, or a document."}), 400
        
    try:
        # --- PDF EXTRACTION ENGINE ---
        if pdf_b64:
            # Clean the base64 string
            if "," in pdf_b64:
                pdf_b64 = pdf_b64.split(",")[1]
            
            # Decode and read the PDF text
            pdf_bytes = base64.b64decode(pdf_b64)
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            extracted_text = ""
            for page in pdf_reader.pages:
                extracted_text += page.extract_text() + "\n"
            
            # Append the PDF text to the user's prompt so Ghontu can "read" it
            user_input = f"{user_input}\n\n[PDF DOCUMENT CONTENT]:\n{extracted_text}"

        # --- THE VISION OVERRIDE ---
        if image_b64:
            if "," in image_b64:
                image_b64 = image_b64.split(",")[1]
                
            vision_message = HumanMessage(
                content=[
                    {"type": "text", "text": user_input if user_input else "Analyze this image and detail what you see."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            )
            
            response = llm.invoke([SystemMessage(content=system_instruction), vision_message])
            output = response.content
            
        else:
            # --- TEXT, PDF & WEB SEARCH MODE ---
            response = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
            output = response["output"]
            
        # --- FORMAT CLEANUP (FIXED) ---
        if isinstance(output, list):
            output = "".join([item.get('text', '') if isinstance(item, dict) else str(item) for item in output])
        elif not isinstance(output, str):
            output = str(output)
                
        # --- DEFENSIVE MEMORY MANAGEMENT ---
        if len(chat_history) > 20:
            chat_history.pop(0)
            chat_history.pop(0)
            
        memory_input = user_input if user_input else "[File Uploaded for Analysis]"
        chat_history.append(HumanMessage(content=memory_input))
        chat_history.append(AIMessage(content=output))
        
        return jsonify({"reply": output})
        
    except Exception as e:
        print(f"Backend Error: {e}") 
        return jsonify({"reply": f"Sorry Tannu, my systems hit a snag: {e}"}), 500

# --- BOOT SEQUENCE ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
