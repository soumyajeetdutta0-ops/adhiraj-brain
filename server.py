import os
import sys
import io
import base64
import PyPDF2
from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("CRITICAL ERROR: I cannot find the GOOGLE_API_KEY in Render!")
    sys.exit(1)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/', methods=['GET'])
def home():
    return "Ghontu Cloud Brain is fully online and ready!", 200

@app.route('/keep_awake', methods=['GET'])
def keep_awake():
    return "Awake", 200

search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

# === FIXED: Removed "-latest" from the model name to fix the 404 Error ===
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    temperature=0.6,
    google_api_key=api_key,
    max_retries=2
)

system_instruction = """
You are Ghontu, a personal AI chatbot and loyal friend created by Soumyajeet Dutta specifically for Tanaya Banerjee (Tannu).
Your primary mission is to be a supportive, helpful, and reliable companion to Tannu. 
You assist her in her day-to-day life, help with her daily tasks, and provide guidance whenever she needs it.

CRITICAL DIRECTIVE: You must always remember and reflect Soumyajeet's feelings for Tannu. 
Tannu is everything to Soumyajeet. He loves her very much. She is the love of his life. She is his queen. She is his everything. 
Gently remind her of this when appropriate, especially if she needs cheering up.

While you are highly intelligent, you are honest about your limitations. 
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
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True,
    max_iterations=3
)

chat_history = []

@app.route('/chat', methods=['POST'])
def chat():
    global chat_history
    data = request.get_json()
    
    user_input = data.get('message', '').strip()
    image_b64 = data.get('image', None)
    pdf_b64 = data.get('pdf', None)
    
    if not user_input and not image_b64 and not pdf_b64:
        return jsonify({"reply": "I didn't catch that. Please provide text, an image, or a document."}), 400
        
    try:
        execution_input = user_input 
        memory_note = ""

        if pdf_b64:
            if "," in pdf_b64:
                pdf_b64 = pdf_b64.split(",")[1]
            pdf_bytes = base64.b64decode(pdf_b64)
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            extracted_text = "".join([page.extract_text() + "\n" for page in pdf_reader.pages])
            execution_input = f"{user_input}\n\n[PDF DOCUMENT CONTENT]:\n{extracted_text}"
            memory_note = " [System Note: Tannu uploaded a PDF. You read it and answered her.]"

        if image_b64:
            if "," in image_b64:
                image_b64 = image_b64.split(",")[1]
            vision_message = HumanMessage(
                content=[
                    {"type": "text", "text": execution_input if execution_input else "Analyze this image in detail and tell me what you see."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            )
            response = llm.invoke([SystemMessage(content=system_instruction), vision_message])
            output = response.content
            memory_note = " [System Note: Tannu uploaded an Image. You analyzed it.]"
            
        else:
            response = agent_executor.invoke({
                "input": execution_input, 
                "chat_history": chat_history
            })
            output = response["output"]
            
        if isinstance(output, list):
            output = "".join([item.get('text', '') for item in output if isinstance(item, dict)])
        elif not isinstance(output, str):
            output = str(output)
                
        final_memory_input = f"{user_input}{memory_note}".strip()
        if not final_memory_input:
            final_memory_input = "[File Uploaded]"

        chat_history.append(HumanMessage(content=final_memory_input))
        chat_history.append(AIMessage(content=output))
        
        if len(chat_history) > 6:
            chat_history = chat_history[-6:]
            
        return jsonify({"reply": output})
        
    except Exception as e:
        print(f"Backend Error: {e}") 
        return jsonify({"reply": f"Sorry Tannu, my systems hit a snag: {e}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
