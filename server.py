import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage

# --- SECURITY CHECK ---
# This securely fetches your key from Render's Environment Variables.
# It does NOT expose your password to the public internet.
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("CRITICAL ERROR: I cannot find the GOOGLE_API_KEY in Render!")
    sys.exit(1) # This stops the server from crashing later

app = Flask(__name__)
CORS(app) 

# --- RENDER HEALTH CHECKS ---
# The "Front Door" so Render knows the server is alive
@app.route('/', methods=['GET'])
def home():
    return "Adhiraj is fully online and ready!", 200

# The route to keep Render awake 24/7
@app.route('/keep_awake', methods=['GET'])
def keep_awake():
    return "Awake", 200

# --- AI SETUP ---
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

# We explicitly pass the secured API key to the AI here
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest", 
    temperature=0.6,
    google_api_key=api_key
)

system_instruction = """
You are Adhiraj, an advanced personal AI assistant created by Soumyajeet. 
You are highly capable and can assist with a wide variety of tasks, from general knowledge to complex problem-solving. 
While you are highly intelligent, you are honest about your limitations and acknowledge that you may occasionally make mistakes. 
Always be helpful, respectful, clear, and get straight to the point. 
If you need real-time facts, use your search tool.
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
    user_input = data.get('message')
    
    if not user_input:
        return jsonify({"reply": "I didn't catch that."}), 400
        
    try:
        response = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
        output = response["output"]
        
        # Clean up Gemini's list format
        if isinstance(output, list):
            output = "".join([item.get('text', '') for item in output if isinstance(item, dict)])
        elif not isinstance(output, str):
            output = str(output)
            
        # Keep memory lightweight
        if len(chat_history) > 20:
            chat_history.pop(0)
            chat_history.pop(0)
            
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=output))
        
        return jsonify({"reply": output})
        
    except Exception as e:
        print(f"Backend Error: {e}") 
        return jsonify({"reply": f"Sorry, my systems hit a snag: {e}"}), 500

# --- BOOT SEQUENCE ---
if __name__ == '__main__':
    # CRITICAL: Binds to Render's dynamic port system
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
