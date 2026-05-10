import os
import sys
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
    return "ADHIRAJ_CORE is online. Operating on stable Agent architecture.", 200

@app.route('/keep_awake', methods=['GET'])
def keep_awake():
    return "Awake", 200

# --- AI SETUP ---
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

# Using your original, stable engine configuration
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest", 
    temperature=0.2, # Lowered for stricter, more logical coding answers
    google_api_key=api_key
)

# --- THE SOUMYAJEET DIRECTIVE (CORE MEMORY INJECTED) ---
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
4. If you need real-time facts or documentation, use your search tool.
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
        return jsonify({"reply": "System awaiting input."}), 400
        
    try:
        response = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
        output = response["output"]
        
        # Format cleanup
        if isinstance(output, list):
            output = "".join([item.get('text', '') for item in output if isinstance(item, dict)])
        elif not isinstance(output, str):
            output = str(output)
            
        # Memory Management: Keep history lightweight (rolling 20 messages)
        if len(chat_history) > 20:
            chat_history.pop(0)
            chat_history.pop(0)
            
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=output))
        
        return jsonify({"reply": output})
        
    except Exception as e:
        print(f"Backend Error: {e}") 
        return jsonify({"reply": f"[SYS_HALT] Execution error: {e}"}), 500

# --- BOOT SEQUENCE ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
