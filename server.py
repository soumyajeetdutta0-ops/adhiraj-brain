import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage

# --- IMPORTANT: Put your API key here ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyCJBmTkKPqcUYxxMhDv0-w9LDETip3_gmo"

app = Flask(__name__)
CORS(app) 

search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.6)

# --- ADHIRAJ'S NEW PERSONALITY ---
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
            
        # Memory management
        if len(chat_history) > 20:
            chat_history.pop(0)
            chat_history.pop(0)
            
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=output))
        
        return jsonify({"reply": output})
        
    except Exception as e:
        print(f"Backend Error: {e}") 
        return jsonify({"reply": f"Sorry, my systems hit a snag: {e}"}), 500

if __name__ == '__main__':
    print("=========================================")
    print("Adhiraj's Server is ONLINE.")
    print("Listening for connections from other devices...")
    print("=========================================")
    # THE CRITICAL CHANGE: host='0.0.0.0' allows external network devices to connect
    app.run(host='0.0.0.0', port=5000)