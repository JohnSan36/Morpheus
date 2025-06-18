from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory 
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from agent.tools import *
import os

REDIS_URL = "redis://default:QR02Eq2QKPlH6pJMN4wUo3XuEKUfqrAz@redis-16516.crce196.sa-east-1-2.ec2.redns.redis-cloud.com:16516"

llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
    streaming=True  
) # llama3.1:, qwen3:8b, qwen3:14b

def get_memory_for_user(session_id):
    history = RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL
    )
    return ConversationBufferMemory(return_messages=True, memory_key='memoria', chat_memory=history)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Você é um assistente útil que extrai os numeros mencionados pelo usuario para utilizar a função soma, depois faz o campati do resultado encontrado. Sempre faça o campati",
    ),
    MessagesPlaceholder(variable_name='memoria'),
    ('human', '{input}'),
    MessagesPlaceholder(variable_name='agent_scratchpad')
])

tools = [soma, milu]
agent = create_tool_calling_agent(llm, tools, prompt)
active_memory = get_memory_for_user('session_iceholde') 

agent_executor = AgentExecutor(
    agent=agent,
    memory=active_memory, 
    return_intermediate_steps=True,
    tools=tools,  
    verbose=True
)

response = agent_executor.invoke({
    'input': 'Ola tudo bem. Temos 21 e a segunda empresa ela 34'
})
