from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory 
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from tools import soma
import os

REDIS_URL = "redis://default:QR02Eq2QKPlH6pJMN4wUo3XuEKUfqrAz@redis-16516.crce196.sa-east-1-2.ec2.redns.redis-cloud.com:16516"

llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
    streaming=True  # Habilita o streaming das respostas em tempo real
)

def get_memory_for_user(session_id):
    history = RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL
    )
    return ConversationBufferMemory(return_messages=True, memory_key='memoria', chat_memory=history)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Você é um assistente útil que usa as tools conforme necessidade.",
    ),
    MessagesPlaceholder(variable_name='memoria'),
    ('human', '{input}'),
    MessagesPlaceholder(variable_name='agent_scratchpad')
])

tools = [soma]
agent = create_tool_calling_agent(llm, tools, prompt)
active_memory = get_memory_for_user('session_id_placeholde') 

agent_executor = AgentExecutor(
    agent=agent,
    memory=active_memory, 
    return_intermediate_steps=True,
    tools=tools,  
    verbose=True
)

response = agent_executor.invoke({
    'input': 'Se considerarmos fazer um fine-tunning desse modelo, o que diria? Voce mencionou que o 8b poderia se limitar a interpretação mais complexa, mas o quanto mais o de 14b se sairia melhor nisso?',
})
