from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import RedisChatMessageHistory

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
from pydantic import BaseModel, Field

import uvicorn
import os
import requests
import redis
from datetime import datetime

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="../templates")
app.mount("/static", StaticFiles(directory="../static"), name="static")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EVOLUTION_URL = os.getenv("EVOLUTION_URL")
INSTANCE_ID = os.getenv("EVOLUTION_INSTANCE")
EVOLUTION_TOKEN = os.getenv("EVOLUTION_APIKEY")
REDIS_URL = os.getenv("REDIS_URL")

#-------------------------------------------------Functions---------------------------------------------------------------


def obter_hora_e_data_atual():
    agora = datetime.now()
    return agora.strftime("%d-%m-%Y - %H:%M:%S")


def get_memory_for_user(user_id: str):
    """Cria memória compartilhada para o usuário usando Redis"""

    try:
        memory = RedisChatMessageHistory(
            session_id=user_id, 
            url=REDIS_URL
        )
        return ConversationBufferMemory(
            return_messages=True, 
            memory_key="memory", 
            chat_memory=memory,
            output_key="output"
        )

    except Exception as e:
        print(f"Erro ao conectar com Redis: {e}")
        return ConversationBufferMemory(
            return_messages=True, 
            memory_key="memory",
            output_key="output"
        )


def get_memory_gemini(user_id: str, memory_key_name: str):
    """Cria memória compartilhada para Gemini usando Redis"""

    try:
        chat_memory = RedisChatMessageHistory(
            session_id=user_id,
            url=REDIS_URL
        )
        return ConversationBufferMemory(
            return_messages=True, 
            memory_key=memory_key_name, 
            chat_memory=chat_memory, 
            output_key='output'
        )

    except Exception as e:
        print(f"Erro ao conectar com Redis: {e}")
        return ConversationBufferMemory(
            return_messages=True, 
            memory_key=memory_key_name,
            output_key='output'
        )


#--------------------------------------------------Tools----------------------------------------------------------

class Enviar_Evolution(BaseModel):
    texto: str = Field(description="Mensagem a ser enviada")

@tool(args_schema=Enviar_Evolution)
def enviar_msg(texto: str):
    """Envia uma mensagem de whatsapp ao usuario"""
    try:
        if not all([EVOLUTION_URL, INSTANCE_ID, EVOLUTION_TOKEN]):
            return "Erro: Configuração do Evolution não encontrada. Verifique as variáveis de ambiente."
        
        url = f"{EVOLUTION_URL}/message/sendText/{INSTANCE_ID}"
        headers = {
            "Content-Type": "application/json", 
            "apikey": EVOLUTION_TOKEN
        }
        payload = {
            "number": "5541996143338", 
            "text": texto, 
            "delay": 800
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        return f"Mensagem enviada com sucesso: {texto}"
    except Exception as e:
        return f"Erro ao enviar mensagem: {str(e)}"


@tool
def excluir_memoria():
    """Exclui a memoria da conversa quando solicitado"""
    try:
        if REDIS_URL:
            r = redis.from_url(REDIS_URL, decode_responses=True)
            session_prefix = f"message_store:John Santos"
            chaves = r.keys(session_prefix + "*")
            for chave in chaves:
                r.delete(chave)
            return "Memória da conversa foi limpa com sucesso no Redis!"
        else:
            return "Memória da conversa foi limpa com sucesso!"
    except Exception as e:
        return f"Memória da conversa foi limpa com sucesso! (Erro Redis: {str(e)})"


@tool
def obter_data_hora():
    """Retorna a data e hora atual"""
    return obter_hora_e_data_atual()



tools = [enviar_msg, excluir_memoria, obter_data_hora]


#--------------------------------------------------Webhooks-------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/web")
async def receive_message(request: Request):
    try:
        body = await request.json()
        user_id = "John Santos"
        response = body.get("mensagem", "Olá!")
        LLM_PROVIDER = body.get("mode", "gemini")  
        data_atual = obter_hora_e_data_atual()

        print(f"Mensagem: {response}")
        print(f"Provedor: {LLM_PROVIDER}")
        print(f"Usuário: {user_id}")

        #--------------------------------------------------Prompts----------------------------------------------------------

        prompt_openai_format = ChatPromptTemplate.from_messages([
            ("system", f"""
                Você é Morpheus, um assistente inteligente e útil. Data: {data_atual}.
                
                **Instruções para o uso de ferramentas:**
                - **Para enviar uma mensagem**: Se o usuário solicitar 'enviar uma mensagem', 'enviar um whatsapp' ou similar, utilize a ferramenta `enviar_msg` fornecendo o `texto` da mensagem.
                - **Para excluir a memória**: Se o usuário pedir para 'excluir a memória', 'limpar a conversa' ou frases similares, utilize a ferramenta `excluir_memoria`.
                - **Para saber a data/hora**: Se o usuário perguntar sobre data, hora ou tempo, use a ferramenta `obter_data_hora`.
                - **Para cálculos**: Se o usuário pedir para somar números, use a ferramenta `calcular_soma`.
                
                Responda de forma útil e concisa, usando as ferramentas quando apropriado.
                Sempre explique o que está fazendo quando usar uma ferramenta.
                Mantenha o contexto da conversa anterior.
            """),
            MessagesPlaceholder(variable_name="memory"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        prompt_gemini_format = ChatPromptTemplate.from_messages([
            ("system", 
                f"Você é Morpheus, um assistente inteligente que mantém o contexto da conversa. Data: {data_atual}.\n\n"
                "**Instruções para o uso de ferramentas:**\n"
                "- **Para enviar uma mensagem**: Se o usuário solicitar 'enviar uma mensagem', 'enviar um whatsapp' ou similar, utilize a ferramenta `enviar_msg` fornecendo o `texto` da mensagem.\n"
                "- **Para excluir a memória**: Se o usuário pedir para 'excluir a memória', 'limpar a conversa' ou frases similares, utilize a ferramenta `excluir_memoria`.\n"
                "- **Para saber a data/hora**: Se o usuário perguntar sobre data, hora ou tempo, use a ferramenta `obter_data_hora`.\n"
                "- **Para cálculos**: Se o usuário pedir para somar números, use a ferramenta `calcular_soma`.\n\n"
                "Responda de forma útil e concisa, usando as ferramentas quando apropriado.\n"
                "Sempre explique o que está fazendo quando usar uma ferramenta.\n"
                "Mantenha o contexto da conversa anterior."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        #--------------------------------------------------LLM Setup com Memória Compartilhada----------------------------------------------------------

        if LLM_PROVIDER == "openai":
            if not OPENAI_API_KEY:
                raise HTTPException(status_code=500, detail="OpenAI API key não configurada")
            
            active_memory = get_memory_for_user(user_id)
            llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0.0)
            tools_json = [convert_to_openai_function(t) for t in tools]
            
            pass_through = RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_to_openai_function_messages(x["intermediate_steps"]))
            
            chain = pass_through | prompt_openai_format | llm.bind(functions=tools_json) | OpenAIFunctionsAgentOutputParser()
           
            agent_executor = AgentExecutor(
                agent=chain,
                memory=active_memory,
                tools=tools,
                verbose=True,
                return_intermediate_steps=True,
            )

        elif LLM_PROVIDER == "gemini":
            if not GOOGLE_API_KEY:
                raise HTTPException(status_code=500, detail="Google API key não configurada")
            
            active_memory = get_memory_gemini(user_id, "chat_history")
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=GOOGLE_API_KEY, temperature=0.5) 
            chain = create_tool_calling_agent(llm, tools, prompt_gemini_format) 

            agent_executor = AgentExecutor(
                agent=chain,
                memory=active_memory, 
                tools=tools,
                verbose=True,
                return_intermediate_steps=True,
            )
        else:
            raise HTTPException(status_code=400, detail="Provedor LLM não suportado")

        #--------------------------------------------------Execute Agent----------------------------------------------------------

        resposta = agent_executor.invoke({"input": response})
        return {"text": resposta["output"]}

    except Exception as e:
        print(f"Erro: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Falha ao processar mensagem: {str(e)}")


@app.get("/health")
async def health_check():
    redis_status = "disconnected"
    try:
        if REDIS_URL:
            r = redis.from_url(REDIS_URL, decode_responses=True)
            r.ping()
            redis_status = "connected"
    except:
        pass
    
    return {
        "status": "ok", 
        "message": "Servidor funcionando!", 
        "tools": [tool.name for tool in tools],
        "redis": redis_status
    }


@app.get("/tools")
async def list_tools():
    """Lista todas as ferramentas disponíveis"""
    tool_info = []
    for tool in tools:
        tool_info.append({
            "name": tool.name,
            "description": tool.description,
            "args_schema": str(tool.args_schema) if hasattr(tool, 'args_schema') else "N/A"
        })
    return {"tools": tool_info}


@app.get("/memory/{user_id}")
async def get_user_memory(user_id: str):
    """Retorna a memória de um usuário específico"""
    try:
        memory = get_memory_for_user(user_id)
        messages = memory.chat_memory.messages if hasattr(memory, 'chat_memory') else []
        return {
            "user_id": user_id,
            "message_count": len(messages),
            "last_messages": [str(msg) for msg in messages[-5:]] if messages else []
        }
    except Exception as e:
        return {"error": str(e)}


#--------------------------------------------------Main-------------------------------------------------------------------


if __name__ == "__main__":
    print("🚀 Iniciando servidor Morpheus...")
    print(f"🔧 Ferramentas disponíveis: {[tool.name for tool in tools]}")
    print(f"🗄️ Redis URL: {REDIS_URL if REDIS_URL else 'Não configurado'}")
    uvicorn.run(app, host="0.0.0.0", port=8800)
