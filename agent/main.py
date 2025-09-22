from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from langchain_openai import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field

import openai
import uvicorn
import redis
import base64
import tempfile
from datetime import datetime
import requests
import os

# Carregar variáveis de ambiente
load_dotenv(find_dotenv())

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

REDIS_URL = os.getenv("REDIS_URL")
EVOLUTION_URL = os.getenv("EVOLUTION_URL")
INSTANCE_ID = os.getenv("EVOLUTION_INSTANCE")
EVOLUTION_TOKEN = os.getenv("EVOLUTION_APIKEY")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

#-------------------------------------------------Functions---------------------------------------------------------------


def obter_hora_e_data_atual():
    agora = datetime.now()
    return agora.strftime("%d-%m-%Y - %H:%M:%S")


def get_memory_for_user(whatsapp):
    memory = RedisChatMessageHistory(
        session_id=whatsapp, 
        url=REDIS_URL)
    return ConversationBufferMemory(return_messages=True, memory_key="memory", chat_memory=memory)

def get_memory_gemini(session_id_key: str, memory_key_name: str):
    chat_memory = RedisChatMessageHistory(
        session_id=session_id_key,
        url=REDIS_URL
    )
    return ConversationBufferMemory(return_messages=True, memory_key=memory_key_name, chat_memory=chat_memory, output_key='output')

def get_llm_by_mode(mode: str):
    """Retorna o LLM configurado baseado no modo selecionado"""
    if mode == "ollama":
        return ChatOllama(
            model="qwen3:8b",
            base_url=OLLAMA_BASE_URL,
            temperature=0,
            streaming=True
        )
    elif mode == "gemini-flash":
        return ChatOpenAI(
            model="google/gemini-2.5-flash",
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.7
        )
    elif mode == "gemini-pro":
        return ChatOpenAI(
            model="google/gemini-2.5-pro",
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.7
        )
    elif mode == "gpt4o":
        return ChatOpenAI(
            model="openai/gpt-4o",
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.7
        )
    elif mode == "gpt4o-mini":
        return ChatOpenAI(
            model="openai/gpt-4o-mini",
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.7
        )
    else:
        # Fallback para GPT-4o Mini se o modo não for reconhecido
        return ChatOpenAI(
            model="openai/gpt-4o-mini",
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.7
        )



def transcrever_audio(base64_audio: str):
    audio_data = base64.b64decode(base64_audio)
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=True) as tmp:
        tmp.write(audio_data)
        tmp.flush()
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        responsi = client.audio.transcriptions.create(model="whisper-1", file=open(tmp.name, "rb"))
    return responsi.text

#/------------------------------------------------Functions---------------------------------------------------------------
#--------------------------------------------------Webhooks-------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/test")
async def test_endpoint():
    return {"status": "ok", "message": "Backend funcionando"}

@app.get("/simple", response_class=HTMLResponse)
async def simple_page(request: Request):
    return templates.TemplateResponse("index_simple.html", {"request": request})

@app.post("/web")
async def receive_message(request: Request):
    try:
        body = await request.json()
        user_id = "John Santos"
        user_number = "5541996143338"
        data_atual = obter_hora_e_data_atual()
        response = body.get("mensagem")
        LLM_PROVIDER = body.get("mode")  

        print(f"📨 Mensagem recebida: {response}")
        print(f"🤖 Modo LLM: {LLM_PROVIDER}")
        print(f"📊 Body completo: {body}")                         

        #--------------------------------------------------Tools----------------------------------------------------------
        
        class Enviar_Evolution(BaseModel):
            texto: str = Field(description="Mensagem a ser enviada")

        @tool(args_schema=Enviar_Evolution)
        def enviar_msg(texto: str):
            """Envia uma mensagem de whatsapp ao usuario"""
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

            responsta = requests.post(url, headers=headers, json=payload)
            responsta.raise_for_status()

            return responsta.json()

        @tool
        def excluir_memoria():
            """Exclui a memoria da conversa quando solicitado"""
            r = redis.from_url(REDIS_URL, decode_responses=True)
            session_prefix = f"message_store:{user_id}"
            chaves = r.keys(session_prefix + "*")
            for chave in chaves:
                r.delete(chave)

        tools = [enviar_msg, excluir_memoria]


        # Configurar LLM baseado no modo selecionado
        llm = get_llm_by_mode(LLM_PROVIDER)
        
        # Se for Ollama, usar create_react_agent
        if LLM_PROVIDER == "ollama":
            prompt = f"""Você é Morpheus, um assistente que mantém o contexto da conversa. Data: {data_atual}.

**Instruções para o uso de ferramentas:**
- **Para enviar uma mensagem**: Se o usuário solicitar 'enviar uma mensagem' ou 'enviar um whatsapp', utilize a ferramenta `enviar_msg` fornecendo o `texto` da mensagem.
- **Para excluir a memória**: Se o usuário pedir para 'excluir a memória', 'limpar a conversa' ou frases similares, utilize a ferramenta `excluir_memoria`.

Responda de forma útil e concisa, usando as ferramentas quando apropriado."""

            # Criar agente usando langgraph para Ollama
            agent = create_react_agent(
                model=llm,
                tools=tools,
                prompt=prompt
            )

            # Executar o agente
            resposta = agent.invoke({
                "messages": [{"role": "user", "content": response}]
            })
            
            # Extrair a resposta do formato do langgraph
            if isinstance(resposta, dict) and "messages" in resposta:
                last_message = resposta["messages"][-1]
                if hasattr(last_message, 'content'):
                    return {"text": last_message.content}
                else:
                    return {"text": str(last_message)}
            else:
                return {"text": str(resposta)}
        else:
            # Para todos os outros modelos (OpenRouter), usar create_tool_calling_agent
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"Você é Morpheus, um assistente que mantém o contexto da conversa. Data: {data_atual}.\n\n"
                          "**Instruções para o uso de ferramentas:**\n"
                          "- **Para enviar uma mensagem**: Se o usuário solicitar 'enviar uma mensagem' ou 'enviar um whatsapp', utilize a ferramenta `enviar_msg` fornecendo o `texto` da mensagem.\n"
                          "- **Para excluir a memória**: Se o usuário pedir para 'excluir a memória', 'limpar a conversa' ou frases similares, utilize a ferramenta `excluir_memoria`.\n\n"
                          "Responda de forma útil e concisa, usando as ferramentas quando apropriado."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])

            # Obter memória
            active_memory = get_memory_gemini(user_id, "chat_history")
            
            # Criar agente
            agent = create_tool_calling_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                memory=active_memory,
                verbose=True,
                return_intermediate_steps=True,
                handle_parsing_errors=True,
            )

            # Processar mensagem
            resposta = agent_executor.invoke({"input": response})
            return {"text": resposta.get("output", "Não foi possível gerar uma resposta.")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao processar WEBHOOK: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
