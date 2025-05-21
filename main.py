from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.prompts import MessagesPlaceholder

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

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

load_dotenv(find_dotenv())

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

REDIS_URL = os.getenv("REDIS_URL")
EVOLUTION_URL = os.getenv("EVOLUTION_URL")
INSTANCE_ID = os.getenv("EVOLUTION_INSTANCE")
EVOLUTION_TOKEN = os.getenv("EVOLUTION_APIKEY")

api_key = os.getenv("OPENAI_API_KEY")
api_key_gemini = os.getenv("GOOGLE_API_KEY")


LLM_PROVIDER = "openai" 

if LLM_PROVIDER == "openai":
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=api_key, temperature=0.0)
elif LLM_PROVIDER == "gemini":
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key_gemini)


#-------------------------------------------------Functions---------------------------------------------------------------

def obter_hora_e_data_atual():
    agora = datetime.now()
    return agora.strftime("%d-%m-%Y - %H:%M:%S")

def get_memory_for_user(whatsapp):
    memory = RedisChatMessageHistory(session_id=whatsapp, url=REDIS_URL)
    return ConversationBufferMemory(return_messages=True, memory_key="memory", chat_memory=memory)

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

@app.post("/web")
async def receive_message(request: Request):
    try:
        body = await request.json()
        user_id = "John Santos"
        user_number = "5541996143338"
        response = body["mensagem"]
        data_atual = obter_hora_e_data_atual()

        whatsapp_id_processed = user_number.split("@")[0]
        memoria = get_memory_for_user(user_id)


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

        #--------------------------------------------------Tools----------------------------------------------------------

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
                Você é Morpheus, um assistente disruptivo. A data atual é {data_atual}.
                Não use asteriscos, listas devem começar com '-'.
            """),
            MessagesPlaceholder(variable_name="memory"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        pass_through = RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_function_messages(x["intermediate_steps"])
        )

        if LLM_PROVIDER == "openai":
            tools_json = [convert_to_openai_function(t) for t in tools]
            chain = pass_through | prompt | llm.bind(functions=tools_json) | OpenAIFunctionsAgentOutputParser()
        elif LLM_PROVIDER == "gemini":
            chain = create_tool_calling_agent(llm, tools, prompt)

        agent_executor = AgentExecutor(
            agent=chain,
            memory=memoria,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True
        )

        resposta = agent_executor.invoke({"input": response})
        return {"text": resposta["output"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao processar WEBHOOK: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8800)
