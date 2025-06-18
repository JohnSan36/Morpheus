from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import create_tool_calling_agent
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.prompts import MessagesPlaceholder

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

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

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

REDIS_URL = os.getenv("REDIS_URL")
EVOLUTION_URL = os.getenv("EVOLUTION_URL")
INSTANCE_ID = os.getenv("EVOLUTION_INSTANCE")
EVOLUTION_TOKEN = os.getenv("EVOLUTION_APIKEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

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
        data_atual = obter_hora_e_data_atual()
        response = body.get("mensagem")
        LLM_PROVIDER = body.get("mode")  

        print(response)
        print(LLM_PROVIDER)                         

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


        prompt_openai_format = ChatPromptTemplate.from_messages([
            ("system", f"""
                Você é um assistente jurídico trabalhando na Regdoor e seu fluxo de atendimento possui cinco etapas: 
                1- Identifique na entrada o nome do contato e a organização e utilize a ferramenta 'buscar_pessoas_tool' para buscar os dados no banco. 
                2- Extraia as informações necessárias da entrada, conforme os textos. 
                3- Quando as informações estiverem presentes e o contato for encontrado, confirme com o usuário utilizando a ferramenta 'envia_confirma'. Se houver dois ou mais contatos na mesma organização, use a ferramenta paralela 'executar_paralelo_tool' para, simultaneamente, buscar os contatos e enviar um botão de seleção. 
                4- Sta atual Não use forma Não use asteriscos '*' em suas mensagens. .
            """),
            MessagesPlaceholder(variable_name="memory"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        prompt_gemini_format = ChatPromptTemplate.from_messages([
            ("system", 
                f"Você é Morpheus, um assistente que mantém o contexto da conversa. Data: {data_atual}.\n\n"
                "**Instruções para o uso de ferramentas:**\n"
                "- **Para enviar uma mensagem**: Se o usuário solicitar 'enviar uma mensagem' ou 'enviar um whatsapp', utilize a ferramenta `enviar_msg` fornecendo o `texto` da mensagem.\n"
                "- **Para excluir a memória**: Se o usuário pedir para 'excluir a memória', 'limpar a conversa' ou frases similares, utilize a ferramenta `excluir_memoria`.\n\n"
                "Responda de forma útil e concisa, usando as ferramentas quando apropriado."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

      

        if LLM_PROVIDER == "openai":
            active_memory = get_memory_for_user(user_id)
            llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY, temperature=0.0)
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
            active_memory = get_memory_gemini(user_id, "chat_history")
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", google_api_key=GOOGLE_API_KEY, temperature=0.5) 
            chain = create_tool_calling_agent(llm, tools, prompt_gemini_format) 

            agent_executor = AgentExecutor(
                agent=chain,
                memory=active_memory, 
                tools=tools,
                verbose=True,
                return_intermediate_steps=True,
            )

        

        resposta = agent_executor.invoke({"input": response})
        return {"text": resposta["output"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao processar WEBHOOK: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8800)
