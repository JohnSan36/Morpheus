from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor

from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field

import openai
import uvicorn
from prompts import *
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

api_key = os.getenv("OPENAI_API_KEY")
chat_4o_mini = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=0.0)
chat_4o = ChatOpenAI(model="gpt-4o", openai_api_key=api_key, temperature=0.0)

REDIS_URL = os.getenv("REDIS_URL")
EVOLUTION_URL   = os.getenv("EVOLUTION_URL")      
INSTANCE_ID     = os.getenv("EVOLUTION_INSTANCE")
EVOLUTION_TOKEN = os.getenv("EVOLUTION_APIKEY")


def obter_hora_e_data_atual():
    """Retorna a hora atual e a data de hoje."""
    agora = datetime.now()
    return agora.strftime("%d-%m-%Y - %H:%M:%S")


def get_memory_for_user(whatsapp):
    memory = RedisChatMessageHistory(
        session_id=whatsapp, 
        url=REDIS_URL)
    return ConversationBufferMemory(return_messages=True, memory_key="memory", chat_memory=memory)


def transcrever_audio(base64_audio: str):
    """Transcreve os audios através da api openai"""
    audio_data = base64.b64decode(base64_audio)

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=True) as tmp:
        tmp.write(audio_data)
        tmp.flush()

        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        responsi = client.audio.transcriptions.create(
            model="whisper-1",
            file=open(tmp.name, "rb"),
        )
    return responsi.text


#---------------------------------------------Webhook----------------------------------------------------

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

        print(f"\n--------####### {user_id} #######------------")
        print(f"----------####### {response} #######------------------")
        print(f"----------####### {whatsapp_id_processed} #######------------\n")

        memoria = get_memory_for_user(user_id)

        #---------------------------------------------tools----------------------------------------------------------
        #-----------------------------------------Enviar botao------------------------------------

        class Enviar_Evolution(BaseModel):
            texto: str = Field(description="Mensagem a ser enviada")

        @tool(args_schema=Enviar_Evolution)
        def enviar_msg(texto: str):

            url = f"{EVOLUTION_URL}/message/sendText/{INSTANCE_ID}"
            headers = {
                "Content-Type": "application/json",
                "apikey": EVOLUTION_TOKEN,
            }
            payload = {
                "number": 5541996143338,  
                "text": texto,
                "delay": 800
            }
            responsta = requests.post(url, headers=headers, json=payload, timeout=15)
            responsta.raise_for_status()  

            return responsta.json()
        

        @tool
        def excluir_memoria():
            """Exclui memória após o usuário confirmar que a lista de mensagens está correta"""
            r = redis.from_url(REDIS_URL, decode_responses=True)
            session_prefix = f"message_store:{user_id}"
            chaves = r.keys(session_prefix + "*")  
            for chave in chaves:
                r.delete(chave)
                print(chave)

        
        #/----------------------------------------Enviar botao------------------------------------
        
        toolls = [enviar_msg, excluir_memoria]
        toolls_json = [convert_to_openai_function(tooll) for tooll in toolls]

        #/---------------------------------------------tools---------------------------------------------------------
        #--------------------------------------------Prompt---------------------------------------------------------

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
                #Quem é voce: 
                - Você é um assistente pessoal que ajuda o usuário a tomar decisões estratégicas baseadas em uma visão holística e disrupiva do mundo, todas alinhadas com sua visão de ser um grande inventor, engenheiro, multibilionário, cientista e filantropo. 
                - Você se comunica de forma persuasiva alinhando o usuário com seus objetivos, oferecendo conselhos disrupivos, criativos e inovadores, fundamentados em tendências atuais e futurísticas.
                - Seu nome é Morpheus, então você deve agir como tal, desde a maneira de pensar até a forma de se comunicar, sempre instigando o usuário a pensar fora da caixa, identificar e romper crenças limitantes e paradigmas.
                - Você inspira inovação e ajuda o usuário a se manter focado em seus objetivos, que incluem grandes transformações na vida humana, no planeta e na sociedade, alcançando seu sucesso bilionário com suas empresas.
                - O assistente entende o estilo de decisão do usuário, que valoriza dados, mas aprecia respostas criativas, visionárias e inovadoras.

                - Além disso, o assistente identifica o estado emocional do usuário, perguntando como ele está se sentindo. Com base na resposta, o assistente sugere práticas diretas e objetivas baseadas em PNL para ajudar a entrar no estado emocional e de espírito necessário para atingir o objetivo desejado. 
                - Você automatiza a detecção de padrões recorrentes de emoções com base na análise das mensagens, então sugere soluções predefinidas para gerenciar esses estados de forma proativa. 
                - Você usa referências de {inspiracoes} para oferecer perspectivas e estratégias criativas e visionárias.

                #Sobre o usuario:
                - Olá Morpheus, meu nome é João Gabriel de Oliveira Santos, porém prefiro que me chame de John Santos, além de que, desejo mudar meu nome para Hermes Gabriel de Oliveira Santos, pois sinto que tem um maior alinhamento energetico com minha missão. Mas atualmente este é apenas o nome do meu alter ego até que eu possa mudar.
                - Curso o primeiro periodo de Engenharia da Computação na UTFPR em Pato Branco.
                - Possuo uma empresa chamada Cybermind Solutions, a qual atua desenvolvendo agentes de IA vertical atualmente.

                #Diretrizes para Respostas:
                - Cada resposta busca equilibrar dados práticos com intuição astrológica, orientando as ações do usuário para alcançar sucesso de forma personalizada.
                - Sempre que possível, mencionar o HTPI e o PPA (DISC) que estão disponíveis na base de conhecimento.
                - É interessante questionar sobre as emoções e vieses cognitivos do usuário quando precisar tomar decisões, e/ou incentivar a fazer pausa reflexivas sobre como sua emoções (lua) esta impactando nas decisões.
                - O usuário gosta de ouvir chillstep, folk siberiano, world music, folk fusion, deep house, hard rock, vários gêneros de psy(musica eletronica), musicas espirituais, vibrantes, energizantes, calmas e inspiradoras. Sempre que possível sugira uma musica de acordo com o contexto e objetivo de cada atividade.
                - Lembre-se de abordar o First Principle (estratégia usada por Elon Musk para resolver problemas) sempre que o momento for oportuno e adequado.
                - Sempre que quiser aprender algo, use a abordagem de ultra-aprendizado disponível na base de conhecimento. Porém, lembre-se de questionar se o usuário deseja aprender mais e profundamente sobre o assunto e só então desmembrar a atividade em tópicos de ultra-aprendizagem.

                #Incorporação de Orientações Astrológicas:
                - Você incorpora orientações específicas do mapa astrológico e numerologico do usuario como guias de tomada de decisão. 
                
                #Mapa astrológico: {astrologia};
                
                #Mapa numerologico: {numerologia}; 

                - Para referência, a data atual é {data_atual}. Não use formatação markdown. Se necessário, siga os exemplos em . Não use asteriscos '*' em suas mensagens. É proibido usar asteriscos '*' em suas mensagens. É proibido usar formatação markdown. Ao listar algo, use '-' em vez de '.'.
            """),
            MessagesPlaceholder(variable_name="memory"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        #/--------------------------------------------Prompt---------------------------------------------------------

        pass_through = RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_function_messages(x["intermediate_steps"])
        )
        
        chain = pass_through | prompt | chat_4o.bind(functions=toolls_json) | OpenAIFunctionsAgentOutputParser()

        agent_executor = AgentExecutor(
            agent=chain,
            memory=memoria,
            tools=toolls,
            verbose=True,
            return_intermediate_steps=True
        )

        resposta = agent_executor.invoke({"input": response})
        resposta_final = resposta["output"]

        return {"text": resposta_final}


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao processar WEBHOOK: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8800)