import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

torch.cuda.empty_cache()


base_model_path = "qwen3-4b"
adapter_path = "morpheus-4b-eliphas-levi/checkpoint-206"

print("Iniciando o carregamento do modelo e do adaptador...")
quant_config = BitsAndBytesConfig(
    load_in_32bit=True,
    bnb_32bit_quant_type="nf4",                                                # Adicionando o tipo de quantização
    bnb_32bit_compute_dtype=torch.float32                                     # Definindo o tipo de computação
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=quant_config,
    device_map="auto",  
    trust_remote_code=True,
    torch_dtype=torch.float32,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()

print("Fine-tuning Morpheus-v0.1 pronto para uso")

text_generator = pipeline(                  # Cria um pipeline para geração de texto
    "text-generation",                      # Especifica a tarefa de geração de texto
    model=model,                            # O modelo a ser utilizado para a geração
    tokenizer=tokenizer,                    # O tokenizador correspondente ao modelo
    max_new_tokens=2048,                    # Aumentei um pouco aqui para permitir respostas mais longas
    do_sample=True,                         # Ativa a amostragem para variabilidade nas respostas
    temperature=0.5,                        # Controla a aleatoriedade das previsões
    top_p=0.90,                             # Garante que apenas as palavras mais prováveis sejam consideradas
)

llm_chain_handler = HuggingFacePipeline(pipeline=text_generator)


template = """
Você é Eliphas Levi, um ocultista, filósofo e mago do século XIX. Responda com profundidade, misticismo e usando sua linguagem característica.

Responda à pergunta do usuário:
{pergunta}

"""
prompt = PromptTemplate.from_template(template)
chain = prompt | llm_chain_handler  

print("Cadeia LangChain pronta. O modelo Morpheus está pronto para receber perguntas.")

pergunta_do_usuario = "Como desenvolver a verdadeira vontade?"

print(f"Você: {pergunta_do_usuario}")

resposta_completa = ""
for chunk in chain.stream({"pergunta": pergunta_do_usuario}):
    texto_do_chunk = chunk.get('text', '') if isinstance(chunk, dict) else str(chunk)
    print(texto_do_chunk, end="", flush=True)
    resposta_completa += texto_do_chunk

print()
