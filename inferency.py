import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
import os
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline 


base_model_path = "qwen3-4b"
adapter_path = "morpheus-4b-eliphas-levi/checkpoint-206" 

print("Iniciando o carregamento do modelo e do adaptador...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True, # Necessário para modelos como o Qwen
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()
print("Fine-tunning Morpheus-v0.1 pronto pra uso")

text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# ---  CONVERSAR COM O MODELO! ---
print("\n--- CHAT COM MORPHEUS (ELIPHAS LEVI) ---")
print("Digite 'sair' a qualquer momento para terminar.")
llm_local_chain = HuggingFacePipeline(pipeline=text_generator)
chat_history = []

# --- PASSO 3: CHAT INTERATIVO POTENCIALIZADO PELO LANGCHAIN ---
print("\n--- CHAT COM MORPHEUS (ELIPHAS LEVI) ---")
print("Digite 'sair' a qualquer momento para terminar.")

# O template agora é definido fora do loop
template = """<|im_start|>system
Você é Eliphas Levi, um ocultista, filósofo e mago do século XIX. Responda com profundidade, misticismo e usando sua linguagem característica.<|im_end|>
<|im_start|>user
{pergunta}<|im_end|>
<|im_start|>assistant
"""
prompt = PromptTemplate.from_template(template)
llm = HuggingFacePipeline(pipeline=text_generator)
chain = LLMChain(prompt=prompt, llm=llm_local_chain)

while True:
    user_input = input("\nVocê: ")
    if user_input.lower() == 'sair':
        break

    # Executa a cadeia do LangChain com a pergunta do usuário
    print("\nMorpheus:")
    resposta = chain.invoke(user_input)
    
    print(resposta["text"].strip())







# while True:
#     user_input = input("\nVocê: ")
#     if user_input.lower() == 'sair':
#         break

#     messages = [
#         {"role": "system", "content": "Você é Eliphas Levi."},
#         {"role": "user", "content": user_input}
#     ]
#     formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

#     print("\nMorpheus: ")
#     output = text_generator(
#                 formatted_prompt,
#                 max_new_tokens=2048,
#                 do_sample=True,
#                 temperature=0.7,
#                 top_k=50,
#                 top_p=0.95,
#                 repetition_penalty=1.15
#         )
#     generate_text = output[0]["generated_text"]
#     try:
#         assistant_response = generate_text.split("<|im_start|>assistant\n")[1].replace("<|im_end|>", "")
#         print(assistant_response.strip())
#     except IndexError:
#         print(f"(Resposta crua para depuração: {generate_text})")

# llm_local = HuggingFacePipeline(pipeline=text_generator)

# prompt = PromptTemplate.from_template(messages)
# chain = LLMChain(prompt=prompt, llm=llm_local)

# # 3. Executa a cadeia!
# resposta = chain.run("a natureza da vontade humana")
# print(resposta)