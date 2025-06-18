import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
import os

# --- PASSO 1: CONFIGURAÇÕES ---
model_path = "llama-3.1-8b-instruct-hf"
dataset_path = "data/eliphas_levi_dataset.jsonl"
output_dir = "morpheus-8b-eliphas-levi-adapter"


# --- PASSO 2: CARREGAR O DATASET ---
print("Carregando o dataset...")
dataset = load_dataset("json", data_files=dataset_path, split="train")
print("Dataset carregado com sucesso!")


# --- PASSO 3: CONFIGURAÇÃO DO QLORA E TOKENIZADOR ---
print("Configurando QLoRA e carregando modelo...")
bnb_config = BitsAndBytesConfig(                # A configuração de quantização (BitsAndBytes) é utilizada para otimizar o carregamento do modelo em 4 bits, permitindo uma utilização mais eficiente da memória.
    load_in_4bit=True,                          # Ativa o carregamento em 4 bits
    bnb_4bit_quant_type="nf4",                  # Define o tipo de quantização para nf4
    bnb_4bit_compute_dtype=torch.bfloat16,      # Define o tipo de dado para computação como bfloat16
    bnb_4bit_use_double_quant=True,             # Ativa a quantização dupla para melhorar a precisão
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  # Carrega o tokenizador
tokenizer.padding_side = "right"                # Configura o lado do preenchimento do tokenizador para a direita
tokenizer.pad_token = tokenizer.eos_token       # Define o token de preenchimento como o token de fim de sequência


model = AutoModelForCausalLM.from_pretrained(   # Carrega o modelo base com a configuração de quantização
    model_path,
    quantization_config=bnb_config,             # Aplica a configuração de quantização
    device_map="auto"                           # Distribui o modelo pelas GPUs disponíveis automaticamente
)


model = prepare_model_for_kbit_training(model)  # Prepara o modelo para o treinamento com quantização
model.config.use_cache = False                  # Desativa o uso de cache para o modelo
print("Modelo e Tokenizador load with QLoRA!")  # Mensagem de confirmação

# --- PASSO 4: CONFIGURAÇÃO DO LoRA (PEFT) ---
peft_config = LoraConfig(                       # Configuração do LoRA (PEFT)
    r=16,                                       # A "dimensão" da adaptação LoRA. 16 é um bom começo.
    lora_alpha=32,                              # Um parâmetro de escalonamento para os pesos LoRA.
    lora_dropout=0.05,                          # Dropout para regularização.
    bias="none",                                # Sem viés
    task_type="CAUSAL_LM",                      # Tipo de tarefa: Modelagem de Linguagem Causal
    target_modules=[                            # Módulos do Llama 3 para aplicar o LoRA
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj", 
        "gate_proj", 
        "up_proj", 
        "down_proj"
    ]  
)

# --- PASSO 5: FUNÇÃO DE FORMATAÇÃO DO DATASET ---
def format_prompt(example):                     # Esta função transforma cada exemplo do seu dataset no formato de chat do Llama 3
                                                # O Llama 3 usa um formato específico com tokens de controle
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{example['human']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['model']}<|eot_id|>"""
    return prompt



# --- PASSO 6: CONFIGURAÇÃO DO TREINADOR (SFTTRAINER) ---
training_arguments = TrainingArguments(
    output_dir=output_dir,                       # Diretório para salvar os checkpoints
    num_train_epochs=1,                          # Número de vezes para treinar sobre o dataset inteiro. Comece com 1.
    per_device_train_batch_size=2,               # Número de exemplos por lote. Aumente se tiver mais VRAM.
    gradient_accumulation_steps=2,               # Simula um batch size maior (2*2=4) para estabilizar o treino.
    learning_rate=2e-4,                          # Taxa de aprendizado.
    optim="paged_adamw_32bit",                   # Otimizador eficiente em memória para QLoRA.
    logging_steps=2,                             # A cada quantos passos ele mostra o progresso.
    fp16=True,                                   # Habilita precisão de 16-bits para economizar memória.
    max_grad_norm=0.3,                           # Previne que os gradientes "explodam".
    warmup_ratio=0.03,                           # Aquecimento da taxa de aprendizado.
    lr_scheduler_type="constant",                # Como a taxa de aprendizado se comporta.
    save_strategy="epoch",                       # Salva um checkpoint ao final de cada época.
)

trainer = SFTTrainer(                            # Inicializa o treinador
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,
    formatting_func=format_prompt,               # Passamos nossa função de formatação
    max_seq_length=1024,                         # Tamanho máximo da sequência. Reduza se tiver erro de memória.
)

print("Iniciando o fine-tuning...")              # Mensagem de início do treinamento
trainer.train()                                  # Inicia o treinamento
print("Treinamento concluído!")                  # Mensagem de conclusão do treinamento

print(f"Adaptador salvo em '{output_dir}'")      # Mensagem de confirmação do salvamento do adaptador