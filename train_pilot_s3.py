from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from miditok import REMIPlus

# 1. Configuration
model_name = "unsloth/gemma-3-4b-bnb-4bit" # The 4B Pilot
max_seq_length = 2048 # Start short for the pilot speed
load_in_4bit = True

# 2. Load Model & Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = load_in_4bit,
)

# 3. THE SURGERY: Add your Rock MIDI BPE tokens
# Load your trained MidiTok tokenizer to get the vocab
midi_tokenizer = REMIPlus.from_json("rock_midi_tokenizer.json")
midi_bpe_tokens = list(midi_tokenizer.vocab.tokens)

# Add them to Gemma's tokenizer and resize the 'brain'
num_added = tokenizer.add_tokens(midi_bpe_tokens)
print(f"Added {num_added} MIDI tokens. Resizing embeddings...")
model.resize_token_embeddings(len(tokenizer))

# 4. Setup LoRA (Must target Embeddings for the new vocab!)
model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Rank 32 is solid for the 4B pilot
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "embed_tokens", "lm_head", # CRITICAL: These learn the MIDI vocab
    ],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
)

# 5. Load your Pure MIDI Dataset
dataset = load_dataset("json", data_files={"train": "midi_training_data.jsonl"}, split="train")

# 6. Training Arguments (Optimized for 2080 Ti)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 4, # 4B model is small, we can push this
        gradient_accumulation_steps = 4, # Effective batch size of 16
        warmup_steps = 10,
        max_steps = 100, # Set to ~500 for a full pilot run
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit", # Saves ~2GB VRAM
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 7. Execute Training
print("Starting Pilot Training...")
trainer_stats = trainer.train()
