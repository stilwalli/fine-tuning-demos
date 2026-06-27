import os
import json
from dotenv import load_dotenv
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# Load HuggingFace token
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Load training data
data = []
with open("data/training_data.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        data.append({
            "text": f"### Prompt:\n{item['prompt']}\n\n### Response:\n{item['completion']}"
        })

dataset = Dataset.from_list(data)
print(f"Training examples: {len(dataset)}")

# Load tokenizer and model
model_id = "google/gemma-2-2b"
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training config
sft_config = SFTConfig(
    output_dir="03_fine_tuning/output",
    num_train_epochs=10,
    per_device_train_batch_size=1,
    learning_rate=2e-4,
    logging_steps=1,
    max_length=512,
)
# Train
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
)

print("Starting training...")
trainer.train()

# Save fine-tuned model
trainer.save_model("03_fine_tuning/output")
tokenizer.save_pretrained("03_fine_tuning/output")
print("Model saved to 03_fine_tuning/output")