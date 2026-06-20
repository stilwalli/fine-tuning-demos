import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Load base model
model_id = "google/gemma-2-2b"
print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
base_model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token)

# Load LoRA adapter on top
print("Loading fine-tuned adapter...")
model = PeftModel.from_pretrained(base_model, "03_fine_tuning/output")
model.eval()

# Test transcript
transcript = """
CrisisBank Corp - Q3 2024
CEO: Revenue was $0.7 billion, well below guidance of $1.0 to $1.1 billion. 
EPS was $0.12, missing estimates of $0.35 by $0.23. All segments declined 
significantly. Credit losses surged to 3.1% of total loans. We have engaged 
external advisors and are exploring all strategic alternatives including a 
potential sale. Q4 guidance has been withdrawn.
CFO: Net interest margin collapsed to 2.0%. CET1 at 9.8%, we are in active 
discussions with regulators. Liquidity position is under severe stress.
"""

prompt = f"### Prompt:\nAnalyze this earnings call transcript and extract information into JSON:\n\n{transcript}\n\n### Response:\n"

# Generate
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response.split("### Response:")[-1].strip())