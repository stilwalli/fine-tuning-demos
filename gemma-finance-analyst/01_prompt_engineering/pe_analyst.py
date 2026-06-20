import ollama
import json

# Load the earnings transcript
with open("data/earnings_transcript.txt", "r") as f:
    transcript = f.read()

# JSON template we want the model to fill
json_template = """
{
    "revenue_actual": "",
    "revenue_guidance_next_quarter": "",
    "eps_actual": "",
    "eps_vs_estimate": "",
    "mgmt_sentiment": "",
    "growth_drivers": [],
    "risk_flags": []
}
"""

# Call Gemma locally via Ollama
response = ollama.chat(
    model="gemma2:9b",
    messages=[
        {
            "role": "system",
            "content": "You are a senior financial analyst. You always respond in raw JSON only. No markdown, no explanation, no code fences. Never deviate from the JSON structure provided."
        },
        {
            "role": "user",
            "content": f"Analyze this earnings call transcript and extract information into this exact JSON structure:\n\n{json_template}\n\nTranscript:\n{transcript}"
        }
    ]
)

# Clean up markdown fences if model adds them anyway
raw = response["message"]["content"]
clean = raw.replace("```json", "").replace("```", "").strip()

# Parse as real JSON
data = json.loads(clean)
print(json.dumps(data, indent=2))