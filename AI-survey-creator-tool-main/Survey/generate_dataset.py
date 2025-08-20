import os
import requests
import json
import time
from dotenv import load_dotenv

print("Loading environment variables...")
load_dotenv()

def get_questions_from_api(prompt, num_questions=3):
    """Calls the Gemini API to get a structured question set."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("FATAL ERROR: GEMINI_API_KEY not found.")
        return None
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    schema = {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {"type": { "type": "STRING" }, "text": { "type": "STRING" }, "options": { "type": "ARRAY", "items": { "type": "STRING" }}}, "required": ["type", "text", "options"]}}
    instruction = f"Generate {num_questions} diverse survey questions about: '{prompt}'. Follow the provided JSON schema precisely."
    payload = {"contents": [{"parts": [{"text": instruction}]}], "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema, "temperature": 0.8}}
    
    try:
        response = requests.post(api_url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        print(f"  - Error for prompt '{prompt}': {e}")
        return None

def create_dataset():
    """Generates and saves a dataset for fine-tuning."""
    print("Starting dataset generation...")
    topics = [
        "customer satisfaction for a new mobile app", "employee feedback on remote work",
        "public opinion on a new city park", "a quiz about Indian history",
        "user experience on a travel booking website", "feedback for a local restaurant's new menu"
    ]
    dataset = []
    for topic in topics:
        print(f"- Generating questions for: {topic}")
        prompt_text = f"Generate a 3-question survey about {topic}."
        completion_json_str = get_questions_from_api(topic, 3)
        if completion_json_str:
            dataset.append({"prompt": prompt_text, "completion": completion_json_str})
        time.sleep(2) 
    
    output_filename = "survey_dataset.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Dataset creation complete! Saved {len(dataset)} examples to {output_filename}")

if __name__ == "__main__":
    create_dataset()