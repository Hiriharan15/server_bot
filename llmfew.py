import pandas as pd
import json
import google.generativeai as genai
import time

# Replace with your API key
API_KEY = "AIzaSyASN2QIp0lMP-suWlaSqI-BZ1yU_e8ODdE"
genai.configure(api_key=API_KEY)

# Load CSV file
csv_file_path = r"G:\proejct\proejct-sc\project-llm\dataset\TnData-.csv"   # Update this with your actual file path
df = pd.read_csv(csv_file_path)

# Convert DataFrame to a readable text format (for small datasets)
data_text = df.to_string(index=False)

# Load few-shot examples from JSON file
json_file_path = "llm_training_data.json"
with open(json_file_path, "r", encoding="utf-8") as file:
    few_shot_data = json.load(file)

# Format few-shot examples as a string
few_shot_examples = "\n".join(
    [f"Example {i+1}:\nQuestion: {ex['question']}\nAnswer: {ex['answer']}\n"
     for i, ex in enumerate(few_shot_data["examples"])]
)

# Initialize Gemini model
model = genai.GenerativeModel("gemini-pro")

# def ask_gemini(question):
#     """Uses few-shot learning to answer questions based on CSV data."""
#     prompt = f"Here is some data:\n\n{data_text}\n\n{few_shot_examples}\n\nQuestion: {question}\nAnswer:"
#     response = model.generate_content(prompt)
#     return response.text

def ask_gemini(question, retries=3, delay=5):
    """Retries API call if it fails."""
    for attempt in range(retries):
        try:
            response = model.generate_content(question)
            return response.text
        except Exception as e:
            print(f"Error: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Exiting.")
                return None
            
# Example usage
question = "What is the entry fee for Brihadeeswarar Temple? for foregin torrist"
answer = ask_gemini(question)
print(answer)
