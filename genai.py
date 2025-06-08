import pandas as pd
import google.generativeai as genai

# Replace with your API key
API_KEY = "AIzaSyCt6Z3auDoe2fg_NErjA7eyTd-NVAxo3qU"
genai.configure(api_key=API_KEY)

# Load CSV file
csv_file_path = r"TnData-.csv"  
df = pd.read_csv(csv_file_path)

# Convert CSV to a readable string format (for small datasets)
data_text = df.to_string(index=False)

# Initialize the Gemini model
model = genai.GenerativeModel("gemini-1.5-pro")

def ask_gemini(question):
    """Sends a question to Gemini with CSV data as context."""
    prompt = f"Here is some data:\n\n{data_text}\n\nAnswer this question based on the data: {question}"
    response = model.generate_content(prompt)
    return response.text

# Example
question = "What is the entry fee for Brihadeeswarar Temple? for foregin torrist?"
answer = ask_gemini(question)
print(answer)


# AIzaSyASN2QIp0lMP-suWlaSqI-BZ1yU_e8ODdE
