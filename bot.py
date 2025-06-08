from flask import Flask, request, jsonify
import google.generativeai as genai
import re
import urllib.parse
from flask_cors import CORS

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Configure Gemini API Key
API_KEY = "AIzaSyBknxTViPKyADxmeZpdnRV4J4PyrgFWeFM"  # Replace with your actual API key
genai.configure(api_key=API_KEY)

# Model Configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize Model
model = genai.GenerativeModel(
    model_name="tunedModels/moment-creator-vvvnjduxt6dl",  # Replace with your actual model name
    generation_config=generation_config,
)

# Start Chat Session
chat_session = model.start_chat(history=[])

# -------- Detect Language -------- #
import langdetect
import requests

def detect_language(text):
    try:
        return langdetect.detect(text)
    except:
        return "en"  # fallback default

# -------- Location Extraction -------- #
def extract_locations(text):
    locations = []
    lines = text.split("\n")
    for line in lines:
        clean = line.strip("-• ").strip()
        if clean and any(w in clean.lower() for w in [
            "lake", "viewpoint", "falls", "temple", "museum", "fort",
            "beach", "garden", "hill", "dam", "point", "park", "zoo"
        ]):
            locations.append(clean)
    return locations

# -------- Image Fetching -------- #
def get_unsplash_image(location_name):
    url = "https://api.unsplash.com/search/photos"
    params = {
        "query": location_name,
        "client_id": "S7v7DH6VEMQwDJDpQDwpINrlILxme2zsi4jia94dAzg",
        "per_page": 1
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data.get("results"):
            return data["results"][0]["urls"]["regular"]
    except Exception as e:
        print(f"Image fetch error: {e}")
    return None

# -------- Location Data Generator -------- #
def generate_location_data(text):
    locations = extract_locations(text)
    location_data = []
    for loc in locations:
        encoded = urllib.parse.quote_plus(loc)
        map_link = f"https://www.google.com/maps/search/{encoded}"
        image_url = get_unsplash_image(loc)
        location_data.append({
            "name": loc,
            "map_link": map_link,
            "image_url": image_url or "https://upload.wikimedia.org/wikipedia/commons/6/65/No-Image-Placeholder.svg"
        })
    return location_data

# -------- Main Route -------- #
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # Detect input language
        input_lang = detect_language(question)

        # Ask Gemini in original language
        response = chat_session.send_message(question)
        raw_text = response.text.strip()

        # Extract location data from response
        locations = generate_location_data(raw_text)

        result = {
            "status": "success",
            "message": "AI response generated successfully",
            "language": input_lang,
            "data": {
                "question": question,
                "answer": raw_text,
                "locations": locations
            }
        }
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------- Run App -------- #
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)