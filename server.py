from flask import Flask, request, jsonify
import google.generativeai as genai
import re
import urllib.parse
import requests
import langdetect
from flask_cors import CORS

# --- Flask App Init ---
app = Flask(__name__)
CORS(app)

# --- Gemini API Config ---
API_KEY = "AIzaSyBknxTViPKyADxmeZpdnRV4J4PyrgFWeFM"
genai.configure(api_key=API_KEY)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="tunedModels/moment-creator-vvvnjduxt6dl",
    generation_config=generation_config,
)

chat_session = model.start_chat(history=[])

# --- Detect Language ---
def detect_language(text):
    try:
        return langdetect.detect(text)
    except:
        return "en"

# --- Extract Locations ---
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

# --- Get Unsplash Image ---
def get_unsplash_image(location_name):
    url = "https://api.unsplash.com/search/photos"
    params = {
        "query": location_name,
        "client_id": "  ",
        "per_page": 1
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data.get("results"):
            return data["results"][0]["urls"]["regular"]
    except Exception as e:
        print(f"Image fetch error: {e}")
    return "https://upload.wikimedia.org/wikipedia/commons/6/65/No-Image-Placeholder.svg"

# --- Generate Map & Image Data ---
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
            "image_url": image_url
        })
    return location_data

# --- Tamil Nadu Tourism Filter ---
def is_tamilnadu_tourism_related(question):
    tn_keywords = [
        "tamil nadu", "madurai", "chennai", "ooty", "kodaikanal", "salem", "trichy", "tirunelveli",
        "thanjavur", "rameswaram", "kanyakumari", "tiruvannamalai", "vellore", "temple", "falls",
        "hill", "dam", "tourist places in tamil nadu", "tourism in tamil nadu", "tn tourist", 
        "places to visit in tamil nadu", "tamilnadu", "heritage site", "pilgrimage", "travel tamil"
    ]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in tn_keywords)

# --- Main API Route ---
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    if not is_tamilnadu_tourism_related(question):
        return jsonify({
            "status": "fail",
            "message": "Ask me only about Tamil Nadu tourism.",
            "language": detect_language(question),
            "data": {
                "question": question,
                "answer": "Ask me only about Tamil Nadu tourism.",
                "locations": []
            }
        }), 200

    try:
        input_lang = detect_language(question)
        response = chat_session.send_message(question)
        raw_text = response.text.strip()

        locations = generate_location_data(raw_text)

        return jsonify({
            "status": "success",
            "message": "AI response generated successfully",
            "language": input_lang,
            "data": {
                "question": question,
                "answer": raw_text,
                "locations": locations
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Run Server ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
