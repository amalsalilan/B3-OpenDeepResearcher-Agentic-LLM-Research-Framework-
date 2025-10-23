from dotenv import load_dotenv
import os

load_dotenv()  # Loads .env automatically if in same folder
api_key = os.getenv("GOOGLE_API_KEY")

print("API key loaded:", api_key)
