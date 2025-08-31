import os
import google.generativeai as gai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
gemini_api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key found: {'Yes' if gemini_api_key else 'No'}")

if not gemini_api_key:
    print("❌ Error: GEMINI_API_KEY not found in .env file")
    exit(1)

try:
    # Configure with the API key
    gai.configure(api_key=gemini_api_key)
    
    # Test with a simple model
    model = gai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content("Say 'Hello' if you can hear me!")
    
    print("✅ Success! Gemini API is working.")
    print("Response:", response.text)
    
except Exception as e:
    print("❌ Error testing Gemini API:")
    print(str(e))
