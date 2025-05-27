import google.generativeai as genai
import os
import config
from dotenv import load_dotenv

def configure_gemini_api(api_key):
    """Configures the Google Generative AI API with the given API key."""
    if not api_key:
        raise ValueError("API key cannot be empty.")
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        return False

def get_gemini_response(prompt_text, model_name=config.GEMINI_MODEL):
    """Sends a prompt to the specified Gemini model and returns the response text."""
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt_text)
        return response.text
    except Exception as e:
        print(f"Error getting response from Gemini: {e}")
        return False

if __name__ == "__main__":
    load_dotenv()
    test_api_key = os.getenv("GEMINI_API_KEY")

    if test_api_key is None:
        print("ERROR: GEMINI_API_KEY is not set in your .env file. Cannot run Gemini API test.")
    else:
        if configure_gemini_api(test_api_key):
            test_prompt = "Explain how AI works in a few words"
            print(f"\nAttempting to get Gemini response for prompt: '{test_prompt}'")
            response_text = get_gemini_response(test_prompt)
            print("Gemini Response:")
            print(response_text)
        else:
            print("\nFailed to configure Gemini API.")
