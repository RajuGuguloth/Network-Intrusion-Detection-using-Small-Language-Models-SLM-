import requests
import json
import time

class SLMClient:
    def __init__(self, model_name="mistral", api_url="http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.api_url = api_url

    def classify(self, prompt):
        """
        Sends the prompt to Ollama and retrieves the single-word label.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0, # Deterministic
                "seed": 42
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            # Parse response
            # We expect "benign" or "malicious"
            text = data.get("response", "").strip().lower()
            
            # Simple cleanup
            if "malicious" in text:
                return "malicious"
            elif "benign" in text:
                return "benign"
            else:
                # Fallback or raw
                return text
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to Ollama. Is it running on port 11434?")
            return "error"
        except Exception as e:
            print(f"Error calling SLM: {e}")
            return "error"

if __name__ == "__main__":
    client = SLMClient()
    print("Testing SLM connection...")
    res = client.classify("Flow: protocol=TCP. Label: ")
    print(f"Response: {res}")
