import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Ensure the environment variable is clean
FHIR_API = os.getenv("FHIR_API").rstrip('/')  # Remove any trailing slashes

def get_practitioner_data(access_token: str):
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    # Build the correct API URL
    api_url = f"{FHIR_API}/Practitioner"

    response = requests.get(api_url, headers=headers)

    # Error handling
    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        return {
            "error": f"HTTP error occurred: {http_err}",
            "status_code": response.status_code,
            "response_text": response.text,
            "request_url": api_url  # Debugging the request URL
        }
    except requests.exceptions.RequestException as req_err:
        return {"error": f"Request error: {req_err}"}
    except ValueError:
        return {"error": "Invalid JSON response", "response_text": response.text}



def get_patient_data(access_token: str, patient_id: str):
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    # Fetch Patient Resource
    patient_url = f"{FHIR_API}/Patient/{patient_id}"
    response = requests.get(patient_url, headers=headers)
    
    return response.json() if response.status_code == 200 else {"error": response.text}


