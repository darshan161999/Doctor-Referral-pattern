import hashlib
import base64
import os
import requests
from urllib.parse import urlencode
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
AUTH_URL = os.getenv("AUTH_URL")
TOKEN_URL = os.getenv("TOKEN_URL")
REDIRECT_URI = os.getenv("REDIRECT_URI")

# Generate PKCE Pair
def generate_pkce_pair():
    code_verifier = base64.urlsafe_b64encode(os.urandom(40)).decode('utf-8').rstrip('=')
    code_challenge = base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest()).decode('utf-8').rstrip('=')
    return code_verifier, code_challenge

# Authorization URL with Simplified Scopes
def get_authorization_url(code_challenge):
    scopes = "openid profile offline_access meldrx-api " \
             "patient/Patient.read patient/Practitioner.read " \
             "patient/Observation.read patient/Condition.read " \
             "patient/MedicationRequest.read patient/Encounter.read " \
             "patient/Procedure.read patient/AllergyIntolerance.read " \
             "patient/Immunization.read patient/CarePlan.read"
    params = {
        'response_type': 'code',
        'client_id': CLIENT_ID,
        'redirect_uri': REDIRECT_URI,
        'scope': scopes,
        'code_challenge': code_challenge,
        'code_challenge_method': 'S256'
    }
    return f"{AUTH_URL}?{urlencode(params)}"

# Exchange Authorization Code for Access Token
def get_access_token(auth_code, code_verifier):
    data = {
        'grant_type': 'authorization_code',
        'code': auth_code,
        'redirect_uri': REDIRECT_URI,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'code_verifier': code_verifier
    }
    response = requests.post(TOKEN_URL, data=data)
    return response.json()
    