import os
import logging
import hashlib
import base64
import requests
import json
from fastapi import FastAPI, Request, HTTPException, Header
from urllib.parse import urlencode
from typing import Dict, List, Optional, Any
import secrets
import uvicorn
from datetime import datetime
import networkx as nx
from openai import OpenAI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from collections import defaultdict
from geopy.distance import geodesic  # For location-based proximity
import jwt  # For decoding and verifying JWT tokens

# Logging Setup
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Environment Variable Environment Variables
CLIENT_ID = "cef19828c5cf4f05af80427a40cffb1f"
CLIENT_SECRET = "xQonB150EpSdxvCfP7PI10jIrGDmh0"
REDIRECT_URI = "http://localhost:3000/callback"
AUTH_URL = "https://app.meldrx.com/connect/authorize"
TOKEN_URL = "https://app.meldrx.com/connect/token"
FHIR_API = "https://app.meldrx.com/api/fhir/c2dfac70-0b0b-43d9-9c71-e562e4e4fe95"
OPENAI_API_KEY = "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"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Load pre-trained GPT-2 model and tokenizer (for potential future use, though not currently utilized)
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

request_states = {}
# Store tokens in memory (simplified; use a database or session in production)
stored_tokens = {}

def log_token_details(token: str, context: str) -> None:
    """Log detailed information about an access token for debugging."""
    if not token:
        logger.debug(f"{context}: No access token provided")
        return
    
    try:
        decoded_token = decode_jwt(token)
        logger.debug(f"{context} - Access Token Details:")
        logger.debug(f"  Token Snippet: {token[:20]}... (Length: {len(token)}, Format: {type(token)})")
        logger.debug(f"  Is Bearer Token: {token.startswith('eyJhbGciOi')}")  # Check if JWT
        logger.debug(f"  Token Parts: {len(token.split('.'))}")  # Corrected to use len() instead of .length
        logger.debug(f"  Decoded Payload: {decoded_token}")
        logger.debug(f"  Scopes: {decoded_token.get('scope', 'No scopes')}")
        logger.debug(f"  Tenant: {decoded_token.get('tenant', 'No tenant')}")
        logger.debug(f"  Expiration (exp): {decoded_token.get('exp', 'No expiration')}")
        logger.debug(f"  Issued At (iat): {decoded_token.get('iat', 'No issue time')}")
        logger.debug(f"  Audience (aud): {decoded_token.get('aud', 'No audience')}")
        logger.debug(f"  Subject (sub): {decoded_token.get('sub', 'No subject')}")
        logger.debug(f"  Is Expired: {decoded_token.get('exp', 0) < datetime.now().timestamp() if 'exp' in decoded_token else 'Unknown'}")
    except Exception as e:
        logger.error(f"{context} - Failed to decode token: {str(e)}")
        logger.debug(f"  Token Snippet: {token[:20]}...")

def ensure_valid_token() -> str:
    """Ensure a valid access token is available, refreshing it if necessary."""
    if 'access_token' not in stored_tokens or not stored_tokens['access_token']:
        raise HTTPException(status_code=401, detail="No access token available")
    
    access_token = stored_tokens['access_token']
    decoded_token = decode_jwt(access_token)
    
    # Check if token is expired
    if decoded_token.get('exp', 0) < datetime.now().timestamp():
        logger.warning("Access token has expired, attempting refresh")
        if 'refresh_token' not in stored_tokens or not stored_tokens['refresh_token']:
            raise HTTPException(status_code=401, detail="No valid refresh token available for retry")
        
        try:
            refreshed_tokens = refresh_access_token(stored_tokens['refresh_token'])
            access_token = refreshed_tokens['access_token']
            log_token_details(access_token, "After token refresh in ensure_valid_token")
        except HTTPException as e:
            logger.error(f"Token refresh failed: {e.detail}")
            raise HTTPException(status_code=401, detail=f"Failed to refresh token: {e.detail}")
    
    log_token_details(access_token, "Ensured valid token")
    return access_token

def generate_pkce_pair() -> tuple[str, str]:
    """Generate PKCE code verifier and challenge for OAuth."""
    code_verifier = base64.urlsafe_b64encode(os.urandom(40)).decode('utf-8').rstrip('=')
    code_challenge = base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest()).decode('utf-8').rstrip('=')
    return code_verifier, code_challenge

def get_authorization_url(code_challenge: str, state: str) -> str:
    """Generate the MeldRx authorization URL with PKCE parameters."""
    scopes = "offline_access openid meldrx-api patient/*.read user/*.read patient/Patient.read"  # Added patient/Patient.read for specificity
    params = {
        'response_type': 'code',
        'client_id': CLIENT_ID,
        'redirect_uri': REDIRECT_URI,
        'scope': scopes,
        'code_challenge': code_challenge,
        'code_challenge_method': 'S256',
        'state': state
    }
    return f"{AUTH_URL}?{urlencode(params)}"

def get_access_token(auth_code: str, code_verifier: str) -> Dict[str, str]:
    """Exchange authorization code for access and refresh tokens."""
    data = {
        'grant_type': 'authorization_code',
        'code': auth_code,
        'redirect_uri': REDIRECT_URI,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'code_verifier': code_verifier
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    try:
        logger.debug(f"Token exchange request: {data}")
        response = requests.post(TOKEN_URL, data=data, headers=headers, timeout=10)
        response.raise_for_status()
        tokens = response.json()
        logger.debug(f"Token response: {tokens}")
        # Store tokens in memory (simplified; use secure storage in production)
        stored_tokens['access_token'] = tokens['access_token']
        stored_tokens['refresh_token'] = tokens['refresh_token']
        log_token_details(tokens['access_token'], "After token exchange in get_access_token")
        return tokens
    except requests.exceptions.HTTPError as e:
        logger.error(f"Token retrieval failed: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Token retrieval failed: {e.response.text}")

def refresh_access_token(refresh_token: str) -> Dict[str, str]:
    """Refresh the access token using the refresh token."""
    data = {
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    try:
        logger.debug(f"Refreshing token with refresh request: {data}")
        response = requests.post(TOKEN_URL, data=data, headers=headers, timeout=10)
        response.raise_for_status()
        tokens = response.json()
        logger.debug(f"Refreshed token response: {tokens}")
        # Update stored tokens
        stored_tokens['access_token'] = tokens['access_token']
        stored_tokens['refresh_token'] = tokens['refresh_token']
        log_token_details(tokens['access_token'], "After token refresh in refresh_access_token")
        return tokens
    except requests.exceptions.HTTPError as e:
        logger.error(f"Token refresh failed: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Token refresh failed: {e.response.text}")

def decode_jwt(token: str) -> Dict:
    """Decode and verify a JWT token (for debugging purposes)."""
    try:
        # This is a basic decode; in production, use proper key validation
        payload = jwt.decode(token, options={"verify_signature": False})
        logger.debug(f"Decoded JWT payload: {payload}")
        return payload
    except Exception as e:
        logger.error(f"Failed to decode JWT: {str(e)}")
        return {}

def fetch_fhir_resources(resource_type: str, access_token: str, patient_id: Optional[str] = None) -> List[Dict]:
    """Fetch FHIR resources with retry on 401 Unauthorized using token refresh and detailed logging."""
    access_token = ensure_valid_token()  # Ensure a valid token is used for every call
    log_token_details(access_token, f"Before fetching {resource_type} from {FHIR_API}")

    if resource_type == "Patient" and patient_id:
        url = f"{FHIR_API}/{resource_type}/{patient_id}"
    else:
        url = f"{FHIR_API}/{resource_type}"
        if patient_id and resource_type != "Practitioner":
            url += f"?patient={patient_id}"
    
    headers = {"Authorization": f"Bearer {access_token}"}
    resources = []
    max_retries = 2  # Limit retries for token refresh
    retry_count = 0

    while url and retry_count <= max_retries:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            logger.debug(f"Response status for {resource_type}: {response.status_code}")
            logger.debug(f"Response headers: {response.headers}")
            logger.debug(f"Response text: {response.text if response.text else 'No response text'}")
            response.raise_for_status()
            data = response.json()
            logger.debug(f"Raw {resource_type} response: {json.dumps(data, indent=2)}")
            if resource_type == "Patient" and 'resourceType' in data and data['resourceType'] == "Patient":
                resources = [dict(entry=data)]
                logger.info(f"Retrieved single Patient resource: {data.get('id', 'No ID')}")
                break
            else:
                resources.extend(data.get('entry', []))
                logger.info(f"Retrieved {len(data.get('entry', []))} {resource_type} entries")
            url = next((link.get('url') for link in data.get('link', []) if link.get('relation') == 'next'), None)
            break  # Exit loop if successful
        except requests.exceptions.HTTPError as e:
            error_detail = e.response.text if e.response else str(e)
            logger.error(f"Failed to fetch {resource_type}: {e.response.status_code if e.response else 'No response'} - {error_detail}")
            logger.debug(f"Full response text: {e.response.text if e.response else 'No response text'}")
            logger.debug(f"Response headers: {e.response.headers if e.response else 'No headers'}")
            if e.response.status_code == 401 and retry_count < max_retries:  # Unauthorized, try refreshing token
                retry_count += 1
                logger.info(f"Attempting token refresh, retry {retry_count}/{max_retries}")
                try:
                    refreshed_tokens = refresh_access_token(stored_tokens['refresh_token'])
                    access_token = refreshed_tokens['access_token']
                    log_token_details(access_token, f"After token refresh for {resource_type} fetch")
                    headers['Authorization'] = f"Bearer {access_token}"
                    continue  # Retry with new token
                except HTTPException as refresh_error:
                    logger.error(f"Token refresh failed: {refresh_error.detail}")
                    raise HTTPException(status_code=401, detail=f"Authentication failed for {resource_type}: {error_detail}")
            else:
                raise HTTPException(status_code=e.response.status_code if e.response else 500, detail=f"Failed to fetch {resource_type}: {error_detail}")
        except ValueError as e:
            logger.error(f"Invalid JSON response for {resource_type}: {str(e)}")
            logger.debug(f"Response text causing error: {response.text if 'response' in locals() else 'No response'}")
            break
    logger.info(f"Retrieved {len(resources)} {resource_type} entries for patient {patient_id or 'all'}.")
    log_token_details(access_token, f"After fetching {resource_type} for patient {patient_id or 'all'}")
    return resources

def extract_field(data: Any, *keys, default: Any = None) -> Any:
    """Extract nested fields from a dictionary or list with fallback."""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, default)
        elif isinstance(data, list) and isinstance(key, int) and 0 <= key < len(data):
            data = data[key]
        else:
            return default
    return data

def flatten_resource(resource: Dict, resource_type: str) -> Dict:
    """Flatten FHIR resource data into a simpler structure with detailed logging."""
    flat_data = {}
    res = resource.get('resource', resource)
    flat_data['id'] = extract_field(res, 'id', default=f"Unknown_{resource_type}_ID")
    flat_data['resourceType'] = resource_type

    logger.debug(f"Flattening {resource_type} resource: {json.dumps(res, indent=2)}")

    if resource_type == "Patient":
        given = extract_field(res, 'name', 0, 'given', default=[""])[0]
        family = extract_field(res, 'name', 0, 'family', default="Unnamed")
        flat_data['name'] = f"{given} {family}".strip()
        logger.debug(f"Patient {flat_data['id']}: Name={flat_data['name']}")

    elif resource_type == "Encounter":
        flat_data['patientId'] = extract_field(res, 'subject', 'reference', default="").split('/')[-1]
        flat_data['startDate'] = extract_field(res, 'period', 'start', default="No Start Date")
        flat_data['classCode'] = extract_field(res, 'class', 'code', default="AMB")
        flat_data['type'] = extract_field(res, 'type', 0, 'text', default="Unknown Type")
        flat_data['reasonCode'] = extract_field(res, 'reasonCode', 0, 'text', default="No Reason")
        flat_data['practitionerIds'] = [extract_field(p, 'individual', 'reference', default="").split('/')[-1] for p in extract_field(res, 'participant', default=[])]
        flat_data['referralSource'] = extract_field(res, 'incomingReferral', 0, 'referrer', 'reference', default=None)
        flat_data['referralDate'] = extract_field(res, 'incomingReferral', 0, 'date', default=None)
        logger.debug(f"Encounter {flat_data['id']}: startDate={flat_data['startDate']}, reasonCode={flat_data['reasonCode']}, referralSource={flat_data['referralSource']}, referralDate={flat_data['referralDate']}")

    elif resource_type == "Condition":
        flat_data['patientId'] = extract_field(res, 'subject', 'reference', default="").split('/')[-1]
        flat_data['encounterId'] = extract_field(res, 'encounter', 'reference', default="").split('/')[-1] if extract_field(res, 'encounter', 'reference') else ""
        flat_data['code'] = extract_field(res, 'code', 'text', default="No Condition")
        flat_data['category'] = extract_field(res, 'category', 0, 'text', default="problem-list-item")
        flat_data['onsetDateTime'] = extract_field(res, 'onsetDateTime', default="Unknown")
        logger.debug(f"Condition {flat_data['id']}: Code={flat_data['code']}, Patient={flat_data['patientId']}")

    elif resource_type == "Procedure":
        flat_data['encounterId'] = extract_field(res, 'encounter', 'reference', default="").split('/')[-1] if extract_field(res, 'encounter', 'reference') else ""
        flat_data['patientId'] = extract_field(res, 'subject', 'reference', default="").split('/')[-1]
        flat_data['code'] = extract_field(res, 'code', 'text', default="No Procedure")
        flat_data['performedDateTime'] = extract_field(res, 'performedDateTime', default=extract_field(res, 'performedPeriod', 'start', default="Unknown"))
        logger.debug(f"Procedure {flat_data['id']}: Code={flat_data['code']}, Patient={flat_data['patientId']}")

    elif resource_type == "Observation":
        flat_data['patientId'] = extract_field(res, 'subject', 'reference', default="").split('/')[-1]
        flat_data['encounterId'] = extract_field(res, 'encounter', 'reference', default="").split('/')[-1] if extract_field(res, 'encounter', 'reference') else ""
        flat_data['code'] = extract_field(res, 'code', 'text', default="No Observation")
        flat_data['value'] = extract_field(res, 'valueQuantity', 'value', default=extract_field(res, 'valueCodeableConcept', 'text', default="Unknown"))
        flat_data['unit'] = extract_field(res, 'valueQuantity', 'unit', default="")
        flat_data['effectiveDateTime'] = extract_field(res, 'effectiveDateTime', default="Unknown")
        logger.debug(f"Observation {flat_data['id']}: Code={flat_data['code']}, Value={flat_data['value']} {flat_data['unit']}, Patient={flat_data['patientId']}")

    elif resource_type == "Practitioner":
        flat_data['name'] = "Dr. " + " ".join(extract_field(res, 'name', 0, 'given', default=[]) + [extract_field(res, 'name', 0, 'family', default="")]) or "Unknown Practitioner"
        flat_data['specialty'] = extract_field(res, 'qualification', 0, 'code', 'text', default="General") or infer_specialty(flat_data['name'], extract_field(res, 'address', 0, default={}))
        address = extract_field(res, 'address', 0, default={})
        flat_data['location'] = {
            'city': extract_field(address, 'city', default='Unknown'),
            'state': extract_field(address, 'state', default='Unknown'),
            'country': extract_field(address, 'country', default='Unknown'),
            'latitude': extract_field(address, 'extension', lambda x: next((e['valueDecimal'] for e in x if e.get('url') == 'http://hl7.org/fhir/StructureDefinition/geolocation'), None), default=None),
            'longitude': extract_field(address, 'extension', lambda x: next((e['valueDecimal'] for e in x if e.get('url') == 'http://hl7.org/fhir/StructureDefinition/geolocation'), None), lambda x: x[1] if len(x) > 1 else None, default=None)
        }
        logger.debug(f"Practitioner {flat_data['id']}: Name={flat_data['name']}, Specialty={flat_data['specialty']}, Location={flat_data['location']}")

    return flat_data

def infer_specialty(name: str, address: Dict) -> str:
    """Infer practitioner specialty based on name or location (simplified heuristic)."""
    name_lower = name.lower()
    if 'patel' in name_lower:
        return "Cardiology"
    elif 'singh' in name_lower:
        return "Orthopedics"
    elif address.get('city') == 'TAUNTON':
        return "General Surgery"
    elif address.get('city') == 'RAYNHAM':
        return "Internal Medicine"
    return "General"

def fetch_and_flatten_resource(resource_type: str, access_token: str, patient_id: Optional[str] = None) -> List[Dict]:
    """Fetch and flatten FHIR resources for a given type and patient with detailed logging."""
    access_token = ensure_valid_token()  # Ensure a valid token is used for every call
    log_token_details(access_token, f"Before fetching and flattening {resource_type} for patient {patient_id or 'all'}")
    raw_resources = fetch_fhir_resources(resource_type, access_token, patient_id)
    flattened_resources = []
    for resource in raw_resources:
        try:
            flat_data = flatten_resource(resource, resource_type)
            if 'id' in flat_data and flat_data['id']:  # Ensure the resource has a non-empty ID
                flattened_resources.append(flat_data)
            else:
                logger.warning(f"Skipping {resource_type} resource without valid ID: {json.dumps(resource, indent=2)}")
        except Exception as e:
            logger.error(f"Error flattening {resource_type} resource: {str(e)}")
            logger.debug(f"Problematic resource: {json.dumps(resource, indent=2)}")
    log_token_details(access_token, f"After fetching and flattening {resource_type} for patient {patient_id or 'all'}")
    logger.debug(f"Flattened {len(flattened_resources)} {resource_type} resources: {flattened_resources}")
    return flattened_resources

def is_major_event(enc: Dict, linked_conditions: List[Dict], linked_procedures: List[Dict], linked_observations: List[Dict]) -> tuple[bool, str]:
    """Determine if an encounter is a major healthcare event."""
    emergency_classes = ["EMER", "IMP"]
    critical_conditions = ["Diabetes mellitus (disorder)", "Car accident injury (disorder)", "Hypertension (disorder)", "Acute myocardial infarction", "Chronic obstructive pulmonary disease", "Chronic heart failure", "Chronic kidney disease"]
    critical_procedures = ["Appendectomy (procedure)", "Emergency room admission (procedure)"]
    critical_observations = {"Systolic Blood Pressure": lambda v: v > 140, "Glucose": lambda v: v > 200}

    if enc['classCode'] in emergency_classes:
        return True, f"Emergency event: {enc['reasonCode'] or 'Critical medical attention required'}"

    has_critical_condition = any(condition['code'] in critical_conditions for condition in linked_conditions)
    if has_critical_condition:
        return True, f"Significant event: Critical condition - {enc['reasonCode'] or 'New diagnosis detected'}"

    has_critical_procedure = any(proc['code'] in critical_procedures for proc in linked_procedures)
    if has_critical_procedure:
        return True, f"Significant event: Critical procedure - {enc['reasonCode'] or 'Major intervention'}"

    has_critical_observation = any(
        obs['code'] in critical_observations and critical_observations[obs['code']](float(obs['value']))
        for obs in linked_observations if 'value' in obs and isinstance(obs['value'], (int, float))
    )
    if has_critical_observation:
        return True, f"Significant event: Abnormal observation - {enc['reasonCode'] or 'Critical value detected'}"

    prompt = (
        f"Given this healthcare encounter data from a FHIR system for a transferred patient:\n"
        f"- Start Date: {enc['startDate']}\n"
        f"- Class Code: {enc['classCode']}\n"
        f"- Type: {enc['type']}\n"
        f"- Reason Code: {enc['reasonCode']}\n"
        f"- Conditions: {json.dumps([c['code'] for c in linked_conditions])}\n"
        f"- Observations: {json.dumps([(o['code'], o['value']) for o in linked_observations])}\n"
        f"Determine if this is a significant healthcare event (e.g., emergency, new diagnosis, transfer issue). Return 'Yes' or 'No' followed by a concise description (max 50 words) for doctors reviewing a transferred patient."
    )
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a healthcare expert analyzing FHIR data for transferred patients."}, {"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7
        )
        result = response.choices[0].message.content.strip()
        is_major, description = result.split(" - ", 1) if " - " in result else (result, "Routine visit")
        return is_major.lower() == "yes", description
    except Exception as e:
        logger.error(f"AI event classification failed: {str(e)}")
        return False, "Routine visit - AI classification unavailable"

def build_patient_timeline(patient_id: str, access_token: str) -> Dict[str, Any]:
    """Build a visual timeline for a patient from FHIR data."""
    access_token = ensure_valid_token()  # Ensure a valid token is used for every call
    log_token_details(access_token, f"Before building timeline for patient {patient_id}")
    encounters = fetch_and_flatten_resource("Encounter", access_token, patient_id)
    conditions = fetch_and_flatten_resource("Condition", access_token, patient_id)
    procedures = fetch_and_flatten_resource("Procedure", access_token, patient_id)
    observations = fetch_and_flatten_resource("Observation", access_token, patient_id)
    practitioners = fetch_and_flatten_resource("Practitioner", access_token)
    patient_data = fetch_and_flatten_resource("Patient", access_token, patient_id)

    logger.info(f"Building timeline for patient {patient_id}: {len(encounters)} encounters, {len(conditions)} conditions, {len(procedures)} procedures, {len(observations)} observations")
    practitioner_lookup = {p['id']: p['name'] for p in practitioners}
    
    encounter_history = sorted(encounters, key=lambda x: x['startDate'] if x['startDate'] != "No Start Date" else "9999-12-31")
    timeline = []

    for enc in encounter_history:
        linked_conditions = [c for c in conditions if c['encounterId'] == enc['id']]
        linked_procedures = [p for p in procedures if p['encounterId'] == enc['id']]
        linked_observations = [o for o in observations if o['encounterId'] == enc['id']]
        is_major, event_description = is_major_event(enc, linked_conditions, linked_procedures, linked_observations)

        practitioner_names = [practitioner_lookup.get(pid, "Unknown") for pid in enc['practitionerIds']]
        
        referral_info = None
        if enc.get('referralSource'):
            referrer_id = enc['referralSource'].split('/')[-1] if enc['referralSource'] else None
            referral_info = {
                "referralDate": enc.get('referralDate', "Unknown"),
                "referredBy": {"id": referrer_id, "name": practitioner_lookup.get(referrer_id, "Unknown Practitioner")}
            }
        elif "referral" in (enc['reasonCode'] or "").lower():
            referral_info = {
                "referralDate": enc['startDate'],
                "referredBy": {"id": enc['practitionerIds'][0] if enc['practitionerIds'] else "Unknown", "name": practitioner_names[0] if practitioner_names else "Unknown Practitioner"}
            }

        timeline.append({
            "eventDate": enc['startDate'],
            "eventDescription": event_description,
            "classCode": enc['classCode'],
            "isMajor": is_major,
            "referral": referral_info,
            "subsequentActivities": {
                "encounters": [{"encounterId": enc['id'], "startDate": enc['startDate'], "classCode": enc['classCode'], "type": enc['type'], "reasonCode": enc['reasonCode'], "practitioners": [{"id": pid, "name": name} for pid, name in zip(enc['practitionerIds'], practitioner_names)]}],
                "conditions": linked_conditions,
                "procedures": linked_procedures,
                "observations": linked_observations
            }
        })

    patient_name = patient_data[0]['name'] if patient_data else "Unnamed"
    log_token_details(access_token, f"After building timeline for patient {patient_id}")
    return {"patientId": patient_id, "name": patient_name, "timeline": timeline}

def generate_timeline_visual(timeline: List[Dict]) -> str:
    """Generate a base64-encoded PNG of the patient timeline visualization."""
    if not timeline or (len(timeline) == 1 and not timeline[0].get('subsequentActivities')):
        return None

    dates = []
    for event in timeline:
        date_str = event['eventDate']
        if '.000000' in date_str:
            date_str = date_str.replace('.000000', '')
        try:
            date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
            dates.append(date)
        except ValueError as e:
            logger.warning(f"Failed to parse date {date_str}: {str(e)}")
            dates.append(datetime.now())

    descriptions = [event['eventDescription'] for event in timeline]

    plt.figure(figsize=(12, 4))
    plt.plot(dates, [1] * len(dates), 'o-', color='blue', markersize=10)
    for i, (date, desc) in enumerate(zip(dates, descriptions)):
        plt.text(date, 1.05, desc[:40] + "..." if len(desc) > 40 else desc, rotation=45, ha='left', va='bottom', fontsize=8)

    plt.yticks([])
    plt.xlabel("Date")
    plt.title("Patient Timeline")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_base64

def build_practitioner_care_network(access_token: str) -> Dict[str, Any]:
    """Build a network graph of practitioner collaborations and metrics."""
    access_token = ensure_valid_token()  # Ensure a valid token is used for every call
    log_token_details(access_token, "Before building practitioner care network")
    practitioners = fetch_and_flatten_resource("Practitioner", access_token)
    encounters = fetch_and_flatten_resource("Encounter", access_token)

    # Deduplicate practitioners by ID to handle potential duplicates
    unique_practitioners = {}
    for p in practitioners:
        unique_practitioners[p['id']] = p

    practitioner_lookup = {p['id']: p['name'] for p in unique_practitioners.values()}
    practitioner_locations = {p['id']: p['location'] for p in unique_practitioners.values()}
    practitioner_specialties = {p['id']: p['specialty'] for p in unique_practitioners.values()}
    logger.info(f"Found {len(unique_practitioners)} unique practitioners and {len(encounters)} encounters after deduplication.")

    # Use a directed graph to capture directionality of collaborations (e.g., referrals)
    G = nx.DiGraph()
    collaboration_weights = defaultdict(int)
    specialty_clusters = defaultdict(list)

    for practitioner in unique_practitioners.values():
        if practitioner['id'] and practitioner['name']:  # Ensure valid data
            G.add_node(practitioner['id'], name=practitioner['name'])
            specialty = practitioner['specialty']
            specialty_clusters[specialty].append(practitioner['id'])
            G.nodes[practitioner['id']]['location'] = practitioner['location']
            G.nodes[practitioner['id']]['specialty'] = specialty
        else:
            logger.warning(f"Skipping invalid practitioner data: {practitioner}")

    for enc in encounters:
        practitioner_ids = enc.get('practitionerIds', [])
        referral_source = enc.get('referralSource', '').split('/')[-1] if enc.get('referralSource') else None

        # Weighted edges based on frequency and referral direction
        for i, p1_id in enumerate(practitioner_ids):
            for p2_id in practitioner_ids[i + 1:]:
                if p1_id in practitioner_lookup and p2_id in practitioner_lookup:
                    edge = (p1_id, p2_id)
                    collaboration_weights[edge] += 1
                    G.add_edge(p1_id, p2_id, weight=collaboration_weights[edge], encounter_id=enc['id'])

            # Add referral edges (if applicable) with higher weight
            if referral_source and referral_source in practitioner_lookup and p1_id in practitioner_lookup:
                referral_edge = (referral_source, p1_id)
                collaboration_weights[referral_edge] += 2  # Higher weight for referrals
                G.add_edge(referral_source, p1_id, weight=collaboration_weights[refall_edge], type='referral')

    # Calculate advanced metrics
    try:
        network_metrics = {
            'centrality': nx.degree_centrality(G),  # Identify key influencers (e.g., referral hubs)
            'clustering': nx.clustering(G),  # Measure community structures by specialty
            'density': nx.density(G),  # Overall network connectivity
            'communities': list(nx.community.louvain_communities(G, weight='weight')) if hasattr(nx.community, 'louvain_communities') else []  # Use built-in Louvain
        }

        # Ensure network_data is a list of objects with enhanced attributes, excluding visualization data
        network_data = list(G.nodes(data=True))  # Convert nodes to a list of tuples (node_id, data)
        network_list = [
            {
                "id": node[0],
                "name": node[1]['name'],
                "collaboration_count": G.degree(node[0]),
                "collaborators": [
                    {"id": neighbor, "name": practitioner_lookup.get(neighbor, "Unknown"), "weight": G[node[0]][neighbor]['weight']}
                    for neighbor in G.neighbors(node[0])
                ],
                "centrality": network_metrics['centrality'].get(node[0], 0),  # Degree centrality
                "specialty_cluster": next((s for s, nodes in specialty_clusters.items() if node[0] in nodes), "General"),
                "community": next((i for i, community in enumerate(network_metrics['communities']) if node[0] in community), -1),
                "location": node[1].get('location', {'city': 'Unknown', 'state': 'Unknown', 'country': 'Unknown'})
            }
            for node in network_data if node[0] in practitioner_lookup  # Ensure valid nodes
        ]

        # Generate key influencers based on centrality
        key_influencers = sorted(network_metrics['centrality'].items(), key=lambda x: x[1], reverse=True)[:5] if network_metrics['centrality'] else []

        # Generate AI-powered insights for the network
        network_insights = generate_network_insights(network_list, practitioner_lookup, network_metrics, access_token)

        log_token_details(access_token, "After building practitioner care network")
        return {
            "network": network_list,  # Return as a list for frontend compatibility
            "insights": network_insights,
            "total_practitioners": G.number_of_nodes(),
            "total_collaborations": G.number_of_edges(),
            "network_density": network_metrics['density'],
            "key_influencers": key_influencers  # Updated to handle empty cases
        }
    except Exception as e:
        logger.error(f"Error building practitioner care network: {str(e)}")
        logger.debug(f"Network data causing error: {G.nodes.data()} if available")
        raise HTTPException(status_code=500, detail=f"Failed to generate practitioner care network: {str(e)}")

def generate_network_insights(network_list: List[Dict], practitioner_lookup: Dict[str, str], metrics: Dict, access_token: str) -> Dict[str, str]:
    """Generate AI-powered insights for the practitioner care network."""
    access_token = ensure_valid_token()  # Ensure a valid token is used for every call
    log_token_details(access_token, "Before generating network insights")
    if not network_list:
        return {"text": "No data available for network insights."}

    top_collaborators = sorted(network_list, key=lambda x: x['collaboration_count'], reverse=True)[:3]
    top_centrality = metrics.get('key_influencers', [])
    density = metrics.get('density', 0.0)
    community_count = len(metrics.get('communities', [])) if metrics.get('communities') else 0

    prompt = (
        f"Analyze this healthcare practitioner care network for clinical insights:\n"
        f"- Total Practitioners: {len(network_list)}\n"
        f"- Total Collaborations: {sum(p['collaboration_count'] for p in network_list) // 2}\n"
        f"- Network Density: {density:.2f}\n"
        f"- Top Collaborators: {', '.join([f'{p['name']} ({p['collaboration_count']} collaborations)' for p in top_collaborators])}\n"
        f"- Key Influencers (by centrality): {', '.join([f'{practitioner_lookup.get(pid, 'Unknown')} ({centrality:.2f})' for pid, centrality in top_centrality])}\n"
        f"- Community Count: {community_count}\n"
        f"Provide concise, actionable insights (max 200 words) for healthcare administrators or doctors, focusing on collaboration patterns, key influencers, and potential referral optimization."
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",  # Use gpt-4 for more advanced analysis (if available in your plan)
            messages=[{"role": "system", "content": "You are a healthcare data scientist analyzing practitioner networks for clinical optimization."}, {"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.5,  # Lower temperature for more focused, deterministic insights
            headers={"Authorization": f"Bearer {access_token}"}  # Ensure token is used for OpenAI API (if required)
        )
        insight_text = response.choices[0].message.content.strip()
        log_token_details(access_token, "After generating network insights")
        return {"text": insight_text}
    except Exception as e:
        logger.error(f"Failed to generate network insights: {str(e)}")
        logger.debug(f"Prompt causing error: {prompt}")
        return {"text": "Error generating insights. Please check data or contact support."}

def predict_referral(timeline: Dict[str, Any], practitioners: List[Dict], access_token: str) -> Dict[str, str]:
    """Predict the next referral for a patient based on timeline and network data."""
    access_token = ensure_valid_token()  # Ensure a valid token is used for every call
    log_token_details(access_token, "Before predicting referral")
    events = timeline["timeline"]
    if not events:
        return {"predictedDate": "Unknown", "predictedReferrer": "No data available", "reason": "No historical events"}

    referral_sequence = []
    practitioner_locations = {p['id']: (p['location']['latitude'], p['location']['longitude']) if p['location'].get('latitude') and p['location'].get('longitude') else None for p in practitioners}
    practitioner_specialties = {p['id']: p['specialty'] for p in practitioners}
    practitioner_centrality = {p['id']: p['centrality'] for p in build_practitioner_care_network(access_token)['network']}

    for event in events:
        if event["referral"]:
            ref_date = event["referral"]["referralDate"]
            ref_by = event["referral"]["referredBy"]["name"]
            reason = event["subsequentActivities"]["encounters"][0]["reasonCode"]
            referral_sequence.append(f"{ref_date} - Referred by {ref_by} for {reason}")

    if not referral_sequence:
        # If no prior referrals, predict based on most central practitioners, location proximity, and conditions
        network_data = build_practitioner_care_network(access_token)
        top_influencers = sorted([(p['id'], p['centrality']) for p in network_data['network']], key=lambda x: x[1], reverse=True)[:3]
        patient_location = next((p['location'] for p in practitioners if p['id'] in [i[0] for i in top_influencers]), {'city': 'Unknown', 'state': 'Unknown'})
        recent_conditions = [c['code'] for e in events for c in e['subsequentActivities']['conditions']][-3:]  # Last 3 conditions
        
        prompt = (
            f"Predict the next referral for a patient with no prior referrals, based on this practitioner network:\n"
            f"- Patient Location: {patient_location['city']}, {patient_location['state']}\n"
            f"- Recent Conditions: {', '.join(recent_conditions) if recent_conditions else 'None'}\n"
            f"- Top Influencers: {', '.join([f'{practitioner_lookup.get(pid, 'Unknown')} (Centrality: {centrality:.2f})' for pid, centrality in top_influencers])}\n"
            f"- Practitioner Specialties and Locations: {json.dumps({p['id']: {'name': p['name'], 'specialty': p['specialty'], 'location': p['location']} for p in practitioners})}\n"
            f"Recommend the next referral (date, referrer, reason) considering clinical need (e.g., {', '.join(recent_conditions)}), proximity, specialty expertise, and centrality. Return in format: 'YYYY-MM-DD - Referred by [Name] for [Reason]' (max 100 words)."
        )
    else:
        # Enhanced prediction using historical data, centrality, location, and conditions
        last_referral_date = datetime.strptime(referral_sequence[-1].split(" - ")[0], "%Y-%m-%dT%H:%M:%SZ") if referral_sequence else datetime.now()
        next_date = (last_referral_date + datetime.timedelta(days=30)).strftime("%Y-%m-%d")  # Default: 30 days later

        network_data = build_practitioner_care_network(access_token)
        top_influencers = sorted([(p['id'], p['centrality']) for p in network_data['network']], key=lambda x: x[1], reverse=True)[:3]
        patient_location = next((e['subsequentActivities']['encounters'][0]['practitioners'][0]['location'] for e in events if e['subsequentActivities']['encounters']), {'city': 'Unknown', 'state': 'Unknown'})
        recent_conditions = [c['code'] for e in events for c in e['subsequentActivities']['conditions']][-3:]  # Last 3 conditions
        
        prompt = (
            f"Predict the next referral for a patient with this history:\n"
            f"- Referral History: {'; '.join(referral_sequence)}\n"
            f"- Patient Location: {patient_location['city']}, {patient_location['state']}\n"
            f"- Recent Conditions: {', '.join(recent_conditions) if recent_conditions else 'None'}\n"
            f"- Top Influencers: {', '.join([f'{practitioner_lookup.get(pid, 'Unknown')} (Centrality: {centrality:.2f})' for pid, centrality in top_influencers])}\n"
            f"- Practitioner Specialties and Locations: {json.dumps({p['id']: {'name': p['name'], 'specialty': p['specialty'], 'location': p['location']} for p in practitioners})}\n"
            f"Recommend the next referral (date, referrer, reason) considering clinical need (e.g., {', '.join(recent_conditions)}), historical patterns, proximity, specialty expertise, and centrality. Return in format: 'YYYY-MM-DD - Referred by [Name] for [Reason]' (max 100 words)."
        )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",  # Use gpt-4 for more advanced analysis (if available)
            messages=[{"role": "system", "content": "You are a healthcare AI predicting referrals for doctors based on FHIR data and network analysis."}, {"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.5,  # Lower temperature for more focused predictions
            headers={"Authorization": f"Bearer {access_token}"}  # Ensure token is used for OpenAI API (if required)
        )
        prediction = response.choices[0].message.content.strip()
        
        if " - Referred by " in prediction:
            pred_date, rest = prediction.split(" - Referred by ", 1)
            pred_referrer, pred_reason = rest.split(" for ", 1) if " for " in rest else (rest, "Clinical need")
        else:
            pred_date, pred_referrer, pred_reason = "Unknown", "Unknown", "Prediction parsing failed"
        
        logger.debug(f"Prediction result: {prediction}")
        log_token_details(access_token, "After predicting referral")
        return {
            "predictedDate": pred_date,
            "predictedReferrer": pred_referrer,
            "reason": pred_reason
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.debug(f"Prediction prompt causing error: {prompt}")
        return {"predictedDate": "Unknown", "predictedReferrer": "Error", "reason": str(e)}

def patient_insights(timeline: Dict[str, Any], access_token: str) -> Dict[str, str]:
    """Generate AI-powered insights for a patient's timeline."""
    access_token = ensure_valid_token()  # Ensure a valid token is used for every call
    log_token_details(access_token, "Before generating patient insights")
    events = timeline["timeline"]
    if not events:
        return {"text": "No data available for patient insights."}

    major_events = [e for e in events if e["isMajor"]]
    routine_events = [e for e in events if not e["isMajor"]]
    conditions = set(c["code"] for e in events for c in e["subsequentActivities"]["conditions"])
    procedures = set(p["code"] for e in events for p in e["subsequentActivities"]["procedures"])
    observations = set(f"{o['code']}: {o['value']}{o['unit']}" for e in events for o in e["subsequentActivities"]["observations"])

    prompt = (
        f"Generate concise, actionable insights (max 200 words) for a patient's healthcare timeline:\n"
        f"- Patient ID: {timeline['patientId']}\n"
        f"- Patient Name: {timeline['name']}\n"
        f"- Major Events: {len(major_events)} (e.g., {', '.join(e['eventDescription'] for e in major_events[:3]) or 'None'})\n"
        f"- Routine Events: {len(routine_events)}\n"
        f"- Conditions: {', '.join(conditions) or 'None'}\n"
        f"- Procedures: {', '.join(procedures) or 'None'}\n"
        f"- Observations: {', '.join(observations) or 'None'}\n"
        f"Provide insights for doctors, focusing on clinical trends, potential risks, and care optimization."
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",  # Use gpt-4 for more advanced analysis (if available)
            messages=[{"role": "system", "content": "You are a healthcare AI analyzing patient data for clinical insights."}, {"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.5,
            headers={"Authorization": f"Bearer {access_token}"}  # Ensure token is used for OpenAI API (if required)
        )
        insight_text = response.choices[0].message.content.strip()
        log_token_details(access_token, "After generating patient insights")
        return {"text": insight_text}
    except Exception as e:
        logger.error(f"Failed to generate patient insights: {str(e)}")
        logger.debug(f"Insights prompt causing error: {prompt}")
        return {"text": "Error generating insights. Please check data or contact support."}

@app.get("/login")
async def login():
    """Initiate OAuth login with MeldRx."""
    state = secrets.token_hex(16)
    code_verifier, code_challenge = generate_pkce_pair()
    request_states[state] = code_verifier
    auth_url = get_authorization_url(code_challenge, state)
    logger.debug(f"Generated auth URL: {auth_url}")
    return {"auth_url": auth_url}

@app.get("/callback")
async def callback(request: Request):
    """Handle OAuth callback from MeldRx."""
    code = request.query_params.get('code')
    received_state = request.query_params.get('state')
    if not code or not received_state:
        raise HTTPException(status_code=400, detail="Invalid callback parameters: code or state missing")
    if received_state not in request_states:
        raise HTTPException(status_code=400, detail="Invalid callback parameters: state not recognized")
    code_verifier = request_states[received_state]
    try:
        tokens = get_access_token(code, code_verifier)
        access_token = tokens.get("access_token")
        log_token_details(access_token, "After successful callback")
        del request_states[received_state]
        return {"message": "Access token retrieved successfully", "access_token": access_token}
    except Exception as e:
        logger.error(f"Token retrieval error: {str(e)}")
        logger.debug(f"Callback request params: {request.query_params}")
        raise HTTPException(status_code=500, detail="Failed to retrieve access token")

@app.get("/patients")
async def get_patients(access_token: str = Header(..., alias="access-token")):
    """Retrieve a list of patient IDs from FHIR data with detailed debugging and proper error handling."""
    access_token = ensure_valid_token()  # Ensure a valid token is used for every call
    log_token_details(access_token, "Received in /patients endpoint")
    try:
        # Debug: Log detailed token information
        decoded_token = decode_jwt(access_token)
        logger.debug(f"Fetching patients with token: {access_token[:20]}... (length: {len(access_token)}, format: {type(access_token)}, decoded: {decoded_token})")
        logger.debug(f"Token scopes: {decoded_token.get('scope', 'No scopes')}")  # Log scopes explicitly
        logger.debug(f"Token tenant: {decoded_token.get('tenant', 'No tenant')}")  # Log tenant explicitly
        logger.debug(f"Request headers: {{'access-token': '{access_token[:20]}...'}}")
        
        patients = fetch_and_flatten_resource("Patient", access_token)
        logger.debug(f"FHIR Patient resources fetched: {len(patients)} entries, sample: {patients[:2] if patients else 'None'}")
        
        if not patients:
            logger.error("No patients found in FHIR response")
            raise HTTPException(status_code=404, detail="No patients found")
        
        patient_ids = [p['id'] for p in patients if 'id' in p and p['id']]  # Ensure 'id' exists and is non-empty
        logger.debug(f"Extracted patient IDs: {patient_ids}")
        
        if not patient_ids:
            logger.error("No valid patient IDs found in flattened resources")
            raise HTTPException(status_code=404, detail="No valid patient IDs found")
        
        logger.info(f"Retrieved {len(patient_ids)} patient IDs: {patient_ids}")
        log_token_details(access_token, "After fetching patients successfully")
        return {"patientIds": patient_ids}
    except HTTPException as e:
        logger.error(f"Failed to fetch patients: {e.detail}")
        logger.debug(f"Patients request error context: access_token={access_token[:20]}..., decoded_token={decode_jwt(access_token)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error fetching patients: {str(e)}")
        logger.debug(f"Full exception details: {str(e)}, traceback: {e.__traceback__}, token={access_token[:20]}...")
        raise HTTPException(status_code=500, detail="Failed to fetch patients due to server error")

@app.get("/patient_timeline_visual/{patient_id}")
async def patient_timeline_visual(patient_id: str, access_token: str = Header(..., alias="access-token")):
    """Generate and return a patient's timeline visualization."""
    access_token = ensure_valid_token()  # Ensure a valid token is used for every call
    log_token_details(access_token, f"Received in /patient_timeline_visual for patient {patient_id}")
    try:
        logger.debug(f"Generating timeline for patient {patient_id} with token: {access_token[:20]}..., decoded: {decode_jwt(access_token)}")
        timeline = build_patient_timeline(patient_id, access_token)
        visual = generate_timeline_visual(timeline["timeline"])
        logger.info(f"Timeline generated for patient {patient_id}")
        log_token_details(access_token, f"After generating timeline for patient {patient_id}")
        return {"patientId": timeline["patientId"], "name": timeline["name"], "timeline": timeline["timeline"]}
    except HTTPException as e:
        logger.error(f"Failed to generate patient timeline for {patient_id}: {e.detail}")
        logger.debug(f"Timeline request error context: patient_id={patient_id}, token={access_token[:20]}..., decoded_token={decode_jwt(access_token)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error generating patient timeline for {patient_id}: {str(e)}")
        logger.debug(f"Full exception details: {str(e)}, traceback: {e.__traceback__}, token={access_token[:20]}...")
        raise HTTPException(status_code=500, detail=f"Failed to generate patient timeline visualization: {str(e)}")

@app.get("/practitioner_care_network")
async def practitioner_care_network(access_token: str = Header(..., alias="access-token")):
    """Generate and return the practitioner care network."""
    access_token = ensure_valid_token()  # Ensure a valid token is used for every call
    log_token_details(access_token, "Received in /practitioner_care_network endpoint")
    try:
        logger.debug(f"Generating practitioner care network with token: {access_token[:20]}..., decoded: {decode_jwt(access_token)}")
        network = build_practitioner_care_network(access_token)
        logger.info("Practitioner care network generated successfully")
        log_token_details(access_token, "After generating practitioner care network")
        return network
    except HTTPException as e:
        logger.error(f"Failed to generate practitioner care network: {e.detail}")
        logger.debug(f"Network request error context: token={access_token[:20]}..., decoded_token={decode_jwt(access_token)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error generating practitioner care network: {str(e)}")
        logger.debug(f"Full exception details: {str(e)}, traceback: {e.__traceback__}, token={access_token[:20]}...")
        raise HTTPException(status_code=500, detail=f"Failed to generate practitioner care network: {str(e)}")

@app.get("/patient_insights/{patient_id}")
async def patient_insights_endpoint(patient_id: str, access_token: str = Header(..., alias="access-token")):
    """Generate and return insights for a patient."""
    access_token = ensure_valid_token()  # Ensure a valid token is used for every call
    log_token_details(access_token, f"Received in /patient_insights for patient {patient_id}")
    try:
        logger.debug(f"Generating insights for patient {patient_id} with token: {access_token[:20]}..., decoded: {decode_jwt(access_token)}")
        timeline = build_patient_timeline(patient_id, access_token)
        insights = patient_insights(timeline, access_token)
        logger.info(f"Insights generated for patient {patient_id}")
        log_token_details(access_token, f"After generating insights for patient {patient_id}")
        return {"patientId": patient_id, "insights": insights}
    except HTTPException as e:
        logger.error(f"Failed to generate insights for patient {patient_id}: {e.detail}")
        logger.debug(f"Insights request error context: patient_id={patient_id}, token={access_token[:20]}..., decoded_token={decode_jwt(access_token)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error generating insights for patient {patient_id}: {str(e)}")
        logger.debug(f"Full exception details: {str(e)}, traceback: {e.__traceback__}, token={access_token[:20]}...")
        raise HTTPException(status_code=500, detail=f"Failed to generate patient insights: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)