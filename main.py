import functions_framework
import google.auth
from googleapiclient.discovery import build
from google import genai
from google.cloud import storage
import json
import os
import logging
import base64
from google.cloud import logging as cloud_logging

# Initialize Google Cloud Logging
client = cloud_logging.Client()
client.setup_logging()

# Authenticate using Google Cloud IAM
credentials, project_id = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

# Initialize Google Cloud Healthcare API client
service = build("healthcare", "v1", credentials=credentials)

# Initialize Google Cloud Storage client
storage_client = storage.Client()

# Set required environment variables for Gemini 2.0 on Vertex AI
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

# Define Google Cloud Healthcare FHIR Store details
FHIR_LOCATION = "us-central1"
FHIR_DATASET_ID = "my-dataset"
FHIR_STORE_ID = "my-fhir-store"

FHIR_PARENT = f"projects/{project_id}/locations/{FHIR_LOCATION}/datasets/{FHIR_DATASET_ID}/fhirStores/{FHIR_STORE_ID}"

# Google Cloud Storage Bucket Name (Replace with your bucket name)
GCS_BUCKET_NAME = "icd10-mappings-bucket"

@functions_framework.http
def process_fhir_entry(request):
    """Cloud Function triggered when a new FHIR resource is created."""
    logging.info("🔹 Cloud Function triggered: Processing new FHIR resource.")

    request_json = request.get_json()
    logging.info(f"📥 Received request payload: {json.dumps(request_json, indent=2)}")

    if not request_json or "message" not in request_json:
        logging.error("❌ Invalid request: No JSON body or missing 'message' field.")
        return {"error": "Invalid request, no JSON body or missing 'message' field"}, 400

    try:
        # Decode the Base64 FHIR resource path
        encoded_data = request_json["message"]["data"]
        decoded_data = base64.b64decode(encoded_data).decode("utf-8")
        logging.info(f"📜 Decoded FHIR Resource Path: {decoded_data}")

        # Extract the Patient ID from the decoded FHIR path
        resource_parts = decoded_data.split("/")
        resource_type = resource_parts[-2]  # e.g., "Patient"
        resource_id = resource_parts[-1]    # Unique ID

        if not resource_id:
            logging.error("❌ Missing resource ID in extracted path.")
            return {"error": "No resource ID found after decoding"}, 400

        logging.info(f"🔍 Extracted Resource: {resource_type}/{resource_id}")

        # List available resources
        available_resources = list_fhir_resources()
        logging.info(f"📋 Available FHIR Resources: {available_resources}")

        # Retrieve the FHIR resource from the store
        fhir_request = service.projects().locations().datasets().fhirStores().fhir().read(
            name=f"{FHIR_PARENT}/fhir/{resource_type}/{resource_id}"
        )
        fhir_file = fhir_request.execute()
        logging.info(f"✅ Successfully retrieved {resource_type}/{resource_id}.")

        # Process data with Vertex AI Gemini
        icd_mapping = map_icd10_cm_to_icd10_pcs(fhir_file)

        # Store the ICD-10 mapping in the FHIR store
        store_icd_mappings(icd_mapping, resource_id)

        # Store the ICD-10 mapping in Google Cloud Storage
        store_icd_mapping_in_gcs(icd_mapping, resource_id)

        logging.info(f"✅ Successfully processed and stored ICD-10 mappings for {resource_id}.")
        
        return {
            "resource_id": resource_id,
            "icd_10_mapping": icd_mapping,
            "update_status": "FHIR Store and GCS Updated Successfully"
        }

    except Exception as e:
        logging.error(f"❌ Error processing FHIR file: {str(e)}")
        return {"error": str(e)}, 500


def list_fhir_resources():
    """Lists available FHIR resources in the store."""
    try:
        logging.info("🔹 Listing available FHIR resources...")

        request = service.projects().locations().datasets().fhirStores().fhir().search(
            parent=FHIR_PARENT
        )
        response = request.execute()
        resources = response.get("entry", [])
        
        return [res["resource"]["id"] for res in resources] if resources else []

    except Exception as e:
        logging.error(f"❌ Error listing FHIR resources: {str(e)}")
        return []


def map_icd10_cm_to_icd10_pcs(fhir_data):
    """Uses Gemini 2.0 on Vertex AI to map ICD-10-CM to ICD-10-PCS."""
    try:
        logging.info("🔹 Sending request to Vertex AI Gemini API...")

        # Initialize Vertex AI Gemini 2.0 model
        client = genai.Client()
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=f"""
            You are an AI specializing in ICD-10 medical coding. Analyze the FHIR resource below and map ICD-10-CM conditions to ICD-10-PCS procedures.

            **FHIR Data:**
            ```json
            {json.dumps(fhir_data, indent=2)}
            ```
            """
        )

        logging.info(f"✅ Gemini API Response: {response.text}")
        return json.loads(response.text)  # Convert response to JSON

    except Exception as e:
        logging.error(f"❌ Error with Gemini API: {str(e)}")
        return {}


def store_icd_mappings(icd_mapping, resource_id):
    """Stores ICD-10 mappings in the FHIR store."""
    try:
        logging.info(f"🔹 Storing ICD-10 mappings for {resource_id}...")

        updated_fhir_data = {
            "resourceType": "Observation",
            "status": "final",
            "code": {
                "text": "ICD-10 Mapping"
            },
            "subject": {
                "reference": f"Patient/{resource_id}"
            },
            "valueString": json.dumps(icd_mapping)
        }

        store_request = service.projects().locations().datasets().fhirStores().fhir().create(
            parent=FHIR_PARENT, body=updated_fhir_data
        )
        store_request.execute()
        logging.info(f"✅ ICD-10 mappings stored in FHIR store for {resource_id}.")

    except Exception as e:
        logging.error(f"❌ Error storing ICD-10 mappings: {str(e)}")


def store_icd_mapping_in_gcs(icd_mapping, resource_id):
    """Stores ICD-10 mappings in Google Cloud Storage as JSON files."""
    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(f"icd10_mappings/{resource_id}.json")
        blob.upload_from_string(json.dumps(icd_mapping), content_type="application/json")
        logging.info(f"✅ ICD-10 mappings stored in Google Cloud Storage for {resource_id}.")

    except Exception as e:
        logging.error(f"❌ Error storing ICD-10 mappings in GCS: {str(e)}")
