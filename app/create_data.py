import os
import json
import random
import uuid
from datetime import datetime, timedelta
import time
from openai import OpenAI
import networkx as nx

# OpenAI API Key
OPENAI_API_KEY = "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"
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Directory to save synthetic bundles
OUTPUT_DIR = "synthetic_fhir_bundles_with_network"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Base date for timeline consistency
BASE_DATE = datetime(2023, 1, 1)

# Helper function to generate AI-driven text with strict JSON format and fallback
def generate_ai_text(prompt: str, max_tokens: int = 100, retries: int = 3, delay: int = 1, fallback: str = None) -> str:
    for attempt in range(retries):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a healthcare data expert creating realistic FHIR resource details with practitioner care network context. Return a single JSON object or string as specified, with no extra text. Use SNOMED CT for conditions/procedures, LOINC for observations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7,
                timeout=10
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI error (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return fallback if fallback else '{"error": "Failed to generate"}'

# Helper function to generate UUID
def generate_uuid() -> str:
    return str(uuid.uuid4())

# AI-driven code generators with JSON parsing
def generate_condition_code(used_codes: set) -> dict:
    prompt = f"Generate a unique, realistic SNOMED CT code and display name for a patient condition (e.g., acute like fractures or chronic like diabetes). Avoid these used codes: {list(used_codes)}. Return as a JSON object: {{\"system\": \"http://snomed.info/sct\", \"code\": \"73211009\", \"display\": \"Diabetes mellitus (disorder)\"}}."
    result = generate_ai_text(prompt, max_tokens=50, fallback='{"system": "http://snomed.info/sct", "code": "73211009", "display": "Diabetes mellitus (disorder)"}')
    
    try:
        data = json.loads(result)
        if not isinstance(data, dict) or "code" not in data or "display" not in data or "system" not in data:
            raise ValueError(f"Invalid condition code format: {result}")
        code = data["code"]
        if code in used_codes:
            print(f"Duplicate code {code} generated, overriding with unique code. Result was: '{result}'")
            code = f"999{random.randint(10000, 99999)}"  # Force unique code
            data["code"] = code
            data["display"] = f"Custom {data['display']} (disorder)"
        used_codes.add(code)
        return data
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"Error in generate_condition_code: {e}, result was: '{result}'")
        fallbacks = [
            {"system": "http://snomed.info/sct", "code": "73211009", "display": "Diabetes mellitus (disorder)"},
            {"system": "http://snomed.info/sct", "code": "59621000", "display": "Fracture of femur (disorder)"},
            {"system": "http://snomed.info/sct", "code": "233606009", "display": "Car accident injury (disorder)"},
            {"system": "http://snomed.info/sct", "code": "38341003", "display": "Hypertension (disorder)"},
            {"system": "http://snomed.info/sct", "code": "44054006", "display": "Type 2 diabetes mellitus (disorder)"},
            {"system": "http://snomed.info/sct", "code": "53741008", "display": "Coronary artery disease (disorder)"}
        ]
        available_fallbacks = [fb for fb in fallbacks if fb["code"] not in used_codes]
        if available_fallbacks:
            fb = random.choice(available_fallbacks)
            used_codes.add(fb["code"])
            return fb
        code = f"999{random.randint(10000, 99999)}"
        used_codes.add(code)
        return {"system": "http://snomed.info/sct", "code": code, "display": "Generic condition (disorder)"}

def generate_encounter_type(class_code: str = "AMB") -> dict:
    prompt = f"Generate a realistic SNOMED CT code and display name for an encounter type matching class '{class_code}'. Return as a JSON object: {{\"system\": \"http://snomed.info/sct\", \"code\": \"162673000\", \"display\": \"General examination of patient (procedure)\"}}."
    result = generate_ai_text(prompt, max_tokens=50, fallback='{"system": "http://snomed.info/sct", "code": "162673000", "display": "General examination of patient (procedure)"}')
    try:
        data = json.loads(result)
        if not isinstance(data, dict) or "code" not in data or "display" not in data:
            raise ValueError(f"Invalid encounter type format: {result}")
        return data
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error in generate_encounter_type: {e}, result was: '{result}'")
        return {"system": "http://snomed.info/sct", "code": "162673000", "display": "General examination of patient (procedure)"}

def generate_procedure_code() -> dict:
    prompt = "Generate a realistic SNOMED CT code and display name for a medical procedure. Return as a JSON object: {\"system\": \"http://snomed.info/sct\", \"code\": \"80146002\", \"display\": \"Appendectomy (procedure)\"}."
    result = generate_ai_text(prompt, max_tokens=50, fallback='{"system": "http://snomed.info/sct", "code": "80146002", "display": "Appendectomy (procedure)"}')
    try:
        data = json.loads(result)
        if not isinstance(data, dict) or "code" not in data or "display" not in data:
            raise ValueError(f"Invalid procedure code format: {result}")
        return data
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error in generate_procedure_code: {e}, result was: '{result}'")
        return {"system": "http://snomed.info/sct", "code": "80146002", "display": "Appendectomy (procedure)"}

def generate_observation_code() -> dict:
    prompt = "Generate a realistic LOINC code, display name, unit, and value range for a clinical observation. Return as a JSON object: {\"system\": \"http://loinc.org\", \"code\": \"8480-6\", \"display\": \"Systolic Blood Pressure\", \"unit\": \"mmHg\", \"min\": 90, \"max\": 120}."
    result = generate_ai_text(prompt, max_tokens=50, fallback='{"system": "http://loinc.org", "code": "8480-6", "display": "Systolic Blood Pressure", "unit": "mmHg", "min": 90, "max": 120}')
    try:
        data = json.loads(result)
        if not all(k in data for k in ["system", "code", "display", "unit", "min", "max"]):
            raise ValueError(f"Invalid observation code format: {result}")
        data["value_range"] = (float(data.pop("min")), float(data.pop("max")))
        return data
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error in generate_observation_code: {e}, result was: '{result}'")
        return {"system": "http://loinc.org", "code": "8480-6", "display": "Systolic Blood Pressure", "unit": "mmHg", "value_range": (90, 120)}

# Validate JSON before saving
def validate_json(data: dict) -> bool:
    try:
        json.dumps(data)
        return data.get("resourceType") == "Bundle" and "entry" in data
    except (TypeError, ValueError) as e:
        print(f"JSON validation failed: {e}")
        return False

# Build a practitioner care network
def build_practitioner_network(practitioners: list) -> nx.Graph:
    G = nx.Graph()
    for prac in practitioners:
        G.add_node(prac["id"], name=prac["name"], role=prac["role"])
    for i, p1 in enumerate(practitioners):
        for p2 in practitioners[i + 1:]:
            if random.random() < 0.3:  # 30% chance of collaboration
                G.add_edge(p1["id"], p2["id"], weight=random.randint(1, 5))
    return G

# Generate a single Bundle with practitioner care network
def create_patient_bundle(patient_count: int) -> None:
    bundle = {"resourceType": "Bundle", "type": "transaction", "entry": []}
    used_condition_codes = set()

    # Patient
    pat_id = generate_uuid()
    full_name = generate_ai_text("Generate a realistic full name for a patient. Return as a string.", max_tokens=20, fallback="John Doe")
    family_name, given_name = full_name.split(" ", 1) if " " in full_name else (full_name, "")
    gender = random.choice(["M", "F"])
    birth_date = BASE_DATE - timedelta(days=random.randint(20 * 365, 80 * 365))

    patient = {
        "resourceType": "Patient",
        "id": pat_id,
        "name": [{"use": "official", "family": family_name, "given": [given_name], "prefix": ["Mr." if gender == "M" else "Ms."]}],
        "gender": "male" if gender == "M" else "female",
        "birthDate": birth_date.strftime("%Y-%m-%d")
    }
    bundle["entry"].append({"fullUrl": f"urn:uuid:{pat_id}", "resource": patient, "request": {"method": "POST", "url": "Patient"}})

    # Practitioners (GP, Specialist, Surgeon)
    practitioners = [
        {"id": generate_uuid(), "name": generate_ai_text("Generate a realistic full name for a General Practitioner. Return as a string.", fallback="Rachel Nguyen"), "role": "GP", "specialty": "408443003"},  # General medical practice
        {"id": generate_uuid(), "name": generate_ai_text("Generate a realistic full name for an Orthopedist. Return as a string.", fallback="Olivia Patel"), "role": "Specialist", "specialty": "419772000"},  # Orthopedics
        {"id": generate_uuid(), "name": generate_ai_text("Generate a realistic full name for a Surgeon. Return as a string.", fallback="James Carter"), "role": "Surgeon", "specialty": "309369003"}  # General surgery
    ]
    for prac in practitioners:
        family, given = prac["name"].split(" ", 1) if " " in prac["name"] else (prac["name"], "")
        prac_resource = {
            "resourceType": "Practitioner",
            "id": prac["id"],
            "name": [{"family": family, "given": [given], "prefix": ["Dr."]}]
        }
        role_resource = {
            "resourceType": "PractitionerRole",
            "id": generate_uuid(),
            "practitioner": {"reference": f"urn:uuid:{prac['id']}"},
            "specialty": [{"coding": [{"system": "http://snomed.info/sct", "code": prac["specialty"]}]}]
        }
        bundle["entry"].extend([
            {"fullUrl": f"urn:uuid:{prac['id']}", "resource": prac_resource, "request": {"method": "POST", "url": "Practitioner"}},
            {"fullUrl": f"urn:uuid:{role_resource['id']}", "resource": role_resource, "request": {"method": "POST", "url": "PractitionerRole"}}
        ])

    # Build practitioner network
    care_network = build_practitioner_network(practitioners)
    print(f"Practitioner Network for Patient {patient_count}: {care_network.edges(data=True)}")

    # Generate timeline with encounters, referrals, and collaborations
    num_encounters = random.randint(5, 7)
    last_end_date = BASE_DATE
    referral_history = {}

    for enc_num in range(1, num_encounters + 1):
        enc_id = generate_uuid()
        class_code = "AMB" if enc_num % 2 == 0 else random.choice(["EMER", "IMP"])
        enc_type = generate_encounter_type(class_code)
        cond_code = generate_condition_code(used_condition_codes)
        reason = f"Management of {cond_code['display']}"

        # Determine practitioners involved (collaboration or referral)
        if enc_num == 1:
            primary_prac = practitioners[0]  # Start with GP
            participants = [primary_prac]
        else:
            if random.random() < 0.4 and care_network.degree(participants[-1]["id"]) > 0:  # 40% chance of referral
                referrer = participants[-1]
                referred_to = random.choice([n for n in care_network.neighbors(referrer["id"])])
                participants = [next(p for p in practitioners if p["id"] == referred_to)]
                referral_history[enc_id] = {"referrer": referrer["id"], "referred_to": referred_to, "date": last_end_date.strftime("%Y-%m-%d")}
                reason = f"Referral from Dr. {referrer['name']} for {cond_code['display']}"
            else:
                participants = [random.choice(practitioners)]
                if random.random() < 0.3:  # 30% chance of adding a collaborator
                    collaborator = random.choice([p for p in practitioners if p["id"] != participants[0]["id"]])
                    participants.append(collaborator)
                    care_network.add_edge(participants[0]["id"], collaborator["id"], weight=1)

        enc_start = last_end_date + timedelta(days=random.randint(30, 90))
        enc_end = enc_start + timedelta(minutes=random.randint(30, 240))

        enc = {
            "resourceType": "Encounter",
            "id": enc_id,
            "status": "finished",
            "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": class_code},
            "type": [{"coding": [enc_type], "text": enc_type["display"]}],
            "subject": {"reference": f"urn:uuid:{pat_id}", "display": f"{'Mr.' if gender == 'M' else 'Ms.'} {given_name} {family_name}"},
            "participant": [{"individual": {"reference": f"urn:uuid:{p['id']}", "display": f"Dr. {p['name']}"}} for p in participants],
            "period": {"start": enc_start.strftime("%Y-%m-%dT%H:%M:%SZ"), "end": enc_end.strftime("%Y-%m-%dT%H:%M:%SZ")},
            "reasonCode": [{"text": reason}]
        }
        if enc_id in referral_history:
            enc["incomingReferral"] = [{"referrer": {"reference": f"urn:uuid:{referral_history[enc_id]['referrer']}"}, "date": referral_history[enc_id]["date"]}]
        bundle["entry"].append({"fullUrl": f"urn:uuid:{enc_id}", "resource": enc, "request": {"method": "POST", "url": "Encounter"}})

        # Condition
        cond_id = generate_uuid()
        condition = {
            "resourceType": "Condition",
            "id": cond_id,
            "code": {"coding": [cond_code], "text": cond_code["display"]},
            "subject": {"reference": f"urn:uuid:{pat_id}"},
            "encounter": {"reference": f"urn:uuid:{enc_id}"},
            "onsetDateTime": enc_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        bundle["entry"].append({"fullUrl": f"urn:uuid:{cond_id}", "resource": condition, "request": {"method": "POST", "url": "Condition"}})

        # Procedure
        proc_id = generate_uuid()
        proc_code = generate_procedure_code()
        procedure = {
            "resourceType": "Procedure",
            "id": proc_id,
            "status": "completed",
            "code": {"coding": [proc_code], "text": proc_code["display"]},
            "subject": {"reference": f"urn:uuid:{pat_id}"},
            "encounter": {"reference": f"urn:uuid:{enc_id}"},
            "performedDateTime": enc_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        bundle["entry"].append({"fullUrl": f"urn:uuid:{proc_id}", "resource": procedure, "request": {"method": "POST", "url": "Procedure"}})

        # Observation
        obs_id = generate_uuid()
        obs_code = generate_observation_code()
        observation = {
            "resourceType": "Observation",
            "id": obs_id,
            "status": "final",
            "code": {"coding": [{"system": obs_code["system"], "code": obs_code["code"], "display": obs_code["display"]}]},
            "subject": {"reference": f"urn:uuid:{pat_id}"},
            "encounter": {"reference": f"urn:uuid:{enc_id}"},
            "effectiveDateTime": enc_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "valueQuantity": {"value": random.uniform(*obs_code["value_range"]), "unit": obs_code["unit"]}
        }
        bundle["entry"].append({"fullUrl": f"urn:uuid:{obs_id}", "resource": observation, "request": {"method": "POST", "url": "Observation"}})

        last_end_date = enc_end

    if validate_json(bundle):
        with open(os.path.join(OUTPUT_DIR, f"bundle_pat_{patient_count}.json"), "w") as f:
            json.dump(bundle, f, indent=2)
        print(f"Successfully saved bundle for patient {patient_count}")
    else:
        print(f"Skipping invalid bundle for patient {patient_count}")

# Main function
def ai_bundle_generator(num_patients: int = 10) -> None:
    print("Generating synthetic FHIR Bundles with Practitioner Care Network...")
    for i in range(num_patients):
        create_patient_bundle(i + 1)
    print(f"All Bundles saved in '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    ai_bundle_generator(num_patients=5)