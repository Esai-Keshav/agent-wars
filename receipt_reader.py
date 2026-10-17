"""Multimodal challan/receipt extraction with Gemini Vision (spec 7)."""
import base64
import os

import config

_EXTRACT_PROMPT = (
    "Read this loading challan and return ONLY the fields exactly as written:"
    " Sender: <company origin>; Pallets: <integer>; Weight: <kg integer>;"
    " Destination: <area among Chennai Port, Sriperumbudur, Avadi, Ambattur,"
    " Guindy>. If a field is missing, write 'None'."
)


def _client():
    from google import genai
    return genai.Client(api_key=config.GOOGLE_API_KEY)


def read_receipt_impl(image_path: str) -> dict:
    """Extract sender, pallets, weight, destination from a challan photo at image_path."""
    if not os.path.exists(image_path):
        return {"status": "ERROR", "reason": "not_found", "detail": image_path}
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        mine = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
        client = _client()
        resp = client.models.generate_content(
            model=config.MODEL_NAME,
            contents=[{
                "role": "user",
                "parts": [
                    {"text": _EXTRACT_PROMPT},
                    {
                        "inline_data": {
                            "mime_type": mine,
                            "data": base64.b64encode(img_bytes).decode(),
                        }
                    },
                ],
            }],
        )
        text = resp.text or ""
        fields = {}
        for line in text.splitlines() + text.split(";"):
            for key in ["Sender", "Pallets", "Weight", "Destination"]:
                if line.strip().lower().startswith(key.lower()):
                    val = line.split(":", 1)[-1].strip()
                    if val and val != "None":
                        fields[key.lower().replace(" ", "_")] = val
        weight_raw = "".join(
            c for c in str(fields.get("weight", "0"))
            if c.isdigit() or c == "."
        )
        return {
            "status": "OK",
            "sender": fields.get("sender", "unknown"),
            "pallets": int(fields["pallets"]) if "pallets" in fields else 0,
            "weight_kg": float(weight_raw) if weight_raw else 0.0,
            "destination_hint": fields.get("destination", "unknown"),
        }
    except Exception as exc:  # spec 6: structured error, never raw raise
        return {"status": "ERROR", "reason": "extraction_failed", "detail": str(exc)}
