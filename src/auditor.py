import base64
import io
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request

from .model import practice_image
from .runtime import write_json


FAILED_RESPONSE = "The central shape is blue. Its outline is square."


def endpoint_url(value):
    parsed = urllib.parse.urlsplit(value)
    local = parsed.hostname in ("localhost", "127.0.0.1", "::1")
    if (parsed.scheme != "https" and not (parsed.scheme == "http" and local)):
        raise ValueError("Use HTTPS, or HTTP on a loopback address")
    if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Endpoint must not contain credentials, query parameters, or a fragment")
    return value.rstrip("/") + "/chat/completions"


def validate_audit(value, failed_response=FAILED_RESPONSE):
    if not isinstance(value, dict) or value.get("action") not in ("repair", "skip"):
        raise ValueError("Auditor must return an object with action repair or skip")
    if set(value) != {"action", "prefix_end", "hint"}:
        raise ValueError("Auditor must return exactly action, prefix_end, and hint")
    boundary = value.get("prefix_end")
    if type(boundary) is not int or not 0 <= boundary <= len(failed_response):
        raise ValueError("prefix_end must be a valid character boundary")
    hint = value.get("hint")
    if not isinstance(hint, str) or len(hint) > 500:
        raise ValueError("hint must be a string of at most 500 characters")
    if value["action"] == "repair" and not hint.strip():
        raise ValueError("A repair requires a hint")
    if value["action"] == "repair" and failed_response == FAILED_RESPONSE:
        # These checks concern the known synthetic fixture, not general medical auditing.
        findings = r"\b(red|blue|green|yellow|black|white|orange|purple|pink|brown|circle|square|triangle|rectangle|oval)\b"
        if re.search(findings, hint, re.IGNORECASE):
            raise ValueError("The practice hint states a color or shape instead of directing inspection")
        if re.search(r"\b(blue|square)\b", failed_response[:boundary], re.IGNORECASE):
            raise ValueError("The preserved practice prefix contains an unsupported finding")
    return value


def auditor_probe(directory=None):
    endpoint = os.environ.get("AUDITOR_BASE_URL", "")
    model = os.environ.get("AUDITOR_MODEL", "")
    if not endpoint or not model:
        return {"status": "blocked", "reason": "auditor_not_configured",
                "next_action": "Set AUDITOR_BASE_URL and AUDITOR_MODEL for an approved image-capable service"}
    url = endpoint_url(endpoint)
    if model == "HuggingFaceTB/SmolVLM2-500M-Video-Instruct":
        raise ValueError("Select a separate auditor model for this study")
    buffer = io.BytesIO()
    practice_image().save(buffer, format="PNG")
    image_data = base64.b64encode(buffer.getvalue()).decode("ascii")
    schema = {
        "type": "object", "additionalProperties": False,
        "required": ["action", "prefix_end", "hint"],
        "properties": {
            "action": {"type": "string", "enum": ["repair", "skip"]},
            "prefix_end": {"type": "integer", "minimum": 0,
                           "maximum": len(FAILED_RESPONSE)},
            "hint": {"type": "string", "maxLength": 500}
        }
    }
    instruction = (
        "Review the image, question, and failed response. Return only an object matching "
        "the supplied JSON schema. action is repair when reasoning needs correction, "
        "otherwise skip. prefix_end is an INTEGER: the zero-based character offset where "
        "the retained prefix ends, with the end excluded. It is not a word, a quotation, "
        f"or a step number. Its valid range is 0 to {len(FAILED_RESPONSE)}. "
        "The retained text must contain only supported reasoning. Use 0 when no complete "
        "reasoning step before the error can be retained. hint must be a short imperative "
        "instruction directing visual reinspection. Do not write a replacement answer, "
        "state a specific color or shape, or copy claims from the failed response. "
        "If action is skip, use an empty hint. JSON schema: " + json.dumps(schema)
    )
    payload = {"model": model, "max_tokens": 256, "temperature": 0, "seed": 7,
               "response_format": {"type": "json_schema", "json_schema": {
                   "name": "practice_audit", "strict": True, "schema": schema}},
               "messages": [
                   {"role": "system", "content": instruction},
                   {"role": "user", "content": [
                       {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_data}},
                       {"type": "text", "text": "Question: Name the color and shape.\nFailed response: " + FAILED_RESPONSE}
                   ]}
               ]}
    if directory is not None:
        write_json(Path(directory) / "auditor_request.json", payload)
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("AUDITOR_API_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, file, code, message, headers, new_url):
            return None

    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=90) as response:
            body = response.read(1_000_001)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Auditor HTTP {error.code}; check endpoint, model, authentication, and supported request fields") from None
    if len(body) > 1_000_000:
        raise ValueError("Auditor response exceeded the probe limit")
    if directory is not None:
        write_json(Path(directory) / "auditor_response.json", {
            "auditor_model": model, "response_text": body.decode("utf-8", errors="replace")
        })
    response = json.loads(body)
    audit = validate_audit(json.loads(response["choices"][0]["message"]["content"]))
    if audit["action"] != "repair":
        raise ValueError("Auditor skipped the deliberately wrong practice response; inspect compatibility")
    return {"status": "passed", "check": "synthetic auditor connectivity and format",
            "auditor_model": model, "audit": audit,
            "request_format": "json_schema", "temperature": 0, "seed": 7,
            "medical_quality": "not_evaluated",
            "independent_model_identity": "must be confirmed by the researcher"}
