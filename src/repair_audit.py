import base64
import io
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image

from .auditor import endpoint_url
from .runtime import write_json


FAMILIES = ("V", "R", "D", "K", "format_only", "unresolved")
AUDIT_VALIDATION_POLICY = "nonrepair-auxiliary-unused-v2"


def local_endpoint(value):
    url = endpoint_url(value)
    if urllib.parse.urlsplit(url).hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("The approved CPU pilot uses a local loopback auditor only")
    return url


def image_content(image):
    clean = Image.frombytes("RGB", image.size, image.convert("RGB").tobytes())
    clean.thumbnail((512, 512))
    buffer = io.BytesIO(); clean.save(buffer, format="PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")}}


def audit_payload(image, question, response, steps, entropy, config):
    schema = {"type": "object", "additionalProperties": False,
        "required": ["action", "first_error_step", "error_family", "prefix_supported", "hint", "evidence"],
        "properties": {"action": {"type": "string", "enum": ["repair", "skip", "unresolved"]},
            "first_error_step": {"type": "integer", "enum": list(range(len(steps) + 1))},
            "error_family": {"type": "string", "enum": list(FAMILIES)},
            "prefix_supported": {"type": "boolean"}, "hint": {"type": "string", "maxLength": 240},
            "evidence": {"type": "string", "maxLength": 500}}}
    instruction = (
        "Audit the provided image, question, and reasoning. No reference answer is provided. "
        "Entropy proposes a region, not proof of an error. Trace dependencies backward and identify the first "
        "consequential unsupported step using visible evidence. Steps are heuristic sentence spans. "
        "Use repair only for a supported medical reasoning error: V visual misreading, R invalid inference, "
        "D differential comparison, K missing prerequisite knowledge. first_error_step is its 1-based step number, "
        "not a character offset. Earlier steps must all be supported; set prefix_supported accordingly. "
        "If the first step is wrong, choose 1 and retain no prefix. The hint must be one short imperative "
        "inspection/reasoning operation. Do not state a finding, diagnosis, answer, option, corrected rationale, "
        "or new clinical evidence. Do not simply repeat a claim from the failed response. "
        "Use skip with first_error_step 0 and empty hint if there is no justified reasoning repair, including "
        "format-only problems. Use unresolved with step 0 and empty hint if evidence is insufficient. "
        "Explain the error/boundary evidence separately; only hint will reach the primary model. Return the JSON schema exactly.")
    visible = {"question": question, "response": response, "steps": steps, "entropy_proposal": entropy}
    return {"model": config["auditor_model"], "temperature": 0, "seed": 7,
        "max_tokens": config["auditor_max_tokens"],
        "response_format": {"type": "json_schema", "json_schema": {"name": "medical_repair_audit", "strict": True, "schema": schema}},
        "messages": [{"role": "system", "content": instruction},
                     {"role": "user", "content": [image_content(image), {"type": "text", "text": json.dumps(visible, ensure_ascii=False)}]}]}


def review_payload(image, question, response, answer, config):
    schema = {"type": "object", "additionalProperties": False,
        "required": ["verdict", "hint_leakage", "reason"],
        "properties": {"verdict": {"type": "string", "enum": ["supported", "unsupported", "unresolved"]},
            "hint_leakage": {"type": "string", "enum": ["none_detected", "suspected", "unresolved"]},
            "reason": {"type": "string", "maxLength": 500}}}
    instruction = ("Review the candidate reasoning and its model-generated answer against the image and question. "
        "No reference answer is provided. Reject unsupported additions, invalid retained claims, repetition, "
        "contradictions or a recovery cue that reveals the answer. Use unresolved when uncertain. "
        "This is a model review, not a clinical validation. Return the JSON schema exactly.")
    return {"model": config["auditor_model"], "temperature": 0, "seed": 7,
        "max_tokens": config["auditor_max_tokens"],
        "response_format": {"type": "json_schema", "json_schema": {"name": "medical_repair_review", "strict": True, "schema": schema}},
        "messages": [{"role": "system", "content": instruction}, {"role": "user", "content": [image_content(image),
            {"type": "text", "text": json.dumps({"question": question, "candidate_response": response, "model_answer_readout": answer}, ensure_ascii=False)}]}]}


def request_json(payload, config, directory, name):
    directory = Path(directory)
    url = local_endpoint(config["auditor_base_url"])
    write_json(directory / (name + "_request.json"), payload)
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("AUDITOR_API_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args):
            return None
    started = time.perf_counter()
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect()).open(request, timeout=config["auditor_timeout_seconds"]) as response:
            content = response.read(1_000_001)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Local auditor HTTP {error.code}; inspect model availability and JSON-schema support") from None
    if len(content) > 1_000_000:
        raise ValueError("Local auditor response exceeded 1 MB")
    raw = content.decode("utf-8", errors="strict")
    write_json(directory / (name + "_response.json"), {"response_text": raw,
               "roundtrip_seconds": round(time.perf_counter() - started, 3)})
    envelope = json.loads(raw)
    return json.loads(envelope["choices"][0]["message"]["content"])


def validate_medical_audit(value, steps):
    required = {"action", "first_error_step", "error_family", "prefix_supported", "hint", "evidence"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Medical audit fields do not match the schema")
    if value["action"] not in ("repair", "skip", "unresolved") or value["error_family"] not in FAMILIES:
        raise ValueError("Unknown audit action or error family")
    step = value["first_error_step"]
    if type(step) is not int or type(value["prefix_supported"]) is not bool:
        raise ValueError("Audit step must be an integer and prefix_supported a boolean")
    if not 0 <= step <= len(steps):
        raise ValueError("Audit step is outside the declared schema range")
    for name, maximum in (("hint", 240), ("evidence", 500)):
        if not isinstance(value[name], str) or len(value[name]) > maximum:
            raise ValueError("Invalid audit text field")
    if value["action"] == "repair":
        if not 1 <= step <= len(steps) or value["error_family"] not in ("V", "R", "D", "K"):
            raise ValueError("A medical repair needs a supported reasoning-step index and family")
        if not value["prefix_supported"] or not value["hint"].strip() or not value["evidence"].strip():
            raise ValueError("A repair requires supported prefix, hint and boundary evidence")
    # Skip/unresolved auxiliary values are unused, as in the paper's Skip branch.
    # Retain them verbatim for inspection; only an explicit repair can continue.
    return value


def safe_hint(hint, reference):
    if "\n" in hint or not re.match(r"\s*(check|recheck|inspect|reinspect|compare|trace|count|verify|locate|revisit|examine|review)\b", hint, re.IGNORECASE):
        return False, "hint_must_be_one_inspection_or_reasoning_instruction"
    if re.search(r"\b(answer|diagnosis is|correct option|should be)\b", hint, re.IGNORECASE):
        return False, "hint_states_a_solution"
    words = set(re.findall(r"\w+", hint.casefold()))
    protected = set(re.findall(r"\w+", reference.casefold()))
    if protected & words:
        return False, "hint_contains_reference_term"
    if reference.strip().isdigit():
        from num2words import num2words
        protected = set(re.findall(r"\w+", num2words(int(reference.strip())).casefold()))
        if protected & words:
            return False, "hint_contains_reference_number_word"
    return True, None


def validate_review(value):
    if not isinstance(value, dict) or set(value) != {"verdict", "hint_leakage", "reason"}:
        raise ValueError("Candidate review fields do not match the schema")
    if value["verdict"] not in ("supported", "unsupported", "unresolved") or value["hint_leakage"] not in ("none_detected", "suspected", "unresolved"):
        raise ValueError("Unknown candidate review verdict")
    if not isinstance(value["reason"], str) or not value["reason"].strip() or len(value["reason"]) > 500:
        raise ValueError("Candidate review requires a bounded explanation")
    return value
