"""Qwen3.5-9B served by Ollama: baseline answers and reference-free audits."""

import json
import time

from PIL import Image

import evaluate_feedback as feedback
from check_primary import local_json
from src.benchmarks import file_sha256, image_path

MODEL = feedback.MODEL
AUDIT_FAMILIES = ("V", "R", "D", "K")


def original_image(data_dir, record):
    path = image_path(data_dir, record["image_path"])
    if file_sha256(path) != record["image_file_sha256"]:
        raise ValueError(f"Image changed for {record['id']}")
    with Image.open(path) as image:
        return image.convert("RGB")


def server_identity(transport=local_json):
    version = transport("/api/version")
    tags = transport("/api/tags").get("models", [])
    tag = next((m for m in tags if m.get("name") == MODEL or m.get("model") == MODEL), None)
    if not tag or not tag.get("digest"):
        raise ValueError(f"{MODEL} is not installed in Ollama")
    details = transport("/api/show", {"model": MODEL})
    if "vision" not in details.get("capabilities", []):
        raise ValueError("Ollama does not report vision support for the baseline model")
    return {"ollama_version": version, "model": MODEL, "model_digest": tag["digest"]}


def answer(data_dir, record, num_predict, transport=local_json, timeout=300):
    """Exactly the 119-run request (paper prompt, think=false, same options) except num_predict."""
    payload = feedback.request_payload(original_image(data_dir, record), record["question"])
    payload["options"]["num_predict"] = num_predict
    started = time.perf_counter()
    reply = transport("/api/chat", payload, timeout=timeout)
    prediction = feedback.response_prediction(reply, record["id"], time.perf_counter() - started)
    prediction["hit_token_limit"] = reply["done_reason"] == "length" or reply["eval_count"] >= num_predict
    return payload, reply, prediction


AUDIT_SCHEMA = {
    "type": "object",
    "required": ["action", "first_error_step", "error_type", "evidence", "hint"],
    "properties": {
        "action": {"type": "string", "enum": ["repair", "skip", "unresolved"]},
        "first_error_step": {"type": "integer"},
        "error_type": {"type": "string", "enum": ["V", "R", "D", "K", "none"]},
        "evidence": {"type": "string"},
        "hint": {"type": "string"},
    },
}

AUDIT_INSTRUCTION = (
    "You audit a model's medical image reasoning. No reference answer is given and you must not guess one. "
    "Read the image, the question and the numbered reasoning steps. Find the FIRST step that is not supported "
    "by the image or by medical knowledge and that changes the conclusion. Classify it: V = visual misreading "
    "(wrong finding, location, modality or laterality), R = invalid inference from correct observations, "
    "D = failed differential comparison, K = missing or wrong prerequisite knowledge. "
    "Every step before first_error_step must be supported. If step 1 is already wrong, use 1. "
    "Write a hint: ONE short imperative instruction (under 30 words) telling the model what to re-examine "
    "or reconsider. The hint must NOT state the answer, a diagnosis, a yes/no verdict, an option, or a corrected "
    "finding. Example of a good hint: 'Compare the signal intensity on both sides of the midline before deciding.' "
    "Use action 'skip' (first_error_step 0, empty hint, error_type none) if the reasoning has no real error "
    "or only a formatting problem. Use 'unresolved' (step 0, empty hint, error_type none) if the image is "
    "insufficient to judge. Return only JSON matching the schema.")


def audit_payload(data_dir, record, response_text, steps, think=False, num_predict=512):
    image = original_image(data_dir, record)
    encoded = feedback.image_content(image)["image_url"]["url"].split(",", 1)[1]
    visible = {"question": record["question"],
               "steps": [{"step": s["step"], "text": s["text"].strip()} for s in steps]}
    return {"model": MODEL, "think": think, "stream": False, "keep_alive": "5m", "format": AUDIT_SCHEMA,
            "messages": [{"role": "system", "content": AUDIT_INSTRUCTION},
                         {"role": "user", "content": json.dumps(visible, ensure_ascii=False), "images": [encoded]}],
            "options": {"temperature": 0, "seed": 7, "num_predict": num_predict, "num_ctx": 6144,
                        "num_thread": 4, "num_gpu": 0}}


def audit(data_dir, record, response_text, steps, transport=local_json, think=False, timeout=600):
    payload = audit_payload(data_dir, record, response_text, steps, think)
    started = time.perf_counter()
    reply = transport("/api/chat", payload, timeout=timeout)
    seconds = round(time.perf_counter() - started, 3)
    content = reply.get("message", {}).get("content", "")
    try:
        value = json.loads(content)
    except (TypeError, ValueError):
        value = None
    return payload, reply, value, seconds


def validate_audit(value, steps):
    """Returns (decision, reason). decision in repair/skip/unresolved/invalid."""
    if not isinstance(value, dict):
        return "invalid", "audit is not a JSON object"
    action = value.get("action")
    step = value.get("first_error_step")
    hint = value.get("hint", "")
    if action in ("skip", "unresolved"):
        return action, "auditor declined repair"
    if action != "repair":
        return "invalid", "unknown action"
    if type(step) is not int or not 1 <= step <= len(steps):
        return "invalid", f"first_error_step {step!r} outside 1..{len(steps)}"
    if value.get("error_type") not in AUDIT_FAMILIES:
        return "invalid", "repair needs error_type V/R/D/K"
    if not isinstance(hint, str) or not hint.strip():
        return "invalid", "repair needs a non-empty hint"
    if len(hint.split()) > 40 or len(hint) > 300:
        return "invalid", "hint too long"
    if not isinstance(value.get("evidence"), str) or not value["evidence"].strip():
        return "invalid", "repair needs evidence"
    return "repair", "valid repair proposal"


# ---------------------------------------------------------------- audit v2 (MedEvo Appendix D.3)

AUDIT_SCHEMA_V2 = {
    "type": "object",
    "required": ["action", "first_error_step", "error_type", "key_error", "hint"],
    "properties": {
        "action": {"type": "string", "enum": ["repair", "skip"]},
        "first_error_step": {"type": "integer"},
        "error_type": {"type": "string", "enum": ["V", "R", "D", "K", "none"]},
        "key_error": {"type": "string"},
        "hint": {"type": "string"},
    },
}

AUDIT_INSTRUCTION_V2 = (
    "You are a senior medical reviewer auditing a FAILED multimodal reasoning trajectory segmented into "
    "numbered steps. Its final answer did not match the reference answer. The reference is hidden from you: "
    "do not guess it and do not state it.\n"
    "Using the image, the question and the dependencies between steps, identify the first consequential "
    "failure: the earliest step whose unsupported observation, invalid inference, incorrect differential "
    "comparison or missing medical principle leads to the failed answer. Trace dependencies backward: if a "
    "later mistake follows from an earlier unsupported premise, report the earlier step. Keep every preceding "
    "step that is medically and logically valid.\n"
    "Error types: V = visual misreading (finding, region, laterality, modality, sequence); R = invalid "
    "inference from correctly read evidence; D = incorrect differential comparison; K = missing or misapplied "
    "medical knowledge.\n"
    "Write one recovery hint for the diagnosed failure, under 30 words: V - direct attention to the relevant "
    "region or visual attribute; R - revisit the unsupported inference or reconnect the evidence to the "
    "question; D - compare a discriminative feature or reconsider an underexamined alternative; K - name the "
    "relevant concept or rule. The hint must not give the corrected finding, the diagnosis, the final answer, "
    "a yes/no verdict or an answer option.\n"
    "Return action 'skip' only when the reasoning content is sound and only the answer formatting is wrong "
    "(for example a correct yes/no conclusion written as 1 or 0); then use first_error_step 0, error_type "
    "'none' and an empty hint. Otherwise return 'repair'. key_error is one sentence describing the failure. "
    "Return only JSON matching the schema.")


def audit_v2(data_dir, record, steps, transport=local_json, think=False, timeout=600):
    payload = audit_payload(data_dir, record, "", steps, think)
    payload["format"] = AUDIT_SCHEMA_V2
    payload["messages"][0]["content"] = AUDIT_INSTRUCTION_V2
    started = time.perf_counter()
    reply = transport("/api/chat", payload, timeout=timeout)
    seconds = round(time.perf_counter() - started, 3)
    try:
        value = json.loads(reply.get("message", {}).get("content", ""))
    except (TypeError, ValueError):
        value = None
    return payload, reply, value, seconds


def validate_audit_v2(value, steps):
    """Returns (decision, reason). decision in repair/skip/invalid."""
    if not isinstance(value, dict):
        return "invalid", "audit is not a JSON object"
    if value.get("action") == "skip":
        return "skip", "auditor judged the failure to be formatting only"
    if value.get("action") != "repair":
        return "invalid", "unknown action"
    step, hint = value.get("first_error_step"), value.get("hint", "")
    if type(step) is not int or not 1 <= step <= len(steps):
        return "invalid", f"first_error_step {step!r} outside 1..{len(steps)}"
    if value.get("error_type") not in AUDIT_FAMILIES:
        return "invalid", "repair needs error_type V/R/D/K"
    if not isinstance(hint, str) or not hint.strip():
        return "invalid", "repair needs a non-empty hint"
    if len(hint.split()) > 40 or len(hint) > 300:
        return "invalid", "hint too long"
    if not isinstance(value.get("key_error"), str) or not value["key_error"].strip():
        return "invalid", "repair needs key_error"
    return "repair", "valid repair proposal"
