import json
import urllib.request
from typing import Any, Dict, List, Optional


def ollama_chat(images_b64_png: List[str], ollama_url: str, model: str, object_class: str) -> Dict[str, Any]:
    url = ollama_url.rstrip("/") + "/api/chat"

    system = (
        "You are a visual captioner. Return STRICT JSON only.\n"
        "Schema:\n"
        "{"
        "\"summary_text\": string,"
        "\"details\": object,"
        "\"face_description\": string|null"
        "}\n"
        "details should include numeric proportions when possible (height, width, length, shoulder_width, head_to_body_ratio, etc.).\n"
        "If object_class != 'character', set face_description to null."
    )
    user = (
        "Describe the object in the images.\n"
        "object_class: {}\n"
        "Return JSON matching the schema."
    ).format(object_class)

    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user, "images": images_b64_png},
        ],
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    data = json.loads(raw)

    content = ""
    if isinstance(data, dict):
        msg = data.get("message") or {}
        content = msg.get("content") or ""

    parsed = None  # type: Optional[Dict[str, Any]]
    if isinstance(content, str) and content.strip():
        try:
            parsed = json.loads(content)  # type: ignore[assignment]
        except Exception:
            parsed = None
    if parsed is None and isinstance(data, dict) and all(k in data for k in ("summary_text", "details")):
        parsed = data  # type: ignore[assignment]

    if parsed is None:
        return {"summary_text": content.strip(), "details": {}, "face_description": None}

    summary_text = (parsed.get("summary_text") or "").strip()
    details = parsed.get("details") or {}
    face_description = parsed.get("face_description", None)
    if object_class != "character":
        face_description = None
    return {
        "summary_text": summary_text,
        "details": details if isinstance(details, dict) else {},
        "face_description": face_description if isinstance(face_description, str) and face_description.strip() else None,
    }

