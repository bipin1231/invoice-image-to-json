"""Send an invoice image to Groq's vision model and return the parsed JSON.

Nothing here knows what a "valid" invoice looks like; see validator.py for that.
"""

import base64
import io
import json
import os
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import groq
from groq import Groq
from PIL import Image, UnidentifiedImageError

# Verified against https://console.groq.com/docs/vision on 2026-11-03. Groq lists this model as
# "Preview", so it can change; override it with GROQ_MODEL without touching the code.
DEFAULT_MODEL = "qwen/qwen3.8-27b"

MAX_IMAGE_BYTES = 20 * 1024 * 1024  # Groq's documented 20 MB image limit
SUPPORTED_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
MAX_COMPLETION_TOKENS = 4096

PROMPT = """You are reading a single invoice image. Extract the data and reply with ONE JSON object and nothing else.

Use exactly this structure (use null for anything that is not visible or not legible):
{
  "invoice_number": string | null,
  "invoice_date": "YYYY-MM-DD" | null,
  "supplier_name": string | null,
  "currency": string | null,
  "items": [
    {"description": string | null, "quantity": number | null, "unit_price": number | null, "line_total": number | null}
  ],
  "subtotal": number | null,
  "vat_rate": number | null,
  "vat_amount": number | null,
  "total": number | null
}

Rules:
- Copy values exactly as printed on the invoice. Do NOT recalculate, correct or "fix" any amount, even if the arithmetic looks wrong.
- Numbers must be plain JSON numbers: no currency symbols, no thousands separators.
- vat_rate is a percentage number (13 for 13%).
- invoice_date: write the printed date as YYYY-MM-DD without converting between calendars.
- currency: the currency code or symbol printed on the invoice, if any.
- supplier_name is the business that issued the invoice, not the customer.
- List every line item in the order shown. Do not invent items, and do not include subtotal/VAT/total rows as items.
"""


class ExtractionError(Exception):
    """Base class for every failure the command line reports without a traceback."""


class ConfigurationError(ExtractionError):
    """The API key (or other required setting) is missing."""


class InvalidImageError(ExtractionError):
    """The input is missing, unreadable, too large, or not a supported image."""


class GroqAPIError(ExtractionError):
    """The Groq request failed."""


class MalformedResponseError(ExtractionError):
    """The model answered, but not with a usable JSON object."""


def get_model_name() -> str:
    return os.environ.get("GROQ_MODEL", "").strip() or DEFAULT_MODEL


def create_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "your_groq_api_key_here":
        raise ConfigurationError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "(create one at https://console.groq.com/keys)."
        )
    return Groq(api_key=api_key, max_retries=2, timeout=60.0)


def load_image(path: Path) -> tuple[bytes, str]:
    """Read and verify the image; returns (bytes, mime type)."""
    if not path.exists():
        raise InvalidImageError(f"Image not found: {path}")
    if not path.is_file():
        raise InvalidImageError(f"Not a file: {path}")
    try:
        size = path.stat().st_size
        if size == 0:
            raise InvalidImageError(f"Image file is empty: {path}")
        if size > MAX_IMAGE_BYTES:
            raise InvalidImageError(
                f"Image is {size / 1024 / 1024:.1f} MB; the limit is {MAX_IMAGE_BYTES // 1024 // 1024} MB."
            )
        data = path.read_bytes()
    except OSError as exc:
        raise InvalidImageError(f"Cannot read {path}: {exc}") from exc

    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = img.format
            img.verify()
    except UnidentifiedImageError as exc:
        raise InvalidImageError(
            f"{path.name} is not a recognised image file. Use a JPEG, PNG or WebP image."
        ) from exc
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(f"{path.name} is damaged or not a valid image ({exc}).") from exc

    if fmt not in SUPPORTED_FORMATS:
        raise InvalidImageError(
            f"Unsupported image format {fmt!r}. Use one of: {', '.join(sorted(SUPPORTED_FORMATS))}."
        )
    return data, SUPPORTED_FORMATS[fmt]


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _loads(text: str) -> Any:
    # parse_float=Decimal keeps amounts exact; floats never touch the money.
    return json.loads(text, parse_float=Decimal)


def parse_model_json(content: str | None) -> dict[str, Any]:
    """Turn the model's reply into a dict, tolerating code fences and stray prose."""
    if not content or not content.strip():
        raise MalformedResponseError("The model returned an empty response.")

    text = _THINK_BLOCK.sub("", content).strip()
    fenced = _CODE_FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        data = _loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start == -1:
            raise MalformedResponseError(
                f"The model did not return JSON. Response began: {text[:200]!r}"
            ) from None
        end = text.rfind("}")
        candidate = text[start : end + 1] if end > start else text[start:]
        try:
            data = _loads(candidate)
        except json.JSONDecodeError as exc:
            raise MalformedResponseError(
                f"The model returned malformed JSON ({exc.msg} at line {exc.lineno}, column {exc.colno}). "
                f"Response began: {text[:200]!r}"
            ) from exc

    if not isinstance(data, dict):
        raise MalformedResponseError(
            f"Expected a JSON object but the model returned a {type(data).__name__}."
        )
    return data


def _translate_api_error(exc: groq.APIError, model: str) -> GroqAPIError:
    if isinstance(exc, groq.AuthenticationError):
        msg = "Groq rejected the API key (401). Check GROQ_API_KEY in your .env file."
    elif isinstance(exc, groq.PermissionDeniedError):
        msg = f"Access denied (403). Your key or organisation may not be allowed to use model '{model}'."
    elif isinstance(exc, groq.NotFoundError):
        msg = (
            f"Model '{model}' was not found (404). It may have been renamed or retired; "
            "see https://console.groq.com/docs/vision and set GROQ_MODEL."
        )
    elif isinstance(exc, groq.BadRequestError):
        msg = f"Groq rejected the request (400): {exc.message} The image may be too large or unsupported."
    elif isinstance(exc, groq.RateLimitError):
        msg = "Groq rate limit reached (429). Wait a moment and try again."
    elif isinstance(exc, groq.APITimeoutError):
        msg = "The request to Groq timed out. Try again."
    elif isinstance(exc, groq.APIConnectionError):
        msg = "Could not connect to Groq. Check your internet connection."
    elif isinstance(exc, groq.APIStatusError):
        msg = f"Groq API error ({exc.status_code}): {exc.message}"
    else:
        msg = f"Groq API error: {exc}"
    return GroqAPIError(msg)


def extract_invoice(
    image_path: str | Path, *, client: Groq | None = None, model: str | None = None
) -> dict[str, Any]:
    """Send the image straight to the vision model and return its JSON answer as a dict."""
    image, mime = load_image(Path(image_path))
    client = client or create_client()
    model = model or get_model_name()
    data_uri = f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            temperature=0,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            response_format={"type": "json_object"},
            # Qwen 3.8 has thinking and instruct modes; "none" selects instruct mode, which is
            # faster and keeps reasoning text out of the JSON reply.
            extra_body={"reasoning_effort": "none"},
        )
    except groq.APIError as exc:
        raise _translate_api_error(exc, model) from exc

    if not completion.choices:
        raise MalformedResponseError("The model returned no choices.")
    choice = completion.choices[0]
    if choice.finish_reason == "length":
        raise MalformedResponseError(
            "The model's reply was cut off before the JSON was complete (token limit reached)."
        )
    return parse_model_json(choice.message.content)
