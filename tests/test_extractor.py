import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import groq
import httpx
import pytest

import extractor
from extractor import (
    ConfigurationError,
    GroqAPIError,
    InvalidImageError,
    MalformedResponseError,
    extract_invoice,
    load_image,
    parse_model_json,
)


def fake_client(content: str | None = "{}", finish_reason: str = "stop") -> MagicMock:
    client = MagicMock()
    message = SimpleNamespace(content=content)
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)]
    )
    return client


def api_error(cls, status: int, **kwargs):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return cls("boom", response=httpx.Response(status, request=request), body=None, **kwargs)


# ---------- image loading ----------

def test_load_image_returns_bytes_and_mime(png_file):
    data, mime = load_image(png_file)
    assert mime == "image/png" and data.startswith(b"\x89PNG")


def test_missing_file(tmp_path):
    with pytest.raises(InvalidImageError, match="not found"):
        load_image(tmp_path / "nope.png")


def test_directory_is_rejected(tmp_path):
    with pytest.raises(InvalidImageError, match="Not a file"):
        load_image(tmp_path)


def test_empty_file(tmp_path):
    path = tmp_path / "empty.png"
    path.write_bytes(b"")
    with pytest.raises(InvalidImageError, match="empty"):
        load_image(path)


def test_text_file_named_png(tmp_path):
    path = tmp_path / "fake.png"
    path.write_text("definitely not an image")
    with pytest.raises(InvalidImageError, match="not a recognised image"):
        load_image(path)


def test_truncated_image(png_file):
    png_file.write_bytes(png_file.read_bytes()[:30])
    with pytest.raises(InvalidImageError, match="damaged|recognised"):
        load_image(png_file)


def test_unsupported_format(tmp_path):
    from PIL import Image

    path = tmp_path / "scan.bmp"
    Image.new("RGB", (10, 10)).save(path, format="BMP")
    with pytest.raises(InvalidImageError, match="Unsupported image format 'BMP'"):
        load_image(path)


def test_oversized_file(tmp_path, monkeypatch):
    monkeypatch.setattr(extractor, "MAX_IMAGE_BYTES", 10)
    path = tmp_path / "big.png"
    path.write_bytes(b"x" * 11)
    with pytest.raises(InvalidImageError, match="limit"):
        load_image(path)


# ---------- reply parsing ----------

def test_plain_json_and_decimal_floats():
    data = parse_model_json('{"subtotal": 100500.10, "n": 3}')
    assert data["subtotal"] == Decimal("100500.10") and isinstance(data["subtotal"], Decimal)
    assert data["n"] == 3


def test_json_in_code_fence():
    assert parse_model_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_json_wrapped_in_prose():
    assert parse_model_json('Here you go: {"a": 1} Hope that helps!') == {"a": 1}


def test_thinking_block_is_removed():
    assert parse_model_json('<think>the total is {oops}</think>{"a": 1}') == {"a": 1}


@pytest.mark.parametrize("content", [None, "", "   \n"])
def test_empty_reply(content):
    with pytest.raises(MalformedResponseError, match="empty"):
        parse_model_json(content)


def test_reply_without_json():
    with pytest.raises(MalformedResponseError, match="did not return JSON"):
        parse_model_json("Sorry, I cannot read this image.")


def test_malformed_json():
    with pytest.raises(MalformedResponseError, match="malformed JSON"):
        parse_model_json('{"invoice_number": "A1", "items": [')


def test_json_array_is_rejected():
    with pytest.raises(MalformedResponseError, match="JSON object"):
        parse_model_json("[1, 2, 3]")


# ---------- extract_invoice with a mocked Groq client ----------

def test_extract_invoice_sends_image_and_returns_dict(png_file, valid_reply):
    client = fake_client(valid_reply)
    result = extract_invoice(png_file, client=client, model="test-model")

    assert result["invoice_number"] == "INV-0001"
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["response_format"] == {"type": "json_object"}
    parts = kwargs["messages"][0]["content"]
    image_part = next(p for p in parts if p["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert any(p["type"] == "text" for p in parts)


def test_prompt_contains_no_invoice_values():
    for forbidden in ("Rivertech", "100,500", "113,565", "100500", "113565"):
        assert forbidden not in extractor.PROMPT


def test_invalid_image_never_reaches_the_api(tmp_path):
    path = tmp_path / "bad.png"
    path.write_text("nope")
    client = fake_client()
    with pytest.raises(InvalidImageError):
        extract_invoice(path, client=client)
    client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize(
    "error, fragment",
    [
        (api_error(groq.AuthenticationError, 401), "API key"),
        (api_error(groq.RateLimitError, 429), "rate limit"),
        (api_error(groq.NotFoundError, 404), "GROQ_MODEL"),
        (api_error(groq.BadRequestError, 400), "too large"),
        (api_error(groq.PermissionDeniedError, 403), "Access denied"),
        (api_error(groq.InternalServerError, 500), "500"),
        (
            groq.APIConnectionError(request=httpx.Request("POST", "https://api.groq.com")),
            "connect",
        ),
        (
            groq.APITimeoutError(request=httpx.Request("POST", "https://api.groq.com")),
            "timed out",
        ),
    ],
)
def test_api_errors_become_friendly_messages(png_file, error, fragment):
    client = MagicMock()
    client.chat.completions.create.side_effect = error
    with pytest.raises(GroqAPIError) as info:
        extract_invoice(png_file, client=client, model="m")
    assert fragment.lower() in str(info.value).lower()


def test_truncated_reply_is_reported(png_file):
    client = fake_client('{"invoice_number": "A', finish_reason="length")
    with pytest.raises(MalformedResponseError, match="cut off"):
        extract_invoice(png_file, client=client)


def test_no_choices(png_file):
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(choices=[])
    with pytest.raises(MalformedResponseError, match="no choices"):
        extract_invoice(png_file, client=client)


def test_malformed_json_from_model(png_file):
    with pytest.raises(MalformedResponseError):
        extract_invoice(png_file, client=fake_client("{not json at all"))


# ---------- configuration ----------

def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
        extractor.create_client()


def test_placeholder_api_key_is_rejected(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "your_groq_api_key_here")
    with pytest.raises(ConfigurationError):
        extractor.create_client()


def test_model_name_from_environment(monkeypatch):
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    assert extractor.get_model_name() == extractor.DEFAULT_MODEL
    monkeypatch.setenv("GROQ_MODEL", "some/other-model")
    assert extractor.get_model_name() == "some/other-model"
