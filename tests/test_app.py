from fastapi.testclient import TestClient

import app
from extractor import InvalidImageError
from tests.conftest import as_decimals, make_png_bytes


client = TestClient(app.app)


def test_index_serves_upload_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "Invoice Extractor" in response.text
    assert 'id="form"' in response.text


def test_extract_returns_json(monkeypatch, valid_reply):
    monkeypatch.setattr(app, "extract_invoice", lambda *a, **k: as_decimals(valid_reply))
    monkeypatch.setattr(app, "get_model_name", lambda: "test-model")

    response = client.post(
        "/extract",
        files={"file": ("invoice.png", make_png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["source_file"] == "invoice.png"
    assert data["ai_model"] == "test-model"
    assert data["extracted"]["total"] == "113565.00"
    assert data["calculated"]["subtotal"] == "100500.00"


def test_extraction_failure_returns_400(monkeypatch):
    def boom(*args, **kwargs):
        raise InvalidImageError("bad image")

    monkeypatch.setattr(app, "extract_invoice", boom)

    response = client.post(
        "/extract",
        files={"file": ("invoice.png", make_png_bytes(), "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "bad image"


def test_empty_upload_is_rejected():
    response = client.post(
        "/extract",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_missing_file_field_is_rejected():
    response = client.post("/extract")
    assert response.status_code == 422
