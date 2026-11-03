"""Shared fixtures. All invoice data here is synthetic; no test calls the Groq API."""

import io
import json
from decimal import Decimal
from typing import Any

import pytest
from PIL import Image


def make_png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 20), "white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def png_file(tmp_path):
    path = tmp_path / "invoice.png"
    path.write_bytes(make_png_bytes())
    return path


@pytest.fixture
def valid_raw() -> dict[str, Any]:
    """What a correct model reply looks like: five items, 13% VAT, internally consistent."""
    return {
        "invoice_number": "INV-0001",
        "invoice_date": "2026-03-15",
        "supplier_name": "Example Supplies Pvt. Ltd.",
        "currency": "NPR",
        "items": [
            {"description": "Sample item 1", "quantity": 10, "unit_price": 2500.00, "line_total": 25000.00},
            {"description": "Sample item 2", "quantity": 5, "unit_price": 4000.00, "line_total": 20000.00},
            {"description": "Sample item 3", "quantity": 20, "unit_price": 750.00, "line_total": 15000.00},
            {"description": "Sample item 4", "quantity": 3, "unit_price": 12500.00, "line_total": 37500.00},
            {"description": "Sample item 5", "quantity": 2, "unit_price": 1500.00, "line_total": 3000.00},
        ],
        "subtotal": 100500.00,
        "vat_rate": 13,
        "vat_amount": 13065.00,
        "total": 113565.00,
    }


@pytest.fixture
def valid_reply(valid_raw) -> str:
    """The same data as the text a model would send back (JSON string)."""
    return json.dumps(valid_raw)


def as_decimals(text: str) -> Any:
    """Parse JSON the same way the extractor does."""
    return json.loads(text, parse_float=Decimal)
