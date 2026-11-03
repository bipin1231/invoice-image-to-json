"""Pydantic models for the extracted invoice, the independent calculations and the report.

All monetary values are ``Decimal``. Floats are never used for money: the extractor
parses model output with ``parse_float=Decimal`` and the helpers below convert anything
else (strings such as "NPR 100,500.00" or "13%") into a ``Decimal``.
"""

import re
from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field, PlainSerializer

# A single number such as 1,200.50 / -3 / .5. Currency codes and symbols around it are ignored.
_NUMBER_TOKEN = re.compile(r"-?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")
_MAX_ABS_VALUE = Decimal("1e15")

# Day-first formats are tried before month-first ones (invoices in Nepal use DD/MM/YYYY).
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%b %d %Y",
    "%B %d %Y",
)


def parse_decimal(value: Any) -> Decimal | None:
    """Convert numbers or strings like 'NPR 100,500.00' or '13%' into a ``Decimal``.

    Returns None for empty or non-numeric placeholders ('', 'N/A', '-').
    Raises ValueError for values that look numeric but cannot be read safely
    (for example '2 x 5'), so the caller can report them instead of guessing.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Expected a number, got {value!r}")
    if isinstance(value, Decimal):
        number = value
    elif isinstance(value, int):
        number = Decimal(value)
    elif isinstance(value, float):
        number = Decimal(str(value))
    elif isinstance(value, str):
        tokens = _NUMBER_TOKEN.findall(value)
        if not tokens:
            return None
        if len(tokens) > 1:
            raise ValueError(f"Ambiguous number: {value!r}")
        number = Decimal(tokens[0].replace(",", ""))
        stripped = value.strip()
        if stripped.startswith("(") and stripped.endswith(")"):
            number = -number  # accounting-style negative
    else:
        raise ValueError(f"Expected a number, got {type(value).__name__}")
    if not number.is_finite() or abs(number) >= _MAX_ABS_VALUE:
        raise ValueError(f"Number out of range: {value!r}")
    return number


def parse_text(value: Any) -> str | None:
    """Strip text; accept numbers (e.g. an invoice number returned as an int)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError(f"Expected text, got {value!r}")
    text = str(value).strip()
    return text or None


def parse_date(value: Any) -> date | None:
    """Parse ISO dates plus a handful of common printed formats."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
        raise ValueError(f"Unrecognised date format: {value!r}")
    raise ValueError(f"Expected a date, got {type(value).__name__}")


def _format_money(value: Decimal | None) -> str | None:
    """Money as a plain string with at least two decimals, e.g. '100500.00'."""
    if value is None:
        return None
    if value.as_tuple().exponent >= -1:  # type: ignore[operator]
        value = value.quantize(Decimal("0.01"))
    return format(value, "f")


def _format_number(value: Decimal | None) -> str | None:
    """Quantities and rates without trailing zeros, e.g. '13' or '2.5'."""
    if value is None:
        return None
    return format(value.normalize(), "f")


Text = Annotated[str | None, BeforeValidator(parse_text)]
Day = Annotated[date | None, BeforeValidator(parse_date)]
Money = Annotated[
    Decimal | None,
    BeforeValidator(parse_decimal),
    PlainSerializer(_format_money, return_type=str | None),
]
Number = Annotated[
    Decimal | None,
    BeforeValidator(parse_decimal),
    PlainSerializer(_format_number, return_type=str | None),
]


class LineItem(BaseModel):
    description: Text = None
    quantity: Number = None
    unit_price: Money = None
    line_total: Money = None


class ExtractedInvoice(BaseModel):
    """Values exactly as read from the invoice (nothing here is recalculated)."""

    invoice_number: Text = None
    invoice_date: Day = None
    supplier_name: Text = None
    currency: Text = None
    items: Annotated[list[LineItem], BeforeValidator(lambda v: [] if v is None else v)] = Field(
        default_factory=list
    )
    subtotal: Money = None
    vat_rate: Number = None  # percentage, e.g. 13 means 13%
    vat_amount: Money = None
    total: Money = None


class CalculatedTotals(BaseModel):
    """Values recomputed independently with Decimal from quantity x unit price."""

    item_totals: list[Money] = Field(default_factory=list)  # aligned with extracted.items
    subtotal: Money = None
    vat_amount: Money = None
    total: Money = None


class Issue(BaseModel):
    severity: Literal["error", "warning"]
    code: str
    field: str | None = None
    message: str


class InvoiceResult(BaseModel):
    source_file: str
    ai_model: str
    status: Literal["ok", "warnings", "errors"]
    extracted: ExtractedInvoice
    calculated: CalculatedTotals
    issues: list[Issue] = Field(default_factory=list)
