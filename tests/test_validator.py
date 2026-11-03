import json
from datetime import date
from decimal import Decimal

import pytest

from models import parse_date, parse_decimal
from validator import build_result

TODAY = date(2026, 11, 3)


def run(raw, today=TODAY):
    return build_result(raw, source_file="test.png", ai_model="test-model", today=today)


def codes(result):
    return {issue.code for issue in result.issues}


def issue(result, code):
    return next(i for i in result.issues if i.code == code)


# ---------- happy path ----------

def test_consistent_invoice_has_no_issues(valid_raw):
    result = run(valid_raw)
    assert result.status == "ok"
    assert result.issues == []


def test_independent_calculation_with_decimal(valid_raw):
    calc = run(valid_raw).calculated
    assert calc.item_totals == [Decimal("25000.00"), Decimal("20000.00"), Decimal("15000.00"), Decimal("37500.00"), Decimal("3000.00")]
    assert calc.subtotal == Decimal("100500.00")
    assert calc.vat_amount == Decimal("13065.00")
    assert calc.total == Decimal("113565.00")


def test_no_float_rounding_errors():
    # 3 x 0.10 would be 0.30000000000000004 as a float
    raw = {
        "invoice_number": "1", "invoice_date": "2026-01-01", "supplier_name": "S", "currency": "NPR",
        "items": [{"description": "x", "quantity": 3, "unit_price": "0.10", "line_total": "0.30"}],
        "subtotal": "0.30", "vat_rate": 0, "vat_amount": "0.00", "total": "0.30",
    }
    result = run(raw)
    assert result.calculated.subtotal == Decimal("0.30")
    assert result.status == "ok"


def test_vat_rounds_half_up():
    raw = {
        "invoice_number": "1", "invoice_date": "2026-01-01", "supplier_name": "S", "currency": "NPR",
        "items": [{"description": "x", "quantity": 1, "unit_price": "10.05", "line_total": "10.05"}],
        "subtotal": "10.05", "vat_rate": 13, "vat_amount": "1.31", "total": "11.36",
    }
    result = run(raw)  # 10.05 x 13% = 1.3065 -> 1.31
    assert result.calculated.vat_amount == Decimal("1.31")
    assert result.status == "ok"


def test_string_amounts_with_currency_and_percent(valid_raw):
    valid_raw["subtotal"] = "NPR 100,500.00"
    valid_raw["total"] = "Rs. 113,565.00"
    valid_raw["vat_rate"] = "13%"
    assert run(valid_raw).status == "ok"


# ---------- calculation mismatches ----------

def test_item_total_mismatch(valid_raw):
    valid_raw["items"][2]["line_total"] = 15500.00
    result = run(valid_raw)
    found = issue(result, "ITEM_TOTAL_MISMATCH")
    assert found.severity == "error"
    assert found.field == "items[2].line_total"
    assert "Item 3" in found.message and "15,000.00" in found.message
    assert result.status == "errors"


def test_subtotal_mismatch(valid_raw):
    valid_raw["subtotal"] = 100000.00
    assert "SUBTOTAL_MISMATCH" in codes(run(valid_raw))


def test_vat_amount_mismatch(valid_raw):
    valid_raw["vat_amount"] = 13000.00
    assert "VAT_AMOUNT_MISMATCH" in codes(run(valid_raw))


def test_total_mismatch(valid_raw):
    valid_raw["total"] = 113000.00
    found = issue(run(valid_raw), "TOTAL_MISMATCH")
    assert "113,000.00" in found.message and "113,565.00" in found.message


def test_one_cent_difference_is_tolerated(valid_raw):
    valid_raw["vat_amount"] = 13065.01
    valid_raw["total"] = 113565.01
    assert run(valid_raw).status == "ok"


# ---------- missing information ----------

def test_empty_extraction_reports_everything_missing():
    result = run({})
    missing = {i.field for i in result.issues if i.code == "MISSING_FIELD"}
    assert {"invoice_number", "invoice_date", "supplier_name", "items", "subtotal", "vat_rate", "vat_amount", "total"} <= missing
    assert result.status == "errors"
    assert result.calculated.subtotal is None


def test_null_items_is_treated_as_missing(valid_raw):
    valid_raw["items"] = None
    result = run(valid_raw)
    assert "items" in {i.field for i in result.issues if i.code == "MISSING_FIELD"}


def test_missing_item_fields(valid_raw):
    valid_raw["items"][0] = {"description": None, "quantity": None, "unit_price": 2500, "line_total": 25000}
    result = run(valid_raw)
    by_field = {i.field: i for i in result.issues if i.code == "MISSING_FIELD"}
    assert by_field["items[0].description"].severity == "warning"
    assert by_field["items[0].quantity"].severity == "error"
    # the printed line total stands in for the unreadable quantity, so the subtotal is still checked
    assert result.calculated.subtotal == Decimal("100500.00")


def test_missing_line_total_is_only_a_warning_when_computable(valid_raw):
    valid_raw["items"][1]["line_total"] = None
    result = run(valid_raw)
    found = next(i for i in result.issues if i.field == "items[1].line_total")
    assert found.severity == "warning"
    assert result.status == "warnings"


def test_missing_vat_rate_means_vat_cannot_be_verified(valid_raw):
    valid_raw["vat_rate"] = None
    result = run(valid_raw)
    assert result.calculated.vat_amount is None
    assert {"MISSING_FIELD", "VAT_AMOUNT_UNVERIFIED", "TOTAL_UNVERIFIED"} <= codes(result)


def test_missing_currency_is_a_warning(valid_raw):
    valid_raw["currency"] = None
    result = run(valid_raw)
    assert result.status == "warnings"
    assert issue(result, "MISSING_FIELD").field == "currency"


# ---------- malformed values are reported, never raised ----------

def test_ambiguous_number_is_reported_and_rest_still_validated(valid_raw):
    valid_raw["items"][0]["quantity"] = "2 x 5"
    result = run(valid_raw)
    invalid = issue(result, "INVALID_FIELD")
    assert invalid.field == "items[0].quantity"
    # not double-reported as missing, and other checks still ran
    assert not any(i.code == "MISSING_FIELD" and i.field == "items[0].quantity" for i in result.issues)
    assert result.calculated.item_totals[1] == Decimal("20000.00")


def test_unparseable_date_is_reported(valid_raw):
    valid_raw["invoice_date"] = "last Tuesday"
    result = run(valid_raw)
    assert issue(result, "INVALID_FIELD").field == "invoice_date"
    assert result.extracted.invoice_date is None


def test_items_not_a_list_and_item_not_a_dict(valid_raw):
    valid_raw["items"] = "five items"
    assert "INVALID_FIELD" in codes(run(valid_raw))

    valid_raw["items"] = ["oops", {"description": "ok", "quantity": 1, "unit_price": 5, "line_total": 5}]
    result = run(valid_raw)
    assert issue(result, "INVALID_FIELD").field == "items[0]"
    assert len(result.extracted.items) == 2


def test_extra_fields_are_ignored(valid_raw):
    valid_raw["notes"] = "thank you"
    assert run(valid_raw).status == "ok"


def test_numeric_invoice_number_is_accepted(valid_raw):
    valid_raw["invoice_number"] = 1042
    assert run(valid_raw).extracted.invoice_number == "1042"


# ---------- sanity warnings ----------

def test_future_date_warns_about_bikram_sambat(valid_raw):
    valid_raw["invoice_date"] = "2083-07-15"
    found = issue(run(valid_raw), "DATE_IN_FUTURE")
    assert found.severity == "warning" and "Bikram Sambat" in found.message


def test_fractional_vat_rate_warns(valid_raw):
    valid_raw["vat_rate"] = 0.13
    assert "VAT_RATE_LOOKS_FRACTIONAL" in codes(run(valid_raw))


def test_negative_and_zero_values_warn(valid_raw):
    valid_raw["items"][0]["quantity"] = 0
    valid_raw["items"][0]["line_total"] = 0
    valid_raw["items"][1]["unit_price"] = -4000
    valid_raw["items"][1]["line_total"] = -20000
    assert {"NON_POSITIVE_QUANTITY", "NEGATIVE_AMOUNT"} <= codes(run(valid_raw))


# ---------- output format ----------

def test_json_output_formats_amounts_as_exact_strings(valid_raw):
    data = json.loads(run(valid_raw).model_dump_json())
    assert data["extracted"]["subtotal"] == "100500.00"
    assert data["extracted"]["vat_rate"] == "13"
    assert data["extracted"]["invoice_date"] == "2026-03-15"
    assert data["extracted"]["items"][0]["quantity"] == "10"
    assert data["calculated"]["total"] == "113565.00"
    assert data["status"] == "ok"


# ---------- parsing helpers ----------

@pytest.mark.parametrize(
    "value, expected",
    [
        ("NPR 100,500.00", Decimal("100500.00")),
        ("Rs. 1,000", Decimal("1000")),
        ("13%", Decimal("13")),
        (".5", Decimal("0.5")),
        ("(1,200.00)", Decimal("-1200.00")),
        (7, Decimal("7")),
        (2.5, Decimal("2.5")),
        ("N/A", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_decimal(value, expected):
    assert parse_decimal(value) == expected


@pytest.mark.parametrize("value", ["2 x 5", "2026-03-15", True, [1], "1" + "0" * 20])
def test_parse_decimal_rejects_unsafe_values(value):
    with pytest.raises(ValueError):
        parse_decimal(value)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("2026-03-15", date(2026, 3, 15)),
        ("15/03/2026", date(2026, 3, 15)),
        ("15 Mar 2026", date(2026, 3, 15)),
        ("March 15, 2026", date(2026, 3, 15)),
    ],
)
def test_parse_date(value, expected):
    assert parse_date(value) == expected
