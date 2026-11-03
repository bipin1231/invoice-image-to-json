"""Validate extracted invoice data and independently recalculate every amount with Decimal."""

import copy
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, localcontext
from typing import Any

from pydantic import ValidationError

from models import CalculatedTotals, ExtractedInvoice, InvoiceResult, Issue

CENT = Decimal("0.01")
TOLERANCE = Decimal("0.01")  # differences up to one cent are treated as rounding
HUNDRED = Decimal(100)
MAX_PARSE_PASSES = 3

REQUIRED_FIELDS = (
    ("invoice_number", "Invoice number"),
    ("invoice_date", "Invoice date"),
    ("supplier_name", "Supplier name"),
    ("subtotal", "Subtotal"),
    ("vat_rate", "VAT rate"),
    ("vat_amount", "VAT amount"),
    ("total", "Total amount"),
)


def money(value: Decimal) -> Decimal:
    """Round to two decimal places, half up (the usual invoicing convention)."""
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def _fmt(value: Decimal) -> str:
    return f"{value:,.2f}"


def _fmt_qty(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _field_path(loc: tuple[int | str, ...]) -> str:
    path = ""
    for part in loc:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += f".{part}" if path else str(part)
    return path


def _clear(data: Any, loc: tuple[int | str, ...]) -> bool:
    """Blank out the value at ``loc`` so the rest of the data can still be validated."""
    if not loc:
        return False
    parent = data
    for part in loc[:-1]:
        try:
            parent = parent[part]
        except (KeyError, IndexError, TypeError):
            return False
    try:
        parent[loc[-1]] = None if isinstance(parent, dict) else {}
    except (KeyError, IndexError, TypeError):
        return False
    return True


def parse_extraction(raw: dict[str, Any]) -> tuple[ExtractedInvoice, list[Issue]]:
    """Validate raw model output with Pydantic.

    Fields that fail validation are reported as INVALID_FIELD issues and blanked, so one
    bad value never prevents the remaining data from being checked.
    """
    data = copy.deepcopy(raw)
    issues: list[Issue] = []
    seen: set[str] = set()
    for _ in range(MAX_PARSE_PASSES):
        try:
            return ExtractedInvoice.model_validate(data), issues
        except ValidationError as exc:
            progressed = False
            for err in exc.errors():
                loc = tuple(err["loc"])
                path = _field_path(loc)
                if path not in seen:
                    seen.add(path)
                    message = err["msg"].removeprefix("Value error, ")
                    if err["type"] != "value_error":
                        message += f" (got {str(err['input'])[:60]!r})"
                    issues.append(
                        Issue(
                            severity="error",
                            code="INVALID_FIELD",
                            field=path or None,
                            message=f"Invalid value for {path or 'invoice'}: {message}",
                        )
                    )
                progressed |= _clear(data, loc)
            if not progressed:
                break
    issues.append(
        Issue(
            severity="error",
            code="UNPARSEABLE",
            message="The extracted data could not be interpreted at all.",
        )
    )
    return ExtractedInvoice(), issues


def calculate_totals(invoice: ExtractedInvoice) -> CalculatedTotals:
    """Recompute item totals, subtotal, VAT and total from quantity x unit price."""
    with localcontext() as ctx:
        ctx.prec = 50
        item_totals: list[Decimal | None] = []
        amounts: list[Decimal | None] = []
        for item in invoice.items:
            if item.quantity is not None and item.unit_price is not None:
                calculated = money(item.quantity * item.unit_price)
            else:
                calculated = None
            item_totals.append(calculated)
            # Fall back to the printed line total only when quantity or price is unreadable.
            amounts.append(calculated if calculated is not None else item.line_total)

        subtotal: Decimal | None = None
        if amounts and all(a is not None for a in amounts):
            subtotal = money(sum((a for a in amounts if a is not None), Decimal(0)))

        vat_amount: Decimal | None = None
        if subtotal is not None and invoice.vat_rate is not None:
            vat_amount = money(subtotal * invoice.vat_rate / HUNDRED)

        total: Decimal | None = None
        if subtotal is not None and vat_amount is not None:
            total = money(subtotal + vat_amount)

    return CalculatedTotals(
        item_totals=item_totals, subtotal=subtotal, vat_amount=vat_amount, total=total
    )


def check_missing(invoice: ExtractedInvoice, reported: set[str | None]) -> list[Issue]:
    """Report required information that could not be read from the invoice."""
    issues: list[Issue] = []

    def missing(field: str, label: str, severity: str = "error") -> None:
        if field not in reported:  # already reported as INVALID_FIELD
            issues.append(
                Issue(severity=severity, code="MISSING_FIELD", field=field, message=f"{label} is missing.")  # type: ignore[arg-type]
            )

    for field, label in REQUIRED_FIELDS:
        if getattr(invoice, field) is None:
            missing(field, label)
    if invoice.currency is None:
        missing("currency", "Currency", "warning")

    if not invoice.items:
        if "items" not in reported:
            issues.append(
                Issue(severity="error", code="MISSING_FIELD", field="items", message="No line items were extracted.")
            )
    for idx, item in enumerate(invoice.items):
        prefix, label = f"items[{idx}]", f"Item {idx + 1}"
        if item.description is None:
            missing(f"{prefix}.description", f"{label} description", "warning")
        if item.quantity is None:
            missing(f"{prefix}.quantity", f"{label} quantity")
        if item.unit_price is None:
            missing(f"{prefix}.unit_price", f"{label} unit price")
        if item.line_total is None:
            computable = item.quantity is not None and item.unit_price is not None
            missing(f"{prefix}.line_total", f"{label} line total", "warning" if computable else "error")
    return issues


def _compare(
    name: str, field: str, label: str, calculated: Decimal | None, extracted: Decimal | None
) -> list[Issue]:
    if extracted is None:
        return []  # absence is already reported by check_missing
    if calculated is None:
        return [
            Issue(
                severity="warning",
                code=f"{name}_UNVERIFIED",
                field=field,
                message=f"{label} could not be verified because the values needed to calculate it are missing.",
            )
        ]
    if abs(calculated - extracted) > TOLERANCE:
        return [
            Issue(
                severity="error",
                code=f"{name}_MISMATCH",
                field=field,
                message=(
                    f"{label} on the invoice is {_fmt(extracted)}, but calculates to "
                    f"{_fmt(calculated)} (difference {_fmt(extracted - calculated)})."
                ),
            )
        ]
    return []


def check_calculations(invoice: ExtractedInvoice, calculated: CalculatedTotals) -> list[Issue]:
    """Compare every extracted amount with its independently calculated value."""
    issues: list[Issue] = []
    for idx, item in enumerate(invoice.items):
        expected = calculated.item_totals[idx]
        if expected is None or item.line_total is None:
            continue
        if abs(expected - item.line_total) > TOLERANCE:
            name = f" ({item.description})" if item.description else ""
            issues.append(
                Issue(
                    severity="error",
                    code="ITEM_TOTAL_MISMATCH",
                    field=f"items[{idx}].line_total",
                    message=(
                        f"Item {idx + 1}{name}: {_fmt_qty(item.quantity)} x {_fmt(item.unit_price)} "  # type: ignore[arg-type]
                        f"= {_fmt(expected)}, but the invoice shows {_fmt(item.line_total)}."
                    ),
                )
            )
    issues += _compare("SUBTOTAL", "subtotal", "Subtotal", calculated.subtotal, invoice.subtotal)
    issues += _compare("VAT_AMOUNT", "vat_amount", "VAT amount", calculated.vat_amount, invoice.vat_amount)
    issues += _compare("TOTAL", "total", "Total amount", calculated.total, invoice.total)
    return issues


def check_sanity(invoice: ExtractedInvoice, today: date) -> list[Issue]:
    """Non-fatal oddities worth a human glance."""
    issues: list[Issue] = []

    def warn(code: str, field: str | None, message: str) -> None:
        issues.append(Issue(severity="warning", code=code, field=field, message=message))

    if invoice.invoice_date is not None and invoice.invoice_date > today:
        warn(
            "DATE_IN_FUTURE",
            "invoice_date",
            f"Invoice date {invoice.invoice_date.isoformat()} is in the future. If the invoice uses "
            "the Bikram Sambat (BS) calendar, the date has not been converted to AD.",
        )
    if invoice.vat_rate is not None:
        if invoice.vat_rate < 0 or invoice.vat_rate > 100:
            warn("VAT_RATE_OUT_OF_RANGE", "vat_rate", f"VAT rate {invoice.vat_rate} is outside 0-100%.")
        elif 0 < invoice.vat_rate < 1:
            warn(
                "VAT_RATE_LOOKS_FRACTIONAL",
                "vat_rate",
                f"VAT rate {invoice.vat_rate} looks like a fraction rather than a percentage (13 means 13%).",
            )
    for idx, item in enumerate(invoice.items):
        if item.quantity is not None and item.quantity <= 0:
            warn("NON_POSITIVE_QUANTITY", f"items[{idx}].quantity", f"Item {idx + 1} has quantity {item.quantity}.")
        for field in ("unit_price", "line_total"):
            value = getattr(item, field)
            if value is not None and value < 0:
                warn("NEGATIVE_AMOUNT", f"items[{idx}].{field}", f"Item {idx + 1} has a negative {field.replace('_', ' ')}.")
    for field in ("subtotal", "vat_amount", "total"):
        value = getattr(invoice, field)
        if value is not None and value < 0:
            warn("NEGATIVE_AMOUNT", field, f"{field.replace('_', ' ').capitalize()} is negative ({value}).")
    return issues


def build_result(
    raw: dict[str, Any], *, source_file: str, ai_model: str, today: date | None = None
) -> InvoiceResult:
    """Validate raw model output, recalculate, and assemble the final report."""
    invoice, issues = parse_extraction(raw)
    reported = {issue.field for issue in issues}
    calculated = calculate_totals(invoice)
    issues += check_missing(invoice, reported)
    issues += check_calculations(invoice, calculated)
    issues += check_sanity(invoice, today or date.today())
    issues.sort(key=lambda issue: issue.severity != "error")  # errors first, order otherwise kept

    if any(issue.severity == "error" for issue in issues):
        status = "errors"
    elif issues:
        status = "warnings"
    else:
        status = "ok"
    return InvoiceResult(
        source_file=source_file,
        ai_model=ai_model,
        status=status,
        extracted=invoice,
        calculated=calculated,
        issues=issues,
    )
