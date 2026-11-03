import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from extractor import ExtractionError, extract_invoice, get_model_name
from models import InvoiceResult
from validator import build_result

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "invoice.json"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_NEEDS_REVIEW = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and validate invoice data from an image using Groq's vision model."
    )
    parser.add_argument("image", type=Path, help="path to the invoice image (JPEG, PNG or WebP)")
    parser.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_OUTPUT, help=f"where to save the JSON (default: {DEFAULT_OUTPUT})"
    )
    return parser.parse_args(argv)


def save_result(result: InvoiceResult, output: Path) -> str:
    """Write the result as JSON and return the JSON text."""
    text = result.model_dump_json(indent=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    return text


def print_report(result: InvoiceResult, output: Path) -> None:
    inv, calc = result.extracted, result.calculated
    errors = sum(1 for i in result.issues if i.severity == "error")
    warnings = len(result.issues) - errors
    lines = [
        "",
        f"Invoice {inv.invoice_number or '?'} from {inv.supplier_name or '?'} ({inv.invoice_date or 'no date'})",
        f"Items extracted: {len(inv.items)}",
        f"Total on invoice: {inv.total if inv.total is not None else '?'}   calculated: {calc.total if calc.total is not None else '?'}",
        f"Status: {result.status.upper()}  ({errors} error(s), {warnings} warning(s))",
    ]
    for issue in result.issues:
        lines.append(f"  [{issue.severity.upper()}] {issue.code}: {issue.message}")
    lines.append(f"Saved to {output}")
    print("\n".join(lines), file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv()
    model = get_model_name()

    try:
        raw = extract_invoice(args.image, model=model)
    except ExtractionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_FAILED

    result = build_result(raw, source_file=str(args.image), ai_model=model)
    try:
        text = save_result(result, args.output)
    except OSError as exc:
        print(f"Error: could not save {args.output}: {exc}", file=sys.stderr)
        return EXIT_FAILED

    print(text)
    print_report(result, args.output)
    return EXIT_NEEDS_REVIEW if result.status == "errors" else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
