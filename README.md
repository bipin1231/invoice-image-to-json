# AI Invoice Extractor

A small command-line tool that sends an invoice image **directly** to Groq's vision model, gets structured JSON back, validates it with Pydantic, and independently recalculates every amount with `Decimal` so that misreads and arithmetic errors are caught.

```
invoice image ──► Groq vision model ──► JSON ──► Pydantic validation ──► Decimal recalculation ──► output/invoice.json
                                                                              └─► report: missing info, mismatches, warnings
```

## Model

The app uses **`qwen/qwen3.8-27b`**, the vision model listed on Groq's [Images and Vision](https://console.groq.com/docs/vision) page (checked 2026-11-03). It accepts text + images, supports JSON mode, and takes up to 3 images / 20 MB per request.

Groq currently labels this model **Preview**, which means it can be changed or retired at short notice, and Groq's vision lineup has changed several times already. If the model is renamed or removed you will get a clear "model not found" error. Fix it by setting `GROQ_MODEL` in `.env` (see the vision page above for the current name); no code change needed.

## Installation

Requires Python 3.10+.

```bash
cd invoice-extractor
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Add your Groq API key (create one at https://console.groq.com/keys):

```bash
cp .env.example .env               # Windows: copy .env.example .env
# then edit .env and set GROQ_API_KEY=...
```

The key is only ever read from the environment / `.env`. `.env` is listed in `.gitignore`.

## Usage

Put the invoice image (JPEG, PNG or WebP) in `input/` and run:

```bash
python main.py input/invoice.png
```

- JSON is printed to **stdout** and saved to `output/invoice.json` (use `-o path.json` to save elsewhere).
- A human-readable report (status, errors, warnings) is printed to **stderr**.

Exit codes: `0` extracted and consistent (warnings allowed), `1` could not extract (bad image, API error, malformed reply, missing key), `2` extracted but the validation found errors that need a human look. A failed run (`1`) never overwrites an existing `output/invoice.json`.

### Output shape

```jsonc
{
  "source_file": "input/invoice.png",
  "ai_model": "qwen/qwen3.8-27b",
  "status": "ok",                 // "ok" | "warnings" | "errors"
  "extracted": { ... },           // exactly what the model read from the invoice
  "calculated": { ... },          // item totals, subtotal, VAT, total recomputed with Decimal
  "issues": [                     // empty when everything checks out
    {"severity": "error", "code": "TOTAL_MISMATCH", "field": "total", "message": "..."}
  ]
}
```

Amounts are JSON **strings** (`"100500.00"`) so no consumer can accidentally round them as floats. Item `quantity` and `vat_rate` are plain strings too (`"10"`, `"13"`; the VAT rate is a percentage).

## What is checked

| Check | Severity |
|---|---|
| Missing invoice number, date, supplier, items, subtotal, VAT rate/amount, total | error |
| Missing item quantity / unit price (or line total when it cannot be computed) | error |
| Missing item description, currency, or a line total that can be computed | warning |
| A value that cannot be read as a number/date (reported, then the rest is still validated) | error |
| `quantity × unit price` ≠ line total | error |
| Sum of items ≠ subtotal | error |
| `subtotal × VAT rate` ≠ VAT amount | error |
| `subtotal + VAT` ≠ total | error |
| An amount that cannot be verified because its inputs are missing | warning |
| Future invoice date (hint: may be a Bikram Sambat date), VAT rate outside 0-100 or written as a fraction, zero/negative quantities or amounts | warning |

Calculations use `Decimal`, round half-up to 2 places, and treat differences of up to 0.01 as rounding. The model is told to copy printed values as-is and **not** to fix the arithmetic, otherwise the recalculation would have nothing to catch.

## Tests

```bash
pytest
```

Tests use synthetic data and a mocked Groq client, so they need no API key and make no network calls. They cover validation and calculation, number/date parsing, invalid images, malformed or truncated JSON, every API error class, and the command-line flow.

## Project layout

```
main.py         command line, report, saving the result
extractor.py    image checks, Groq request, API-error handling, JSON parsing
validator.py    Pydantic validation + independent Decimal calculations + issue reporting
models.py       Pydantic models and number/date parsing
tests/          pytest suite (mocked AI responses)
input/          put invoice images here
output/         invoice.json is written here
.env            your API key (not committed)   .env.example  template
```

## Limitations

- One invoice image per run; multi-page PDFs are not supported (convert pages to images first).
- The schema has no fields for discounts, shipping or other extra charges. An invoice that has them will show up as a subtotal/total mismatch for a person to review.
- Dates are read as printed. Bikram Sambat dates are not converted to AD.
- Day-first formats (`15/03/2026`) are assumed when the model returns something other than ISO dates.
- Image size is capped at Groq's documented 20 MB per image. Very large base64 payloads may still be rejected by the API, in which case you get a clear error.
