import json

import pytest

import main
from extractor import GroqAPIError, InvalidImageError
from tests.conftest import as_decimals


@pytest.fixture(autouse=True)
def hermetic(monkeypatch):
    monkeypatch.setattr(main, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GROQ_MODEL", raising=False)


def test_success_saves_json_and_exits_zero(monkeypatch, tmp_path, valid_reply, capsys):
    monkeypatch.setattr(main, "extract_invoice", lambda *a, **k: as_decimals(valid_reply))
    output = tmp_path / "nested" / "invoice.json"

    code = main.main(["invoice.png", "--output", str(output)])

    assert code == main.EXIT_OK
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["status"] == "ok"
    assert saved["extracted"]["total"] == "113565.00"
    assert saved["calculated"]["subtotal"] == "100500.00"
    assert len(saved["extracted"]["items"]) == 5
    captured = capsys.readouterr()
    assert json.loads(captured.out) == saved  # stdout is pure JSON
    assert "Status: OK" in captured.err


def test_validation_errors_still_save_and_exit_two(monkeypatch, tmp_path, valid_reply, capsys):
    raw = as_decimals(valid_reply)
    raw["total"] = 999
    monkeypatch.setattr(main, "extract_invoice", lambda *a, **k: raw)
    output = tmp_path / "invoice.json"

    code = main.main(["invoice.png", "-o", str(output)])

    assert code == main.EXIT_NEEDS_REVIEW
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["status"] == "errors"
    assert any(i["code"] == "TOTAL_MISMATCH" for i in saved["issues"])
    assert "TOTAL_MISMATCH" in capsys.readouterr().err


@pytest.mark.parametrize("error", [InvalidImageError("bad image"), GroqAPIError("rate limited")])
def test_extraction_failure_exits_one_and_keeps_previous_output(monkeypatch, tmp_path, capsys, error):
    def boom(*args, **kwargs):
        raise error

    monkeypatch.setattr(main, "extract_invoice", boom)
    output = tmp_path / "invoice.json"
    output.write_text('{"previous": true}')

    code = main.main(["invoice.png", "-o", str(output)])

    assert code == main.EXIT_FAILED
    assert str(error) in capsys.readouterr().err
    assert output.read_text() == '{"previous": true}'  # a failed run never clobbers a good result


def test_unwritable_output_path(monkeypatch, tmp_path, valid_reply, capsys):
    monkeypatch.setattr(main, "extract_invoice", lambda *a, **k: as_decimals(valid_reply))
    blocker = tmp_path / "file"
    blocker.write_text("x")

    code = main.main(["invoice.png", "-o", str(blocker / "invoice.json")])

    assert code == main.EXIT_FAILED
    assert "could not save" in capsys.readouterr().err


def test_default_output_location():
    assert main.DEFAULT_OUTPUT.parts[-2:] == ("output", "invoice.json")
