from pathlib import Path
from backend.remote_sensing.validator import FileValidator, ValidationResult
from backend.config import settings


def test_validator_valid_geotiff(sample_geotiff_rgb: Path):
    result: ValidationResult = FileValidator.validate(sample_geotiff_rgb)
    assert result.valid is True
    assert result.width == 256
    assert result.height == 256
    assert result.bands == 3
    assert len(result.errors) == 0


def test_validator_valid_png(sample_png: Path):
    result: ValidationResult = FileValidator.validate(sample_png)
    assert result.valid is True
    assert result.width == 200
    assert result.height == 200
    assert result.bands == 3


def test_validator_nonexistent_file(tmp_path: Path):
    missing_path = tmp_path / "does_not_exist.tif"
    result = FileValidator.validate(missing_path)
    assert result.valid is False
    assert any("not found" in err.lower() for err in result.errors)


def test_validator_invalid_extension(tmp_path: Path):
    invalid_file = tmp_path / "script.py"
    invalid_file.write_text("print('hello')")

    result = FileValidator.validate(invalid_file)
    assert result.valid is False
    assert any("unsupported extension" in err.lower() for err in result.errors)


def test_validator_empty_file(tmp_path: Path):
    empty_file = tmp_path / "empty.tif"
    empty_file.write_bytes(b"")

    result = FileValidator.validate(empty_file)
    assert result.valid is False
    assert any("empty" in err.lower() for err in result.errors)
