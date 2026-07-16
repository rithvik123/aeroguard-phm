"""Command-line inference entrypoint for the AeroGuard-PHM final release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from aeroguard.inference.predictor import AeroGuardPredictor


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_ready(payload), handle, indent=2, allow_nan=False)


def _read_batch_directory(path: Path) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(path.glob("*.csv")):
        frame = pd.read_csv(csv_path)
        if "engine_id" not in frame.columns:
            frame.insert(0, "engine_id", csv_path.stem)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No CSV files found in batch directory: {path}")
    return frames


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run frozen AeroGuard-PHM inference.")
    parser.add_argument("--manifest", required=True, help="Path to frozen_system_manifest.json.")
    parser.add_argument("--input", help="Single engine-history CSV.")
    parser.add_argument("--batch-dir", help="Directory containing one CSV per engine.")
    parser.add_argument("--output", help="Output JSON path for single or batch results.")
    parser.add_argument("--output-json", help="Alias for --output.")
    parser.add_argument("--output-csv", help="Optional flattened output CSV for batch results.")
    parser.add_argument("--device", default="cpu", help="Torch device for checkpoint inference.")
    parser.add_argument("--explanation-level", default="standard", choices=["none", "standard", "full"])
    parser.add_argument("--validation-only", action="store_true", help="Validate input without prediction.")
    args = parser.parse_args(argv)

    predictor = AeroGuardPredictor.from_manifest(args.manifest, device=args.device)
    output_path = Path(args.output_json or args.output) if (args.output_json or args.output) else None

    if args.batch_dir:
        frames = _read_batch_directory(Path(args.batch_dir))
        if args.validation_only:
            results = [predictor.validate_engine(frame) for frame in frames]
        else:
            results = predictor.predict_batch(frames)
    else:
        if not args.input:
            parser.error("--input is required when --batch-dir is not provided.")
        frame = pd.read_csv(args.input)
        results = predictor.validate_engine(frame) if args.validation_only else predictor.predict_engine(frame)

    if args.explanation_level == "none" and not args.validation_only:
        targets = results if isinstance(results, list) else [results]
        for item in targets:
            item["explanation"] = []

    if output_path is not None:
        _write_json(output_path, results)
    else:
        print(json.dumps(_json_ready(results), indent=2, allow_nan=False))

    if args.output_csv and isinstance(results, list):
        flat_rows = [{key: value for key, value in row.items() if not isinstance(value, (list, dict))} for row in results]
        pd.DataFrame(flat_rows).to_csv(args.output_csv, index=False)
    return 0


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
