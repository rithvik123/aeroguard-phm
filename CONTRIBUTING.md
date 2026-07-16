# Contributing

Thanks for your interest in AeroGuard-PHM.

## Scope

Good contributions include:

- Bug fixes in inference, API, dashboard or tests
- Documentation improvements
- Reproducibility improvements
- Additional validation on public prognostics datasets
- Safer model-governance or monitoring checks

Please avoid submitting changes that retrain or retune the frozen v1.0.0 system without clearly separating them from the frozen release.

## Development Setup

```powershell
python -m pip install -e ".[api,dashboard,dev]" -c requirements/constraints.txt
$env:PYTHONPATH = ".\src"
python -m pytest tests\unit\test_final_release.py -q
```

## Pull Request Expectations

- Keep changes focused.
- Add or update tests for behavioral changes.
- Do not commit raw datasets, local paths, secrets or temporary reports.
- Preserve third-party attribution and dataset citations.
- Keep contributions compatible with Apache License 2.0.
- Update `README.md`, `MODEL_CARD.md` or `REPRODUCIBILITY.md` when user-facing behavior changes.

## Frozen Release Policy

The v1.0.0 frozen release should remain reproducible. New research experiments should write to new report/artifact folders and should not overwrite frozen final-release evidence.
