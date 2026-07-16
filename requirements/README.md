# Dependency Strategy

`pyproject.toml` is the canonical package metadata. The files in this directory are installer-friendly views of the same direct dependency groups, constrained to versions verified in the existing local `aerostat-ai` environment.

The project uses split requirement files instead of a full environment export because the Conda environment contains platform-specific and unrelated packages. These files intentionally list only direct dependencies needed for frozen inference, CLI use, FastAPI, Streamlit, monitoring demonstration, tests, README validation and release validation.

Use the constraints file when reproducing this release:

```powershell
python -m pip install -e ".[api,dashboard,dev]" -c requirements/constraints.txt
```

`torch` is constrained to the public version `2.12.1`; the local verification environment used a CUDA build with local version suffix `+cu126`.
