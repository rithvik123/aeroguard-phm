# Third-Party Notices

This file records copied, adapted, referenced or previously bundled third-party material identified during the public-release cleanup.

## Project License

- AeroGuard-PHM source code, documentation, release scripts and generated release metadata are distributed under Apache License 2.0 unless a file-specific notice says otherwise.
- Copyright holder: Copyright 2026 Yarroju Rithvik.
- No third-party NOTICE file was found among retained copied or adapted source-code components, so no separate root `NOTICE` file is required for this release.

## NASA C-MAPSS Turbofan Degradation Dataset

- Source: NASA Prognostics Center of Excellence C-MAPSS turbofan degradation benchmark.
- Project use: Benchmark dataset for remaining-useful-life modelling and offline evaluation.
- Files: Raw local dataset files are intentionally ignored under `data/raw/cmapss/` and should not be staged unless redistribution rights are reviewed.
- Attribution: Dataset citation is documented in `docs/references.md` and `CITATION.cff`.

## Production PDM System Reference Copy

- Detected paths before cleanup: `references/production-pdm-system/`, `extracted-code/production-pdm-system/`, and `data/reference-derived/production-pdm-system/`.
- License observed: MIT License, copyright `(c) 2026 Predictive Maintenance Manufacturing System`.
- Project use: Reference material only; not imported by the AeroGuard-PHM package after cleanup.
- Public-release action: Copied repository trees and derived non-AeroGuard data were removed from the working directory; `.gitignore` keeps those paths excluded if they are restored locally.
- Attribution impact: If any code from that project is later copied into `src/aeroguard`, the MIT copyright and permission notice must be preserved with the adapted files.

## Original Aircraft Predictive-Maintenance Notebook Repository

- Detected paths before cleanup: `references/original-aircraft-pm/`, `extracted-code/original-aircraft-pm/`, and `notebooks/original-aircraft-pm/`.
- License observed: No explicit license file was found in the copied tree.
- Project use: Historical notebook/reference material only; not imported by the AeroGuard-PHM package after cleanup.
- Public-release action: Copied notebook/code trees were removed from the working directory; `.gitignore` keeps those paths excluded if they are restored locally. Do not publish copied notebooks or code without license review.
- Attribution impact: License is unresolved, so substantial reuse should be treated as blocked until permission is verified.

## Generated README Images

- Source: User-provided generated image assets in the local `images/` folder.
- Project use: README visuals after inspection.
- Public-release action: Two images are rendered in README; three are retained only as non-rendered assets because of visible generated-text errors.
- License treatment: Treated as project-provided release assets under Apache License 2.0 for repository distribution.
- Mapping and validation: `docs/assets/README_IMAGE_MAPPING.md`.
