# Public software distribution — 2026-11-17

[Home](../README.md) · [OmicVerse integration](omicverse_integration.md)

The canonical [scSLIDE-py repository](https://github.com/LCGaoZzz/scSLIDE-py) now
contains the clean standalone software history. The original research repository
object, branches, PRs and historical records are preserved separately and remain
private. No research Git parents or campaign directories were imported here.

- Clean root commit: `57605d55054a3c26b0d30721f696ef67eab1809d` (184 tracked files).
- Numerical source tree: `1d178a697055f00411c4a82fec7e18cee2e9663c`.
- GitHub repository ID: `1461743072`.
- [Initial software CI](https://github.com/LCGaoZzz/scSLIDE-py/actions/runs/44137259242): five jobs passed, one optional R environment probe skipped.
- The [machine-readable publication receipt](../validation/PUBLICATION.json)
  records anonymous downloads, exact artifact identities and installation checks.

## Exact public artifacts

| Artifact | SHA256 |
|---|---|
| [0.3.3 wheel](https://raw.githubusercontent.com/LCGaoZzz/scSLIDE-py/57605d55054a3c26b0d30721f696ef67eab1809d/artifacts/scslide_py-0.3.3-py3-none-any.whl) | `02043e3c1215320beca93c9ce58645f4fdebf29796dbbdc30d7ce00a580aacbd` |
| [0.3.3 sdist](https://raw.githubusercontent.com/LCGaoZzz/scSLIDE-py/57605d55054a3c26b0d30721f696ef67eab1809d/artifacts/scslide_py-0.3.3.tar.gz) | `782f4f662dc156a428ad4fcb40c306cc542781cdc1f6b2850a41b8a56d2d6bc0` |

Both files were retrieved without GitHub credentials and their complete byte
hashes matched. Public repository metadata is anonymously accessible; the private
research repository is not. Follow the [installation command](../README.md#installation)
to install the exact wheel with the verified Python 3.11 constraints.

The public wheel and constraints were installed with pip into an isolated target
using Python 3.11.15. All 29 dependency distributions satisfied the selected
`[anndata]` dependency closure, and all 14 installed scSLIDE Python modules
matched the downloaded wheel byte for byte. The real OmicVerse integration case
passed (1 passed, 0 skipped) against adapter commit
`4d77e62894d7635788d81fddb48df98c04014718`, comparing direct and wrapped sample
counts/diffusion coordinates and checking input immutability and sample axes.
This installation recheck uses the existing integration case; it is not an
additional unique case or a new scientific experiment.

The original [export manifest](../validation/PUBLIC_SNAPSHOT_MANIFEST.json)
records preparation-time flags before publication; this receipt records the later
public transition. The approved wheel and sdist were not rebuilt or replaced.
Source checkout builds with newer documentation remain distinct packaging artifacts.

GitHub distribution is complete. No PyPI/TestPyPI upload, GitHub Release/tag,
upstream merge, tutorial submodule update or website deployment is implied.
