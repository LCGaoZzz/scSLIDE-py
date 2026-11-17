# OmicVerse integration status

[Home](../README.md) · [Documentation](README.md) · [Validation receipt](omicverse_validation.md)

scSLIDE-py is the independent native Python implementation. The optional
`ov.single.scslide` adapter delegates numerical work to this package and returns
sample-by-landmark AnnData separately from the input cell AnnData.

## Contributions awaiting upstream review

- [OmicVerse adapter PR #1098](https://github.com/omicverse/omicverse/pull/1098), targeting `master`.
- [Companion tutorial PR #145](https://github.com/omicverse/omicverse-tutorials/pull/145), targeting `main`.

Both contributions now have real-backend verification and an executed synthetic
notebook. The API is proposed upstream; it is not yet part of an accepted
OmicVerse release. The standalone package works independently of OmicVerse.

The exact reviewed backend is `scslide-py-0.3.3-02043e3c1215`; use
[the artifact manifest](../artifacts/manifest.json) to distinguish it from earlier
private candidates with the same version string. Source builds remain distinct
from the preserved wheel and sdist. No PyPI publication is claimed.

## Release sequence

1. Preserve the original research repository privately and establish the clean
   software repository/public installation endpoint; verify the downloaded wheel
   against its SHA256. The clean software export and exact artifacts are prepared.
2. Maintainers review and merge the linked OmicVerse code and tutorial PRs in
   their chosen order once the public backend is available.
3. Only after the tutorial is accepted, update `omicverse_guide` to the accepted,
   fetchable tutorial commit and let maintainers synchronize the website.

Repository preparation, public distribution, upstream merge and website deployment
are separate states. The tutorial submodule and published website have not been
changed by this preparation. Native algorithms, frozen baselines and historical
scientific results remain unchanged.
