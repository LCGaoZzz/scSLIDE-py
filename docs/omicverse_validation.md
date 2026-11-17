# OmicVerse pre-merge verification — 2026-11-17

[Integration status](omicverse_integration.md)

## Tested identities

- Backend: `scslide-py` 0.3.3; numerical source tree `1d178a697055f00411c4a82fec7e18cee2e9663c`.
- Exact wheel SHA256: `02043e3c1215320beca93c9ce58645f4fdebf29796dbbdc30d7ce00a580aacbd`.
- Exact sdist SHA256: `782f4f662dc156a428ad4fcb40c306cc542781cdc1f6b2850a41b8a56d2d6bc0`.
- Adapter source: `omicverse/omicverse` PR #1098 at `4d77e62894d7635788d81fddb48df98c04014718`.
- Executed tutorial source: `0d0773f97f7b69d3e36c614354ddfb934f788974` (tutorial PR #145).
- Environment: Linux x86_64, Python 3.11.15 from the OmicOS sandbox interpreter;
  an isolated import target held the exact wheel. All 14 installed Python modules
  were compared byte-for-byte with the wheel, rather than importing an editable checkout.

## Observed checks

| Check | Result and scope |
|---|---|
| Standalone regression tests on the exact wheel | 43 passed |
| Retained synthetic R-reference comparisons | 47 matched, 0 failed; all 6 corruption controls detected |
| Adapter contract plus real-backend differential test | 36 passed, 0 skipped; direct and wrapped counts/embeddings agree |
| Explicit offline pip installation of the exact wheel | Passed; real-backend differential test passed again after installation (same test, not an extra unique case) |
| Full public import and actual registry | `ov.single.scslide` resolves and is registered in `single` |
| Architecture and registry suite | Earlier local run at the same adapter commit: 2,390 passed; retained evidence, not rerun here |
| Tutorial notebook | 5 code cells executed in a new kernel; sample plot, H5AD round-trip and CSV export passed |
| API index | Existing registry generator validates the actual scSLIDE public symbol; scoped English/Chinese index update |
| scSLIDE-only Sphinx render (guide, executed notebook, actual API) | Passed with `-W --keep-going`, zero warnings |
| Full English Sphinx HTML build | Succeeded; final cached build reported 110 warnings, with no scSLIDE-specific warning/error |
| Exact release seed | Wheel/sdist hashes verified; 14 wheel/source modules matched; 6 staging safety tests passed |
| Standalone software export | Allowlisted tree and local links checked; exact source/artifact identities preserved; no research directories or old Git history copied |

The notebook uses only synthetic data: 360 cells, 40 genes, 12 samples and 60
landmarks. Execution and Sphinx rendering are separate checks. The first full-site attempt was terminated during HTML output; the cached build
completed successfully with the final committed notebook and guide. Other modules
have existing missing references, optional API imports and docstring diagnostics;
this is not a warning-free whole-site claim.

Ordinary dependency
and exploratory-inference warnings are not hidden; no new live-R comparison,
human-cohort run, benchmark or confirmatory inference claim is made.

## Maintainer-controlled follow-up

[Code PR #1098](https://github.com/omicverse/omicverse/pull/1098) and
[tutorial PR #145](https://github.com/omicverse/omicverse-tutorials/pull/145) must be
reviewed and accepted upstream. The public backend installation gate has been completed; see [publication verification](PUBLICATION.md).
Only an accepted tutorial commit may be used for the later `omicverse_guide`
update; website deployment remains with maintainers.

## Upstream CI observation

The [Python 3.12 full-suite job](https://github.com/omicverse/omicverse/actions/runs/44108879185/job/134128694886)
reported 8,574 passed, 1,363 skipped and one failure in
`tests/space/test_track.py::test_mitosis_signal_and_refinement`: the runner lacks
`cv2`. That tracking test, its implementation and the CI dependency declaration
are unchanged by the adapter PR. The resolver/install smoke jobs passed. This
existing full-suite dependency issue is recorded for maintainers rather than
reported as a green whole-repository CI run.
