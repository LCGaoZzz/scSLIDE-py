# Comparison with the R implementation

[Home](../README.md) · [Complete API map](algorithm.md) · [Migration](R_TO_PY_MIGRATION.md) · [Validation](../validation/README.md)

scSLIDE-py implements the original sample-level workflow natively. Its core does not invoke Rscript or rpy2. R remains the reference for compatibility testing, not a hidden numerical backend.

## References and matching inputs

The stored references were generated with scSLIDE 0.1.1 (source `968dbde299458100346c792aaf60f620c69f41fd`) and Seurat 6.0.0 (source `e6ba2b24149efc0ad62f9c48a9da527dfda4466d`). The full legacy suite ran in a compatibility dependency stack with Seurat 6.0.0, rather than an untouched historical environment. Synthetic inputs, outputs, their hashes and exporter scripts are retained under [validation](../validation/README.md).

Inputs, cell/gene/sample IDs, preprocessing, feature sets, landmarks and reference profiles must match. Exact RANN versus Python exact is different from default Annoy versus exact. Seurat's retained pre-filter residuals and ScaleData upper clipping are part of the tested contract; they are not interchangeable preprocessing decisions.

## Retrospective matched real-case results

| Readout | COVID | SEA-AD Microglia-PVM |
|---|---:|---:|
| Cells | 104,447 | 40,000 |
| Shared eligible genes | 8,355 | 14,733 |
| Exact-R LMC counts | identical | identical |
| Signed trajectory rank correlation | +1 | −1 |
| Signed coefficient correlation | +0.99997068 | −0.9999992166 |
| Through-zero coefficient scale | +0.9972623 | −0.99769023 |

The SEA-AD orientation is reversed. A coordinate transformation t'=a−t reverses its gene slope; matching structure is not raw sign equality. Anchor the direction to a scientifically defined endpoint before interpreting upregulation or disease progression. For COVID the recorded Annoy-reference coefficient correlation is 0.99979564; approximate-neighbor comparisons are not bit-exact claims.

The numbers above are retained aggregate findings from saved matched outputs, with their compact [source table](../validation/results/r_case_summary.tsv). Raw human cell matrices and clinical donor tables are not redistributed. COVID used 40 draws from 36 subjects. The SEA-AD native scientific funnel was 89 donors → 76 filtered → 75 DE → 74 QC, across 227 specimens; its performance-only cohort is different.

## What was actually revalidated?

The exact approved package has historical direct installed-package checks: 43 unique package tests, 47 saved-reference checks and six detecting negative controls. The final bounded repair changed exact-worker metadata and narrow permission diagnostics, not the estimator. Saved case outputs were recalculated and compared to the earlier frozen implementation; the final repair did not rerun every complete disease case or live R oracle.

This public repository relocates the synthetic reference suite without changing its numerical checks, and makes failures produce a failing exit code. New workflow runs provide their own test records; they are not automatically evidence of new R generation.

## Scientific interpretation, not marketing equivalence

A numerical reimplementation, matching scientific readouts and superiority to another method are separate claims. Composition can be enough for composition shifts; pseudobulk can be strong for mean changes. Distribution-sensitive tasks are an important use case, but scSLIDE is not claimed universally best.

The core descriptive TrajDE follows the specified R model. Response-guided selection and trajectory fitting can change inferential interpretation. OOF/count-splitting/permutation extensions remain exploratory and do not carry a universal FDR/FWER guarantee. See [scientific usage](scientific_usage_and_migration.md).
