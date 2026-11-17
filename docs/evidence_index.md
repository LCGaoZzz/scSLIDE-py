# Evidence index

[Home](../README.md) · [Validation guide](../validation/README.md)

| Claim | Material retained in the software repository |
|---|---|
| Exact approved installed artifact | [Artifact manifest](../artifacts/manifest.json), [SHA256SUMS](../artifacts/SHA256SUMS) |
| R-to-Python capability coverage | [19-capability map](algorithm.md), package tests |
| Stored R regression | [Synthetic golden files](../validation/golden), [hashes](../validation/golden/SHA256SUMS), [profile provenance](../validation/golden/PROVENANCE.json) |
| Executable comparison and deliberate failures | [Comparison runner](../validation/tools/run_parity_suite.py), [labelled comparator](../validation/tools/compare_labelled.py) |
| R source-generation procedure | [Full exporter](../validation/r_reference/export_full_r_oracles.R), [normalization exporter](../validation/r_reference/export_normalization_oracle.R) |
| Real-case aggregate compatibility | [Case summary](../validation/results/r_case_summary.tsv), [signed interpretation](comparison_with_R.md) |
| Historical resource observations | [Unchanged figure source CSV](assets/showcase/historical_release_acceptance.csv), [limitations](benchmark.md) |
| Plots and reproducible example | [Gallery](gallery.md), [executable example](../examples/r_to_python_migration_quickstart.py) |
| Environment and usage limits | [Support](SUPPORTED_ENVIRONMENTS.md), [scientific usage](scientific_usage_and_migration.md) |

Historical experiments and newly run validation must not be conflated. Stored R golden checks are not live R regeneration. Saved human-case outputs support scoped retrospective conclusions, not independent external confirmation. The public reference set is synthetic; raw human matrices and private research operations are not distributed.
