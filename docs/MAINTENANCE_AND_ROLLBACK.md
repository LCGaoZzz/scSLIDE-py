# Maintenance and reproducibility

Use a dedicated environment and record distribution version, artifact SHA256, numerical profile, optional backend, dependency versions, input identifiers and sample/subject mapping. Historical private candidates shared version 0.3.3; use [the exact artifact manifest](../artifacts/manifest.json) to distinguish the approved build.

Install from the verified wheel or from a pinned source revision. A source rebuild with updated documentation is a different packaging artifact even when all numerical modules remain byte-identical. Never silently replace a previously distributed file under the same identity.

For problems, report a minimal synthetic reproduction, Python/OS/dependency information, input shape and axes, backend/worker settings and the complete redacted error. Do not post patient matrices, private clinical tables, tokens or signed download URLs. This project's GitHub Issues are the feedback entry point.

Changes affecting numerical behavior require regression/reference review; documentation or packaging changes require their corresponding installation/link checks. Preserve supported behavior or document migration and deprecation. Exact and approximate backends are distinct profiles.

To roll back, create a clean environment and install a previously verified artifact whose checksum is retained by your project. Do not overwrite working data or rely on an untracked developer environment. The approved artifact in this tree is directly recoverable without internal research directories.

The package is independent of OmicVerse. Feedback on the optional adapter should identify both package versions and the adapter commit. Public issue visibility does not constitute a clinical support or service-level guarantee.
