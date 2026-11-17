"""Small labelled CSV comparison; not an R-parity or scientific certificate.

First column is the row ID; remaining header cells are column IDs. This utility
rejects missing/duplicate labels, ragged matrices and non-finite values. Numeric
mode needs explicit tolerances and their rationale. Integer/count contracts
should use exact mode, which compares Decimal values without float rounding.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def load_matrix(path: Path) -> tuple[list[str], list[str], dict[str, list[Decimal]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.reader(handle))
    if len(records) < 2 or len(records[0]) < 2:
        raise ValueError("matrix needs a header, a data row and a value column")
    cols = records[0][1:]
    if any(not x for x in cols) or len(set(cols)) != len(cols):
        raise ValueError("column IDs must be nonempty and unique")
    rows: list[str] = []
    data: dict[str, list[Decimal]] = {}
    for line, record in enumerate(records[1:], 2):
        if len(record) != len(cols) + 1:
            raise ValueError(f"ragged CSV row {line}")
        key = record[0]
        if not key or key in data:
            raise ValueError(f"empty or duplicate row ID at row {line}")
        try:
            values = [Decimal(x) for x in record[1:]]
        except InvalidOperation as exc:
            raise ValueError(f"nonnumeric value at row {line}") from exc
        if any(not x.is_finite() for x in values):
            raise ValueError(f"non-finite value at row {line}")
        rows.append(key)
        data[key] = values
    return rows, cols, data


def compare(reference: Path, candidate: Path, *, mode: str = "exact",
            atol: float | None = None, rtol: float | None = None,
            tolerance_note: str | None = None) -> dict[str, Any]:
    if mode not in {"exact", "numeric"}:
        raise ValueError("mode must be exact or numeric")
    if mode == "numeric":
        if atol is None or rtol is None or not tolerance_note or not tolerance_note.strip():
            raise ValueError("numeric mode requires atol, rtol and tolerance_note")
        if any(not math.isfinite(x) or x < 0 for x in (atol, rtol)):
            raise ValueError("tolerances must be finite and nonnegative")
    elif atol is not None or rtol is not None or tolerance_note is not None:
        raise ValueError("exact mode does not accept numeric tolerance options")
    rr, rc, rd = load_matrix(reference)
    cr, cc, cd = load_matrix(candidate)
    if set(rr) != set(cr) or set(rc) != set(cc):
        raise ValueError("row/column ID sets differ; intersection-only comparison is forbidden")
    column_positions = {name: i for i, name in enumerate(cc)}
    mismatches: list[dict[str, str]] = []
    mismatch_count = 0
    max_decimal = Decimal(0)
    error_norm = 0.0
    reference_norm = 0.0
    for row in rr:
        for j, col in enumerate(rc):
            a = rd[row][j]
            b = cd[row][column_positions[col]]
            # Local precision avoids Decimal's default rounding for large integers.
            from decimal import localcontext
            with localcontext() as context:
                context.prec = max(len(a.as_tuple().digits), len(b.as_tuple().digits),
                                   abs(a.adjusted() - b.adjusted())) + 64
                delta = abs(a - b)
            max_decimal = max(max_decimal, delta)
            if mode == "exact":
                matches = a == b
            else:
                af, bf, df = float(a), float(b), float(delta)
                if not all(math.isfinite(x) for x in (af, bf, df)):
                    raise ValueError("numeric mode exceeds finite float range; use an appropriate comparator")
                bound = atol + rtol * abs(af)
                if not math.isfinite(bound):
                    raise ValueError("tolerance bound overflow")
                matches = df <= bound
                error_norm = math.hypot(error_norm, df)
                reference_norm = math.hypot(reference_norm, af)
            if not matches:
                mismatch_count += 1
                if len(mismatches) < 10:
                    mismatches.append({"row_id": row, "column_id": col,
                                       "reference": str(a), "candidate": str(b)})
    report: dict[str, Any] = {
        "scope": "labelled_matrix_comparison_only_not_parity_certificate",
        "mode": mode, "shape": [len(rr), len(rc)],
        "raw_row_order_equal": rr == cr, "raw_column_order_equal": rc == cc,
        "comparison_alignment": "candidate reordered to reference IDs",
        "matched_after_id_alignment": mismatch_count == 0,
        "mismatch_count": mismatch_count, "first_mismatches": mismatches,
        "max_abs_decimal": str(max_decimal),
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "reference_origin_verified": False,
    }
    if mode == "numeric":
        if not math.isfinite(error_norm) or not math.isfinite(reference_norm):
            raise ValueError("norm overflow; use a scale-aware comparator")
        report.update(atol=atol, rtol=rtol, tolerance_note=tolerance_note,
                      absolute_frobenius=error_norm,
                      relative_frobenius=error_norm / reference_norm if reference_norm else None)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--mode", choices=["exact", "numeric"], default="exact")
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--tolerance-note")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        if args.out is not None and args.out.exists():
            raise ValueError("output already exists; evidence will not be overwritten")
        result = compare(args.reference, args.candidate, mode=args.mode,
                         atol=args.atol, rtol=args.rtol, tolerance_note=args.tolerance_note)
        encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.out is not None:
            with args.out.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
        else:
            print(encoded, end="")
        return 0 if result["matched_after_id_alignment"] else 1
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
