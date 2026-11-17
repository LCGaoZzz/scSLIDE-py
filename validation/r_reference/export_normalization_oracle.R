#!/usr/bin/env Rscript
# Design starter: execute a real installed backend, never a Python substitute.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop("Usage: Rscript export_normalization_oracle.R <scSLIDE|Seurat> <counts.csv> <new-output-directory>")
}
backend <- args[[1L]]
if (!(backend %in% c("scSLIDE", "Seurat"))) stop("Unsupported backend")
if (!requireNamespace(backend, quietly = TRUE)) stop("Requested R backend is not installed: ", backend)
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite is required to write metadata")
input <- normalizePath(args[[2L]], mustWork = TRUE)
out <- path.expand(args[[3L]])
if (file.exists(out) || dir.exists(out)) stop("Output path already exists; no overwrite")
if (!dir.exists(dirname(out))) stop("Output parent directory must exist")
df <- read.csv(input, check.names = FALSE, stringsAsFactors = FALSE)
if (ncol(df) < 2L || nrow(df) < 1L) stop("Need landmark IDs and at least one sample column")
ids <- as.character(df[[1L]])
if (anyNA(ids) || any(ids == "") || anyDuplicated(ids)) stop("Invalid landmark IDs")
samples <- colnames(df)[-1L]
if (anyNA(samples) || any(samples == "") || anyDuplicated(samples)) stop("Invalid sample IDs")
if (!all(vapply(df[-1L], is.numeric, logical(1L)))) stop("Count columns must be numeric")
x <- as.matrix(df[-1L])
storage.mode(x) <- "double"
rownames(x) <- ids
if (any(!is.finite(x)) || any(x < 0) || sum(x) <= 0) stop("Input must be finite nonnegative with positive total")
fun <- getExportedValue(backend, "NormalizeChiSquared")
method <- getS3method("NormalizeChiSquared", "default", optional = TRUE, envir = asNamespace(backend))
if (!dir.create(out, recursive = FALSE)) stop("Cannot create output directory")
out <- normalizePath(out, mustWork = TRUE)
if (!file.copy(input, file.path(out, "input.csv"), overwrite = FALSE)) stop("Cannot copy oracle input")
writeLines(capture.output(sessionInfo()), file.path(out, "sessionInfo.txt"))
writeLines(deparse(fun), file.path(out, "NormalizeChiSquared_generic.R.txt"))
if (!is.null(method)) writeLines(deparse(method), file.path(out, "NormalizeChiSquared_default.R.txt"))
meta <- list(schema_version = "normalization-oracle-1", status = "STARTED",
             backend = backend, backend_version = as.character(packageVersion(backend)),
             backend_path = find.package(backend), r_version = R.version.string,
             input_path = input, landmark_ids = ids, sample_ids = samples,
             input_shape = dim(x), input_dtype = "double", parameters = list(verbose = FALSE),
             started_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
             source_identity_note = "Bind exact package/source commit and SHA256 externally; version alone is insufficient.",
             scope = "One real backend normalization call only; not whole-package parity.")
write_meta <- function(value) jsonlite::write_json(value, file.path(out, "execution.json"),
                                                auto_unbox = TRUE, pretty = TRUE, null = "null")
write_meta(meta)
warnings <- character()
value <- tryCatch(withCallingHandlers(fun(x, verbose = FALSE), warning = function(w) {
  warnings <<- c(warnings, conditionMessage(w))
  invokeRestart("muffleWarning")
}), error = function(e) {
  meta$status <- "FAILED"
  meta$error <- conditionMessage(e)
  meta$warnings <- warnings
  write_meta(meta)
  stop(e)
})
y <- as.matrix(value)
if (!identical(dim(y), dim(x))) {
  meta$status <- "FAILED_OUTPUT_SHAPE"
  meta$output_shape <- dim(y)
  write_meta(meta)
  stop("Backend output shape changed")
}
saveRDS(value, file.path(out, "backend_output.rds"))
write.csv(y, file.path(out, "pearson_residuals_backend.csv"), row.names = TRUE)
# This is a mathematical sidecar, NOT an output returned by the backend.
e <- outer(rowSums(x), colSums(x)) / sum(x)
dimnames(e) <- dimnames(x)
write.csv(e, file.path(out, "independence_expected_formula_sidecar.csv"), row.names = TRUE)
meta$status <- "EXECUTED"
meta$finished_at_utc <- format(Sys.time(), tz = "UTC", usetz = TRUE)
meta$warnings <- warnings
meta$output_dimnames <- dimnames(value)
meta$output_finite <- all(is.finite(y))
meta$sidecar_note <- "independence_expected_formula_sidecar.csv is computed from the input, not exported by NormalizeChiSquared."
meta$comparison_status <- "NOT_COMPARED_TO_PYTHON"
write_meta(meta)
writeLines("Hash all files with an external SHA256 manifest; do not hash the manifest inside itself.", file.path(out, "HASHING_NOTE.txt"))
cat("Executed backend normalization; Python comparison not yet performed: ", out, "\n", sep = "")
