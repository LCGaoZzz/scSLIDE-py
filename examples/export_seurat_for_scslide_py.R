#!/usr/bin/env Rscript
# One-time portable export of an R Seurat (v5/v6) object for scslide-py migration.
# Usage:
#   Rscript export_seurat_for_scslide_py.R input_seurat.rds ./exported_seurat_bundle [assay=RNA] [reduction=pca]
#
# Core scslide-py never calls R at runtime; this standalone helper exports an
# existing R .rds object into standard MatrixMarket + TXT/CSV/JSON files
# (`counts_cells_by_genes.mtx`, `cells.txt`, `genes.txt`, `cell_metadata.csv`,
# optional `variable_features.txt`, optional `embedding_<reduction>.csv`,
# `export_summary.txt`, and `export_manifest.json`) with dimension & ID checks.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) {
  stop("Usage: Rscript export_seurat_for_scslide_py.R <input.rds> <out_dir> [assay=RNA] [reduction=pca]")
}
rds_path <- args[[1L]]
out_dir <- args[[2L]]
assay_name <- if (length(args) >= 3L) args[[3L]] else "RNA"
reduc_name <- if (length(args) >= 4L) args[[4L]] else "pca"

suppressPackageStartupMessages({
  library(SeuratObject)
  library(Matrix)
})

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
obj <- readRDS(rds_path)
cts <- as(LayerData(obj[[assay_name]], layer = "counts"), "dgCMatrix")

genes <- rownames(cts)
cells <- colnames(cts)
if (anyDuplicated(genes) > 0L || anyDuplicated(cells) > 0L) {
  stop("Duplicate gene or cell identifiers detected in Seurat counts layer.")
}

# Export transposed cells x genes MatrixMarket so Python scipy.io.mmread lands directly in AnnData orientation
cells_by_genes <- Matrix::t(cts)
Matrix::writeMM(cells_by_genes, file = file.path(out_dir, "counts_cells_by_genes.mtx"))
writeLines(cells, con = file.path(out_dir, "cells.txt"), useBytes = TRUE)
writeLines(genes, con = file.path(out_dir, "genes.txt"), useBytes = TRUE)

meta <- obj@meta.data[cells, , drop = FALSE]
meta$cell_id <- cells
utils::write.csv(meta, file = file.path(out_dir, "cell_metadata.csv"), row.names = FALSE)

var_feats <- VariableFeatures(obj[[assay_name]])
has_vf <- length(var_feats) > 0L
if (has_vf) {
  writeLines(var_feats, con = file.path(out_dir, "variable_features.txt"), useBytes = TRUE)
}

has_reduc <- reduc_name %in% names(obj@reductions)
emb_file <- NULL
if (has_reduc) {
  emb <- Embeddings(obj, reduction = reduc_name)[cells, , drop = FALSE]
  emb_file <- paste0("embedding_", reduc_name, ".csv")
  utils::write.csv(emb, file = file.path(out_dir, emb_file), row.names = TRUE)
}

summary_lines <- c(
  sprintf("assay=%s", assay_name),
  sprintf("n_cells=%d", length(cells)),
  sprintf("n_genes=%d", length(genes)),
  sprintf("nnz_counts=%d", length(cts@x)),
  sprintf("total_umi=%.0f", sum(cts@x)),
  sprintf("has_variable_features=%s", as.character(has_vf)),
  sprintf("has_reduction=%s", as.character(has_reduc)),
  sprintf("reduction_name=%s", reduc_name)
)
writeLines(summary_lines, con = file.path(out_dir, "export_summary.txt"), useBytes = TRUE)

manifest_json <- sprintf(
  paste0(
    "{\n",
    "  \"schema_version\": \"scslide-r-export-1\",\n",
    "  \"assay\": \"%s\",\n",
    "  \"n_cells\": %d,\n",
    "  \"n_genes\": %d,\n",
    "  \"nnz_counts\": %d,\n",
    "  \"total_umi\": %.0f,\n",
    "  \"counts_file\": \"counts_cells_by_genes.mtx\",\n",
    "  \"cells_file\": \"cells.txt\",\n",
    "  \"genes_file\": \"genes.txt\",\n",
    "  \"metadata_file\": \"cell_metadata.csv\",\n",
    "  \"variable_features_file\": %s,\n",
    "  \"embedding_file\": %s\n",
    "}"
  ),
  assay_name,
  length(cells),
  length(genes),
  length(cts@x),
  sum(cts@x),
  if (has_vf) "\"variable_features.txt\"" else "null",
  if (has_reduc) sprintf("\"%s\"", emb_file) else "null"
)
writeLines(manifest_json, con = file.path(out_dir, "export_manifest.json"), useBytes = TRUE)
cat("Exported portable Seurat bundle to:", normalizePath(out_dir), "\n")
