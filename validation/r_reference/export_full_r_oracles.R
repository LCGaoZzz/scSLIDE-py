#!/usr/bin/env Rscript
# Comprehensive real-backend R oracle exporter for both pinned reference profiles:
#   - backend = "scSLIDE" (legacy: LCGaoZzz/scSLIDE@968dbde299458100346c792aaf60f620c69f41fd; full saved profile used Seurat 6.0.0)
#   - backend = "Seurat"  (v6: satijalab/seurat@e6ba2b24149efc0ad62f9c48a9da527dfda4466d = CRAN Seurat 6.0.0)
#
# Never substitutes Python or hand-counted formulas for upstream R functions.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop("Usage: Rscript export_full_r_oracles.R <scSLIDE|Seurat> <lmc_contract.json> <new-output-dir>")
}
backend <- args[[1L]]
if (!(backend %in% c("scSLIDE", "Seurat"))) stop("backend must be 'scSLIDE' or 'Seurat'")
lmc_json_path <- normalizePath(args[[2L]], mustWork = TRUE)
out_dir <- path.expand(args[[3L]])
if (file.exists(out_dir) || dir.exists(out_dir)) {
  stop("Output directory already exists; refusing to overwrite evidence: ", out_dir)
}
dir.create(out_dir, recursive = TRUE)
out_dir <- normalizePath(out_dir, mustWork = TRUE)

suppressPackageStartupMessages({
  library(jsonlite)
  library(Matrix)
  library(SeuratObject)
  library(Seurat)
  library(pls)
  library(spls)
  library(princurve)
  library(glmGamPoi)
  library(destiny)
  library(BPCells)
})
if (backend == "scSLIDE") {
  # Attach scSLIDE AFTER Seurat so scSLIDE::* symbols take precedence and getExportedValue("scSLIDE", ...) calls scSLIDE's exact compiled C++/R code!
  suppressPackageStartupMessages(library(scSLIDE))
} else {
  if ("scSLIDE" %in% rownames(installed.packages(lib.loc = .libPaths()[1L]))) {
    stop("Seurat v6 profile library must not contain legacy scSLIDE")
  }
}

ns_pkg <- asNamespace(backend)
writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))

get_fn <- function(name) {
  if (name %in% getNamespaceExports(ns_pkg)) {
    return(getExportedValue(backend, name))
  }
  return(NULL)
}

manifest <- list(
  backend = backend,
  backend_version = as.character(packageVersion(backend)),
  seurat_version = as.character(packageVersion("Seurat")),
  seurat_object_version = as.character(packageVersion("SeuratObject")),
  pls_version = as.character(packageVersion("pls")),
  spls_version = as.character(packageVersion("spls")),
  destiny_version = as.character(packageVersion("destiny")),
  glmgampoi_version = as.character(packageVersion("glmGamPoi")),
  bpcells_version = as.character(packageVersion("BPCells")),
  princurve_version = as.character(packageVersion("princurve")),
  r_version = R.version.string,
  lib_paths = .libPaths(),
  exported_symbols = sort(getNamespaceExports(ns_pkg)),
  started_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  operations = list()
)

# ---------------------------------------------------------------------------
# 1. W1: Programmatic Seurat object + Neighbor from lmc_contract.json -> REAL GenerateSampleObject & NormalizeChiSquared
# ---------------------------------------------------------------------------
spec <- jsonlite::fromJSON(lmc_json_path)
q_ids <- as.character(spec$query_ids)
s_for_q <- factor(as.character(spec$sample_for_query), levels = as.character(spec$sample_order))
lm_ids <- as.character(spec$landmark_ids)
nn_idx <- matrix(as.integer(spec$neighbor_index), nrow = length(q_ids), ncol = ncol(spec$neighbor_index))
nn_dist <- matrix(0.25, nrow = nrow(nn_idx), ncol = ncol(nn_idx))

# Minimal valid count matrix for CreateSeuratObject (4 genes x 8 cells)
toy_counts <- matrix(1L, nrow = 4L, ncol = length(q_ids),
                     dimnames = list(paste0("G", 1:4), q_ids))
meta_toy <- data.frame(
  Sample = s_for_q,
  Donor = as.character(s_for_q),
  CellGroup = c("T", "M", "B", "T", "T", "M", "B", "M"),
  row.names = q_ids,
  stringsAsFactors = FALSE
)
toy_obj <- CreateSeuratObject(counts = toy_counts, meta.data = meta_toy)
toy_obj[["LANDMARK"]] <- subset(toy_obj[["RNA"]], cells = lm_ids)
toy_nn <- new(
  Class = "Neighbor",
  nn.idx = nn_idx,
  nn.dist = nn_dist,
  alg.info = list(),
  cell.names = q_ids
)
toy_obj[["weighted.nn"]] <- toy_nn

gen_fn <- get_fn("GenerateSampleObject")
norm_fn <- get_fn("NormalizeChiSquared")

# Note: s_for_q has 3 factor levels A, B, C where C only has landmark L3 (0 non-landmark queries!)
lmc_raw <- gen_fn(
  object = toy_obj,
  nn.name = "weighted.nn",
  k.nn = as.integer(spec$k),
  sketch.assay = "LANDMARK",
  return.seurat = FALSE,
  group.by = "Sample",
  rename.group.by = "CellGroup",
  verbose = FALSE
)
write.csv(lmc_raw, file.path(out_dir, "w1_lmc_counts_backend.csv"), row.names = TRUE)

# Also run return.seurat = TRUE with ChiSquared and LogNormalize
sample_seurat_chi <- gen_fn(
  object = toy_obj,
  nn.name = "weighted.nn",
  k.nn = as.integer(spec$k),
  sketch.assay = "LANDMARK",
  return.seurat = TRUE,
  group.by = "Sample",
  rename.group.by = "CellGroup",
  normalization.method = "ChiSquared",
  add.meta.data = TRUE,
  verbose = FALSE
)
chi_mat <- as.matrix(LayerData(sample_seurat_chi[["LMC"]], layer = "data"))
write.csv(chi_mat, file.path(out_dir, "w1_pearson_residuals_seurat_obj.csv"), row.names = TRUE)
write.csv(sample_seurat_chi@meta.data, file.path(out_dir, "w1_sample_metadata.csv"), row.names = TRUE)

sample_seurat_log <- gen_fn(
  object = toy_obj,
  nn.name = "weighted.nn",
  k.nn = as.integer(spec$k),
  sketch.assay = "LANDMARK",
  return.seurat = TRUE,
  group.by = "Sample",
  rename.group.by = "CellGroup",
  normalization.method = "LogNormalize",
  scale.factor = 10000,
  add.meta.data = TRUE,
  verbose = FALSE
)
log_mat <- as.matrix(LayerData(sample_seurat_log[["LMC"]], layer = "data"))
write.csv(log_mat, file.path(out_dir, "w1_log_normalized_seurat_obj.csv"), row.names = TRUE)

manifest$operations$W1_LMC <- list(status = "EXECUTED", shape = dim(lmc_raw))

# ---------------------------------------------------------------------------
# 2. Construct deterministic 12-donor, 360-cell, 60-gene frozen benchmark cohort
#    (Shared inputs exported to CSV so Python runs on the EXACT same bytes!)
# ---------------------------------------------------------------------------
set.seed(20261113L)
n_donors <- 12L
cells_per_donor <- 30L
n_cells <- n_donors * cells_per_donor
n_genes <- 60L

donor_ids <- sprintf("D%02d", seq_len(n_donors))
donor_disease <- rep(c("HV", "MILD", "SEV"), each = 4L)
donor_batch <- rep(c("B1", "B2"), times = 6L)
donor_age <- seq(36, 80, length.out = n_donors)
donor_sex <- rep(c("F", "M", "M", "F"), times = 3L)

cell_ids <- sprintf("cell_%04d", seq_len(n_cells))
gene_ids <- sprintf("GENE_%03d", seq_len(n_genes))

cell_donor <- rep(donor_ids, each = cells_per_donor)
cell_disease <- rep(donor_disease, each = cells_per_donor)
cell_batch <- rep(donor_batch, each = cells_per_donor)
cell_age <- rep(donor_age, each = cells_per_donor)
cell_sex <- rep(donor_sex, each = cells_per_donor)
cell_type <- rep(c("Mono", "CD4T", "NK"), times = n_cells / 3L)
sev_score <- ifelse(cell_disease == "HV", 0.0, ifelse(cell_disease == "MILD", 1.0, 2.4))

rates <- matrix(0.8, nrow = n_genes, ncol = n_cells)
mono_cols <- which(cell_type == "Mono")
rates[1:10, mono_cols] <- sweep(rates[1:10, mono_cols, drop = FALSE], 2, 2.0 + 1.6 * sev_score[mono_cols], `+`)
rates[11:20, which(cell_type == "CD4T")] <- 3.5
rates[21:30, which(cell_type == "NK")] <- 3.5
# Batch shift on genes 31:38 for B1
rates[31:38, which(cell_batch == "B1")] <- rates[31:38, which(cell_batch == "B1")] + 1.8

counts_mat <- matrix(rpois(n_genes * n_cells, lambda = as.vector(rates)),
                     nrow = n_genes, ncol = n_cells,
                     dimnames = list(gene_ids, cell_ids))
meta_cohort <- data.frame(
  Donor = cell_donor,
  Disease = factor(cell_disease, levels = c("HV", "MILD", "SEV")),
  SeverityNum = sev_score,
  Batch = cell_batch,
  Age = cell_age,
  Sex = cell_sex,
  cell_type = cell_type,
  row.names = cell_ids,
  stringsAsFactors = FALSE
)

coh_obj <- CreateSeuratObject(counts = as(counts_mat, "dgCMatrix"), meta.data = meta_cohort)
coh_obj <- NormalizeData(coh_obj, normalization.method = "LogNormalize", scale.factor = 10000, verbose = FALSE)
coh_obj <- FindVariableFeatures(coh_obj, nfeatures = 35L, verbose = FALSE)
coh_obj <- ScaleData(coh_obj, features = VariableFeatures(coh_obj), verbose = FALSE)
coh_obj <- RunPCA(coh_obj, features = VariableFeatures(coh_obj), npcs = 10L, verbose = FALSE)

# Export shared inputs once so Python parity runner consumes identical bytes
write.csv(counts_mat, file.path(out_dir, "shared_counts_genes_by_cells.csv"), row.names = TRUE)
write.csv(as.matrix(LayerData(coh_obj[["RNA"]], layer = "data")), file.path(out_dir, "shared_lognorm_genes_by_cells.csv"), row.names = TRUE)
write.csv(meta_cohort, file.path(out_dir, "shared_cell_metadata.csv"), row.names = TRUE)
write.csv(Embeddings(coh_obj, "pca"), file.path(out_dir, "shared_pca_embeddings.csv"), row.names = TRUE)
writeLines(VariableFeatures(coh_obj), file.path(out_dir, "shared_initial_hvgs.txt"))

# Fixed deterministic training (15 per donor = 180 cells) and landmark (48 cells) subsets
train_cells <- unlist(lapply(split(cell_ids, cell_donor), function(x) x[1:15]), use.names = FALSE)
lm_cells <- unlist(lapply(split(train_cells, meta_cohort[train_cells, "Donor"]), function(x) x[1:4]), use.names = FALSE)
writeLines(train_cells, file.path(out_dir, "shared_training_cells.txt"))
writeLines(lm_cells, file.path(out_dir, "shared_landmark_cells.txt"))

coh_obj[["TRAINING"]] <- subset(coh_obj[["RNA"]], cells = train_cells)
coh_obj[["LANDMARK"]] <- subset(coh_obj[["RNA"]], cells = lm_cells)

# ---------------------------------------------------------------------------
# 3. QuickCorTest (F10): numeric + categorical + grouped
# ---------------------------------------------------------------------------
qct_fn <- get_fn("QuickCorTest")
qct_cat_grp <- qct_fn(
  coh_obj, assay = "TRAINING", layer = "data",
  group.by = "cell_type", Y = "Disease",
  top.prop = 0.15, cor.cutoff = 0.15, min.pct = 0.1, min.cells = 40,
  unlist = TRUE, verbose = FALSE
)
qct_num <- qct_fn(
  coh_obj, assay = "TRAINING", layer = "data",
  group.by = NULL, Y = "SeverityNum",
  top.prop = 0.20, cor.cutoff = 0.10, min.pct = 0.1, min.cells = 40,
  unlist = TRUE, verbose = FALSE
)
writeLines(as.character(qct_cat_grp), file.path(out_dir, "f10_quickcortest_cat_grouped.txt"))
writeLines(as.character(qct_num), file.path(out_dir, "f10_quickcortest_numeric.txt"))

all_hvg <- union(VariableFeatures(coh_obj), qct_cat_grp)
writeLines(all_hvg, file.path(out_dir, "shared_all_hvgs.txt"))

# ---------------------------------------------------------------------------
# 4. RunPLS (plsr, cppls, spls), kernelpls/cppls_ondisk (BPCells), ProjectCellEmbeddings, PredictPLS
# ---------------------------------------------------------------------------
DefaultAssay(coh_obj) <- "TRAINING"
coh_obj <- ScaleData(coh_obj, assay = "TRAINING", features = all_hvg, verbose = FALSE)
if (backend == "scSLIDE") {
  # Record genuine upstream bug: scSLIDE::RunPLS(..., save.model=TRUE) calls base::rowMeans on sparse dgCMatrix 'data' layer
  sparse_err <- tryCatch({
run_pls_fn <- get_fn("RunPLS")
    run_pls_fn(coh_obj, assay = "TRAINING", features = all_hvg, ncomp = 2L,
               Y = "SeverityNum", pls.function = "plsr", save.model = TRUE, verbose = FALSE)
    NULL
  }, error = function(e) conditionMessage(e))
  manifest$upstream_exceptions <- list(
    legacy_RunPLS_save_model_sparse_data_bug = sparse_err
  )
  # Convert TRAINING 'data' layer to dense matrix so legacy save.model=TRUE & PredictPLS can proceed
  LayerData(coh_obj[["TRAINING"]], layer = "data") <- as.matrix(LayerData(coh_obj[["TRAINING"]], layer = "data"))
}
train_scaled <- as.matrix(LayerData(coh_obj[["TRAINING"]], layer = "scale.data")[all_hvg, train_cells])
write.csv(train_scaled, file.path(out_dir, "shared_training_scaled_hvgs.csv"), row.names = TRUE)


# 4a. plsr (single numeric Y with ncomp=5 -> proves B05 in R too!)
if (backend == "scSLIDE") {
  coh_obj <- run_pls_fn(coh_obj, assay = "TRAINING", features = all_hvg, ncomp = 5L,
                        Y = "SeverityNum", pls.function = "plsr",
                        reduction.name = "pls_num", save.model = TRUE, verbose = FALSE)
} else {
  coh_obj <- run_pls_fn(coh_obj, assay = "TRAINING", features = all_hvg, ncomp = 5L,
                        Y = "SeverityNum", pls.function = "plsr",
                        reduction.name = "pls_num", verbose = FALSE)
}
write.csv(Embeddings(coh_obj, "pls_num"), file.path(out_dir, "f13_plsr_single_y_scores.csv"), row.names = TRUE)
write.csv(Loadings(coh_obj, "pls_num"), file.path(out_dir, "f13_plsr_single_y_projection.csv"), row.names = TRUE)

# 4b. cppls on categorical Y = Disease (3 indicator columns -> highlights v6 vs legacy_ondisk Y-centering!)
if (backend == "scSLIDE") {
  coh_obj <- run_pls_fn(coh_obj, assay = "TRAINING", features = all_hvg, ncomp = 6L,
                        Y = "Disease", pls.function = "cppls",
                        reduction.name = "pls", save.model = TRUE, verbose = FALSE)
} else {
  coh_obj <- run_pls_fn(coh_obj, assay = "TRAINING", features = all_hvg, ncomp = 6L,
                        Y = "Disease", pls.function = "cppls",
                        reduction.name = "pls", verbose = FALSE)
}
write.csv(Embeddings(coh_obj, "pls"), file.path(out_dir, "f13_cppls_disease_scores.csv"), row.names = TRUE)
write.csv(Loadings(coh_obj, "pls"), file.path(out_dir, "f13_cppls_disease_projection.csv"), row.names = TRUE)

# 4c. spls on numeric Y
if (backend == "scSLIDE") {
  # Record genuine upstream bug #2: legacy RunPLS.StdAssay forwards threads=1L via ... into spls::spls -> unused argument
  spls_seurat_err <- tryCatch({
    run_pls_fn(coh_obj, assay = "TRAINING", features = all_hvg, ncomp = 4L,
               Y = "SeverityNum", pls.function = "spls", eta = 0.5,
               reduction.name = "spls_num", verbose = FALSE)
    NULL
  }, error = function(e) conditionMessage(e))
  manifest$upstream_exceptions$legacy_RunPLS_StdAssay_spls_threads_leak_bug <- spls_seurat_err
  # Call exported RunPLS.default directly on train_scaled (without leaked threads=1L) to get exact legacy spls numeric oracle
  spls_red <- run_pls_fn(train_scaled, assay = "TRAINING", ncomp = 4L,
                         Y = meta_cohort[train_cells, "SeverityNum"],
                         pls.function = "spls", eta = 0.5,
                         reduction.name = "spls_num", verbose = FALSE)
  coh_obj[["spls_num"]] <- spls_red
} else {
  coh_obj <- run_pls_fn(coh_obj, assay = "TRAINING", features = all_hvg, ncomp = 4L,
                        Y = "SeverityNum", pls.function = "spls", eta = 0.5,
                        reduction.name = "spls_num", verbose = FALSE)
}
write.csv(Embeddings(coh_obj, "spls_num"), file.path(out_dir, "f13_spls_scores.csv"), row.names = TRUE)
write.csv(Loadings(coh_obj, "spls_num"), file.path(out_dir, "f13_spls_projection.csv"), row.names = TRUE)

# 4d. BPCells on-disk kernelpls & cppls_ondisk (F18, F19)!
bp_dir <- file.path(out_dir, "bpcells_train_store")
bp_mat <- BPCells::write_matrix_dir(as(train_scaled, "dgCMatrix"), dir = bp_dir)
y_dis_mat <- model.matrix(~ . + 0, data = meta_cohort[train_cells, "Disease", drop = FALSE])

kpls_fn <- get_fn("kernelpls")
cppls_od_fn <- get_fn("cppls_ondisk")
res_kpls_od <- kpls_fn(bp_mat, Y = y_dis_mat, ncomp = 5L)
res_cppls_od <- cppls_od_fn(bp_mat, Y = y_dis_mat, ncomp = 5L)

rownames(res_kpls_od$scores) <- train_cells
colnames(res_kpls_od$scores) <- paste0("PLS_", 1:5)
rownames(res_cppls_od$scores) <- train_cells
colnames(res_cppls_od$scores) <- paste0("PLS_", 1:5)
write.csv(unclass(res_kpls_od$scores), file.path(out_dir, "f19_kernelpls_ondisk_scores.csv"), row.names = TRUE)
write.csv(unclass(res_cppls_od$scores), file.path(out_dir, "f18_cppls_ondisk_scores.csv"), row.names = TRUE)
rownames(res_cppls_od$projection) <- all_hvg
colnames(res_cppls_od$projection) <- paste0("PLS_", 1:5)
write.csv(unclass(res_cppls_od$projection), file.path(out_dir, "f18_cppls_ondisk_projection.csv"), row.names = TRUE)

# 4e. ProjectCellEmbeddings (RNA <- TRAINING pls) & PredictPLS (legacy F08)
DefaultAssay(coh_obj) <- "RNA"
proj_pls <- ProjectCellEmbeddings(
  query = coh_obj, query.assay = "RNA",
  reference = coh_obj, reference.assay = "TRAINING",
  reduction = "pls", dims = 1:6, scale = TRUE,
  normalization.method = "LogNormalize", verbose = FALSE
)
coh_obj[["proj.pls"]] <- CreateDimReducObject(embeddings = proj_pls, key = "pPLS_", assay = "RNA")
write.csv(proj_pls, file.path(out_dir, "f08_project_cell_embeddings_pls.csv"), row.names = TRUE)

pred_fn <- get_fn("PredictPLS")
if (!is.null(pred_fn)) {
  pred_y <- pred_fn(coh_obj, newdata = coh_obj, reduction = "pls", assay = "RNA", layer = "data")
  write.csv(pred_y, file.path(out_dir, "f08_predict_pls_seurat.csv"), row.names = TRUE)
  manifest$operations$F08_PredictPLS <- "EXECUTED"
} else {
  manifest$operations$F08_PredictPLS <- "NOT_EXPORTED_IN_BACKEND"
}

# ---------------------------------------------------------------------------
# 5. FindsmNN (F04) and FindmmNN (F03: fixed weights AND learned weights!)
# ---------------------------------------------------------------------------
smnn_fn <- get_fn("FindsmNN")
mmnn_fn <- get_fn("FindmmNN")

coh_obj <- smnn_fn(
  coh_obj, sketch.assay = "LANDMARK", reduction = "proj.pls",
  dims = 1:6, k.nn = 5L, l2.norm = TRUE, nn.name = "single.nn", verbose = FALSE
)
sm_idx <- coh_obj[["single.nn"]]@nn.idx
rownames(sm_idx) <- coh_obj[["single.nn"]]@cell.names
colnames(sm_idx) <- paste0("nn_", seq_len(ncol(sm_idx)))
write.csv(sm_idx, file.path(out_dir, "f04_findsmnn_idx.csv"), row.names = TRUE)
coh_obj <- smnn_fn(
  coh_obj, sketch.assay = "LANDMARK", reduction = "proj.pls",
  dims = 1:6, k.nn = 5L, l2.norm = TRUE, nn.method = "rann",
  nn.name = "single.rann.nn", verbose = FALSE
)
sm_rann_idx <- coh_obj[["single.rann.nn"]]@nn.idx
rownames(sm_rann_idx) <- coh_obj[["single.rann.nn"]]@cell.names
colnames(sm_rann_idx) <- paste0("nn_", seq_len(ncol(sm_rann_idx)))
write.csv(sm_rann_idx, file.path(out_dir, "f04_findsmnn_rann_exact_idx.csv"), row.names = TRUE)

coh_obj <- mmnn_fn(
  coh_obj, sketch.assay = "LANDMARK",
  reduction.list = list("pca", "proj.pls"),
  dims.list = list(1:10, 1:6),
  k.nn = 5L, knn.range = 48L, l2.norm = TRUE,
  fix.wnn.weights = c(0.5, 0.5),
  nn.name = "weighted.nn", verbose = FALSE
)
mm_idx <- coh_obj[["weighted.nn"]]@nn.idx
mm_dist <- coh_obj[["weighted.nn"]]@nn.dist
rownames(mm_idx) <- rownames(mm_dist) <- coh_obj[["weighted.nn"]]@cell.names
colnames(mm_idx) <- colnames(mm_dist) <- paste0("nn_", seq_len(ncol(mm_idx)))
write.csv(mm_idx, file.path(out_dir, "f03_findmmnn_fixed_idx.csv"), row.names = TRUE)
write.csv(mm_dist, file.path(out_dir, "f03_findmmnn_fixed_dist.csv"), row.names = TRUE)

coh_obj <- mmnn_fn(
  coh_obj, sketch.assay = "LANDMARK",
  reduction.list = list("pca", "proj.pls"),
  dims.list = list(1:10, 1:6),
  k.nn = 5L, knn.range = 48L, l2.norm = TRUE,
  fix.wnn.weights = NULL,
  nn.name = "weighted.learned.nn", verbose = FALSE
)
mml_idx <- coh_obj[["weighted.learned.nn"]]@nn.idx
rownames(mml_idx) <- coh_obj[["weighted.learned.nn"]]@cell.names
colnames(mml_idx) <- paste0("nn_", seq_len(ncol(mml_idx)))
write.csv(mml_idx, file.path(out_dir, "f03_findmmnn_learned_idx.csv"), row.names = TRUE)

# ---------------------------------------------------------------------------
# 6. GenerateSampleObject (F05) -> ScaleData(center) -> RunDiffusionMap (F12) -> princurve
# ---------------------------------------------------------------------------
sample_obj <- gen_fn(
  coh_obj, sketch.assay = "LANDMARK", nn.name = "weighted.nn",
  group.by = "Donor", rename.group.by = "cell_type",
  k.nn = 5L, normalization.method = "ChiSquared",
  add.meta.data = TRUE, return.seurat = TRUE, verbose = FALSE
)
sample_cts <- as.matrix(LayerData(sample_obj[["LMC"]], layer = "counts"))
sample_res <- as.matrix(LayerData(sample_obj[["LMC"]], layer = "data"))
write.csv(sample_cts, file.path(out_dir, "f05_cohort_lmc_counts.csv"), row.names = TRUE)
write.csv(sample_res, file.path(out_dir, "f06_cohort_pearson_residuals.csv"), row.names = TRUE)

sample_obj <- ScaleData(sample_obj, assay = "LMC", features = rownames(sample_obj),
                        do.scale = FALSE, do.center = TRUE, verbose = FALSE)
dm_fn <- get_fn("RunDiffusionMap")
sample_obj <- dm_fn(
  sample_obj, assay = "LMC", layer = "scale.data",
  features = rownames(sample_obj), ncomp = 4L, seed.use = 101L, verbose = FALSE
)
dm_emb <- Embeddings(sample_obj, "DiffMap")
dm_ev <- sample_obj[["DiffMap"]]@stdev
write.csv(dm_emb, file.path(out_dir, "f12_diffusion_embeddings.csv"), row.names = TRUE)
write.csv(data.frame(component = paste0("DC_", seq_along(dm_ev)), eigenvalue = dm_ev),
          file.path(out_dir, "f12_diffusion_eigenvalues.csv"), row.names = FALSE)

pcurve <- princurve::principal_curve(dm_emb[, 1:2, drop = FALSE], plot = FALSE, maxit = 20L)
sample_obj[["traj_DC1"]] <- dm_emb[, 1]
sample_obj[["traj_overall"]] <- pcurve$lambda
pcurve_df <- data.frame(
  sample_id = rownames(dm_emb),
  lambda = as.numeric(pcurve$lambda),
  s1 = pcurve$s[, 1],
  s2 = pcurve$s[, 2],
  ord = as.integer(pcurve$ord)
)
write.csv(pcurve_df, file.path(out_dir, "f12_principal_curve.csv"), row.names = FALSE)

# ---------------------------------------------------------------------------
# 7. AggregateExpression -> TrajDETest (F16)
# ---------------------------------------------------------------------------
bulk_obj <- AggregateExpression(coh_obj, assays = "RNA", group.by = c("Donor", "cell_type"), return.seurat = TRUE, verbose = FALSE)
mdata_pb <- coh_obj@meta.data[, c("Donor", "cell_type")]
mdata_pb <- mdata_pb[!duplicated(mdata_pb), ]
rownames(mdata_pb) <- paste0(gsub("_", "-", mdata_pb$Donor), "_", gsub("_", "-", mdata_pb$cell_type))
bulk_obj <- AddMetaData(bulk_obj, metadata = mdata_pb)
bulk_obj$nCount_RNA <- colSums(LayerData(bulk_obj[["RNA"]], layer = "counts"))
bulk_obj$nFeature_RNA <- colSums(LayerData(bulk_obj[["RNA"]], layer = "counts") > 0)

sub_bulk <- subset(bulk_obj, subset = cell_type == "Mono")
s_meta_join <- sample_obj@meta.data[, c("Donor", "Age", "Sex", "Disease", "traj_DC1", "traj_overall")]
new_m <- merge(sub_bulk@meta.data[, "Donor", drop = FALSE], s_meta_join, by = "Donor", all.x = TRUE, sort = FALSE)
rownames(new_m) <- rownames(sub_bulk@meta.data)
sub_bulk <- AddMetaData(sub_bulk, metadata = new_m)
sub_bulk$log_count <- log10(sub_bulk$nCount_RNA)

write.csv(as.matrix(LayerData(sub_bulk[["RNA"]], layer = "counts")), file.path(out_dir, "f16_mono_pseudobulk_counts.csv"), row.names = TRUE)
write.csv(as.matrix(LayerData(sub_bulk[["RNA"]], layer = "data")), file.path(out_dir, "f16_mono_pseudobulk_data.csv"), row.names = TRUE)
write.csv(sub_bulk@meta.data, file.path(out_dir, "f16_mono_pseudobulk_meta.csv"), row.names = TRUE)

traj_de_fn <- get_fn("TrajDETest")
de_res <- traj_de_fn(
  sub_bulk, assay = "RNA", layer = "counts",
  traj.var = "traj_DC1", latent.vars = c("Age", "Sex", "log_count"),
  complete.results = TRUE, verbose = FALSE
)
write.csv(de_res, file.path(out_dir, "f16_trajdetest_complete.csv"), row.names = TRUE)

# ---------------------------------------------------------------------------
# 8. Legacy-specific exports: CorrectSampleComBat (F02), cellanova_calc_BE (F17),
#    BuildLandmarkObject (F01), PlotLandmarkObject (F07), SampleLevelDimPlot (F14), RunAndProjectUMAP (F11)
# ---------------------------------------------------------------------------
combat_fn <- get_fn("CorrectSampleComBat")
if (!is.null(combat_fn)) {
  s_combat_reg <- combat_fn(
    sample_obj, assay = "LMC", layer = "data",
    batch_key = "Batch", condition_key = "Disease", control_label = "HV",
    par.prior = TRUE, method = "regression",
    correct.attenuation = TRUE, floor.beta = 0, verbose = FALSE
  )
  write.csv(as.matrix(LayerData(s_combat_reg[["LMC"]], layer = "data")),
            file.path(out_dir, "f02_combat_regression_atten.csv"), row.names = TRUE)

  # Record genuine upstream bug #4: .combat_int_eprior crashes with NA when any landmark has 0 variance within a control batch
  np_err <- tryCatch({
    combat_fn(sample_obj, assay = "LMC", layer = "data",
              batch_key = "Batch", condition_key = "Disease", control_label = "HV",
              par.prior = FALSE, method = "subtraction", verbose = FALSE)
    NULL
  }, error = function(e) conditionMessage(e))
  manifest$upstream_exceptions$legacy_CorrectSampleComBat_nonparam_zero_ctrl_var_bug <- np_err

  # Export par.prior=TRUE subtraction on full sample_obj AND par.prior=FALSE subtraction on positive-variance landmarks
  s_combat_sub <- combat_fn(
    sample_obj, assay = "LMC", layer = "data",
    batch_key = "Batch", condition_key = "Disease", control_label = "HV",
    par.prior = TRUE, method = "subtraction", verbose = FALSE
  )
  write.csv(as.matrix(LayerData(s_combat_sub[["LMC"]], layer = "data")),
            file.path(out_dir, "f02_combat_param_subtraction.csv"), row.names = TRUE)

  lmc_dat <- as.matrix(LayerData(sample_obj[["LMC"]], layer = "data"))
  b1_hv <- colnames(sample_obj)[sample_obj$Batch == "B1" & sample_obj$Disease == "HV"]
  b2_hv <- colnames(sample_obj)[sample_obj$Batch == "B2" & sample_obj$Disease == "HV"]
  pos_var_lm <- rownames(lmc_dat)[matrixStats::rowVars(lmc_dat[, b1_hv, drop=FALSE]) > 1e-8 &
                                  matrixStats::rowVars(lmc_dat[, b2_hv, drop=FALSE]) > 1e-8]
  if (length(pos_var_lm) >= 4L) {
    sub_so <- subset(sample_obj, features = pos_var_lm)
    s_combat_np <- combat_fn(
      sub_so, assay = "LMC", layer = "data",
      batch_key = "Batch", condition_key = "Disease", control_label = "HV",
      par.prior = FALSE, method = "subtraction", verbose = FALSE
    )
    write.csv(as.matrix(LayerData(s_combat_np[["LMC"]], layer = "data")),
              file.path(out_dir, "f02_combat_nonparam_posvar_subtraction.csv"), row.names = TRUE)
  }
  manifest$operations$F02_CorrectSampleComBat <- "EXECUTED"
} else {
  manifest$operations$F02_CorrectSampleComBat <- "NOT_EXPORTED_IN_BACKEND"
}

ca_fn <- get_fn("cellanova_calc_BE")
if (!is.null(ca_fn)) {
  coh_ca <- ca_fn(
    coh_obj, assay = "RNA", layer = "scale.data",
    integrate_key = "Donor", features = VariableFeatures(coh_obj),
    control_dict = list(ctrl = c("D01", "D02", "D03", "D04")),
    reduction = "pca", k_select = 3L,
    new.assay.name = "CORRECTED", store.diagnostics = TRUE, verbose = FALSE
  )
  ca_mat <- as.matrix(LayerData(coh_ca[["CORRECTED"]], layer = "data"))
  write.csv(ca_mat, file.path(out_dir, "f17_cellanova_corrected.csv"), row.names = TRUE)
  manifest$operations$F17_cellanova_calc_BE <- "EXECUTED"
} else {
  manifest$operations$F17_cellanova_calc_BE <- "NOT_EXPORTED_IN_BACKEND"
}

blo_fn <- get_fn("BuildLandmarkObject")
if (!is.null(blo_fn)) {
  lm_obj <- blo_fn(
    object = coh_obj, sample.obj = sample_obj,
    nn.name = "weighted.nn", landmark.assay.name = "LANDMARK",
    features.to.test = c("traj_DC1", "traj_overall"),
    k.nn = 10L, min.dist = 0.3
  )
  cor_cols <- grep("^cor\\.coef_", colnames(lm_obj@meta.data), value = TRUE)
  write.csv(lm_obj@meta.data[, cor_cols, drop = FALSE], file.path(out_dir, "f01_landmark_correlations.csv"), row.names = TRUE)
  write.csv(Embeddings(lm_obj, "wnn.umap"), file.path(out_dir, "f01_landmark_wnn_umap.csv"), row.names = TRUE)

  plo_fn <- get_fn("PlotLandmarkObject")
  p_res <- plo_fn(lm_obj, order = TRUE)
  manifest$operations$F01_F07_LandmarkObject <- list(status = "EXECUTED", p1_class = class(p_res$p1))
} else {
  manifest$operations$F01_F07_LandmarkObject <- "NOT_EXPORTED_IN_BACKEND"
}

rap_fn <- get_fn("RunAndProjectUMAP")
if (!is.null(rap_fn)) {
  coh_umap <- rap_fn(
    coh_obj, nn.name = "weighted.nn", k.nn = 5L,
    landmark.assay.name = "LANDMARK", reduction.name = "sketched.umap", verbose = FALSE
  )
  write.csv(Embeddings(coh_umap, "proj.sketched.umap"), file.path(out_dir, "f11_projected_umap.csv"), row.names = TRUE)
  manifest$operations$F11_RunAndProjectUMAP <- "EXECUTED"
} else {
  manifest$operations$F11_RunAndProjectUMAP <- "NOT_EXPORTED_IN_BACKEND"
}

# 9. End-to-end PrepareSampleObject wrapper verification (F09 + F15)
prep_fn <- get_fn("PrepareSampleObject")
coh_e2e <- subset(coh_obj, cells = cell_ids)
DefaultAssay(coh_e2e) <- "RNA"
coh_e2e <- FindVariableFeatures(coh_e2e, nfeatures = 35L, verbose = FALSE)
coh_e2e <- prep_fn(
  object = coh_e2e, assay = "RNA", add.hvg = TRUE,
  group.by.CorTest = "cell_type", Y = "Disease",
  sketch.training = TRUE, group.by.Sketch = "Donor",
  ncells.per.group = 15L, ncells.landmark = 48L,
  ncomp = 5L, pls.function = "cppls",
  modality = "multi", k.nn = 5L, knn.range = 48L,
  name.reduction.1 = "pca", dims.reduction.1 = 1:10,
  fix.wnn.weights = c(0.5, 0.5), verbose = FALSE
)
s_e2e <- gen_fn(
  coh_e2e, sketch.assay = "LANDMARK", nn.name = "weighted.nn",
  group.by = "Donor", k.nn = 5L, normalization.method = "ChiSquared",
  return.seurat = TRUE, verbose = FALSE
)
write.csv(as.matrix(LayerData(s_e2e[["LMC"]], layer = "counts")), file.path(out_dir, "f09_prepare_e2e_lmc_counts.csv"), row.names = TRUE)
manifest$operations$F09_PrepareSampleObject <- "EXECUTED"

manifest$finished_at_utc <- format(Sys.time(), tz = "UTC", usetz = TRUE)
manifest$status <- "COMPLETED"
jsonlite::write_json(manifest, file.path(out_dir, "oracle_manifest.json"), auto_unbox = TRUE, pretty = TRUE)
cat("FULL R ORACLE EXPORT COMPLETED:", backend, "->", out_dir, "\n")
