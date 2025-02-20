library(dplyr)
library(impute)
library(tidyr)
library(ggplot2)
library(sva)
# library(umap)
library(viridis)
source('R/calc.R')
source('R/impute.R')
source('R/normalise.R')
source('R/plot.R')
source('R/utils.R')
theme_set(theme_bw(base_size = 7))


file <- 'data/astral/raw/all_sample.csv'
processed <- read.csv(file, row.names = 1)
uniprot_annot <- processed[, 1:2]

# Only pg_matrix has QC samples
# pg_matrix is scaled according to scaling factors for each sample
# pg_matrix is different from processed
file <- 'data/astral/raw/report.pg_matrix.tsv'
pg_matrix <- read.table(file, sep = '\t', header = T, row.names = 1)
colnames(pg_matrix) <- gsub('\\.', '-', colnames(pg_matrix))
colnames(pg_matrix)[1:5] <- substring(colnames(pg_matrix)[1:5], 10)
# Normalise matrix
sample_sums <- colSums(pg_matrix, na.rm = T)
scaling_ratio <- sample_sums / max(sample_sums, na.rm = T)
norm_pgmatrix <- sweep(pg_matrix, 2, scaling_ratio, '/')
lnmatrix <- log2(norm_pgmatrix)
lmatrix <- log2(pg_matrix)

# # Check that Novogene normalised values tally with ours
# # Novogene normalised values tallies with ours!
# file <- 'data/astral/raw/renorm-all_sample.csv'
# reprocessed <- read.csv(file, row.names = 1)
# reprocessed1 <- log2(reprocessed[, 3:ncol(reprocessed)])
# reprocessed1[1:5, paste0('QC', 1:5)]

# Metadata
# Inferring time of run
file <- 'data/astral/raw/raw-files.txt'
time_info <- read.table(file, header = TRUE, sep = '')
idx_qc <- grep('QC', time_info$Name)
rownames(time_info)[idx_qc] <- 
  substring(time_info$Name[idx_qc], 53, 63)
idx_fpep <- grep('FPEP', time_info$Name)
rownames(time_info)[idx_fpep] <- 
  substring(time_info$Name[idx_fpep], 51, 66)
time_info$DateTime <- 
  as.POSIXct(paste(time_info$Date, time_info$Time), format = '%m-%d-%Y %H:%M')
qc_sids <- grep('QC', rownames(time_info), value = T)[c(3:4, 2, 5, 1)]
qc_sids_short <- substring(qc_sids, 9)
# Experimental metadata - All studies
# Rownames: Novogene ID
file <- 'data/astral/metadata/experimental-metadata.csv'
expt_meta <- read.csv(file)
rownames(expt_meta) <- expt_meta$Polypeptide.Novogene.ID
expt_meta$Run.DateTime <- time_info[rownames(expt_meta), 'DateTime']
expt_meta$Sample.Name <- sub('_00', '_0', expt_meta$Sample.Name)
expt_meta$Sample.Name <- sub('_06', '_6', expt_meta$Sample.Name)

# Append QC metadata
qc_meta <- data.frame(matrix(
  NA, 5, 11,
  dimnames = list(qc_sids_short, colnames(expt_meta))
))
qc_meta$Run.DateTime <- time_info[qc_sids, 'DateTime']
qc_meta$Sample.Name <- qc_sids_short
qc_meta$Polypeptide.Novogene.ID <- qc_sids_short
expt_meta1 <- rbind(expt_meta, qc_meta)
 
file <- 'data/astral/metadata/metadata-all.csv'
metadata_all <- read.csv(file, row.names=1)

# Append class information
metadata <- merge(
  expt_meta1, metadata_all,
  by.x = 'Sample.Name', by.y = 'row.names', all.x = T
)
rownames(metadata) <- metadata$Sample.Name
metadata[qc_sids_short, 'Extraction.Date'] <- 'Not applicable'
metadata[qc_sids_short, 'Class'] <- 'QC'
metadata[qc_sids_short, 'Study'] <- 'QC'

# Rename colnames of lnmatrix
idx <- match(colnames(lnmatrix), metadata$Polypeptide.Novogene.ID)
colnames(lnmatrix) <- rownames(metadata)[idx]
colnames(lmatrix) <- rownames(metadata)[idx]

# Subset lyriks study
lyriks_qc_sids <- sort(grep('^[L|Q]', colnames(lnmatrix), value = T))
lyriks <- lnmatrix[, lyriks_qc_sids]
lyriks_unnorm <- lmatrix[, lyriks_qc_sids]
lyriks1 <- lyriks[, 1:402]

zlyriks <- lyriks
zlyriks[is.na(zlyriks)] <- 0

# Metadata10 has been processed to contain all lyriks samples in proteomics data
file <- 'data/astral/metadata/metadata10-lyriks.csv'
metadata10 <- read.csv(file, row.names = 1)
metadata10[metadata10 == ''] <- NA
metadata10$sid <- rownames(metadata10) 
metadata_lyriks_qc <- subset(metadata, Study %in%  c('LYRIKS', 'QC'))

metadata_lyriks <- merge(
  metadata_lyriks_qc, metadata10,
  by.x = 'Sample.Name', by.y = 'sid', all.x = TRUE
)
metadata_lyriks[403:407, c('label', 'final_label', 'period')] <- 'QC'
metadata_lyriks$period <- as.factor(metadata_lyriks$period)
rownames(metadata_lyriks) <- metadata_lyriks$Sample.Name

colnames(metadata_lyriks)
metadata_lyriks$Polypeptide.Novogene.ID
metadata_lyriks$final_label
sum(metadata_lyriks$label == 'relapse')
sum(metadata_lyriks$label == 'remit')
metadata_lyriks$final_label[metadata_lyriks$label == 'relapse']

# file <- 'data/astral/metadata/metadata-csa.csv'
# metadata_csa <- read.csv(file, row.names = 1)

# # Combine metadata
# lyriks_class <- metadata10[colnames(data)[1:402], 'final_label']
# lyriks_class <- recode(
#   lyriks_class,
#   ctrl = 'Healthy control', cvt = 'Convert',
#   mnt = 'Maintain', rmt = 'Remit'
# )
# csa_class <- metadata_csa[colnames(data)[403:599], 'class']
# metadata_all <- data.frame(
#   class = c(lyriks_class, csa_class, rep('Bipolar', 41)),
#   study = c(rep('lyriks', 402), rep('csa', 197), rep('abgn', 41)),
#   row.names = colnames(data)
# )


# LYRIKS: Detailed metadata
# file <- 'data/lyriks/metadata/metadata_57.csv'
# metadata57 <- read.csv(file, row.names=1)
# file <- 'data/lyriks/metadata/metadata_74.csv'
# metadata74 <- read.csv(file, row.names=1)
# file <- 'data/lyriks/metadata/metadata_521.csv'
# metadata521 <- read.csv(file)
# rownames(metadata521) <- paste(metadata521$sn, metadata521$Period, sep = '_')

# # Medication
# # group into psychotropics
# # only report the most intensive medication
# drug_colnames <- paste('drug', 1:3, sep = '_')
# drugs <- metadata57[, drug_colnames]
# drugs <- replace(drugs, drugs == 'Antihistamines', 'Medication')
# psychotropics <- c(
#   'Antidepressants', 'Antipsychotics', 'Anxiolytics', 'Mood stabilisers')
# drugs$drug_1[drugs$drug_1 %in% psychotropics] <- 'Psychotropics'
# drugs$drug_2[drugs$drug_2 %in% psychotropics] <- 'Psychotropics'
# drugs$drug_3[drugs$drug_3 %in% psychotropics] <- 'Psychotropics'
# drug_type <- rep('Nil', length = nrow(metadata57))
# contains_supplements <- apply(drugs == 'Supplements', 1, any)
# contains_medication <- apply(drugs == 'Medication', 1, any)
# is_psychotropic <- apply(drugs == 'Psychotropics', 1, any)
# # assign drug type
# drug_type[contains_supplements] <- 'Supplements'
# drug_type[contains_medication] <- 'Medication'
# drug_type[is_psychotropic] <- 'Psychotropics'
# drug_levels <- c('Psychotropics', 'Medication', 'Supplements', 'Nil')
# drug_type <- factor(drug_type, levels = drug_levels)
# metadata58 <- mutate(metadata57, drug_type = drug_type)
# 
# # Insert missing samples into metadata
# missing_sids <- colnames(lyriks)[!(colnames(lyriks) %in% rownames(metadata58))]
# metadata58[missing_sids,] <- NA
# metadata74[missing_sids,] <- NA
# 
# missing_sids1 <- colnames(lyriks)[!(colnames(lyriks) %in% rownames(metadata521))]
# metadata521[missing_sids1,] <- NA

##### EDA #####

# # Distribution of missingness
# pdf('tmp/fig/astral/features-pct-zero.pdf')
# hist(feature_pct_zero, breaks = 20)
# dev.off()
# 
# sample_pct_zero <- colSums(data == 0) / nrow(data)
# qc <- cbind(metadata_all, pct_zero = sample_pct_zero)
# 
# ax <- ggplot(qc) +
#   geom_point(
#     aes(x = study, y = pct_zero, color = class),
#     position = position_jitterdodge(jitter.width = 0.1, dodge.width = 0.8)
#   )
# file <- 'tmp/fig/astral/samples-pct-zero.pdf'
# ggsave(file, ax, width = 8, height = 5)
# 
# # remove outliers with pct zero < 0.46
# sum(sample_pct_zero < 0.46)
# 
# # plot distribution of protein expression
# long_data <- gather(data, key = 'sid', value = 'expression')
# long_data$class <- metadata_all[long_data$sid, 'class']
# long_data$study <- metadata_all[long_data$sid, 'study']
# 
# ax <- ggplot(long_data) +
#   facet_wrap(~ study + class, nrow = 2, scales = 'free') +
#   geom_density(
#     aes(x = expression, group = sid, col = class),
#     alpha = 0.5
#   )
# file <- 'tmp/fig/astral/samples-density.pdf'
# ggsave(file, ax, width = 15, height = 5)
# 
# # Filter out sparse features
# feat_nonsparse <- names(feature_pct_zero)[feature_pct_zero == 0]
# annot_nonsparse <- uniprot_annot[feat_nonsparse, ]
# write.csv(annot_nonsparse, 'tmp/astral/annot_nonsparse.csv')
# 
# ax <- ggplot_pca(
#   data[feat_nonsparse, ], metadata_all,
#   color = 'class', shape = 'study'
# )
# ggsave('tmp/fig/astral/fltr223-pca-all.pdf', ax, width = 10, height = 6)
# 
# # Plot: PCA elbow plot
# pca_obj <- prcomp(t(data))
# eigenvalues <- pca_obj$sdev ^ 2 / sum(pca_obj$sdev ^ 2)
# 
# pdf('tmp/fig/astral/pca-elbow.pdf')
# plot(eigenvalues[1:20])
# dev.off()
# 
# # Plot: Top PCs
# # increase dodge width
# ax <- ggplot_top_pc(
#   data, metadata_all,
#   'class', col = 'class', shape = 'study',
#   jitter.width = 0.4
# ) +
#   theme(axis.text.x = element_text(angle = 45, hjust = 1))
# ggsave('tmp/fig/astral/top_pcs.pdf', ax, width = 22, height = 8)
# 
# # Plot: UMAP
# ax <- ggplot_umap(
#   data[feat_nonsparse, ], metadata_all,
#   col = 'class', shape = 'study',
#   cex = 1.5, alpha = 0.7
# )
# ggsave('tmp/fig/astral/fltr223-umap.pdf', ax, width = 8, height = 5)
# 
# # Plot: Features (non-sparse)
# # Features: Fully present
# lyriks_feature_pct_zero <- rowSums(is.na(lyriks)) / ncol(lyriks)
# lyriks_sample_pct_zero <- colSums(is.na(lyriks)) / nrow(lyriks)
# 
# # lyriks[1:5, 1:10]
# # lyriks_unnorm[1:5, 1:10]
# 
# pct <- 0.1
# # idx <- lyriks_feature_pct_zero > pct & lyriks_feature_pct_zero <= (pct + 0.1)
# 
# idx <- lyriks_feature_pct_zero == 0
# 
# feats <- names(lyriks_feature_pct_zero)[idx]
# print(length(feats))
# set.seed(0)
# feats1 <- sample(feats, 50)
# 
# for (i in feats1) {
#   x <- unlist(lyriks_unnorm[i, ])
#   data <- cbind(expr = x, metadata_lyriks)
#   ax <- ggplot(data) +
#     facet_wrap(~final_label, nrow = 1, scales = 'free_x') +
#     geom_point(
#       aes(y = expr, x = period, col = Run.DateTime, shape = Extraction.Date),
#       position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8)
#     ) +
#     # scale_colour_viridis_d(option = 'plasma') +
#     theme(axis.text.x = element_text(angle = 45, hjust = 1))
#   file <- sprintf('tmp/astral/fig/features/lyriks_unnorm-runtime-feat-%s.pdf', i)
#   ggsave(file, ax, width = 12, height = 3.5)
#   print(file)
# }
# 
# for (i in feats1) {
#   x <- unlist(lyriks_unnorm[i, ])
#   data <- cbind(expr = x, metadata_lyriks)
#   ax <- subset(data, expr > 0 & Extraction.Date != 'Not applicable') %>%
#     ggplot() +
#       geom_density(
#         aes(x = expr, col = Extraction.Date, group = Extraction.Date)
#       ) # +
#       # scale_colour_viridis_d(option = 'plasma') +
#       # theme(axis.text.x = element_text(angle = 45, hjust = 1))
#   file <- sprintf('tmp/astral/fig/features/lyriks_unnorm-density-00-%s.pdf', i)
#   ggsave(file, ax, width = 6, height = 3.5)
#   print(file)
# }
# 
# feature_avg <- apply(lyriks, 1, function(x) mean(x[x != 0]))
# data <- data.frame(pct_zero = lyriks_feature_pct_zero, avg = feature_avg)
# ax <- ggplot(data) +
#   geom_point(aes(x = avg, y = pct_zero))
# file <- 'tmp/astral/fig/feat-avg_pct_zero.pdf'
# ggsave(file, ax, width = 8, height = 5)
# 
# # Plot: Missing values
# lyriks_sample_pct_zero <- colSums(is.na(lyriks1)) / nrow(lyriks1)
# sample_min <- apply(lyriks1, 2, min, na.rm = T)
# sample_avg <- colMeans(lyriks1, na.rm = T) 
# sample_sum <- colSums(lyriks1, na.rm = T) 
# identical(names(sample_min), colnames(lyriks1))
# 
# sample_stats <- cbind(
#   min = sample_min, 
#   avg = sample_avg, 
#   sum = sample_sum,
#   pct_zero = lyriks_sample_pct_zero,
#   metadata_lyriks[colnames(lyriks1), ]
# )
# 
# # Averages still differ for unnormalised data
# ax <- ggplot(sample_stats) +
#   geom_point(
#     aes(x = Extraction.Date, y = avg, col = Run.DateTime, shape = class.x),
#     position = position_jitterdodge(jitter.width = 0.1, dodge.width = 1)
#   ) +
#   theme(axis.text.x = element_text(angle = 45, hjust = 1))
# file <- 'tmp/astral/fig/sample_unnorm-avg.pdf'
# ggsave(file, ax, width = 8, height = 5)
# 
# ax <- ggplot(sample_stats) +
#   geom_point(
#     aes(x = Extraction.Date, y = pct_zero, col = Run.DateTime, shape = Class),
#     position = position_jitterdodge(jitter.width = 0.11, dodge.width = 0.8),
#     cex = 1,
#   ) +
#   labs(
#     x = 'Extraction date',
#     y = 'Percentage missing',
#     color = 'Run date',
#   ) +
#   theme(legend.key.size = unit(4, "mm")) +
#   guides(color = guide_colorbar(barheight = unit(10, "mm")))
# file <- 'tmp/astral/fig/sample-pct_na.pdf'
# ggsave(file, ax, width = 3, height = 1.8)
# 
# ax <- ggplot(sample_stats) +
#   geom_point(
#     aes(x = pct_zero, y = avg, col = Extraction.Date),
#     cex = 1,
#   ) +
#   labs(
#     x = 'Percentage missing',
#     y = 'Average expression',
#     color = 'Extraction date',
#   ) +
#   theme(legend.key.size = unit(4, "mm"))
# file <- 'tmp/astral/fig/avg-pct_na.pdf'
# ggsave(file, ax, width = 3.2, height = 1.8)
# 
# # Plot: MNAR relationship
# ext_date <- metadata_lyriks[colnames(lyriks1), 'Extraction.Date']
# lyriks_batches <- split_cols(lyriks1, ext_date)
# names(lyriks_batches)
# 
# compute_stats <- function(x) {
#   data.frame(
#     pct_na = rowSums(is.na(x)) / ncol(x),
#     avg = rowMeans(x, na.rm = TRUE) 
#   )
# }
# list_stats <- lapply(lyriks_batches, compute_stats)
# list_stats1 <- lapply(names(list_stats), function(name) {
#   list_stats[[name]]['date'] <- name
#   return(list_stats[[name]])
# })
# mnar_stats <- do.call(rbind, list_stats1)
# mnar_stats[is.na(mnar_stats)] <- 0
# head(mnar_stats)
# 
# ax <- ggplot(mnar_stats) +
#   geom_point(aes(x = avg, y = pct_na, color = date), cex = 1, shape = 1) +
#   labs(
#     x = 'Average expression',
#     y = 'Percentage missing',
#     color = 'Extraction date',
#   ) +
#   theme(legend.key.size = unit(4, "mm"))
#   # guides(color = guide_colorbar(barheight = unit(15, "mm")))
# file <- 'tmp/astral/fig/batches-avg_pct_na.pdf'
# ggsave(file, ax, width = 3.2, height = 1.8)

##### Imputation #####

# Remove remit (not early or late), relapse and QC patients
identical(rownames(metadata_lyriks), colnames(lyriks))
idx <- subset(
  metadata_lyriks,
  !(label %in% c('remit', 'relapse', 'QC'))
) %>%
  rownames()
print(idx)
slyriks <- lyriks[, idx]
metadata_slyriks <- metadata_lyriks[idx, ]

file <- 'data/astral/processed/metadata-lyriks.csv'
write.csv(metadata_slyriks, file)

# Impute in batch aware fashion
# Only features with < 50% missing values in all batches
# Assign QC samples to 4/9/2024 arbitrarily 
ext_date <- metadata_slyriks[colnames(slyriks), 'Extraction.Date']
lyriks_batches <- split_cols(slyriks, ext_date)

# Features have to be < 50% missing in all batches
PCT_MISSING <- 0.5
pct_na <- sapply(lyriks_batches, function(x) rowSums(is.na(x)) / ncol(x))
to_impute <- apply(pct_na < PCT_MISSING, 1, all)
prot_607 <- names(to_impute)[to_impute]
flyriks <- slyriks[prot_607, ]
flyriks_batches <- split_cols(flyriks, ext_date)
print(length(prot_607))
rowSums(is.na(flyriks))
dim(slyriks)

# flyriks_ft_pct_na <- rowSums(is.na(flyriks)) / ncol(flyriks)
# ft_na30 <- names(flyriks_ft_pct_na)[flyriks_ft_pct_na > 0.3]
# 
# flyriks1 <- flyriks
# flyriks1[is.na(flyriks1)] <- 0
# feat_nona <- rownames(flyriks)[rowSums(is.na(flyriks)) == 0]
# ax <- ggplot_pca(
#   flyriks1, metadata_lyriks,
#   col = 'age', shape = 'Extraction.Date'
# )
# file <- 'tmp/astral/fig/pca-flyriks-age.pdf'
# ggsave(file, ax, width = 10, height = 6)

# MVI: MinProb
set.seed(0)
iflyriks_batches <- flyriks_batches %>%
  lapply(impute.MinProb, q = 0.1) %>%
  lapply(data.matrix)
iflyriks <- do.call(cbind, iflyriks_batches)

flyriks[1:20, idx[1:5]]
iflyriks[1:20, 1:5]

# MVI: kNN
# TODO: Document exact details (kNN) 
knn_batches <- flyriks_batches %>%
  lapply(function(x) impute.knn(data.matrix(x), k = 5)) %>%
  lapply(function(x) x$data)
knn_lyriks <- do.call(cbind, knn_batches)
sum(is.na(knn_lyriks))
prod(dim(knn_lyriks))

ax <- ggplot_pca(
  knn_lyriks, metadata_slyriks,
  col = 'final_label', shape = 'Extraction.Date'
)
file <- 'tmp/astral/fig/pca-knn.pdf'
# ggsave(file, ax, width = 10, height = 6)

### MVI: Evaluation ###

# Features: Fully present
lyriks_feature_pct_zero <- rowSums(is.na(slyriks)) / ncol(slyriks)
lyriks_sample_pct_zero <- colSums(is.na(slyriks)) / nrow(slyriks)
# Evaluating imputation methods
f265lyriks <- slyriks[lyriks_feature_pct_zero == 0, ]
f265lyriks_batches <- split_cols(f265lyriks, ext_date)

# # Simulate MAR (batch)
# x <- seq(8, 16, 0.01)
# plot(x, 1 - sigmoid(x, 3, -12))
# x <- seq(0, 1, 0.01)
# pdf <- dbeta(x, shape1 = 0.8, shape2 = 9)
# y <- rbeta(200, shape1 = 0.8, shape2 = 3)
# plot(x, pdf)
#  max(y)

# Mixture of two types of missing values 
# 1. Sigmoid 2. Beta distribution
dropout <- function(x, r, s, shape1, shape2) {
  feat_means <- rowMeans(x)
  p_na <- 1 - sigmoid(feat_means, r, s)
  n_samples <- ncol(x)
  for (i in seq_len(nrow(x))) {
    idx <- rbinom(n_samples, 1, prob = p_na[i])
    x[i, as.logical(idx)] <- NA
    p <- rbeta(1, shape1 = shape1, shape2 = shape2)
    idx <- rbinom(n_samples, 1, prob = p)
    x[i, as.logical(idx)] <- NA
  }
  x
}

# Drop out 
set.seed(0)
mlyriks_batches <- lapply(
  f265lyriks_batches, dropout,
  r = 3, s = -11, shape1 = 0.8, shape2 = 4
)
pct_zero_batches <- mlyriks_batches %>%
  sapply(function(x) rowSums(is.na(x)) / ncol(x))
idx <- rownames(f265lyriks)[rowSums(pct_zero_batches > 0.5) == 0]
print(length(idx))
f238lyriks_batches <- f265lyriks_batches %>%
  lapply(function(x) x[idx, ])
mlyriks_batches1 <- mlyriks_batches %>%
  lapply(function(x) x[idx, ])
lapply(mlyriks_batches1, dim)

# pct_zero_batches1 <- mlyriks_batches1 %>%
#   sapply(function(x) rowSums(is.na(x)) / ncol(x))
# str(mlyriks_batches1)

# prod(dim(pct_zero_batches))

# Evaluate MVI methods
knn_batches3 <- mlyriks_batches1 %>%
  lapply(function(x) impute.knn(data.matrix(x), k = 3)) %>%
  lapply(function(x) x$data)
rmse_knn3 <- mapply(
  function(x, y) mean((data.matrix(x) - y) ^ 2) ^ 0.5,
  f238lyriks_batches, knn_batches
)

knn_batches5 <- mlyriks_batches1 %>%
  lapply(function(x) impute.knn(data.matrix(x), k = 5)) %>%
  lapply(function(x) x$data)
rmse_knn5 <- mapply(
  function(x, y) mean((data.matrix(x) - y) ^ 2) ^ 0.5,
  f238lyriks_batches, knn_batches
)

knn_batches10 <- mlyriks_batches1 %>%
  lapply(function(x) impute.knn(data.matrix(x), k = 10)) %>%
  lapply(function(x) x$data)
rmse_knn10 <- mapply(
  function(x, y) mean((data.matrix(x) - y) ^ 2) ^ 0.5,
  f238lyriks_batches, knn_batches
)

# minprob_batches <- mlyriks_batches1 %>%
#   lapply(impute.MinProb, q = 0.1) %>%
#   lapply(data.matrix)
# rmse_minprob10 <- mapply(
#   function(x, y) mean((data.matrix(x) - y) ^ 2) ^ 0.5,
#   f238lyriks_batches, minprob_batches
# )
# 
# minprob_batches <- mlyriks_batches1 %>%
#   lapply(impute.MinProb, q = 0.01) %>%
#   lapply(data.matrix)
# rmse_minprob01 <- mapply(
#   function(x, y) mean((data.matrix(x) - y) ^ 2) ^ 0.5,
#   f238lyriks_batches, minprob_batches
# )

##### Batch correction #####

# metadata_slyriks$period <- as.numeric(as.character(metadata_slyriks$period))
table(metadata_slyriks$Extraction.Date, metadata_slyriks$label)

# TODO: Decide on class factors to include: (period, class, sex)
# TODO: How to handle the difference in period better (low priority)

# ComBat - Modelling class covariate
knn_lyriks <- knn_lyriks[, rownames(metadata_slyriks)]
all(colnames(knn_lyriks) %in% rownames(metadata_slyriks))
mod <- model.matrix(~label, data = metadata_slyriks)
combat_lyriks <- ComBat(
  knn_lyriks,
  batch = metadata_slyriks$Extraction.Date, mod = mod,
  par.prior = TRUE, ref.batch = '5/9/24'
)
knn_lyriks[1:20, 1:5]
combat_lyriks[1:20, 1:5]

# Class-specific ComBat
label <- metadata_slyriks[colnames(knn_lyriks), 'label']
knn_lyriks_labels <- split_cols(knn_lyriks, label)
str(knn_lyriks_labels[-2])

# Do not correct cvt as they all come from the same batch
combat_lyriks_labels <- lapply(knn_lyriks_labels[-2], function(x) {
  ComBat(
    x,
    batch = metadata_slyriks[colnames(x), 'Extraction.Date'],
    ref.batch = '5/9/24' 
  )
})
combat_lyriks_labels1 <- c(combat_lyriks_labels, knn_lyriks_labels[2])
cscombat_lyriks <- do.call(cbind, combat_lyriks_labels1)

colnames(metadata_slyriks)
ax <- ggplot_pca(
  cscombat_lyriks, metadata_slyriks,
  color = 'label', shape = 'period'
)
file <- 'tmp/astral/fig/pca-cscombat-class.pdf'
ggsave(file, ax, width = 5, height = 4)

# # DE features between batches 
# metadata_slyriks1 <- metadata_slyriks[colnames(knn_lyriks), ]
# ctrl_5924 <- rownames(subset(
#   metadata_slyriks1,
#   final_label == 'ctrl' & Extraction.Date == '5/9/24'
# ))
# ctrl_4924 <- rownames(subset(
#   metadata_slyriks1,
#   final_label == 'ctrl' & Extraction.Date == '4/9/24'
# ))
# pvals <- calc_univariate(
#   t.test,
#   knn_lyriks[, ctrl_4924],
#   knn_lyriks[, ctrl_5924]
# )
# feat_top_p <- names(head(sort(pvals), 30))

ax <- ggplot_pca(
  combat_lyriks, metadata_slyriks,
  color = 'Run.DateTime', shape = 'Extraction.Date'
)
file <- 'tmp/astral/fig/pca-combat_knn-batch.pdf'
ggsave(file, ax, width = 10, height = 6)

set.seed(0)
feats <- sample(rownames(combat_lyriks), 40)

##### Save data ##### 

dim(combat_lyriks)
file <- 'data/astral/processed/combat_knn5_lyriks.csv'
write.csv(combat_lyriks, file)

##### Plot: Feature #####

# Log-normalised data
# i <- 100
# idx <- rownames(lyriks1)[lyriks_feature_pct_zero == 0]
# prot <- idx[i]
prot <- ft_na30[8]
print(prot) # P05362

prot <- 'P04632'
prot <- 'P05362'
prot <- 'P08514'
print(prot)

x <- flyriks[prot, ] %>%
  unlist() %>%
  replace_na(0)
metadata_sorted <- metadata_lyriks[names(x), ]
data <- cbind(expr = x, metadata_sorted)
ax <- ggplot(data) +
  facet_wrap(~Class, nrow = 1, scales = 'free_x') +
  geom_point(
    aes(y = expr, x = period, col = Run.DateTime, shape = Extraction.Date),
    position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8),
    cex = 1
  ) +
  labs(
    x = 'Period',
    y = prot,
    color = 'Run date',
    shape = 'Extraction date',
  ) +
  theme(legend.key.size = unit(4, "mm")) +
  guides(color = guide_colorbar(barheight = unit(15, "mm")))
file <- sprintf('tmp/astral/fig/feature-%s.pdf', prot)
ggsave(file, ax, width = 5, height = 1.8)

# Batch corrected
x <- unlist(combat_lyriks[prot, ])
metadata_sorted <- metadata_lyriks[names(x), ]
data <- cbind(expr = x, metadata_sorted)
ax <- ggplot(data) +
  facet_wrap(~Class, nrow = 1, scales = 'free_x') +
  geom_point(
    aes(y = expr, x = period, col = Run.DateTime, shape = Extraction.Date),
    position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8),
    cex = 1
  ) +
  labs(
    x = 'Period',
    y = prot,
    color = 'Run date',
    shape = 'Extraction date',
  ) +
  theme(legend.key.size = unit(4, "mm")) +
  guides(color = guide_colorbar(barheight = unit(15, "mm")))
file <- sprintf('tmp/astral/fig/feature-knn-combat-%s.pdf', prot)
ggsave(file, ax, width = 5, height = 1.8)

set.seed(0)
feats <- sample(rownames(knn_lyriks), 60)
feats <- sample(feat_nona, 20)
print(head(feats))

for (i in feat_top_p) {
  x <- unlist(knn_lyriks[i, ])
  metadata_sorted <- metadata_slyriks[names(x), ]
  data <- cbind(expr = x, metadata_sorted)
  ax <- ggplot(data) +
    facet_wrap(~label, nrow = 1, scales = 'free_x') +
    geom_point(
      aes(y = expr, x = period, col = Run.DateTime, shape = Extraction.Date),
      position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8)
    ) +
    # scale_colour_viridis_d(option = 'plasma') +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  file <- sprintf('tmp/astral/fig/features/lyriks_knn20_top_p-%s.pdf', i)
  ggsave(file, ax, width = 12, height = 3.5)
  print(file)
}

for (i in feat_top_p) {
  x <- unlist(combat_lyriks[i, ])
  metadata_sorted <- metadata_slyriks[names(x), ]
  data <- cbind(expr = x, metadata_sorted)
  ax <- ggplot(data) +
    facet_wrap(~label, nrow = 1, scales = 'free_x') +
    geom_point(
      aes(y = expr, x = period, col = Run.DateTime, shape = Extraction.Date),
      position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8)
    ) +
    # scale_colour_viridis_d(option = 'plasma') +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  file <- sprintf('tmp/astral/fig/features/lyriks_combat_top_p-%s.pdf', i)
  ggsave(file, ax, width = 12, height = 3.5)
  print(file)
}

for (i in feat_top_p) {
  x <- unlist(cscombat_lyriks[i, ])
  metadata_sorted <- metadata_slyriks[names(x), ]
  data <- cbind(expr = x, metadata_sorted)
  ax <- ggplot(data) +
    facet_wrap(~label, nrow = 1, scales = 'free_x') +
    geom_point(
      aes(y = expr, x = period, col = Run.DateTime, shape = Extraction.Date),
      position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8)
    ) +
    # scale_colour_viridis_d(option = 'plasma') +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  file <- sprintf('tmp/astral/fig/features/lyriks_cscombat_top_p-%s.pdf', i)
  ggsave(file, ax, width = 12, height = 3.5)
  print(file)
}
lnmatrix[1:10, 1:2]
colnames(lnmatrix)
head(metadata)

##### PCA #####

lnmatrix1 <- lnmatrix[, seq(6, ncol(lnmatrix))]
feature_pct_zero <- rowSums(is.na(lnmatrix1)) / ncol(lnmatrix1)
sum(feature_pct_zero == 0)

ax <- ggplot_pca(
  lnmatrix1[feature_pct_zero == 0, ], metadata,
  color = 'Study', shape = 'Class', cex = 1
) + 
  scale_shape_manual(values = seq(0, 9)) +
  theme(legend.key.size = unit(3, "mm"))
ggsave('tmp/astral/fig/pca-all-223-class.pdf', ax, width = 4, height = 2.2)

ax <- ggplot_pca(
  lnmatrix1[feature_pct_zero == 0, ], metadata,
  color = 'Run.DateTime', shape = 'Extraction.Date', cex = 1
) +
  labs(
    color = 'Run date',
    shape = 'Extraction date',
  ) +
  theme(legend.key.size = unit(4, "mm"))
ggsave('tmp/astral/fig/pca-all-223-runtime.pdf', ax, width = 3.5, height = 2.2)

### Lyriks ###

lyriks_feature_pct_zero <- rowSums(is.na(lyriks1)) / ncol(lyriks1)
sum(lyriks_feature_pct_zero == 0)
ax <- ggplot_pca(
  lyriks1[lyriks_feature_pct_zero == 0, ], metadata,
  color = 'Class', shape = 'Extraction.Date', cex = 1
) +
  labs(
    shape = 'Extraction date',
  ) +
  theme(
    legend.key.size = unit(4, "mm"),
    legend.spacing.y = unit(1, "mm"),
  )
ggsave('tmp/astral/fig/pca-lyriks-batch.pdf', ax, width = 3, height = 1.8)

ax <- ggplot_pca(
  lyriks_unnorm, metadata,
  color = 'Run.DateTime', shape = 'Extraction.Date'
)
ggsave('tmp/astral/fig/pca-lyriks_unnorm-batch.pdf', ax, width = 10, height = 6)

ax <- ggplot_pca(
  lyriks, metadata,
  color = 'class', shape = 'Extraction.Date'
) # +
  scale_shape_manual(values = seq(0, 9))
ggsave('tmp/astral/fig/pca-lyriks-class.pdf', ax, width = 10, height = 6)

lyriks_sids <- colnames(lyriks)[1:(ncol(lyriks) - 5)]

lyriks_sids_b1 <- lyriks_sids[
  metadata[lyriks_sids, 'Extraction.Date'] == '5/9/24']
lyriks_sids_b2 <- lyriks_sids[
  metadata[lyriks_sids, 'Extraction.Date'] != '5/9/24']

table(
  metadata10[lyriks_sids_b1, 'period'],
  metadata10[lyriks_sids_b1, 'final_label']
)

table(
  metadata10[lyriks_sids_b2, 'period'],
  metadata10[lyriks_sids_b2, 'final_label']
)


# Important features 
lyriks_ft_pct_zero <- rowSums(lyriks == 0) / ncol(lyriks)
lyriks_sample_pct_zero <- colSums(lyriks == 0) / nrow(lyriks)

lyriks_ft_nonsparse <- rownames(lyriks)[lyriks_ft_pct_zero < 0.2]
lyriks_ft_nonzero <- rownames(lyriks)[lyriks_ft_pct_zero == 0]

# All features
ax <- ggplot_pca(
  lyriks[lyriks_ft_nonsparse, ], metadata,
  color = 'class', shape = 'Extraction.Date'
)
ggsave('tmp/astral/fig/pca-lyriks-nonsparse-class.pdf', ax, width = 10, height = 6)

ax <- ggplot_pca(
  lyriks, lyriks_gundam,
  color = 'Run.DateTime', shape = 'Extraction.Date'
)
ggsave('tmp/astral/fig/pca-lyriks-runtime.pdf', ax, width = 8, height = 5)

# Features with no zeros
length(lyriks_ft_nonsparse)
metadata_lyriks <- metadata_all[colnames(lyriks), ]
metadata_lyriks1 <- cbind(metadata_lyriks, pct_zero = lyriks_sample_pct_zero)
ax <- ggplot_pca(
  lyriks[lyriks_ft_nonsparse, ], metadata_lyriks1,
  color = 'pct_zero', shape = 'batch'
)
ggsave('tmp/astral/fig/pca-lyriks_fltr519_qc.pdf', ax, width = 8, height = 5)

# Only controls
sid_lyriks_ctrl <- metadata_all %>%
  subset(study == 'lyriks' & class == 'Healthy control') %>%
  rownames()
ax <- ggplot_pca(
  lyriks[lyriks_ft_nonsparse, sid_lyriks_ctrl], metadata_all,
  color = 'class', shape = 'batch'
)
ggsave('tmp/astral/fig/pca-lyriks_fltr519_ctrl.pdf', ax, width = 8, height = 5)

ax <- ggplot_pca(lyriks, metadata10, col = 'final_label', shape = 'caarms_status')
ggsave('tmp/fig/astral/pca-lyriks-class1.pdf', ax, width = 8, height = 5)

ax <- ggplot_pca(lyriks, metadata58, color = 'drug_type', shape = 'final_label')
ggsave('tmp/fig/astral/pca-lyriks-class2.pdf', ax, width = 8, height = 5)

# Clinical features
ax <- ggplot_pca(lyriks, metadata10, color = 'age', shape = 'gender')
ggsave('tmp/fig/astral/pca-lyriks-clinical1.pdf', ax, width = 8, height = 5)
ax <- ggplot_pca(lyriks, metadata74, color = 'eth', shape = 'smoke_stat')
ggsave('tmp/fig/astral/pca-lyriks-clinical2.pdf', ax, width = 8, height = 5)

clinical_features <- c('psle_score', 'psy_ill')
metadata_cf <- metadata521[colnames(lyriks), clinical_features]
metadata_cf <- replace(metadata_cf, metadata_cf == -9999, NA)
metadata_cf$psy_ill <- replace(metadata_cf$psy_ill, metadata_cf$psy_ill == 0, NA)
metadata_cf$psy_ill <- ifelse(metadata_cf$psy_ill == 1, 'yes', 'no')

ax <- ggplot_pca(lyriks, metadata_cf, color = 'psle_score', shape = 'psy_ill')
ggsave('tmp/fig/astral/pca-lyriks-clinical3.pdf', ax, width = 8, height = 5)

panel_names <- c(
  'caarms_Wtot', 'panss_tot', 'cdss_score',
  'bai_score', 'hisoc_avg', 'gaf_score', 'ctq_tot', 
  'pbi_mum_ctrl', 'pbi_mum_care', 'pbi_dad_ctrl', 'pbi_dad_care'
)
panels <- metadata521[colnames(lyriks), panel_names]
panels <- replace(panels, panels == '#NULL!', NA)
panels <- replace(panels, panels == -9999, NA)
panels[] <- lapply(panels, as.numeric)
rownames(panels)[402] <- missing_sids1
panels$final_label <- metadata10[rownames(panels), 'final_label']

for (panel in panel_names) {
  ax <- ggplot_pca(lyriks, panels, col = panel, shape = 'final_label')
  file <- sprintf('tmp/fig/astral/pca-lyriks-%s.pdf', panel)
  ggsave(file, ax, width = 8, height = 5)
  print(file)
}

# cognitive_features <- c(
#   'bacs_vm', 'bacs_ds', 'bacs_tmt', 'bacs_vf_ani', 'bacs_vf_frt',
#   'bacs_vf_veg', 'bacs_vf', 'bacs_sc', 'bacs_tol', 'gaf_score', 'sofas_drop_per', 
#   'babble_long_phrase', 'babble_tot_words', 'bdit_tot', 'cpt_avg'
# )

# Assign batches
pca_obj <- prcomp(t(data))
all_pca <- pca_obj$x[, 1:2]

pdf('tmp/fig/astral/pca-all-batch.pdf')
plot(all_pca, col = batch)
intercept <- 0
slope <- -1.25

abline(a = intercept, b = slope, col = 'red') 
dev.off()

x <- all_pca[, 1]
y <- all_pca[, 2]
batch <- ifelse(y < intercept + slope * x, '1', '2')
metadata_all$batch <- batch

file <- 'data/astral/metadata/metadata-all.csv'
write.csv(metadata_all, file)

ax <- ggplot_pca(
  lyriks, metadata_all,
  color = 'class', shape = 'batch'
)
file <- 'tmp/astral/fig/pca-lyriks-cluster.pdf'
ggsave(file, ax, width = 8, height = 5)

# DEA: Cluster 1 (288) v.s. 2 (114)
# DEA only on genes that are present > 80%
# TODO: T.test, wilcoxon rank sum,

length(lyriks_ft_nonsparse)

cvt_0 <- metadata10[lyriks_sids_b1, ] %>%
  subset(final_label == 'cvt' & period == 0) %>%
  rownames()
mnt_24 <- metadata10[lyriks_sids_b1, ] %>%
  subset(final_label == 'mnt' & period == 24) %>%
  rownames()

pvals <- sort(calc_univariate(
  t.test,
  lyriks[lyriks_ft_nonsparse, cvt_0],
  lyriks[lyriks_ft_nonsparse, mnt_24]
))

n <- length(lyriks_ft_nonsparse)
print(n)
sig_p <- pvals[pvals < 0.05]
length(sig_p)
qvals <- p.adjust(pvals, method = 'BH')
res_ttest <- data.frame(pvals, qvals, rank = seq_len(length(pvals)))
# q <- res_ttest$pvals / res_ttest$rank * n
sum(pvals < 0.05)
sum(qvals < 0.05)
sig_q <- qvals[qvals < 0.05]
annot_sigp <- cbind(
  uniprot_annot[names(sig_p), ],
  p = signif(sig_p, digits = 3)
)
write.csv(annot_sigp, 'tmp/astral/annot_sigp69.csv')
length(sig_q)

lyriks_ft_nonzero <- rownames(lyriks)[lyriks_ft_pct_zero == 0]
annot_nonzero <- uniprot_annot[lyriks_ft_nonzero, ]
write.csv(annot_nonzero, 'tmp/astral/annot_nonzero265.csv')
length(lyriks_ft_nonzero)

annot_sigq <- read.csv('data/astral/misc/annot_sigq365.csv', row.names = 1)
annot_nonzero265 <- read.csv('data/astral/misc/annot_nonzero265.csv', row.names = 1)

# TODO: Pathway / complex analysis?
library(KEGGREST)

# Convert UniProt IDs to KEGG IDs
uniprot_ids <- rownames(annot_sigp)
kegg_ids <- keggConv('genes', paste0('uniprot:', uniprot_ids))

# keggLink may have hard-coded limit of 100 IDs
# Split IDs into groups of 100 to avoid limit
npergroup <- 100
ngrps <- ceiling(length(kegg_ids) / npergroup)
groups <- gl(ngrps, npergroup, length(kegg_ids))
list_ids <- split(kegg_ids, groups)
kegg_pathways <- character()
for (grp in list_ids) {
  print(length(grp))
  # some KEGG gene IDs do not have pathways while others have multiple
  kegg_pathways <- c(kegg_pathways, keggLink('pathway', grp))
}
# Option 2: Under 100 IDs
kegg_pathways <- c(kegg_pathways, keggLink('pathway', kegg_ids))
kegg_ids %in% names(kegg_pathways)
pathway_freq <- table(kegg_pathways) %>%
  sort(decreasing = TRUE) %>%
  data.frame() %>%
  head(20)

# Alternative method to get pathway names
# pathway_names <- lapply(kegg_pathways, function(pw) keggGet(pw)[[1]]$NAME)
pathway_annot <- keggList('pathway')
names(pathway_annot) <- names(pathway_annot) %>%
  substring(4) %>%
  paste0('path:hsa', .)
idx <- match(pathway_freq$kegg_pathways, names(pathway_annot))
kegg_names <- pathway_annot[idx]
names_freq <- cbind(name = kegg_names, pathway_freq)

ax <- ggplot(names_freq) +
  geom_col(aes(
    x = Freq,
    y = reorder(name, Freq, sum)
  )) +
  labs(y = "KEGG")
  # scale_x_continuous(breaks = seq(0, max(kegg$Freq), by = 1))
file <- "tmp/astral/fig/annot_sigp-kegg.pdf"
ggsave(file, ax, width = 9, height = 4)

# TODO: Obtain Mongan proteins
file <- 'data/astral/misc/mongan-etable5.csv'
mongan <- read.csv(file)
colnames(mongan) <- c('uniprot', 'name', 'F', 'p', 'q')
rownames(mongan) <- mongan$uniprot

mongan56 <- mongan[mongan$p < 0.05, ]
mongan35 <- mongan[mongan$q < 0.05, ]
mongan10_uniprot <- c(
  A2M = 'P01023', IGHM = 'P01871', C4BPA = 'P04003', PROS = 'P07225',
  FBLN1 = 'P23142', TTHY = 'P02766', PGRP2 = 'Q96PD5',
  VTDB = 'P02774', CLUS = 'P10909', C6 = 'P13671'
)
mongan10 <- mongan[mongan10_uniprot, ]

file <- 'data/astral/misc/byrne-99.csv'
# 99 proteins are present in >70% of samples
byrne99 <- read.csv(file)
byrne10 <- head(byrne99, 10)

library(ggVennDiagram)

sets <- list(lyriks_p = rownames(annot_sigp),  mongan = mongan35$uniprot) 
ax <- ggVennDiagram(sets) + 
  scale_fill_gradient(low = "white", high = "blue") +
  scale_x_continuous(expand = expansion(mult = .2))
file <- 'tmp/astral/fig/venn-lyriks_sigp_mongan.pdf'
ggsave(file, ax, width = 8, height = 5)

sets <- list(lyriks_q = annot_sigq$Gene,  byrne = byrne99$gene_symbol)
ax <- ggVennDiagram(sets) + 
  scale_fill_gradient(low = "white", high = "blue") +
  scale_x_continuous(expand = expansion(mult = .2))
file <- 'tmp/astral/fig/venn-lyriks_sigq_byrne.pdf'
ggsave(file, ax, width = 8, height = 5)

# TODO: DEA on cvt v.s. non-cvt
# TODO: DEA on just control samples (b1 v.s. b2)?
# TODO: DEA on paired convert samples

### Differential expression analysis ###
# TODO: Restrict analysis to single batch
# (convert v.s. non-convert v.s. control
table(metadata10$period, metadata10$final_label, metadata10$batch)

sid_lyriks_b2 <- rownames(subset(metadata_all, study == 'lyriks' & batch == 2))
lyriks_b2 <- lyriks[, sid_lyriks_b2]
metadata_lyriks_b2 <- metadata10[sid_lyriks_b2, ]

sid_cvt_24 <- metadata_lyriks_b2 %>%
  subset(final_label == 'cvt' & period == 24) %>%
  rownames()
sid_noncvt_24 <- metadata_lyriks_b2 %>%
  subset(final_label %in% c('mnt', 'rmt') & period == 24) %>%
  rownames()

pvals <- calc_univariate(
  t.test, lyriks[, sid_cvt_24], lyriks[, sid_noncvt_24]
)
pvals_fltr <- pvals[!is.na(pvals)]

sum(is.na(pvals))
length(pvals)

pdf('tmp/fig/astral/pval-cvt24_noncvt24.pdf')
hist(pvals_fltr)
dev.off()

# TODO: Pathway analysis of DEPs. Mongan: Complement and coagulation

# TODO: Mongan: non-cvt v.s. cvt (M24) both jjuu

# TODO: Prediction models on LYRIKS (validation?)
# Prediction: Clinical and proteomics, ablation tests
# Prediction: Top 10 DEPs
# Batch effects!


# TODO: Look at CSA and ABGN. Validate against literature? Look at pathway?

# TODO: All possible metadata CSA and ABGN
