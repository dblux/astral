library(dplyr)
library(tidyr)
library(impute)
library(sva)
library(ggplot2)
library(pheatmap)
library(RColorBrewer)
library(viridis)
source('R/utils.R')
source('R/impute.R')
source('R/plot.R')
theme_set(theme_bw(base_size = 7))


##### Load data #####

file <- 'data/astral/metadata/metadata-all_645_13.csv'
metadata <- read.csv(file, row.names=1, header=TRUE)

file <- 'data/astral/processed/metadata-lyriks407.csv'
metadata_lyriks <- read.csv(file, row.names=1, header=TRUE)
head(metadata_lyriks)

batch_assignment <- c(
  '28/8/24' = '1',
  '4/9/24' = '2',
  '5/9/24' = '3'
)
metadata$Batch <- batch_assignment[metadata$Extraction.Date]
metadata_lyriks$Batch <- batch_assignment[metadata_lyriks$Extraction.Date]

# Encode as time
metadata$Run.DateTime <- as.POSIXct(metadata$Run.DateTime, format = "%Y-%m-%d %H:%M:%S")
metadata_lyriks$Run.DateTime <- as.POSIXct(metadata_lyriks$Run.DateTime, format = "%Y-%m-%d %H:%M:%S")

# pg_matrix is un-normalised matrix with QC samples
file <- 'data/astral/raw/report.pg_matrix.tsv'
pg_matrix <- read.table(file, sep = '\t', header = T, row.names = 1)
colnames(pg_matrix) <- gsub('\\.', '-', colnames(pg_matrix))
colnames(pg_matrix)[1:5] <- substring(colnames(pg_matrix)[1:5], 10)

# Normalise matrix
# pg_matrix is scaled according to scaling factors for each sample
sample_sums <- colSums(pg_matrix, na.rm = T)
scaling_ratio <- sample_sums / max(sample_sums, na.rm = T)
norm_pgmatrix <- sweep(pg_matrix, 2, scaling_ratio, '/')
lnmatrix <- log2(norm_pgmatrix)
# Rename colnames of lnmatrix
idx <- match(colnames(lnmatrix), metadata$Polypeptide.Novogene.ID)
colnames(lnmatrix) <- rownames(metadata)[idx]

# Subset LYRIKS study
lyriks_sids <- sort(grep('^L', colnames(lnmatrix), value = T))
lyriks <- lnmatrix[, lyriks_sids]
dim(lyriks)


file <- "data/astral/processed/combat_knn5_lyriks-605_402.csv"
lyriks_final <- read.csv(file, row.names=1, header=TRUE)


##### EDA #####

# # Distribution of missingness
# pdf('tmp/fig/astral/features-pct-zero.pdf')
# hist(feature_pct_zero, breaks = 20)
# dev.off()
# 
sample_pct_zero <- colSums(data == 0) / nrow(data)
qc <- cbind(metadata_all, pct_zero = sample_pct_zero)

ax <- ggplot(qc) +
  geom_point(
    aes(x = study, y = pct_zero, color = class),
    position = position_jitterdodge(jitter.width = 0.1, dodge.width = 0.8)
  )
file <- 'tmp/fig/astral/samples-pct-zero.pdf'
ggsave(file, ax, width = 8, height = 5)

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

feature_avg <- apply(lyriks, 1, function(x) mean(x[x != 0]))
data <- data.frame(pct_zero = lyriks_feature_pct_zero, avg = feature_avg)
ax <- ggplot(data) +
  geom_point(aes(x = avg, y = pct_zero))
file <- 'tmp/astral/fig/feat-avg_pct_zero.pdf'
ggsave(file, ax, width = 8, height = 5)

# Plot: Missing values
lyriks_sample_pct_zero <- colSums(is.na(lyriks)) / nrow(lyriks)
sample_min <- apply(lyriks, 2, min, na.rm = T)
sample_avg <- colMeans(lyriks, na.rm = T) 
sample_sum <- colSums(lyriks, na.rm = T) 
identical(names(sample_min), colnames(lyriks))

sample_stats <- cbind(
  min = sample_min, 
  avg = sample_avg, 
  sum = sample_sum,
  pct_zero = lyriks_sample_pct_zero,
  metadata_lyriks[colnames(lyriks), ]
)

# Averages still differ for unnormalised data
ax <- ggplot(sample_stats) +
  geom_point(
    aes(x = Batch, y = avg, col = Run.DateTime, shape = class.x),
    position = position_jitterdodge(jitter.width = 0.1, dodge.width = 1)
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
file <- 'tmp/astral/fig/sample_unnorm-avg.pdf'
ggsave(file, ax, width = 8, height = 5)

ax <- ggplot(sample_stats) +
  geom_point(
    aes(x = Batch, y = pct_zero, col = Run.DateTime, shape = Class),
    position = position_jitterdodge(jitter.width = 0.11, dodge.width = 0.8),
    cex = 1,
  ) +
  labs(
    y = 'Percentage missingness',
    color = 'Run date',
  ) +
  theme(legend.key.size = unit(4, "mm")) +
  guides(color = guide_colorbar(barheight = unit(10, "mm")))
file <- 'tmp/astral/fig/suppl/sample-pct_na.pdf'
ggsave(file, ax, width = 3, height = 1.8)

ax <- ggplot(sample_stats) +
  geom_point(
    aes(x = pct_zero, y = avg, col = Batch),
    cex = 1,
  ) +
  labs(
    x = 'Percentage missingness',
    y = 'Average log intensity',
  ) +
  theme(legend.key.size = unit(4, "mm"))
file <- 'tmp/astral/fig/suppl/avg-pct_na.pdf'
ggsave(file, ax, width = 3.2, height = 1.8)

# Plot: MNAR relationship
batch <- metadata_lyriks[colnames(lyriks), 'Batch']
lyriks_batches <- split_cols(lyriks, batch)
names(lyriks_batches)

compute_stats <- function(x) {
  data.frame(
    pct_na = rowSums(is.na(x)) / ncol(x),
    avg = rowMeans(x, na.rm = TRUE) 
  )
}
list_stats <- lapply(lyriks_batches, compute_stats)
list_stats1 <- lapply(names(list_stats), function(name) {
  list_stats[[name]]['Batch'] <- name
  return(list_stats[[name]])
})
mnar_stats <- do.call(rbind, list_stats1)
mnar_stats[is.na(mnar_stats)] <- 0
head(mnar_stats)

ax <- ggplot(mnar_stats) +
  geom_point(aes(x = avg, y = pct_na, color = Batch), cex = 1, shape = 1) +
  labs(
    x = 'Average log intensity',
    y = 'Percentage missingness',
  ) +
  theme(legend.key.size = unit(4, "mm"))
  # guides(color = guide_colorbar(barheight = unit(15, "mm")))
file <- 'tmp/astral/fig/suppl/batches-avg_pct_na.pdf'
ggsave(file, ax, width = 3.2, height = 1.8)

##### Imputation #####

# Impute in batch aware fashion
# Only features with < 50% missing values in all batches
# Assign QC samples to 4/9/2024 arbitrarily 
extr_date <- metadata_lyriks[colnames(lyriks), 'Extraction.Date']
lyriks_batches <- split_cols(lyriks, extr_date)

# Features have to be < 50% missing in all batches
PCT_MISSING <- 0.5
pct_na <- sapply(lyriks_batches, function(x) rowSums(is.na(x)) / ncol(x))
to_impute <- apply(pct_na < PCT_MISSING, 1, all)
prots_nonsparse <- names(to_impute)[to_impute]
flyriks <- lyriks[prots_nonsparse, ]
flyriks_batches <- split_cols(flyriks, extr_date)

# MVI: kNN
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
lyriks_feature_pct_zero <- rowSums(is.na(lyriks)) / ncol(lyriks)
lyriks_full <- lyriks[lyriks_feature_pct_zero == 0, ]
print(dim(lyriks_full))
lyriks_full_batches <- split_cols(lyriks_full, extr_date)

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

RMSE <- function(x, y) mean((data.matrix(x) - y) ^ 2) ^ 0.5

# Drop out (original seed = 2)
set.seed(2)
n_sim <- 1000
res_knn <- list()
res_minprob <- list()
for (i in seq_len(n_sim)) {
  print("============")
  print(paste("Simulation", i))
  print("============")
  mlyriks_batches <- lapply(
    lyriks_full_batches, dropout,
    r = 3, s = -11, shape1 = 0.8, shape2 = 4
  )
  # Removing proteins with >50% missing in any batch
  pct_zero_batches <- mlyriks_batches %>%
    sapply(function(x) rowSums(is.na(x)) / ncol(x))
  prots_gt50_all <- rownames(lyriks_full)[rowSums(pct_zero_batches > 0.5) == 0]
  print(length(prots_gt50_all))
  fmlyriks_batches <- mlyriks_batches %>%
    lapply(function(x) x[prots_gt50_all, ])
  # True values
  lyriks_full_fltr_batches <- lyriks_full_batches %>%
    lapply(function(x) x[prots_gt50_all, ])
  lyriks_full_fltr <- do.call(cbind, lyriks_full_fltr_batches)
  colnames(lyriks_full_fltr) <- sub(".*\\.", "", colnames(lyriks_full_fltr))
  # Evaluate multiple k values
  for (k in seq(3, 8)) {
    rmse <- fmlyriks_batches  %>%
      lapply(function(x) impute.knn(data.matrix(x), k = k)) %>%
      lapply(function(x) x$data) %>%
      do.call(cbind, .) %>%
      RMSE(lyriks_full_fltr, .)
    idx <- paste0("knn_", k)
    res_knn[[idx]] <- c(res_knn[[idx]], rmse)
  }
  # MVI: MinProb
  for (q in seq(0.1, 0.9, 0.1)) {
    lyriks_minprob <- fmlyriks_batches %>%
      lapply(impute.MinProb, q = q) %>%
      lapply(data.matrix) %>%
      do.call(cbind, .)
    rmse <- RMSE(lyriks_full_fltr, lyriks_minprob)
    idx <- paste0("minprob_", q)
    res_minprob[[idx]] <- c(res_minprob[[idx]], rmse)
  }
}

# save as RDS
saveRDS(res_knn, 'tmp/astral/lyriks402/res_knn.rds')
saveRDS(res_minprob, 'tmp/astral/lyriks402/res_minprob.rds')

# Calculate mean SD of log expression values across all full proteins
mus <- rowMeans(lyriks_full)
sigmas <- sqrt(rowVars(lyriks_full))
print(median(mus)) # 16.4 
print(median(sigmas)) # 0.498

intensity_stats <- data.frame(mean = mus, SD = sigmas)

# Mean-SD relationship
ax <- ggplot(intensity_stats, aes(x = mean, y = SD)) +
  geom_point(alpha = 0.5, size = 1) +
  geom_vline(xintercept = median(mus), linetype = "dashed", color = "blue") +
  geom_hline(yintercept = median(sigmas), linetype = "dashed", color = "red") +
  labs(title = "Log protein intensities", x = "Mean", y = "SD") +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
  )
file <- 'tmp/astral/lyriks402/fig/scatter-sd_mean.pdf'
ggsave(file, ax, width = 2.3, height = 2)

# # Violin plot of SD
# ax <- ggplot(intensity_stats, aes(x = 1, y = SD)) +
#   geom_violin(fill = "lightblue", alpha = 0.5) +
#   geom_jitter(alpha = 0.5, size = 1) +
#   geom_hline(yintercept = median(sigmas), linetype = "dashed", color = "red") +
#   labs(x = NULL, y = "SD") +
#   theme(
#     panel.grid.major = element_blank(),
#     panel.grid.minor = element_blank(),
#     axis.text.x = element_blank(),
#     axis.ticks.x = element_blank()
#   )
# file <- 'tmp/astral/lyriks402/fig/violin-sigma.pdf'
# ggsave(file, ax, width = 1.2, height = 2)

knn_rmse <- data.frame(
  k = seq(3, 8),
  mu = sapply(res_knn, mean),
  sigma = sapply(res_knn, sd)
)
minprob_rmse <- data.frame(
  q = seq(0.1, 0.9, 0.1),
  mu = sapply(res_minprob, mean),
  sigma = sapply(res_minprob, sd)
)

write.csv(knn_rmse, 'tmp/astral/lyriks402/knn_rmse.csv')
write.csv(minprob_rmse, 'tmp/astral/lyriks402/minprob_rmse.csv')

ax <- ggplot(knn_rmse, aes(x = k, y = mu)) +
  geom_point(cex = 1) +
  geom_line() +
  geom_errorbar(aes(
    ymin = mu - sigma,
    ymax = mu + sigma
  ), width = 0.1) +
  labs(title = "k-nearest neighbours", x = "k", y = "RMSE") +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )
file <- 'tmp/astral/lyriks402/fig/knn-rmse.pdf'
ggsave(file, ax, width = 2.3, height = 2)

ax <- ggplot(minprob_rmse, aes(x = q, y = mu)) +
  geom_point(cex = 1) +
  geom_line() +
  geom_errorbar(aes(
    ymin = mu - sigma,
    ymax = mu + sigma
  ), width = 0.02) +
  labs(title = "MinProb", x = "q", y = "RMSE") +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )
file <- 'tmp/astral/lyriks402/fig/minprob-rmse.pdf'
ggsave(file, ax, width = 2.3, height = 2)


##### Plot: PCA #####
ax <- ggplot_pca(
  lyriks_final, metadata_lyriks,
  color = 'Class', shape = 'Extraction.Date',
  cex = 2, alpha = 0.7
) +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )
ggsave('tmp/astral/lyriks402/fig/pca-lyriks_final.pdf', ax, width = 4, height = 2.5)
