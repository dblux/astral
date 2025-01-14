#!/usr/bin/env Rscript

#
# Script to process the featureCounts data file and merge it with annotations from the GENCODE GTF file as well as the sample annotations
#

suppressPackageStartupMessages(library("optparse", character.only=T, warn.conflicts=F, quietly=T, verbose=F))

# get the command line arguments
args <- commandArgs()

option_list <- list(
  make_option(c("-c", "--count"), action="store", type="character", help="featureCounts gene count file"),
  make_option(c("-f", "--gtf"), action="store", type="character", help="GENCODE GTF file used in featureCounts"),
  make_option(c("-g", "--gene_id"), action="store", type="character", default="gene_id", help="GENCODE GTF gene_id field, default is gene_id"),
  make_option(c("-a", "--gene_annotations"), action="store", type="character", default="gene_name,gene_type", help="GENCODE GTF gene annotation fields for extraction separated by commas, default is gene_name,gene_type"),
  make_option(c("-p", "--pheno"), action="store", type="character", help="Phenotype file, must contain a column entitled sample with the sample name in the counts file with the suffix .bam"),
  make_option(c("-o", "--output"), action="store", type="character", help="Results file basename")
)
parser <- OptionParser(option_list=option_list)
opt <- parse_args(parser)

#opt <- c(opt, list(count="gene_counts", output="testing", gtf="gencode.v26.annotation.gtf", pheno="phenotype.txt"))
#opt <- c(opt, list(count="../20160919_Olaf_GSE49622_mouse_sperm_development_RNA_Seq/gene_counts.txt", output="testing", gtf="../20160919_Olaf_GSE49622_mouse_sperm_development_RNA_Seq/gencode.vM10.annotation.gtf", pheno="../20160919_Olaf_GSE49622_mouse_sperm_development_RNA_Seq/phenotype.txt"))
#opt <- c(opt, list(count="../20180119_Olaf_point_recombination_rna_seq_MUX6171/gene_counts", output="testing", gtf="../20180119_Olaf_point_recombination_rna_seq_MUX6171/gencode.vM15.annotation.gtf", pheno="../20180119_Olaf_point_recombination_rna_seq_MUX6171/phenotype.txt"))
#opt <- c(opt, list(count="../20180313_Anand_combined_epithelial_RNA_Seq_MUX4953_MUX5548/gene_counts", output="testing", gtf="../20180313_Anand_combined_epithelial_RNA_Seq_MUX4953_MUX5548/gencode.v24.annotation.gtf", pheno="../20180313_Anand_combined_epithelial_RNA_Seq_MUX4953_MUX5548/phenotype.txt"))
#opt <- c(opt, list(count="../20170904_Anand_RNA_Seq_RNA0191_MUX5548_MUX5613/gene_counts", output="testing", gtf="../20170904_Anand_RNA_Seq_RNA0191_MUX5548_MUX5613/gencode.v26.annotation.gtf", pheno="../20170904_Anand_RNA_Seq_RNA0191_MUX5548_MUX5613/phenotype.txt"))
#opt <- c(opt, list(count="../20171005_Amit_TB_RNA_Seq_MUX5708/mutants/gene_counts", output="testing", gtf="../20171005_Amit_TB_RNA_Seq_MUX5708/mutants/tb.gtf", pheno="../20171005_Amit_TB_RNA_Seq_MUX5708/mutants/phenotype_no_outliers.txt"))
#opt <- c(opt, list(count="../20180604_ShanShan_novogene_sequencing/gene_counts", output="testing", gtf="../20180604_ShanShan_novogene_sequencing/gencode.vM16.annotation.gtf", pheno="../20180604_ShanShan_novogene_sequencing/phenotype.txt"))
#opt <- c(opt, list(count="/home/bernett/ngs/20200513_Anand_tempus_heat_inactivation_comparison/gene_counts", output="testing", gtf="/home/bernett/ngs/20200513_Anand_tempus_heat_inactivation_comparison/gencode.v31.annotation.gtf", pheno="/home/bernett/ngs/20200513_Anand_tempus_heat_inactivation_comparison/phenotype.txt"))

if (is.null(opt$count) || !file.exists(opt$count)) {
  print_help(parser)
  stop("Input featureCounts count file required.")
}

if (is.null(opt$gtf) || !file.exists(opt$gtf)) {
  print_help(parser)
  stop("Input GENCODE GTF file required.")
}

if (is.null(opt$pheno) || !file.exists(opt$pheno)) {
  print_help(parser)
  stop("Input phenotype file required.")
}

if (is.null(opt$output)) {
  print_help(parser)
  stop("Output file basename required.")
}

# process the gene_annotation option
opt$gene_annotations <- strsplit(opt$gene_annotations, ",", fixed=T)[[1]]

# # source the Stats.R file which should be located this script
# scriptPath <- normalizePath(gsub("^--file=", "", args[grep("^--file=", args)]))
# if (length(scriptPath)==0) {
#   source("Stats2.R")
# } else {
#   source(paste(dirname(scriptPath), "Stats2.R", sep="/"))
# }
# 
# # source the GeneSetEnrichment.R file which should be located this script
# scriptPath <- normalizePath(gsub("^--file=", "", args[grep("^--file=", args)]))
# if (length(scriptPath)==0) {
#   source("GeneSetEnrichment.R")
# } else {
#   source(paste(dirname(scriptPath), "GeneSetEnrichment.R", sep="/"))
# }

# load the libraries
libraries <- "stringr,reshape2,bernett,edgeR,futile.logger,rtracklayer,ggplot2,ggdendro,openxlsx,data.table,ggdendro,ggrepel,jsonlite"
libraries <- strsplit(libraries, ",")[[1]]
for (name in libraries) {
  suppressPackageStartupMessages(library(name, character.only=T, warn.conflicts=F, quietly=T, verbose=F))
}

# set up the loggers, wrap in invisible to prevent it from being printed
invisible(flog.threshold(DEBUG))
# remove the log file if it exists
if (file.exists(paste(opt$output, ".log", sep=""))) {
  invisible(file.remove(paste(opt$output, ".log", sep="")))
}
invisible(flog.appender(appender.tee(paste(opt$output, ".log", sep=""))))
flog.info("Command line arguments used:")
flog.info(paste(sapply(names(opt), function(x) ifelse(x!="help", paste("--", x, "=", paste(opt[[x]], collapse=","), sep=""), "")), collapse=" "))
flog.info(sprintf("CMD: %s %s", gsub("^--file=", "", grep("^--file=", commandArgs(), value=T)), paste(commandArgs(trailingOnly=T), collapse=" ")))

# read the counts file
d <- read.table(file=opt$count, sep="\t", header=T, quote="", na.string="", as.is=T, check.names=F, comment.char="#")
flog.info(paste("Counts data read with", nrow(d), "rows and", ncol(d), "columns"))
d <- rename(d, c(Geneid="gene_id", Length="length"))
d$Chr <- NULL
d$Start <- NULL
d$End <- NULL
d$Strand <- NULL

# read the gencode file for the symbol and type
gencode <- readGFF(opt$gtf, filter=list(type="gene"), columns=character(0), tags=c(opt$gene_id, opt$gene_annotations))
gencode$gene_id <- gencode[[opt$gene_id]]
rownames(gencode) <- gencode$gene_id
flog.info(paste("GTF data read with", nrow(gencode), "genes"))

# read the phenotype file
pheno <- read.table(file=opt$pheno, sep="\t", header=T, quote="", na.string="", as.is=T, check.names=F, comment.char="#")
if (!"sample" %in% colnames(pheno)) {
  flog.fatal(paste("Unable to locate a sample column in the phenotype file:", opt$pheno))
  stop()
}
flog.info("Phenotype file contains %d sample information", nrow(pheno))

# prepare the gene level data
fd <- d[, c("gene_id", "length")]
rownames(fd) <- fd$gene_id
for (aname in opt$gene_annotations) {
  if (aname %in% colnames(gencode)) {
    fd[[aname]] <- gencode[rownames(fd), aname]
  }
}

# process the phenotype file
rownames(pheno) <- pheno$sample

# prepare the expression data
sampleNames <- colnames(d)[!colnames(d) %in% c("gene_id", "length")]
d1 <- d[, sampleNames, drop=F]
rownames(d1) <- d$gene_id
d1 <- as.matrix(d1)
sampleNames <- gsub("\\_Aligned.sortedByCoord.out.bam", "", sampleNames)
sampleNames <- gsub("^.*/", "", sampleNames)
colnames(d1) <- sampleNames
flog.info("Counts file contains %d sample information", ncol(d1))

# intersect the sample names from counts and phenotype file
csamples <- intersect(colnames(d1), pheno$sample)
flog.info("%d samples are in common between the phenotype and count files", length(csamples))

# check that there are some overlap in the samples
if (length(csamples) == 0) {
  flog.fatal("Unable to match any of the samples to the phenotype file.")
  stop()
}

# check that all the genes in the counts files have data in the gencode file
if (!all(rownames(d1) %in% rownames(fd))) {
  flog.fatal("Unable to match all the gene_ids in the count file to the GENCODE file.")
  stop()
}

# filter to the common samples
d1 <- d1[, csamples, drop=F]

# construct the DGEList
e <- DGEList(counts=d1, genes=fd[rownames(d1), ])
pheno <- merge(e$sample, pheno, by="row.names")
pheno$Row.names <- NULL
rownames(pheno) <- pheno$sample
e$samples <- pheno[rownames(e$samples), ]
e <- calcNormFactors(e) # compute the normalization factors
flog.info("Finished constructing DGEList")

# save the object
ofilename <- paste(opt$output, ".RData", sep="")
save(e, file=ofilename)
flog.info(paste("Finished writing out DGEList object to file:", ofilename))

# generate the barchart of library size
png(file=paste(opt$output, "_library_size_bar_chart.png", sep=""), width=(100 + ncol(d1)*20), height=400, type="cairo")
ggplot(e$samples, aes(sample, lib.size)) + geom_bar(stat='identity') + xlab("sample") + ylab("Library size (read counts)") + ggtitle("Sample library sizes") + theme(axis.text.x = element_text(angle = 90, hjust = 1))
dev.off()
flog.info("Done plotting library size bar chart")

# convert to log cpm
o.cpm <- cpm(e, normalized.lib.sizes=T, log=T)
o.cpm <- reshape2::melt(o.cpm)
colnames(o.cpm) <- c("gene_id", "sample", "log2_cpm")

# convert to log rpkm
o.rpkm <- rpkm(e, normalized.lib.sizes=T, log=T)
o.rpkm <- reshape2::melt(o.rpkm)
colnames(o.rpkm) <- c("gene_id", "sample", "log2_rpkm")

# raw counts
o.raw <- reshape2::melt(e$counts)
colnames(o.raw) <- c("gene_id", "sample", "count")

# merge the data
o <- merge(as.data.table(o.rpkm), o.cpm, by=c("gene_id", "sample"))
o <- merge(o, o.raw, by=c("gene_id", "sample"))

# compute the tpm
o[, total_rpkm := sum(2 ^ log2_rpkm), .(sample)]
o$log2_tpm <- log2((2 ^ o$log2_rpkm) / o$total_rpkm * 1E6)
o$total_rpkm <- NULL

# gene annotations
o <- merge(o, e$genes, by=c("gene_id"))

# write it out
fwrite(o, file=paste(opt$output, "_counts.txt", sep=""), quote=F, sep="\t", na="", row.names=F, col.names=T)
flog.info("Finished writing out the counts data into a text file.")

# write out the phenotype information
write.table(e$samples, file=paste(opt$output, "_sample_annotations.txt", sep=""), quote=F, sep="\t", na="", row.names=F, col.names=T)
flog.info("Finished writing out the sample annotation data into a text file.")

# write out combined count and phenotype file for external use
o1 <- merge(o, e$samples, by="sample")
nnames <- gsub("[ \\-\\.]+", "_", tolower(colnames(o1)))
# MySQL reserved words
nnames[nnames=="group"] <- "dgelist_group"
# check for duplicate names
if (length(nnames)!=length(unique(nnames))) {
  flog.error("There are duplicate column names in the combined counts and phenotype file.")
  flog.error(sprintf("Column names        : %s", paste(sort(nnames), collapse=", ")))
  flog.error(sprintf("Unique column names : %s", paste(sort(unique(nnames)), collapse=", ")))
  stop()
}
colnames(o1) <- nnames
write.table(o1, file=paste(opt$output, "_counts_and_annotations.txt", sep=""), quote=F, sep="\t", na="", row.names=F, col.names=T)
flog.info("Finished writing out the combined count and sample annotation data into a text file.")

# generate the raw boxplot
png(file=paste(opt$output, "_log2_raw_count_boxplot.png", sep=""), width=(100 + nrow(e$samples)*20), height=400, type="cairo")
ggplot(o, aes(sample, log2(count+0.1))) + geom_boxplot() + ylab("log2 counts") + xlab("sample") + ggtitle("Box plot of log2 raw counts") + theme(axis.text.x = element_text(angle = 90, hjust = 1))
dev.off()
flog.info("Done plotting raw count box plot")

# generate the rpkm boxplot
png(file=paste(opt$output, "_log2_rpkm_boxplot.png", sep=""), width=(100 + nrow(e$samples)*20), height=400, type="cairo")
ggplot(o, aes(sample, log2_rpkm)) + geom_boxplot() + ylab("log2 RPKM") + xlab("sample") + ggtitle("Box plot of log2 RPKM") + theme(axis.text.x = element_text(angle = 90, hjust = 1))
dev.off()
flog.info("Done plotting log2 RPKM box plot")

# generate the PDF of the basic QC
flog.info("Generating QC plots")
pdf(file=paste(opt$output, "_qc_plots.pdf", sep=""), width=11.7, height=8.3)
# density of raw counts
o1 <- o[, {
  fit <- density(log2(count+0.1))
  list(x=fit$x, y=fit$y)
}, .(sample)]
p <- ggplot(o1, aes(x=x, y=y, col=sample)) + geom_line() + theme_bw() + ggtitle("Density plot of log2(raw counts+0.1)") + ylab("Fraction of genes") + xlab("log2(raw counts + 0.1)")
print(p)
# density of raw counts count >= 1
o1 <- o[count >= 1, {
  fit <- density(log2(count+0.1))
  list(x=fit$x, y=fit$y)
}, .(sample)]
p <- ggplot(o1, aes(x=x, y=y, col=sample)) + geom_line() + theme_bw() + ggtitle("Density plot of log2(raw counts+0.1) (at least 1 count)") + ylab("Fraction of genes") + xlab("log2(raw counts + 0.1)")
print(p)
# density of log2 RPKM
o1 <- o[, {
  fit <- density(log2_rpkm)
  list(x=fit$x, y=fit$y)
}, .(sample)]
p <- ggplot(o1, aes(x=x, y=y, col=sample)) + geom_line() + theme_bw() + ggtitle("Density plot of log2 RPKM") + ylab("Fraction of genes") + xlab("log2 RPKM")
print(p)
# density of log2 RPKM count >= 1
o1 <- o[count >= 1, {
  fit <- density(log2_rpkm)
  list(x=fit$x, y=fit$y)
}, .(sample)]
p <- ggplot(o1, aes(x=x, y=y, col=sample)) + geom_line() + theme_bw() + ggtitle("Density plot of log2 RPKM (at least 1 count)") + ylab("Fraction of genes") + xlab("log2 RPKM")
print(p)
# read depth
read_depth <- data.table()
for (rdepth in c(1, 5, 10, 25, 50, 100)) {
  read_depth <- rbind(read_depth, o[count >= rdepth, .(n=.N, depth=rdepth), .(sample)])
}
read_depth <- merge(read_depth, o[, .(total=.N), .(sample)], by="sample")
read_depth$percent <- read_depth$n / read_depth$total * 100
p <- ggplot(read_depth, aes(x=sample, y=n, fill=sample)) + geom_col() + facet_wrap(~ depth) + theme_bw() + ggtitle("Bar plot of number of genes and read depth") + ylab("Number of genes") + xlab("samples") + theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(p)
p <- ggplot(read_depth, aes(x=sample, y=percent, fill=sample)) + geom_col() + facet_wrap(~ depth) + theme_bw() + ggtitle("Bar plot of percentage of genes and read depth") + ylab("Percentage of genes") + xlab("samples") + theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(p)
# box plots
p <- ggplot(o[count>0, ], aes(sample, log2(count))) + geom_boxplot() + ylab("log2 counts") + xlab("sample") + ggtitle("Box plot of log2 raw counts (no zeroes)") + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + theme_bw() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(p)
p <- ggplot(o[count>0, ], aes(sample, log2_rpkm)) + geom_boxplot() + ylab("log2 RPKM") + xlab("sample") + ggtitle("Box plot of log2 RPKM (no zeroes)") + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + theme_bw() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(p)
dev.off()

# write out to Excel file
flog.info("Writing data to Excel file")
o1 <- as.data.table(rpkm(e, normalized.lib.sizes=T, log=T))
o1$gene_id <- rownames(e$counts)
o1 <- merge(as.data.table(e$genes), o1, by="gene_id")
wb <- createWorkbook("Bernett")
writeWorksheet(wb, "log2RPKM", o1)
writeWorksheet(wb, "Samples", e$samples)
writeWorksheet(wb, "Read depth (n)", dcast(read_depth, sample ~ depth, value.var="n"))
writeWorksheet(wb, "Read depth (%)", dcast(read_depth, sample ~ depth, value.var="percent"))
saveWorkbook(wb, paste(opt$output, "_log2rpkm_wide_data.xlsx", sep=""), overwrite=T)
flog.info("Finished writing out the log2 RPKM to an Excel file.")

# compute the tpm
o1 <- as.data.table(rpkm(e, normalized.lib.sizes=T, log=T))
o1$gene_id <- rownames(e$counts)
o1 <- melt(o1, id.vars="gene_id", variable.name="sample", value.name="log2_rpkm")
o1[, total_rpkm := sum(2 ^ log2_rpkm), .(sample)]
o1$log2_tpm <- log2((2 ^ o1$log2_rpkm) / o1$total_rpkm * 1E6)
o1$total_rpkm <- NULL
o1 <- dcast(o1, gene_id ~ sample, value.var="log2_tpm")
o1 <- merge(as.data.table(e$genes), o1, by="gene_id")
wb <- createWorkbook("Bernett")
writeWorksheet(wb, "log2TPM", o1)
writeWorksheet(wb, "Samples", e$samples)
writeWorksheet(wb, "Read depth (n)", dcast(read_depth, sample ~ depth, value.var="n"))
writeWorksheet(wb, "Read depth (%)", dcast(read_depth, sample ~ depth, value.var="percent"))
saveWorkbook(wb, paste(opt$output, "_log2tpm_wide_data.xlsx", sep=""), overwrite=T)
flog.info("Finished writing out the log2 TPM to an Excel file.")

# run the PCA and dendrogram only if there are more than 2 samples
if (nrow(e$samples) > 2) {

  # run the phenotype summary
  pheno <- e$samples
  pheno <- pheno[, !(colnames(pheno) %in% c("group", "lib.size", "norm.factors")), drop=F]
  flog.info("Computing phenotype summary")
  pheno_summary <- computeSummaryStatistics(pheno)

  # generate the PCA data
  flog.info("Computing PCA")
  o3 <- t(rpkm(e, normalized.lib.sizes=T, log=T))
  o3 <- o3[, apply(o3, 2, sum)>0]
  # run the PCA
  fit <- runPCA(o3)
  # process the components
  fit$components <- cbind(sample=rownames(fit$components), fit$components)
  fit$components <- merge(pheno, fit$components, by="sample")
  # run hclust on the first 2 components and compute the clusters
  flog.info("Running hclust on PC1 and PC2")
  fit2 <- hclust(dist(as.matrix(fit$components[, c("PC1", "PC2")])))
  for (k in 2:min(10, nrow(fit$components))) {
    fit$components[[sprintf("hclust_pc1_pc2_K%02d", k)]] <- cutree(fit2, k)
  }

  # write out the PCA text files
  flog.info("Writing out PCA text files")
  write.table(fit$variance, file=paste(opt$output, "_rpkm_pca_variance.txt", sep=""), quote=F, sep="\t", na="", row.names=F, col.names=T)
  write.table(fit$components, file=paste(opt$output, "_rpkm_pca_components.txt", sep=""), quote=F, sep="\t", na="", row.names=F, col.names=T)
  write.table(fit$loadings, file=paste(opt$output, "_rpkm_pca_loadings.txt", sep=""), quote=F, sep="\t", na="", row.names=F, col.names=T)

  # write out the PCA json file
  cat(toJSON(list(pca_components=fit$components, pca_variance=fit$variance), dataframe="columns", digits=NA), file=sprintf("%s_rpkm_pca_data.json", opt$output))

  results <- list()

  # run the PC vs pheno analysis
  flog.info("Runnning analysis of PC vs pheno")
  dv1s <- grep("^PC\\d+$", colnames(fit$components), value=T)
  dv2s <- pheno_summary[pheno_summary$use_as_numeric=="numeric", ]$name
  config <- computeTestCombinations(dv1s, dv2s)
  if (nrow(config) > 0) {
    results[["PC ann Spearman"]] <- runParallelSpearmanRankCorrelation(fit$components, config)
    if (sum(results[["PC ann Spearman"]]$status=="Test ok.") > 0) {
      plotScatterplots(paste(opt$output, "_rpkm_pca_pheno_spearman_scatter_plots.pdf", sep=""), results[["PC ann Spearman"]], fit$components, "x", "y", "Spearman rank correlation of %s vs %s\nrho=%0.3f, rho2=%0.3f, P=%0.3E, P_adj=%0.3E, n=%d", c("x", "y", "rho", "rho2", "pvalue", "pvalue_adj", "n"), by_rank=T)
    }
  }
  dv1s <- grep("^PC\\d+$", colnames(fit$components), value=T)
  dv2s <- pheno_summary[pheno_summary$use_as_categorical=="yes" & pheno_summary$num_min_count_group>1, ]$name
  config <- computeTestCombinations(dv1s, dv2s)
  if (nrow(config) > 0) {
    results[["PC ann Kruskal-Wallis"]] <- runParallelKruskalWallisTest(fit$components, config)
    results[["PC ann Dunn's post hoc"]] <- runParallelKruskalWallisPostHocTest(fit$components, config)
    if (sum(results[["PC ann Kruskal-Wallis"]]$status=="Test ok.") > 0) {
      plotStripplots(paste(opt$output, "_rpkm_pca_pheno_kruskal_wallis_strip_plots.pdf", sep=""), results[["PC ann Kruskal-Wallis"]], fit$components, "x", "y", "Kruskal-Wallis test of %s vs %s\nP=%0.3E, P_adj=%0.3E, n=%d", c("x", "y", "pvalue", "pvalue_adj", "n"))
    }
  }

  # run the clusters vs pheno analysis
  flog.info("Runnning analysis of Hclust vs pheno")
  dv1s <- pheno_summary[pheno_summary$use_as_numeric=="numeric", ]$name
  dv2s <- grep("^hclust_pc1_pc2_K\\d+$", colnames(fit$components), value=T)
  config <- computeTestCombinations(dv1s, dv2s)
  if (nrow(config) > 0) {
    results[["Hclust ann Kruskal-Wallis"]] <- runParallelKruskalWallisTest(fit$components, config)
    results[["Hclust ann Dunn's post hoc"]] <- runParallelKruskalWallisPostHocTest(fit$components, config)
    if (sum(results[["Hclust ann Kruskal-Wallis"]]$status=="Test ok.") > 0) {
      plotStripplots(paste(opt$output, "_rpkm_pca_hclust_pheno_kruskal_wallis_strip_plots.pdf", sep=""), results[["Hclust ann Kruskal-Wallis"]], fit$components, "x", "y", "Kruskal-Wallis test of %s vs %s\nP=%0.3E, P_adj=%0.3E, n=%d", c("x", "y", "pvalue", "pvalue_adj", "n"))
    }
  }
  dv1s <- grep("^hclust_pc1_pc2_K\\d+$", colnames(fit$components), value=T)
  dv2s <- pheno_summary[pheno_summary$use_as_categorical=="yes" & pheno_summary$num_min_count_group>1, ]$name
  config <- computeTestCombinations(dv1s, dv2s)
  if (nrow(config) > 0) {
    results[["Hclust ann Fisher's exact"]] <- runParallelFisherChisquareTest(fit$components, config)
    results[["Hclust ann Fisher table"]] <- getChisquareTable(fit$components, config)
  }

  # run the GO for PC
  for (pname in 1:2) {
    pname <- sprintf("PC%d", pname)
    flog.info(sprintf("Running Gene Ontology analysis with %s", pname))
    selected <- rownames(fit$loadings)[order(abs(fit$loadings[[pname]]), decreasing=T)[1:ceiling(nrow(fit$loadings)*0.1)]]
    goresults <- tryCatch(runTopGo(selected, rownames(fit$loadings)), error=function(e) NULL)
    if (!is.null(goresults)) {
      results[[sprintf("%s results", pname)]] <- goresults$results
      results[[sprintf("%s mapping", pname)]] <- goresults$mapping[significant=="Y" & goid %in% goresults$results[pvalue < 0.05, ]$goid, ]
    }
  }

  # write out the PCA Excel file
  wb <- createWorkbook("Bernett")
  writeWorksheet(wb, "Components", fit$components)
  writeWorksheet(wb, "Variance", fit$variance)
  writeWorksheet(wb, "Loadings", fit$loadings)
  for (nname in names(results)) {
    writeWorksheet(wb, nname, results[[nname]])
  }
  saveWorkbook(wb, paste(opt$output, "_rpkm_pca.xlsx", sep=""), overwrite=T)

  # write out the PCA plots
  pdf(file=paste(opt$output, "_rpkm_pca.pdf", sep=""), width=11.7, height=8.3)
  # print out the variance accounted for
  p <- ggplot(fit$variance, aes(factor(pc))) + geom_bar(aes(weight=variance*100)) + ylab("variance (%)") + xlab("component") + ggtitle("Variance per component")
  print(p)
  # plot the PCA with the sample names
  p <- ggplot(fit$components, aes(PC1, PC2, label=sample)) + geom_point(aes(color=factor(fit$components[["sample"]]))) + ggtitle(paste("PCA plot colored by column", "sample")) + xlab(sprintf("PC1 (%0.3f%%)", fit$variance[fit$variance$pc==1, ]$variance*100)) + ylab(sprintf("PC2 (%0.3f%%)", fit$variance[fit$variance$pc==2, ]$variance*100)) + geom_text_repel(size=2) + theme_bw()
  print(p)
  for (cname in c(pheno_summary[pheno_summary$unique_n>1, ]$name, grep("^hclust_pc1_pc2_K\\d+$", colnames(fit$components), value=T))) {
    print(cname)
    p <- ggplot(fit$components, aes(PC1, PC2)) + geom_point(aes(color=factor(fit$components[[cname]]))) + ggtitle(paste("PCA plot colored by column", cname)) + xlab(sprintf("PC1 (%0.3f%%)", fit$variance[fit$variance$pc==1, ]$variance*100)) + ylab(sprintf("PC2 (%0.3f%%)", fit$variance[fit$variance$pc==2, ]$variance*100)) + theme_bw()
    print(p)
  }
  dev.off()

  flog.info("Done with PCA")

  # generate the summary
  o.sum <- computeSummaryStatistics(pheno)

  # generate the dendrogram
  o.rpkm <- rpkm(e, normalized.lib.sizes=T, log=T)
  pdf(file=paste(opt$output, "_rpkm_dendrograms.pdf", sep=""), width=4+(0.2*ncol(o.rpkm)), height=8)
  pnames <- o.sum[o.sum$use_as_categorical=="yes" & o.sum$num_min_count_group > 1, "name"]
  if (length(pnames) > 0) {
    for (pname in pnames) {
      print(pname)
      dtypes <- list("euclidean"="as.dendrogram(hclust(dist(t(o.rpkm1))))", "spearman"="as.dendrogram(hclust(as.dist(1-cor(o.rpkm1, method='spearman'))))")
      o.rpkm1 <- o.rpkm
      for (dname in names(dtypes)) {
        print(dname)
        o.dend <- eval(parse(text=dtypes[[dname]]))
        o1 <- dendro_data(o.dend)
        o1$labels$group <- pheno[as.character(o1$labels$label), pname]
        p <- ggplot() + geom_segment(data=o1$segments, aes(x=x, y=y, xend=xend, yend=yend)) + geom_text(data=o1$labels, aes(x=x, y=y, label=label, hjust=1, angle=90, color=group)) + scale_y_continuous(expand=c(0.3, 0)) + theme(axis.line.x=element_blank(), axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), panel.background=element_rect(fill="white"), panel.grid=element_blank()) + ggtitle(sprintf("%s distance colored by %s", dname, pname)) + ylab("")
        print(p)
      }
    }
  }
  dev.off()

}

# log the R session info
flog.info(c("", "R session info", "", capture.output(sessionInfo()), ""))
