#!/usr/bin/env Rscript

#
# Script to analyze a RNA-Seq expt as a DGEList object using DESeq2
#

# get the command line arguments
args <- commandArgs()

# source the Stats.R file which should be located with this script
scriptPath <- normalizePath(gsub("^--file=", "", args[grep("^--file=", args)]))
if (length(scriptPath)==0) {
  source("Stats.R")
} else {
  source(paste(dirname(scriptPath), "Stats.R", sep="/"))
}

# load the libraries
libraries <- "optparse,futile.logger,limma,edgeR,data.table,DESeq2"
libraries <- strsplit(libraries, ",")[[1]]
for (name in libraries) {
  suppressPackageStartupMessages(library(name, character.only=T, warn.conflicts=F, quietly=T, verbose=F))
}

# create the command line option list
option_list <- list(
  make_option(c("-i", "--input"), action="store", type="character", help="Input RData with variable e which is a DGEList/DESeqDataSet object."),
  make_option(c("-o", "--output"), action="store", type="character", help="Results file basename."),
  make_option(c("-p", "--pheno"), action="store", type="character", help="Phenotype file, must contain a column entitled sample"),
  make_option(c("--cpm_iqr"), action="store", type="double", default=0.5, help="CPM IQR filtering, use -1 to unset"),
  make_option(c("--one_model_per_parameter"), action="store_true", default=FALSE, help="Use one model per parameter which is the old behavior, the new behavior is to use one model per comparison"),
  make_option(c("-c", "--categorical_max_unique_n"), action="store", type="integer", default=10, help="Maximum unique values for a numerical/categorical column to be used in categorical analysis"),
  make_option(c("--protein_only"), action="store_true", default=FALSE, help="Run analysis on protein coding genes only")
)
parser <- OptionParser(option_list=option_list)
opt <- parse_args(parser)

#opt <- c(opt, list(input="edger_test_data.RData", output="testing"))
#opt <- list(input="y.RData", output="testing")
#opt <- list(input="edger_test_data_e1.RData", output="testing") # different DGEList name test
#opt <- list(input="../20161011_ImmGen_RNA_Seq_PRJNA281360/immgen_rna_seq.RData", output="testing")
#opt <- list(input="../20161010_Alessandra_Gout_study/microarray_E0158/eset.RData", output="testing") # fail case
#opt <- list(input="../20170314_John_flu_pDC_kallisto_analysis/star/flu_star.RData", output="testing")
#opt <- list(input="../20160824_Amit_metformin_treatment_during_bcg_mtb_infection/ly6c_monocytes.RData", output="testing", pheno="../20160824_Amit_metformin_treatment_during_bcg_mtb_infection/ly6c_phenotype.txt")
#opt <- c(opt, list(input="../20180619_Amit_MUX7156_RNA0277/amit_mux7156_rna0277_deseq.RData", output="testing"))
#opt <- c(opt, list(input="../20160824_Amit_metformin_treatment_during_bcg_mtb_infection/spleen_analysis/amit_spleen_t_subsets.RData", output="testing"))
#opt <- c(opt, list(input="../20180607_Amit_MUX7006_RNA0275/mux7006_rna0275.RData", pheno="../20180607_Amit_MUX7006_RNA0275/phenotype_sorted.txt", output="testing"))
#opt <- c(opt, list(input="/home/ppa/ngs/20201202_AS_Carey_Tb_Mac_RNA0438/xoutlier_rna_seq_data.RData", output="testing"))

# check for input
if (is.null(opt$input)) {
  print_help(parser)
  stop("Input file required.")
} else if (!file.exists(opt$input)) {
  print_help(parser)
  stop("Input file does not exists.")
}

# check for output
if (is.null(opt$output)) {
  print_help(parser)
  stop("Output file basename required.")
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
flog.info(sprintf("CMD: %s %s", scriptPath, paste(commandArgs(trailingOnly=T), collapse=" ")))

# load the DGEList object, will attempt to find if it is not named as e
newVariables <- load(opt$input)
# check if e exists and is a DGEList
if ("e" %in% newVariables && class(e) == "DGEList") {
  flog.info("Found variable e which is a DGELIst object")
} else if ("e" %in% newVariables && class(e) == "DESeqDataSet") {
  flog.info("Found variable e which is a DESeqDataSet object")
} else {
  # try to find the variable
  for (cname in newVariables) {
    if (class(get(cname)) == "DGEList") {
      e <- get(cname)
      rm(list=cname)
      flog.info(sprintf("Variable e does not exists, but managed to find variable %s which is a DGEList, using this instead", cname))
      break
    }
    else if (class(get(cname)) == "DGEList") {
      e <- get(cname)
      rm(list=cname)
      flog.info(sprintf("Variable e does not exists, but managed to find variable %s which is a DESeqDataSet, using this instead", cname))
      break
    }
  }
  if(!("e" %in% ls()) || !(class(e) %in% c("DGEList", "DESeqDataSet"))) {
    flog.fatal("Variable e does not exists.")
    stop()
  }
}

# convert the DGEList object to DESeqDataSet
if (class(e)=="DGEList") {
  flog.info("Converting the DGEList to a DESeqDataSet")
  dds <- DESeqDataSetFromMatrix(countData=e$counts, colData=e$samples, design=~1)
  mcols(dds) <- e$genes
  e <- dds
  flog.info("Created DESeqDataSet for analysis")
}

#dds <- dds[1:1000, 1:10] # testing only

flog.info(paste("DESeqDataSet in variable e loaded with", ncol(e), "samples and", nrow(e), "genes"))

# read the phenotype file if there is any
if (!is.null(opt$pheno)) {
  if (file.exists(opt$pheno)) {
    flog.info(sprintf("Using external phenotype file %s", opt$pheno))
    pheno <- read.table(file=opt$pheno, sep="\t", header=T, quote="", na.string="", as.is=T, check.names=F, comment.char="#")
    # check for the sample column for the matching
    if ("sample" %in% colnames(pheno)) {
      rownames(pheno) <- pheno$sample
      commonSamples <- intersect(pheno$sample, colnames(e))
      flog.info(sprintf("DGEList contains %d samples and phenotype file contains %d of which %d is in common", ncol(e), nrow(pheno), length(commonSamples)))
      flog.info(sprintf("Keeping the %d common samples for analysis", length(commonSamples)))
      e <- e[, commonSamples]
      colData(e) <- as(pheno[rownames(colData(e)), ], "DataFrame")
    }
    else {
      msg <- sprintf("Unable to locate a column entitled sample in the phenotype file %s", opt$pheno)
      flog.error(msg)
      stop(msg)
    }
  }
  else {
    msg <- sprintf("Unable to locate phenotype file %s", opt$pheno)
    flog.error(msg)
    stop(msg)
  }
}

# remove genes which are all zeros
ok_genes <- apply(counts(e), 1, sum)!=0
flog.info(sprintf("%d genes removed due to all zeros", sum(!ok_genes)))
e <- e[ok_genes, ]

# compute the cpm
cpm <- log2(t(t(assay(e)) / apply(assay(e), 2, sum) * 1E6) + 0.01)

# filter using IQR
cpm.iqr <- apply(cpm, 1, function(x) IQR(x))
if (opt$cpm_iqr != -1) {
  e <- e[labels(cpm.iqr[cpm.iqr>opt$cpm_iqr]), ]
  flog.info(sprintf("After IQR filtering using IQR > %0.3f, %d samples and %d genes are left", opt$cpm_iqr, dim(e)[2], dim(e)[1]))
} else {
  flog.info("No CPM IQR filtering specified.")
}

# filter for protein encoding genes
if (opt$protein_only) {
  if ("gene_type" %in% colnames(rowData(e)) && "protein_coding" %in% unique(rowData(e)[["gene_type"]])) {
    e <- e[rowData(e)[["gene_type"]]=="protein_coding", ]
    flog.info(sprintf("After filtering for protein encoding genes, %d samples and %d genes are left", dim(e)[2], dim(e)[1]))
  }
  else {
    flog.error("Unable to either locate a gene annotation column entitled 'gene_type' or 'protein_coding' is not a value.")
    stop()
  }
}

# final number of genes and samples to be used for analysis
flog.info(paste("After all filtering,", ncol(e), "samples and", nrow(e), "genes are left"))

# generate the summary of the phenotype data
psummary <- computeSummaryStatistics(colData(e), categorical_min_level_n=2, categorical_max_unique_number=opt$categorical_max_unique_n)

# run the DESeq analysis
results <- data.table()
for (cname in psummary[psummary$use_as_categorical=="yes" & psummary$num_min_count_group > 1, "name"]) {
  flog.info(sprintf("Running DESeq2 analysis for parameter %s", cname))
  x1 <- as.data.table(table(e[[cname]]))
  f1Levels <- as.character(x1[N>=2, ]$V1)
  flog.info(sprintf("Parameter %s has %d comparison groups", cname, length(f1Levels)))
  if (length(f1Levels)>1) {
    # old behavior of one model per parameter
    if (opt$one_model_per_parameter) {
      flog.info(sprintf("Using one model per parameter for %s", cname))
      # create a subset for the analysis
      e1 <- e[, !is.na(e[[cname]]) & as.character(e[[cname]]) %in% f1Levels]
      e1[[cname]] <- factor(e1[[cname]])
      design(e1) <- as.formula(sprintf("~ %s", cname))
      e1 <- DESeq(e1)
      comparisons <- as.data.frame(t(combn(as.character(levels(e1[[cname]])), 2)), stringsAsFactors=F)
      for (i in 1:nrow(comparisons)) {
        comparison <- sprintf("%s vs %s", comparisons[i, "V2"], comparisons[i, "V1"])
        o <- as.data.frame(results(e1, contrast=c(cname, comparisons[i, "V2"], comparisons[i, "V1"]), independentFiltering=FALSE))
        o <- cbind(mcols(e1)[colnames(mcols(e))], o)
        o$parameter <- cname
        o$comparison <- comparison
        o <- o[order(o$pvalue), ]
        flog.info(sprintf("Parameter [%s] comparison [%s] has %d nominally significant and %d significant results", cname, comparison, sum(o$pvalue<0.05, na.rm=T), sum(o$padj<0.05, na.rm=T)))
        results <- rbind(results, o)
      }
    }
    # new behavior of one model per comparison
    else {
      flog.info(sprintf("Using one model per comparison for parameter for %s", cname))
      # compute the comparisons
      comparisons <- as.data.table(t(combn(f1Levels, 2)))
      colnames(comparisons) <- c("x", "y")
      flog.info(sprintf("Parameter %s has %d unique pairwise comparisons", cname, nrow(comparisons)))
      for (i in 1:nrow(comparisons)) {
        e1 <- e[, !is.na(e[[cname]]) & as.character(e[[cname]]) %in% c(comparisons[i, ]$x, comparisons[i, ]$y)]
        e1[[cname]] <- factor(e1[[cname]])
        design(e1) <- as.formula(sprintf("~ %s", cname))
        e1 <- DESeq(e1)
        o <- as.data.frame(results(e1, contrast=c(cname, comparisons[i, ]$y, comparisons[i, ]$x), independentFiltering=FALSE))
        o <- cbind(mcols(e1)[colnames(mcols(e))], o)
        o$parameter <- cname
        o$comparison <- sprintf("%s vs %s", comparisons[i, ]$y, comparisons[i, ]$x)
        o <- o[order(o$pvalue), ]
        flog.info(sprintf("Parameter [%s] comparison [%s] has %d nominally significant and %d significant results", cname, o$comparison[[1]], sum(o$pvalue<0.05, na.rm=T), sum(o$padj<0.05, na.rm=T)))
        results <- rbind(results, o)
      }
    }
  }
}
results <- as.data.table(results)
write.table(results, file=paste(opt$output, "_deseq2_pairwise_comparison_results.txt", sep=""), quote=F, sep="\t", na="", row.names=F, col.names=T)
results1 <- results[, .(tested=.N, nominal_degs=sum(pvalue<0.05, na.rm=T), corrected_degs=sum(padj<0.05, na.rm=T)), .(parameter, comparison)]
write.table(results1, file=paste(opt$output, "_deseq2_pairwise_comparison_result_summary.txt", sep=""), quote=F, sep="\t", na="", row.names=F, col.names=T)

# write out the results to an Excel file
wb <- createWorkbook("Bernett")
# write out the summary
results1$worksheet <- paste("Comparison", 1:nrow(results1))
writeWorksheet(wb, "Summary", results1)
# write out the results into individual worksheets
for (i in 1:nrow(results1)) {
  print(results1[i, ]$comparison)
  writeWorksheet(wb, results1[i, ]$worksheet, results[parameter==results1[i, ]$parameter & comparison==results1[i, ]$comparison, ][order(pvalue), ])
}
# save the Excel file
saveWorkbook(wb, paste(opt$output, "_deseq2_pairwise_comparison_results.xlsx", sep=""), overwrite=T)

# log r session info
flog.info(c("", "R session info", "", capture.output(sessionInfo()), ""))
