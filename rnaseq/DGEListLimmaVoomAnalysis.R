#!/usr/bin/env Rscript

#
# Script to analyze a RNA-Seq expt as a DGEList object using Limma/Voom
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
libraries <- "optparse,futile.logger,limma,edgeR,data.table"
libraries <- strsplit(libraries, ",")[[1]]
for (name in libraries) {
  suppressPackageStartupMessages(library(name, character.only=T, warn.conflicts=F, quietly=T, verbose=F))
}

# create the command line option list
option_list <- list(
  make_option(c("-i", "--input"), action="store", type="character", help="Input RData with variable e which is a DGEList object."),
  make_option(c("-o", "--output"), action="store", type="character", help="Results file basename."),
  make_option(c("-p", "--pheno"), action="store", type="character", help="Phenotype file, must contain a column entitled sample")
)
parser <- OptionParser(option_list=option_list)
opt <- parse_args(parser)

#opt <- list(input="edger_test_data.RData", output="testing")
#opt <- list(input="edger_test_data_e1.RData", output="testing") # different DGEList name test
#opt <- list(input="../20161011_ImmGen_RNA_Seq_PRJNA281360/immgen_rna_seq.RData", output="testing")
#opt <- list(input="../20161010_Alessandra_Gout_study/microarray_E0158/eset.RData", output="testing") # fail case
#opt <- list(input="../20170314_John_flu_pDC_kallisto_analysis/star/flu_star.RData", output="testing")
#opt <- list(input="../20170614_Evans_Servier_RNA_Seq_RNA0173/rna0173.RData", output="testing")

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
if(!("e" %in% newVariables) || class(e) != "DGEList") {
  # try to find the variable
  for (cname in newVariables) {
    if (class(get(cname)) == "DGEList") {
      e <- get(cname)
      rm(list=cname)
      flog.info(sprintf("Variable e does not exists, but managed to find variable %s which is a DGEList, using this instead", cname))
      break
    }
  }
  if(!("e" %in% ls()) || class(e) != "DGEList") {
    flog.fatal("Variable e does not exists.")
    stop()
  }
}
flog.info(paste("DGEList in variable e loaded with", nrow(e$samples), "samples and", nrow(e$genes), "genes"))

# check that we have a sample column in the phenotype and it matches the rownames
if ("sample" %in% colnames(e$samples)) {
  if (!all(e$samples$sample==rownames(e$samples))) {
    flog.fatal("Sample column in the phenotype data does not match the rownames")
    stop()
  }
} else {
  # create the sample column which is the same as the rownames
  e$samples$sample <- rownames(e$samples)
  flog.info("No sample column found, creating a new sample column using the rownames")
}

# read the phenotype file if there is any
if (!is.null(opt$pheno)) {
  if (file.exists(opt$pheno)) {
    flog.info(sprintf("Using external phenotype file %s", opt$pheno))
    pheno <- read.table(file=opt$pheno, sep="\t", header=T, quote="", na.string="", as.is=T, check.names=F, comment.char="")
    # check for the sample column for the matching
    if ("sample" %in% colnames(pheno)) {
      rownames(pheno) <- pheno$sample
      commonSamples <- intersect(pheno$sample, rownames(e$samples))
      flog.info(sprintf("DGEList contains %d samples and phenotype file contains %d of which %d is in common", nrow(e$samples), nrow(pheno), length(commonSamples)))
      flog.info(sprintf("Keeping the %d common samples for analysis", length(commonSamples)))
      e <- e[, colnames(e$counts) %in% commonSamples]
      ecolumns <- c("lib.size", "norm.factors")
      if (!"group" %in% colnames(pheno)) {
        ecolumns <- append(ecolumns, "group")
      }
      pcolumns <- colnames(pheno)[!colnames(pheno) %in% ecolumns]
      e$samples <- cbind(e$samples[, ecolumns], pheno[colnames(e$counts), pcolumns])
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

# create the cpm data for filtering purposes
cpm <- cpm(e, log=T)

# filter using IQR
cpm.iqr <- apply(cpm, 1, function(x) IQR(x))
e <- e[labels(cpm.iqr[cpm.iqr>0.5]), ]
flog.info(paste("After IQR filtering,", nrow(e$samples), "samples and", nrow(e$genes), "genes are left"))

# generate the summary of the phenotype data
psummary <- computeSummaryStatistics(e$samples, categorical_max_unique_number=20)

# run the Limma/Voom pairwise comparisons
results <- data.frame()
for (cname in psummary[psummary$use_as_categorical=="yes", "name"]) {
  flog.info(sprintf("Running limma analysis for parameter %s", cname))
  x1 <- count(e$samples[!is.na(e$samples[[cname]]), cname])
  f1Levels <- as.character(x1[x1$freq>=3, "x"])
  flog.info(sprintf("Parameter %s has %d comparison groups", cname, length(f1Levels)))
  if (length(f1Levels)>1) {
    # create a subset for the analysis
    e1 <- e[, !is.na(e$samples[[cname]]) & as.character(e$samples[[cname]]) %in% f1Levels]
    f1 <- factor(as.character(e1$samples[[cname]]))
    my.contrasts <- createPairwiseComparisonContrastMatrix(f1)
    design <- model.matrix(~0+f1)
    colnames(design) <- levels(f1)
    e1 <- calcNormFactors(e1)
    e1 <- voom(e1, design, plot=F)
    fit <- lmFit(e1, design)
    fit2 <- contrasts.fit(fit, my.contrasts)
    fit2 <- eBayes(fit2)
    for (i in 1:length(colnames(my.contrasts))) {
      o <- topTable(fit2, coef=i, num=Inf, adjust.method="BH")
      o$parameter <- cname
      o$comparison <- colnames(my.contrasts)[i]
      flog.info(sprintf("Parameter [%s] comparison [%s] has %d nominally significant and %d significant results", cname, colnames(my.contrasts)[i], sum(o$P.Value<0.05), sum(o$adj.P.Val<0.05)))
      results <- rbind(results, o)
    }    
  }
}
write.table(results, file=paste(opt$output, "_limma_voom_pairwise_comparison_results.txt", sep=""), quote=F, sep="\t", na="", row.names=F, col.names=T)
results1 <- ddply(results, .(parameter, comparison), function(x) {
  return(data.frame(
    nominal_degs=sum(x$P.Value<0.05),
    corrected_degs=sum(x$adj.P.Val<0.05)
  ))
})
write.table(results1, file=paste(opt$output, "_limma_voom_pairwise_comparison_result_summary.txt", sep=""), quote=F, sep="\t", na="", row.names=F, col.names=T)

# run the correlation analysis using rpkm
d <- as.data.frame(t(rpkm(e, log=T)))
d$sample <- rownames(d)
d <- merge(e$samples, d, by="sample")
dv1 <- psummary[psummary$use_as_numeric=="yes", "name"]
dv2 <- rownames(e$genes)
config <- computeTestCombinations(dv1, dv2)
if (nrow(config)>0) {
  flog.info(sprintf("Running %d Spearman Correlations", nrow(config)))
  results <- runParallelSpearmanRankCorrelation(d, config)
  results <- merge(results, e$genes, by.x="dv2", by.y="row.names")
  # correct for pvalue by dv1
  results <- ddply(results, .(dv1), function(x) {
    x$pvalue_adj <- p.adjust(x$pvalue, "BH")
    return(x)
  })
  results <- results[order(results$pvalue), ]
  flog.info(sprintf("Spearman correlations shows %d significant results with rho > 0.3", sum(results$pvalue_adj<0.05 & abs(results$rho)>=0.3)))
  flog.info(sprintf("Spearman correlations shows %d significant results with rho > 0.6", sum(results$pvalue_adj<0.05 & abs(results$rho)>=0.6)))
  flog.info(sprintf("Spearman correlations shows %d significant results with rho > 0.9", sum(results$pvalue_adj<0.05 & abs(results$rho)>=0.9)))
  write.table(results, file=paste(opt$output, "_spearman_correlation_results.txt", sep=""), quote=F, sep="\t", na="", row.names=F, col.names=T)
  results1 <- ddply(results, .(dv1), function(x) {
    return(data.frame(
      `rho >= 0.3`=sum(x$pvalue_adj<0.05 & abs(x$rho)>=0.3),
      `rho >= 0.6`=sum(x$pvalue_adj<0.05 & abs(x$rho)>=0.6),
      `rho >= 0.9`=sum(x$pvalue_adj<0.05 & abs(x$rho)>=0.9),
      check.names=F
    ))
  })
  write.table(results1, file=paste(opt$output, "_spearman_correlation_result_summary.txt", sep=""), quote=F, sep="\t", na="", row.names=F, col.names=T)    
}

# log r session info
flog.info(c("", "R session info", "", capture.output(sessionInfo()), ""))
