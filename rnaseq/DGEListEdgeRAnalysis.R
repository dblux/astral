#!/usr/bin/env Rscript

#
# Script to analyze a RNA-Seq expt as a DGEList object
#

# get the command line arguments
args <- commandArgs()

# source the Stats.R file which should be located with this script
scriptPath <- normalizePath(gsub("^--file=", "", args[grep("^--file=", args)]))
if (length(scriptPath)==0) {
  source("Stats2.R")
} else {
  source(paste(dirname(scriptPath), "Stats2.R", sep="/"))
}


# load the libraries
libraries <- "optparse,futile.logger,limma,edgeR,data.table,openxlsx"
libraries <- strsplit(libraries, ",")[[1]]
for (name in libraries) {
  suppressPackageStartupMessages(library(name, character.only=T, warn.conflicts=F, quietly=T, verbose=F))
}

# create the command line option list
option_list <- list(
  make_option(c("-i", "--input"), action="store", type="character", help="Input RData with variable e which is a DGEList object."),
  make_option(c("-o", "--output"), action="store", type="character", help="Results file basename."),
  make_option(c("-p", "--pheno"), action="store", type="character", help="Phenotype file, must contain a column entitled sample"),
  make_option(c("-q", "--qlf"), action="store_true", default=FALSE, help="Use the glmQLFit and glmQLFTest rather than the glmFit and glmLRT"),
  make_option(c("--cpm_iqr"), action="store", type="double", default=0.5, help="CPM IQR filtering, use -1 to unset"),
  make_option(c("--one_model_per_parameter"), action="store_true", default=FALSE, help="Use one model per parameter which is the old behavior, the new behavior is to use one model per comparison"),
  make_option(c("-c", "--categorical_max_unique_n"), action="store", type="integer", default=10, help="Maximum unique values for a numerical/categorical column to be used in categorical analysis"),
  make_option(c("--protein_only"), action="store_true", default=FALSE, help="Run analysis on protein coding genes only")
)
parser <- OptionParser(option_list=option_list)
opt <- parse_args(parser)

#opt <- c(opt, list(input="edger_test_data.RData", output="testing"))
#opt <- c(opt, list(input="edger_test_data_e1.RData", output="testing")) # different DGEList name test
#opt <- c(opt, list(input="../20161011_ImmGen_RNA_Seq_PRJNA281360/immgen_rna_seq.RData", output="testing"))
#opt <- c(opt, list(input="../20161010_Alessandra_Gout_study/microarray_E0158/eset.RData", output="testing")) # fail case
#opt <- c(opt, list(input="../20180214_Anis_MUX6279_macrophage_age_inflammation/2ng_no_outliers/anis_mux6279_macrophage_2ng_no_outliers_edger_dgelist.RData", output="testing"))
#opt <- c(opt, list(input="../20170904_Anand_RNA_Seq_RNA0191_MUX5548_MUX5613/rna0191_no_outliers2.RData", output="testing", pheno="../20170904_Anand_RNA_Seq_RNA0191_MUX5548_MUX5613/phenotype_atopy_analysis.txt"))

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
flog.info(sprintf("CMD: %s %s", gsub("^--file=", "", grep("^--file=", commandArgs(), value=T)), paste(commandArgs(trailingOnly=T), collapse=" ")))

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
    pheno <- read.table(file=opt$pheno, sep="\t", header=T, quote="", na.string="", as.is=T, check.names=F, comment.char="#")
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
e.cpm <- cpm(e, log=T)

# filter using IQR
cpm.iqr <- apply(e.cpm, 1, function(x) IQR(x))
if (opt$cpm_iqr != -1) {
  e <- e[labels(cpm.iqr[cpm.iqr>opt$cpm_iqr]), ]
  flog.info(sprintf("After IQR filtering using IQR > %0.3f, %d samples and %d genes are left", opt$cpm_iqr, nrow(e$samples), nrow(e$genes)))
} else {
  flog.info("No CPM IQR filtering specified.")
}

# filter for protein encoding genes
if (opt$protein_only) {
  if ("gene_type" %in% colnames(e$genes) && "protein_coding" %in% unique(e$genes$gene_type)) {
    e <- e[e$genes$gene_type=="protein_coding", ]
    flog.info(sprintf("After filtering for protein encoding genes, %d samples and %d genes are left", nrow(e$samples), nrow(e$genes)))
  }
  else {
    flog.error("Unable to either locate a gene annotation column entitled 'gene_type' or 'protein_coding' is not a value.")
    stop()
  }
}

# final number of genes and samples to be used for analysis
flog.info(paste("After all filtering,", nrow(e$samples), "samples and", nrow(e$genes), "genes are left"))

# generate the summary of the phenotype data
psummary <- computeSummaryStatistics(e$samples, categorical_max_unique_number=opt$categorical_max_unique_n)

# run the edgeR pairwise comparisons
results <- data.table()
for (cname in psummary[psummary$use_as_categorical=="yes", "name"]) {
  flog.info(sprintf("Running edgeR analysis for parameter %s", cname))
  x1 <- count(e$samples[!is.na(e$samples[[cname]]), cname])
  f1Levels <- as.character(x1[x1$freq>=2, "x"])
  flog.info(sprintf("Parameter %s has %d comparison groups", cname, length(f1Levels)))
  if (length(f1Levels)>1) {
    flog.info(sprintf("Running analysis for parameter %s which has %d comparison groups", cname, length(f1Levels)))
    # compute the comparisons
    comparisons <- as.data.table(t(combn(f1Levels, 2)))
    colnames(comparisons) <- c("x", "y")
    flog.info(sprintf("Parameter %s has %d unique pairwise comparisons", cname, nrow(comparisons)))
    # old behavior of one model per parameter
    if (opt$one_model_per_parameter) {
      flog.info(sprintf("Using one model per parameter for %s", cname))
      # create a subset for the analysis
      e1 <- e[, !is.na(e$samples[[cname]]) & as.character(e$samples[[cname]]) %in% f1Levels]
      f1 <- factor(as.character(e1$samples[[cname]]))
      my.contrasts <- createPairwiseComparisonContrastMatrix(f1)
      design <- model.matrix(~0+f1)
      colnames(design) <- levels(f1)
      e1 <- calcNormFactors(e1)
      e1 <- estimateDisp(e1, design=design)
      if (opt$qlf) {
        flog.info("Using the QLF form of the tests.")
        fit <- glmQLFit(e1, design)
        for (i in 1:length(colnames(my.contrasts))) {
          o <- as.data.frame(topTags(glmQLFTest(fit, contrast=my.contrasts[, i]), n=Inf, adjust.method="BH"))
          o$parameter <- cname
          o$comparison <- colnames(my.contrasts)[i]
          o$group1 <- strsplit(o$comparison[1], " vs ", fixed=T)[[1]][1]
          o$group2 <- strsplit(o$comparison[1], " vs ", fixed=T)[[1]][2]
          o$group1_n <- sum(f1==o$group1[1])
          o$group2_n <- sum(f1==o$group2[1])
          flog.info(sprintf("Parameter [%s] comparison [%s] has %d nominally significant and %d significant results", cname, colnames(my.contrasts)[i], sum(o$PValue<0.05), sum(o$FDR<0.05)))
          results <- rbind(results, o)
        }
      } else {
        flog.info("Using the LRT form of the tests.")
        fit <- glmFit(e1, design)    
        for (i in 1:length(colnames(my.contrasts))) {
          o <- as.data.frame(topTags(glmLRT(fit, contrast=my.contrasts[, i]), n=Inf, adjust.method="BH"))
          o$parameter <- cname
          o$comparison <- colnames(my.contrasts)[i]
          o$group1 <- strsplit(o$comparison[1], " vs ", fixed=T)[[1]][1]
          o$group2 <- strsplit(o$comparison[1], " vs ", fixed=T)[[1]][2]
          o$group1_n <- sum(f1==o$group1[1])
          o$group2_n <- sum(f1==o$group2[1])
          flog.info(sprintf("Parameter [%s] comparison [%s] has %d nominally significant and %d significant results", cname, colnames(my.contrasts)[i], sum(o$PValue<0.05), sum(o$FDR<0.05)))
          results <- rbind(results, o)
        }
      }
    # new behavior of one model per comparison
    } else {
      flog.info(sprintf("Using one model per comparison for parameter for %s", cname))
      for (i in 1:nrow(comparisons)) {
        flog.info(sprintf("Running comparison %s vs %s for parameter %s", comparisons[i, ]$x, comparisons[i, ]$y, cname))
        # create a subset for the analysis
        e1 <- e[, !is.na(e$samples[[cname]]) & as.character(e$samples[[cname]]) %in% c(comparisons[i, ]$x, comparisons[i, ]$y)]
        f1 <- factor(as.character(e1$samples[[cname]]))
        design <- model.matrix(~f1)
        e1 <- calcNormFactors(e1)
        e1 <- estimateDisp(e1, design=design)
        if (opt$qlf) {
          flog.info("Using the QLF form of the tests.")
          fit <- glmQLFit(e1, design)
          o <- cbind(as.data.frame(topTags(glmQLFTest(fit, coef=2), n=Inf, adjust.method="BH")), parameter=cname, comparison=sprintf("%s vs %s", levels(f1)[2], levels(f1)[1]), group1=levels(f1)[1], group2=levels(f1)[2], group1_n=sum(f1==levels(f1)[1]), group2_n=sum(f1==levels(f1)[2]))
          flog.info(sprintf("Parameter [%s] comparison [%s] has %d nominally significant and %d significant results", cname, o$comparison[[1]], sum(o$PValue<0.05), sum(o$FDR<0.05)))
          results <- rbind(results, o)
        } else {
          flog.info("Using the LRT form of the tests.")
          fit <- glmFit(e1, design)
          o <- cbind(as.data.frame(topTags(glmLRT(fit, coef=2), n=Inf, adjust.method="BH")), parameter=cname, comparison=sprintf("%s vs %s", levels(f1)[2], levels(f1)[1]), group1=levels(f1)[1], group2=levels(f1)[2], group1_n=sum(f1==levels(f1)[1]), group2_n=sum(f1==levels(f1)[2]))
          flog.info(sprintf("Parameter [%s] comparison [%s] has %d nominally significant and %d significant results", cname, o$comparison[[1]], sum(o$PValue<0.05), sum(o$FDR<0.05)))
          results <- rbind(results, o)
        }
      }
    }
  }
}
results$comparison <- as.character(results$comparison)
results$parameter <- as.character(results$parameter)
fwrite(results, file=paste(opt$output, "_edger_pairwise_comparison_results.txt", sep=""), quote=F, sep="\t", na="")

# generate the summary
results1 <- results[, .(tests=.N, nominal_degs=sum(PValue<0.05), corrected_degs=sum(FDR<0.05)), .(parameter, comparison, group1, group2, group1_n, group2_n)]
fwrite(results1, file=paste(opt$output, "_edger_pairwise_comparison_result_summary.txt", sep=""), quote=F, sep="\t", na="")

# write out the results to an Excel file
wb <- createWorkbook("Bernett")
# write out the summary
results1$worksheet <- paste("Comparison", 1:nrow(results1))
writeWorksheet(wb, "Summary", results1)
# write out the results into individual worksheets
for (i in 1:nrow(results1)) {
  print(results1[i, ]$comparison)
  writeWorksheet(wb, results1[i, ]$worksheet, results[parameter==results1[i, ]$parameter & comparison==results1[i, ]$comparison, ][order(PValue), ])
}
# save the Excel file
saveWorkbook(wb, paste(opt$output, "_edger_pairwise_comparison_results.xlsx", sep=""), overwrite=T)

# log r session info
flog.info(c("", "R session info", "", capture.output(sessionInfo()), ""))
