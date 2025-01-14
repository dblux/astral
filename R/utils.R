#' Compute features from PCA coordinates of SingleCellExperiment
#'
#' Be mindful of patients with >3 time points
#'
#' @param sce SingleCellExperiment object with PCA coordinates
#' @param func function that computes feature from PCA data.frame
#' @param pc numeric indicating which PCs to take while computing features
#' @param pid_remove character of patient IDs to remove 
#'
compute_feature <- function(
  sce, func, pcs = c(1, 2),
  pid_remove = pid_less3tp
) {
  sce <- sce[, !(sce$sn %in% pid_remove)]
  X_pca <- reducedDim(sce, "PCA")[, pcs]
  # Assume: Rows of data frame are arranged from earliest to latest time points
  patients <- split(data.frame(X_pca), sce$sn)
  features <- sapply(patients, func)
  # Patient metadata
  metadata <- colData(sce)
  label <- metadata[match(names(features), metadata$sn), "final_label"]
  data.frame(value = features, label = label)
}


#' Annotates affymetrix probesets to gene ID of choice
#'
#' Annotates affymetrix probesets according to platform and gene
#' naming convention provided in the annotation file. Option of
#' returning vector of annotaions with probesets as names
#'
#' Removes probesets with no matching ID. If multiple probesets
#' map to a single ID, the probeset with the max sum is used.
#' @param X data.frame with affy probesets as rownames. Assumes that X
#'   does not contain ambiguous and affymetrix probesets.
#' @param file name of the annotation file
#' @param ret.annot logical indicating whether to return vector of annotations
#'
affy2id <- function(X, file, return_annot = F) { 
  # probesets are rownames of dataframe
  annot_table <- read.table(
    file, sep = "\t", header = T, row.names = 1,
    stringsAsFactors = F, strip.white = T, na.strings = ""
  )
  orig_rownames <- annot_table[rownames(X), ]
  msg_no_id <- sprintf( 
    "No. of probesets with no ID removed = %d\n",
    sum(is.na(orig_rownames))
  )
  cat(msg_no_id)
   
  # Indices of probe sets with no corresponding ID to be deleted and probesets
  # that are not present in annot_table
  idx_del <- which(is.na(orig_rownames))
   
  # Identifies genes that have multiple probesets mapping to it 
  freq_gene <- table(orig_rownames[!is.na(orig_rownames)])
  dup_genes <- rownames(freq_gene)[freq_gene > 1]
  for (gene in dup_genes) {
    # False & NA = False
    idx <- !is.na(orig_rownames) & orig_rownames == gene
    # subset rows of dataframe with the same id
    same_rows <- X[idx, , drop = F] 
    # assign numeric indices as rownames 
    rownames(same_rows) <- which(orig_rownames == gene)
    # rows that do not have the maximum sum are deleted 
    row_del <- as.numeric(
      rownames(same_rows)[-which.max(rowSums(same_rows))]
    )
    # concat with existing list of indices to be deleted 
    idx_del <- c(idx_del, row_del)
  }
  idx_del <- unique(idx_del) 
  msg_total <- sprintf(
    "Total no. of probesets removed (incl. probesets mapping to same gene): %d\n", 
    length(idx_del)
  ) 
  cat(msg_total)
  
  # Rows are deleted 
  X_genes <- X[-idx_del, ]
  fltr_rownames <- orig_rownames[-idx_del]
  names(fltr_rownames) <- rownames(X)[-idx_del]
  # Assigning id to X
  rownames(X_genes) <- fltr_rownames
  
  if (return_annot) {
    orig_rownames[idx_del] <- NA
    names(orig_rownames) <- rownames(X)
    return(list(X = X_genes, mapping = orig_rownames))
  }

  X_genes
}


#' Sorts sample IDs so that they are paired
sort_sid <- function(x) {
  pid <- sapply(x, substring, 1, 4)
  time <- sapply(x, substring, 6, 7)
  time_pid <- mapply(paste0, time, pid) # 'D0P023'
  return(x[order(time_pid)])
}


#' Generates sample IDs from patient IDs
get_sid <- function(pid) {
  c(paste0(pid, '_D0'), paste0(pid, '_D8'))
}


#' Boolean function checking that dataframe provided has matching pair names
#' @param X dataframe with paired samples
#' @return logical indicating if it is paired or not
is_paired <- function(x) {
  #' @param character of sample ids
  #' @return logical indicating if it is paired or not
  .is_paired <- function(sid) {
    n <- length(sid)
    pid <- substring(sid, 1, 4)
    all(pid[1:(n / 2)] == pid[(n / 2 + 1):n])
  }
  
  if (is.data.frame(x) | is.matrix(x)) {
    return(.is_paired(colnames(x)))
  } else if (is.character(x)) {
    return(.is_paired(x))
  } else{
    stop("x is not a dataframe, matrix or character.")
  }
}

#' @param y_true numeric vector of true labels with 1 as positive
calc_recall <- function(y_true, y_pred) {
  sum(y_pred[y_true == 1] == 1) / sum(y_true == 1)
}


#' @param y_true numeric vector of true labels with 1 as positive
calc_specificity <- function(y_true, y_pred) {
  sum(y_pred[y_true == 0] == 0) / sum(y_true == 0)
}

# Substring without n last characters
substring_head <- function(string, n) substring(string, 1, nchar(string)-n)

# Substring tail starting from length - n characters
substring_tail <- function(string, n) substring(string, nchar(string)-n+1)


#' Splits subsettable objects according to their columns
#'
#' @param x subsettable object to be split
#' @param f vector or list of vectors indicating the grouping of columns
#' @param drop logical indicating if levels that do not occur should be dropped
#' @param ... optional arguments to [split()]
#'
#' @returns List of objects split by columns
#'
#' @examples
#'
#' X <- matrix(1:60, 10, 6)
#' cond <- rep(1:3, each = 2)
#' split_cols(X, cond)
#'
#' @export
#'
split_cols <- function(x, f, drop = FALSE, ...) {
  if (is.list(f)) {
    stopifnot(all(sapply(f, length) == ncol(x)))
  } else {
    stopifnot(length(f) == ncol(x))
  }
  lapply(
    split(seq_len(ncol(x)), f, drop, ...),
    function(ind) x[, ind, drop = FALSE]
  )
}


#' Sigmoid function
#'
#' @param x numeric scalar/vector/matrix
#' @param r inverse scale parameter of the sigmoid function
#' @param s midpoint parameter of the sigmoid function
#'
#' @returns A numeric scalar/vector/matrix of the same dimensions containing
#'   the transformed values.
#'
#' @examples
#'
#' p <- sigmoid(0.5)
#'
#' @export
#'
sigmoid <- function(x, r = 1, s = 0) 1 / (1 + exp(-(r * (x + s))))
