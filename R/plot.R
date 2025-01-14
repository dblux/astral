#' Plots PCA plot using ggplot2
#'
#'
#' Plots PCA plot using ggplot2
#'
#' @import ggplot2
#' @importFrom tibble rownames_to_column
#' @param X dataframe with features as rows and samples as columns
#' @param metadata dataframe of metadata with samples as rows
#' @param newdata dataframe of data to be predicted by prcomp object
#' @param x character indicating PC to plot on x-axis
#' @param y character indicating PC to plot on y-axis
#' @param ... optional arguments are passed to aes_string in ggplot. Optional
#' parameters have to match column names in metadata
ggplot_pca <- function(
  X,
  metadata,
  cex = 2,
  label = FALSE,
  newdata = NULL,
  x = "PC1", y = "PC2",
  show.legend = TRUE,
  ...
) {
  x_idx <- as.numeric(substring(x, 3))
  y_idx <- as.numeric(substring(y, 3))
  
  # PCA
  pca_obj <- prcomp(t(X))
  Z <- data.frame(pca_obj$x)
  eigenvalues <- (pca_obj$sdev)^2
  var_pc <- eigenvalues/sum(eigenvalues)
  pc_labels <- sprintf("%s (%.2f%%)", colnames(Z), var_pc * 100)

  # Projects newdata into PCA space
  if (!is.null(newdata)) {
    Z_new <- predict(pca_obj, newdata = t(newdata))
    # Remove duplicate rows
    Z_new <- Z_new[!(rownames(Z_new) %in% rownames(Z)), ]
    Z <- rbind(Z, Z_new)
  }
  
  # Concat with metadata
  metadata_cols <- unlist(list(...))
  metadata1 <- metadata[rownames(Z), metadata_cols, drop = F]
  Z_metadata <- cbind(tibble::rownames_to_column(Z), metadata1)

  ax <- ggplot(
    Z_metadata,
    aes_string(x = x, y = y, label = "rowname", ...),
  ) +
    labs(x = pc_labels[x_idx], y = pc_labels[y_idx]) +
    geom_vline(xintercept = 0, color = "black") +
    geom_hline(yintercept = 0, color = "black")
  
  # Plot text labels instead of points
  if (label) {
    return(
      ax +
        geom_text(col = "black", cex = cex)
    )
  }

  ax + geom_point(cex = cex, show.legend = show.legend)
}


#' Plots pre-computed PCA coordinates in SingleCellExperiment
#'
#' S3 method for PCA plotting of data.
ggplot_pca.sce <- function(
  sce,
  x = "PC1",
  y = "PC2",
  label = FALSE,
  show.legend = TRUE,
  cex = 2,
  ...
) {
  if (!requireNamespace("SingleCellExperiment", quietly = TRUE))
    stop("Please install SingleCellExperiment package!")

  # Concat with metadata
  # Z <- data.frame(reducedDim(sce, "PCA"))
  # pct_var <- attr(Z, "percentVar")
  # print(pct_var)
  xidx <- as.numeric(substring(x, 3))
  yidx <- as.numeric(substring(y, 3))
  Z <- sce@int_colData@listData$reducedDims@listData$PCA
  pct_var <- attr(Z, "percentVar")
  x_lab <- sprintf("PC%d (%.2f%%)", xidx, pct_var[xidx])
  y_lab <- sprintf("PC%d (%.2f%%)", yidx, pct_var[yidx])
  metadata <- colData(sce)
  metadata_cols <- unlist(list(...))
  metadata1 <- metadata[rownames(Z), metadata_cols, drop = F]
  Z_metadata <- cbind(tibble::rownames_to_column(data.frame(Z)), metadata1)
  ax <- ggplot(
    Z_metadata,
    aes_string(x = x, y = y, label = "rowname", ...),
  ) +
    labs(x = x_lab, y = y_lab) +
    geom_vline(xintercept = 0, color = "black") +
    geom_hline(yintercept = 0, color = "black")
  
  # Plot text labels instead of points
  if (label) {
    return(ax + geom_text(cex = cex))
  }

  ax + geom_point(cex = cex, show.legend = show.legend)
}


#' Plots top PCs using ggplot
#'
#' @import ggplot2
#' @importFrom tidyr gather
#' @export
#' @param X dataframe with features as rows and samples as columns
#' @param metadata dataframe of metadata with samples as rows
ggplot_top_pc <- function(
  X, metadata, x_axis, n = 8, nrow = 2,
  cex = 2, jitter.width = 1, show.legend = FALSE,
  show.xaxis = TRUE, newdata = NULL, ...
) {
  # PCA
  pca_obj <- prcomp(t(X))
  if (ncol(pca_obj$x) < n) {
    n <- ncol(pca_obj$x)
    cat(sprintf("PCA transformed matrix only has %d dimensions!\n", n))
  }
  Z <- data.frame(pca_obj$x[, seq(n)])
  eigenvalues <- (pca_obj$sdev) ^2
  var_pc <- eigenvalues[seq(n)] / sum(eigenvalues)
  pc_labels <- sprintf("PC%d (%.2f%%)", seq(n), var_pc*100)
  names(pc_labels) <- paste0('PC', seq_len(length(pc_labels)))

  # Projects newdata into PCA space
  if (!is.null(newdata)) {
    Z_new <- predict(pca_obj, newdata = t(newdata))[, 1:3]
    # Remove duplicate rows
    Z_new <- Z_new[!(rownames(Z_new) %in% rownames(Z)), ]
    Z <- rbind(Z, Z_new)
  }

  # Concat with metadata
  plot_factors <- unlist(list(...))
  plot_factors <- unique(c(x_axis, plot_factors))
  metadata1 <- metadata[rownames(Z), plot_factors, drop = F]
  Z_metadata <- cbind(Z, metadata1)
  # Convert data to long format
  Z_long <- tidyr::gather(Z_metadata, key = "PC", value = "value", -plot_factors)
  
  ax <- ggplot(Z_long, aes_string(x = x_axis, y = "value", ...)) +
    facet_wrap(
      ~PC, scales = 'free_y', nrow = nrow,
      labeller = as_labeller(pc_labels),
    ) +
    geom_point(
      position = position_jitterdodge(jitter.width = jitter.width),
      cex = cex, alpha = 0.8,
      show.legend = show.legend
    )
  
  if (!show.xaxis) {
    ax <- ax +
      theme(
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank()
      )
  }
  
  ax 
}


#' Plots UMAP plot using ggplot2
#'
#' @import ggplot2
#' @importFrom umap umap
#' @param obj UMAP object from umap function
#' @param ... optional arguments are passed to aes_string in ggplot
ggplot_umap <- function(
  X, metadata,
  cex = 2,
  alpha = 0.2,
  plot_label = FALSE,
  return_data = FALSE, ...
) {
  obj <- umap(t(X))
  Z <- data.frame(obj$layout)
  colnames(Z) <- c("UMAP1", "UMAP2")
  
  metadata_cols <- unlist(list(...))
  metadata1 <- metadata[colnames(X), metadata_cols, drop = F]
  Z_metadata <- cbind(tibble::rownames_to_column(Z), metadata1)
  Z_metadata$rowname <- substring(Z_metadata$rowname, 1, 4)
  
  ax <- ggplot(
    Z_metadata,
    aes_string(x = 'UMAP1', y = 'UMAP2', label = "rowname", ...)
  ) 
  
  # Plot text labels instead of points
  if (plot_label) {
    ax <- ax +
      geom_point(color = "black", cex = cex, alpha = alpha) +
      geom_text(cex = cex)
  } else {
    ax <- ax +
      geom_point(cex = cex, alpha = alpha)
  }

  if (return_data)
    return(list(plot = ax, X = Z_metadata))

  ax
}


#' Plots t-SNE plot using ggplot2
#'
#' @import ggplot2
#' @import Rtsne
#' @param obj UMAP object from umap function
#' @param ... optional arguments are passed to aes_string in ggplot
ggplot_tsne <- function(X, metadata, cex = 2, alpha = 1, ...) {
  obj <- Rtsne(normalize_input(t(X)))
  Z <- data.frame(obj$Y)
  colnames(Z) <- c("tSNE1", "tSNE2")
  metadata_cols <- unlist(list(...))
  metadata1 <- metadata[colnames(X), metadata_cols, drop = F]
  Z_metadata <- cbind(Z, metadata1)
  
  ggplot(Z_metadata, aes_string(x = "tSNE1", y = "tSNE2" , ...)) +
    geom_point(cex = cex, alpha = alpha)
}


#' Plots ROC curve
#'
#' @import ggplot2 pROC
#'
#' @param X dataframe containing scores / predictions and labels
#' @param response character containing name of column in X with labels
#' @param predictor character containing name of column/s in X of predictors
#' @param direction character belonging to c("auto", "<", ">"). control (0) < case (1)
#' no character vectors allowed to specific multiple different directions!
#' @param pauc.limits numeric vector of length 2 indicating limits
#' @param show.names logical indicating whether to include names in AUC legend
#' @param plot.names character containing names of each plot in same order as predictor.
#' Default NULL, predictor used for names
ggplot_roc <- function(
  X, response, predictor,
  direction = "auto",
  pauc.limits = FALSE,
  pauc.axis = c("specificity", "sensitivity"),
  pauc.correct = TRUE,
  plot.names = NULL,
  show.names = TRUE,
  return.auc = FALSE,
  lwd = 1
) {
  pauc.axis = match.arg(pauc.axis)
  
  auc_caption <- "AUC"
  if (!is.logical(pauc.limits)) {
    # If pAUC limits are provided
    pauc.limits <- sort(pauc.limits, decreasing = TRUE)
    auc_caption <- "pAUC"
  }
  
  if (length(predictor) == 1) {
    # Single ROC curve
    roc_objs <- pROC::roc(
      X[[response]],
      X[[predictor]],
      direction = direction,
      partial.auc = pauc.limits,
      partial.auc.focus = pauc.axis,
      partial.auc.correct = pauc.correct
    )
    aucs <- roc_objs$auc
    d <- data.frame(
      FPR = 1 - roc_objs$specificities,
      TPR = roc_objs$sensitivities
    )
    d <- d[nrow(d):1, ]
    d <- cbind(names = predictor, d)
  } else if (length(predictor) > 1) {
    # To ensure that direction can be subsetted
    if (length(direction) == 1) {
      direction <- rep(direction, length(predictor))
    } else if (length(predictor) != length(direction)) {
      stop("Length of direction not equals to that of predictor!")
    }
    
    roc_objs <- lapply(seq_along(predictor),
      function(i) {
        pROC::roc(
          X[[response]],
          X[[predictor[i]]],
          direction = direction[i],
          partial.auc = pauc.limits,
          partial.auc.focus = pauc.axis,
          partial.auc.correct = pauc.correct
        )
      }
    )
    aucs <- sapply(roc_objs, function(obj) obj$auc)
    list_d <- lapply(roc_objs, function(obj) data.frame(
      FPR = 1 - obj$specificities,
      TPR = obj$sensitivities
    ))
    list_d <- lapply(list_d, function(d) d[nrow(d):1, ])
    d <- do.call(rbind, list_d)
    n_rows <- sapply(list_d, nrow)
    predictors_col <- rep(predictor, n_rows)
    d <- cbind(names = predictors_col, d)
  } else {
    stop("arg predictor is of non-positive length.")
  }
        
  ## Plot labels
  if (show.names) {
    if (!is.null(plot.names)) {
      if (length(plot.names) != length(predictor))
        stop("length of plot names does not match number of predictors!")
      
      # plot.names in same order as predictor
      labels <- sprintf("%s (%s: %.3f)", plot.names, auc_caption, aucs)
      labels[is.na(aucs)] <- plot.names[is.na(aucs)]
    } else {
      labels <- sprintf("%s (%s: %.3f)", predictor, auc_caption, aucs)
      labels[is.na(aucs)] <- predictor[is.na(aucs)]
    }
  } else {
    labels <- sprintf("%s = %.3f", auc_caption, aucs)
    labels[is.na(aucs)] <- "pAUC = NA"
  }
  # Order according to lexical order of predictor
  labels <- labels[order(predictor)]

  ax_roc <- ggplot() +
    geom_segment(
      aes(x = 0, y = 0, xend = 1, yend = 1),
      inherit.aes = FALSE,
      lty = "dotted", lwd = lwd,
      colour = "black", alpha = .4
    ) +
    geom_line(
      data = d,  # to avoid mutiple plotting of geom_segment
      aes(x = FPR, y = TPR, col = names),  # OPTION lty = names
      direction = "hv", lwd = lwd
    ) +
    scale_color_discrete(
      name = element_blank(),
      label = labels
    ) +
#     scale_linetype_discrete(
#       name = element_blank(),
#       label = labels
#     ) +
    theme_bw() +
    labs(x = "FPR", y = "TPR") +
    theme(
      legend.position = c(.95, .05),
      legend.justification = c("right", "bottom"),
      legend.background = element_rect(fill = NA)
    )
  
  if (is.logical(pauc.limits)) {
    # if no pauc.limits is provided
    ax_roc <- ax_roc +
      coord_cartesian(xlim = c(0, 1)) +
      coord_cartesian(ylim = c(0, 1))
  } else if (pauc.axis == "specificity") {
    fpr_limits <- 1 - pauc.limits
    d_rect <- data.frame(
      xmin = fpr_limits[1], xmax = fpr_limits[2],
      ymin = 0, ymax = 1
    )
    ax_roc <- ax_roc +
#       coord_cartesian(xlim = fpr_limits) +
      geom_rect(
        data = d_rect,
        aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
        fill = "blue", alpha = 0.2
      )
  } else if (pauc.axis == "sensitivity") {
    d_rect <- data.frame(
      xmin = 0, xmax = 1,
      ymin = pauc.limits[2], ymax = pauc.limits[1]
    )
    ax_roc <- ax_roc +
#       coord_cartesian(ylim = pauc.limits) +
      geom_rect(
        data = d_rect,
        aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
        fill = "blue", alpha = 0.2
      )
  }
  
  if (return.auc) {
    return(list(plot = ax_roc, auc = aucs))
  }
  ax_roc
}


#' Plots vectors from PCA coordinates (3 time points)
#'
#' Assumption: Patients are ordered from earlier to later timepoints 
#'
#' @param sce SingleCellExperiment with computed PCA coordinates 
#' @param xidx numeric indicating which PC to plot on the x-axis
#' @param yidx numeric indicating which PC to plot on the y-axis
#' @param pid_remove character of patient IDs to remove 
#'
#' @importFrom reshape reshape
#'
plot_vectors <- function(
  sce,
  pid_remove = pid_less3tp,
  pcs = c(1, 2),
  cex = 2, alpha = 0.5
) {
  # Plot labels
  xidx <- pcs[1]
  yidx <- pcs[2]
  # Assumption: sce has not been subsetted (i.e. percentVar is present)
  pct_var <- attr(reducedDim(sce, "PCA"), "percentVar")
  x_lab <- sprintf("PC%d (%.2f%%)", xidx, pct_var[xidx])
  y_lab <- sprintf("PC%d (%.2f%%)", yidx, pct_var[yidx])
  sce <- sce[, !(sce$sn %in% pid_remove)]
  sce <- sce[, order(colnames(sce))]
  # getting plot limits
  Z <- reducedDim(sce, "PCA")
  xmax <- max(Z[, xidx]) + 0.5
  xmin <- min(Z[, xidx]) - 0.5
  ymax <- max(Z[, yidx]) + 0.5
  ymin <- min(Z[, yidx]) - 0.5
  sces <- split_cols(sce, sce$final_label)
  get_vector <- function(sce) {
    data <- data.frame(
      sn = sce$sn,
      period = sce$period,
      reducedDim(sce)[, c(xidx, yidx)]
    )
    stats::reshape(
      data, idvar = "sn", timevar = "period", direction = "wide"
    )
  }
  vector_grps <- lapply(sces, get_vector)
  .plot_vector <- function(x) {
    if (any(endsWith(colnames(x), "6"))) {
      # cvt patients
      ftnames <- sprintf(
        "PC%d.%d",
        rep(c(xidx, yidx), 3),
        rep(c(0, 6, 24), each = 2)
      )
    } else {
      ftnames <- sprintf(
        "PC%d.%d",
        rep(c(xidx, yidx), 3),
        rep(c(0, 12, 24), each = 2)
      )
    }
    ax <- ggplot(x) +
      geom_point(
        aes_string(x = ftnames[1], y = ftnames[2]), 
        shape = 21, size = cex, fill = "red", show.legend = T
      ) +
      geom_point(
        aes_string(x = ftnames[3], y = ftnames[4]),
        shape = 21, size = cex, fill = "green", show.legend = F
      ) +
      geom_point(
        aes_string(x = ftnames[5], y = ftnames[6]),
        shape = 21, size = cex, fill = "blue", show.legend = F
      ) +
      geom_segment(
        aes_string(
          x = ftnames[1], y = ftnames[2],
          xend = ftnames[3], yend = ftnames[4]
        ),
        arrow = arrow(length = unit(0.3, "cm")), col = "red", alpha = alpha
      ) +
      geom_segment(
        aes_string(
          x = ftnames[3], y = ftnames[4],
          xend = ftnames[5], yend = ftnames[6]
        ),
        arrow = arrow(length = unit(0.3, "cm")), col = "green", alpha = alpha
      ) +
      geom_segment(
        aes_string(
          x = ftnames[1], y = ftnames[2],
          xend = ftnames[5], yend = ftnames[6]
        ),
        arrow = arrow(length = unit(0.3, "cm")), col = "blue", alpha = alpha
      ) +
      # geom_text(
      #   aes(x = ftnames[1], y = ftnames[2], label = rownames(x)),
      #   position = position_nudge(x = 4, y = 2), size = 2.5
      # ) +
      # scale_color_manual(values = COL_LABEL) +
      xlab(x_lab) + ylab(y_lab) +
      xlim(xmin, xmax) + ylim(ymin, ymax)
    ax
  }
  lapply(vector_grps, .plot_vector)
}


# 3D PCA plot
rglplot_scatter <- function(
  df, colour = "blue", pch = 21, pc_labels = NULL,
  ratio_list = list(2,1,1)
) {
  # RGL plot parameters
  rgl.open()
  rgl.bg(color="white")
  rgl.viewpoint(zoom = 0.8)
  # rgl.viewpoint(theta = 110, phi = 5, zoom = 0.8)
  par3d(windowRect = c(50, 20, 500, 500))
  pch3d(df[,1], df[,2], df[,3], bg = colour,
        pch = pch, cex = 0.5, lwd = 1.5)
  box3d(col = "black")
  # title3d(xlab = pc_labels[1], ylab = pc_labels[2],
  #         zlab = pc_labels[3], col = "black")
  # Plot aspect ratios of axis according to variance
  do.call(aspect3d, ratio_list)
}


# 3D PCA plot
plot3d_pca <- function(
  df, colour, pch, pc_labels = NULL,
  cex = 0.5, lwd = 0.5,
  width = 600, height = 400,
  ratio_list = list(2, 1, 1)
) {
  if (is.null(pc_labels)) {
    print("PCA performed!")
    pca_obj <- prcomp(t(df)) 
    pca_df <- as.data.frame(pca_obj$x[,1:3])
    eigenvalues <- (pca_obj$sdev)^2
    var_pc <- eigenvalues[1:3]/sum(eigenvalues)
    pc_labels <- sprintf("PC%d (%.2f%%)", 1:3, var_pc*100)
  } else {
    print("No PCA performed!")
    pca_df <- as.data.frame(df)
  }
  
  # RGL plot parameters
  if (rgl.cur() == 0) {
    rgl.open()
    rgl.bg(color = "white")
  }
  rgl.clear(type = "all")  
  # rgl.viewpoint(zoom = 0.8)
  # rgl.viewpoint(theta = 110, phi = 5, zoom = 0.8)
  
  # figure size
  window <- c(0, 0, width, height) + 50
  par3d(windowRect = window) 
  
  with(pca_df, pch3d(
    PC1, PC2, PC3,
    bg = colour, pch = pch,
    cex = cex, lwd = 0.5
  ))
  box3d(col = "black")
  title3d(
    xlab = pc_labels[1], ylab = pc_labels[2],
    zlab = pc_labels[3], col = "black"
  )
  
  # Plot aspect ratios of axis according to variance
  do.call(aspect3d, ratio_list)
}


# Plot venn diagram
jacc_coeff <- function(vec1, vec2) {
  # Generate overlap list
  overlap_list <- calculate.overlap(list(vec1,vec2))
  # Calculate venndiagram areas
  venn_area <- sapply(overlap_list, length)
  grid.newpage()
  venn_plot <- draw.pairwise.venn(venn_area[1], venn_area[2], venn_area[3],
                                  category = c("D1", "D2"),
                                  cex = 3, fontfamily = "sans",
                                  cat.cex = 3, cat.fontfamily = "sans",
                                  margin = 0.1)
  union <- (venn_area[1] + venn_area[2] - venn_area[3])
  print(unname(venn_area[3]/union))
  return(venn_plot)
}


#' Get column names in same order as pheatmap plot
#' @param obj pheatmap object
get_colnames <- function(obj) {
  obj$tree_col$labels[obj$tree_col$order]
}


#' Get row names in same order as pheatmap plot
#' @param obj pheatmap object
get_rownames <- function(obj) {
  obj$tree_row$labels[obj$tree_row$order]
}


#' Generate default ggplot colours
ggplot_palette <- function(n) {
  hues = seq(15, 375, length = n + 1)
  return(hcl(h = hues, c = 100, l = 65)[1:n])
}
