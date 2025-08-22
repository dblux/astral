library(data.table)
library(ggplot2)
library(PepMapViz)
library(jsonlite)


# Require: sample_id, class, timepoint, batch(?)
# Protein.Group, Genes, Precursor.Charge, Precursor.Normalised
file <- 'data/astral/processed/report-matched-M0-P19827.csv'
report <- read.csv(
  file, header = TRUE,
  stringsAsFactors = FALSE, check.names = FALSE
)
data <- as.data.table(report)

# Exon information
file <- 'data/tmp/spliceoforms/itih1-1.json'
itih1_1 <- fromJSON(file)
print(itih1_1)
names(itih1_1)

# Skipped functions: strip_sequence_DIANN, obtain_mod_DIANN 
# No need to strip sequence as we have that already
# No need to obtain PTM because we do not want to plot

whole_seq <- data.frame(
  Region_Sequence = c(itih1_1$sequence),
  Genes = c("ITIH1")
)
print(whole_seq)

matching_result <- match_and_calculate_positions(
  data,
  column = "Stripped.Sequence", # name of column with sequence
  whole_seq,
  match_columns = c("Genes"), # match column from data with whole_seq
  column_keep = colnames(report)[1:3] # information to keep 
)
head(matching_result)
head(whole_seq)

# TODO: Raise issue - Allow matching_result to not have reps and PTM_position
matching_result$reps <- NA
matching_result$PTM_position <- NA
data_with_quantification <- peptide_quantification(
  whole_seq,
  matching_result,
  matching_columns = c("Genes"),
  distinct_columns = colnames(report)[1:3], # not aggregated for these conditions
  quantify_method = "PSM",
  # area_column = "Precursor.Normalised",
  with_PTM = FALSE,
  reps = FALSE
)
head(data_with_quantification)

# Domain information
domain <- itih1_1$exons
domain["domain_color"] <- c(rep("#F8766D", 8), rep("#B79F00", 14)) 
domain

p <- create_peptide_plot(
  data_with_quantification,
  y_axis_vars = c("label"),
  x_axis_vars = NULL, # Facets according to region. Option: NULL 
  y_expand = c(0.2, 0.2),
  x_expand = c(0.5, 0.5),
  theme_options = list(),
  labs_options = list(title = "PSM Plot", x = "Position", fill = "PSM"),
  color_fill_column = 'PSM',
  fill_gradient_options = list(),
  label_size = 5,
  add_domain = TRUE, # TODO: Even if FALSE, domain will be plotted
  domain = domain, # TODO: ERROR if domain not provided
  domain_start_column = "start",
  domain_end_column = "end",
  domain_type_column = "exon",
  domain_fill_color_column = "domain_color",
  # PTM = FALSE,
  # PTM_type_column = "PTM_type",
  # PTM_color = PTM_color,
  # add_label = TRUE,
  # label_column = "Character",
  # column_order = list(Region_1 = 'VH,VL')
)
file = "tmp/astral/fig/spliceoforms/peptides-ITIH1-id.pdf"
ggsave(p, filename = file, width = 10, height = 20)

