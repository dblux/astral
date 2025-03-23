library(KEGGREST)
library(biomaRt)
library(magrittr)
library(ggplot2)
theme_set(theme_bw(base_size = 7))


##### Load data #####

file <- 'data/astral/processed/combat_knn5_lyriks.csv'
lyriks <- read.csv(file, row.names = 1)
prots_lyriks <- rownames(lyriks)

file <- 'tmp/astral/1a-prognostic_ancova.csv'
prognostic_1a <- read.csv(file, row.names = 1)
prots_prognostic_1a <- rownames(prognostic_1a)[prognostic_1a$p < 0.01]
length(prots_prognostic_1a)

file <- 'tmp/astral/1a-psychotic_ttest.csv'
psychotic_1a <- read.csv(file, row.names = 1)
prots_psychotic_1a <- rownames(psychotic_1a)[psychotic_1a$p < 0.01]
length(prots_psychotic_1a)

file <- 'tmp/astral/2b-prognostic_ancova.csv'
prognostic_2b <- read.csv(file, row.names = 1)
prots_prognostic_2b <- rownames(prognostic_2b)[prognostic_2b$p < 0.01]
length(prots_prognostic_2b)

file <- 'tmp/astral/uhr_3a-ancova.csv'
uhr_3a <- read.csv(file, row.names = 1)
prots_uhr_3a <- rownames(uhr_3a)[uhr_3a$p < 0.01]
length(prots_uhr_3a)

file <- 'data/astral/etc/mongan-etable5.csv'
mongan <- read.csv(file, row.names = 1)
mongan_prots <- rownames(mongan)[mongan$q < 0.05]
length(mongan_prots)

file <- 'data/astral/etc/perkins.csv'
perkins <- read.csv(file, row.names = 1)
perkins_prots <- perkins$UniProt.CAS[startsWith(perkins$UniProt.CAS, 'P')]
length(perkins_prots)

file <- 'tmp/astral/biomarkers-1a-prognostic_ancova.csv'
file <- 'tmp/astral/biomarkers-1a-psychotic_ttest.csv'
file <- 'tmp/astral/biomarkers-2b-prognostic_ancova.csv'
biomarkers <- read.csv(file, row.names = 1)

##### KEGG ######

#' Compute KEGG pathway frequencies of UniProt IDs 
#'
#' @param uniprot_ids UniProt IDs
#' @param n numeric of top number of rows to show
kegg_frequency <- function(uniprot_ids) {
  kegg_ids <- keggConv('genes', paste0('uniprot:', uniprot_ids))
  # keggLink may have hard-coded limit of 100 IDs
  kegg_limit <- 100
  if (length(kegg_ids) > kegg_limit) {
    # Split IDs into groups of 100 to avoid limit
    ngrps <- ceiling(length(kegg_ids) / kegg_limit)
    groups <- gl(ngrps, kegg_limit, length(kegg_ids))
    list_ids <- split(kegg_ids, groups)
    kegg_pathways <- character()
    for (grp in list_ids) {
      # KEGG gene IDs can have zero or multiple pathways 
      kegg_pathways <- c(kegg_pathways, keggLink('pathway', grp))
    }
  } else {
    kegg_pathways <- keggLink('pathway', kegg_ids)
    # print(kegg_ids %in% names(kegg_pathways))
  }
  pathway_freq <- table(kegg_pathways) %>%
    sort(decreasing = TRUE) %>%
    data.frame()
    # head(n)
  pathway_names <- keggList('pathway')
  names(pathway_names) <- names(pathway_names) %>%
    substring(4) %>%
    paste0('path:hsa', .)
  idx <- match(pathway_freq$kegg_pathways, names(pathway_names))
  kegg_names <- pathway_names[idx]
  kegg_freq <- cbind(name = kegg_names, pathway_freq)
  return(kegg_freq)
}

kegg_freq <- kegg_frequency(prots_lyriks)
m <- length(prots_lyriks)
head(kegg_freq)
print(m)

freq <- head(kegg_freq, 10)
ax <- ggplot(freq) +
  geom_col(aes(
    x = Freq,
    y = reorder(name, Freq, sum)
  )) +
  labs(
     title = sprintf('Astral proteins (m = %d)', m),
     y = 'KEGG'
  ) +
  scale_x_continuous(breaks = seq(0, max(freq$Freq), by = 10))
file <- 'tmp/astral/fig/kegg-astral.pdf'
ggsave(file, ax, width = 4, height = 2)

# TODO: Fisher's exact test for each signature (only for KEGG)
# Filter only genes that map to KEGG
# Filter pathways that are COVID, too little proteins, pathways with small sizes?
dim(kegg_freq)
sum(kegg_freq$Freq > 1)


##### GO ######

### biomaRt ###

go_frequency <- function(uniprot_ids, mart) {
  attributes <- c(
    # 'ensembl_gene_id',
    'uniprot_gn_id',
    'go_id',
    'name_1006',
    'namespace_1003'
    # 'definition_1006'
  )
  mapping <- getBM(
    attributes = attributes,
    filters = 'uniprot_gn_id',
    values = uniprot_ids,
    mart = mart
  )
  # Genes may map to zero or multiple GO terms 
  mapping[mapping == ''] <- NA
  mapping <- na.omit(mapping)
  freq <- mapping['go_id'] %>%
    table() %>%
    sort(decreasing = TRUE)
    # head(n)
  go_names <- mapping[match(names(freq), mapping[['go_id']]), 'name_1006']
  go_freq <- as.data.frame(freq, stringsAsFactors = FALSE)
  colnames(go_freq) <- c('GO Term', 'Frequency')
  go_freq['Name'] <- go_names
  return(go_freq)
}

mart <- useEnsembl(
  biomart = "genes",
  dataset = "hsapiens_gene_ensembl",
  mirror = "useast"
)

go_freq <- go_frequency(prots_psychotic_1a, mart)
m <- length(prots_psychotic_1a)
head(go_freq)

freq <- head(go_freq, 10)
ax <- ggplot(freq) +
  geom_col(aes(
    x = Frequency,
    y = reorder(Name, Frequency, sum)
  )) +
  labs(
     title = sprintf('Psychosis conversion proteins (m = %d)', m),
     y = 'GO'
  ) +
  scale_x_continuous(breaks = seq(0, max(freq$Frequency), 1))
file <- 'tmp/astral/fig/GO-psychosis_conversion.pdf'
ggsave(file, ax, width = 4, height = 2)

# biomaRt attributes 
description <- listAttributes(ensembl, what = 'description')
description[1:263] # most important attributes
description[44:48] # GO terms
attributes <- listAttributes(ensembl) # returns data frame
attributes[44:48,]

library(org.Hs.eg.db)
library(AnnotationDbi)

# Annotate GO using org.Hs.eg.db
columns(org.Hs.eg.db)
go_annot <- select(
  org.Hs.eg.db, keys = ent_genes,
  columns = "GO", keytype = "ENSEMBL"
)

# GO annotation mapping
go_info <- as.list(GOTERM)

go_descriptions <- sapply(go_info[na.omit(go_annot$GO)], function(x) x@Term)

top_20  <- table(go_descriptions) %>%
  sort(decreasing = TRUE) %>%
  head(40) %>%
  data.frame()

ax <- ggplot(top_20) +
  geom_col(aes(
    x = Freq,
    y = reorder(go_descriptions, Freq, sum)
  )) +
  labs(y = "GO")
file <- "tmp/fig/GO-20.pdf"
ggsave(file, ax, width = 6, height = 4)

