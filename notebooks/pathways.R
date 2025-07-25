library(KEGGREST)
library(biomaRt)
library(magrittr)
library(ggplot2)
theme_set(theme_bw(base_size = 7))


##### Load data #####

file <- 'data/astral/processed/combat_knn5_lyriks-605_402.csv'
lyriks <- read.csv(file, row.names = 1)
uids_lyriks  <- rownames(lyriks)

file <- 'data/astral/raw/reprocessed-data.csv'
reprocessed <- read.csv(file, row.names = 1)
annot <- reprocessed[1:2]

### Load biomarkers
file <- 'tmp/astral/lyriks402/new/biomarkers/biomarkers-elasticnet.csv'
bm_enet <- read.csv(file, row.names = 1)
uid_enet <- rownames(bm_enet)

file <- 'tmp/astral/lyriks402/new/biomarkers/biomarkers-ancova.csv'
bm_ancova <- read.csv(file, row.names = 1)
uid_ancova <- rownames(bm_ancova)

file <- 'tmp/astral/lyriks402/new/biomarkers/mongan-etable5.csv'
mongan <- read.csv(file, row.names = 1)
uid_mongan <- rownames(mongan)[mongan$q < 0.05]
uid_mongan33 <- uid_mongan[uid_mongan %in% uids_lyriks]
length(uid_mongan33)

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
  res <- list(
    kegg_ids = kegg_ids,
    kegg_pathways = kegg_pathways,
    kegg_freq = kegg_freq
  )
  res
}

ancova_res <- kegg_frequency(uid_ancova)
enet_res <- kegg_frequency(uid_enet)
mongan_res <- kegg_frequency(uid_mongan33)

hsa_pathways <- keggList('pathway', 'hsa')

uniprot_2_kegg <- function(
  gene_symbol, res
) {
  idx <- match(gene_symbol, annot$Gene)
  stopifnot(is.numeric(idx))
  uid <- rownames(annot)[idx]
  uid <- paste0('up:', uid)
  if (!(uid %in% names(res$kegg_ids)))
    stop(sprintf('UniProt ID %s not found in KEGG mapping', uid))
  kegg_id <- res$kegg_ids[uid]
  message(sprintf('UniProt ID %s maps to KEGG ID %s', uid, kegg_id))
  if (!kegg_id %in% names(res$kegg_pathways))
    stop(sprintf('KEGG ID %s not found in KEGG pathways', kegg_id))
  idx <- names(res$kegg_pathways) == kegg_id
  pathways <- res$kegg_pathways[idx]
  pathways <- substring(gsub(':', '', pathways), 5)
  hsa_pathways[pathways]
}

prots <- c(
  'PCYOX1', 'LCAT', 'F9', 'RPSA', 'ZNF607', 'KRT9', 'C1S', 'ITIH4', 'APOB',
  'SERPINA4', 'PZP', 'PON1', 'CALU', 'KRT14', 'IGKV1-5'
)

res_lyriks <- mapply(c, ancova_res[1:2], enet_res[1:2])

for (prot in prots) {
  print(prot)
  try(print(uniprot_2_kegg(prot, res_lyriks)))
  cat('==========\n\n')
}

uniprot_2_kegg('PCYOX1', ancova_res)

annot['P04114',]
any(rownames(annot) == 'P05546')

ancova_res$kegg_pathways
enet_res$kegg_pathways
enet_kegg <- enet_res$kegg_ids
mongan_kegg <- mongan_res$kegg_ids

# gene_info$ORTHOLOGY
# gene_info$BRITE
# gene_info$NETWORK
# gene_info$PATHWAY

# Retrieve gene info and extract KO term
# Substitute with protein from KEGG pathway if present, else from KO
# Identify number of terms that cannot be substituted
uniprot_2_ko <- function(uids) {
  results <- lapply(uids, function(x) {
    list(
      uniprot = x,
      keggid = NA, 
      # gene_info = NA, 
      pathways = NA, 
      pathway_constituents =  NA,
      ko =  NA,
      ko_alternatives = NA, 
      replacement = NA 
    )
  })
  names(results) <- uids
  results <- lapply(results, function(res) {
    print(res$uniprot)
    # Convert to KEGG ID
    res$keggid <- keggConv('genes', paste0('uniprot:', res$uniprot))
    # print(res$keggid)
    if (length(res$keggid) == 0) {
      warning('No KEGG ID found for UniProt ID: ', res$uniprot)
      return(res)
    }
    gene_info <- keggGet(res$keggid)[[1]]
    pathways <- gene_info$PATHWAY
    if (!is.null(pathways)) {
      pathway <- sample(pathways, 1)
      res$pathways <- pathways
      names(res$pathways) <- pathways 
      # TODO: Save chosen pathway and pick randomly
    } else {
      warning('No pathways found for KEGG ID: ', res$keggid)
      # TODO: What if there is no KO term or mutiple KOs?
      if (is.null(gene_info$ORTHOLOGY)) {
        warning('No KO term found for KEGG ID: ', res$keggid)
        return(res)
      }
      res$ko <- names(gene_info$ORTHOLOGY)
      # Get information of KO term
      ko_info <- keggGet(res$ko)[[1]]
      ko_genes <- ko_info$GENES
      # extract strings starting with 'HSA'
      ko_hsa <- grep('HSA:', ko_genes, value = TRUE)
      stopifnot(length(ko_hsa) == 1)
      ko_kids_symbols <- strsplit(substring(ko_hsa, 6), ' ')[[1]]
      ko_kids <- sapply(ko_kids_symbols, function(x) {
          kid <- paste0('hsa:', sub('\\(.*', '', x))
          gene_symbol <- sub('.*\\((.*)\\)', '\\1', x)
          names(kid) <- gene_symbol
          return(kid)
      }, USE.NAMES = FALSE)
      res$ko_alternatives <- setdiff(ko_kids, res$keggid)
      if (length(res$ko_alternatives) == 0) {
        warning('No alternative KO terms found for KEGG ID: ', res$keggid)
      }
    }
    return(res)
  })
  return(results)
}

keggLink('hsa04610')
keggLink('pathway','hsa:01100')

res_enet <- uniprot_2_ko(uid_enet)
length(res_unet)
str(res_enet)

res_ancova <- uniprot_2_ko(uid_ancova)
length(res_ancova)
str(res_ancova)

res_mongan33 <- uniprot_2_ko(uid_mongan33)
str(res_mongan33)

for (res in res_ancova) {
  print(res$keggid)
  print('==========')
  if (class(res$pathways) != 'character') {
    print('No pathways found')
    if (length(res$ko_alternatives) < 2) {
      print("No alternative KO!!!")
    }
  }
}


# some uniprot IDs have no KEGG IDs
# some KEGG IDs have no pathways
# total number of KEGG pathways - 5

# biggest problem: alot of KEGG IDs have no corresponding pathways
# KO terms have no alternative IDs as well

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


##### Enrichment analysis #####

# N = 605 proteins

### KEGG

# M = ? pathways
# Correct for multiple testing for m pathways

# Get protein-pathways mapping

#' Compute KEGG pathway frequencies of UniProt IDs 
#'
#' @param uniprot_ids UniProt IDs
#' @param n numeric of top number of rows to show
kegg_mapping <- function(uniprot_ids, api_limit = 100) {
  # check that uniprot IDs are all unique
  stopifnot(length(unique(uniprot_ids)) == length(uniprot_ids))
  ids <- paste0('up:', uniprot_ids)
  m <- length(ids)
  # Some uniprot IDs will not have KEGG IDs (many-to-one)
  # Some KEGG IDs will not map to any pathways (one-to-many)
  id_mapping <- rep(NA, m) 
  names(id_mapping) <- ids
  kegg_ids <- keggConv('genes', ids)
  id_mapping[names(kegg_ids)] <- kegg_ids
  pathway_mapping <- rep(list(c()), length(kegg_ids))
  names(pathway_mapping) <- kegg_ids
  # keggLink may have hard-coded limit of 100 IDs
  if (length(kegg_ids) > api_limit) {
    print('luffy!')
    # Split IDs into groups of 100 to avoid limit
    ngrps <- ceiling(length(kegg_ids) / api_limit)
    groups <- gl(ngrps, api_limit, length(kegg_ids))
    list_ids <- split(kegg_ids, groups)
    for (grp in list_ids) {
      # KEGG gene IDs can have zero or multiple pathways 
      kegg_pathways <- keggLink('pathway', grp)
      for (i in seq_len(length(kegg_pathways))) {
        pathway_mapping[[names(kegg_pathways)[i]]] <- c(
          pathway_mapping[[names(kegg_pathways)[i]]],
          kegg_pathways[i]
        )
      }
    }
  } else {
    kegg_pathways <- keggLink('pathway', kegg_ids)
    for (i in seq_len(length(kegg_pathways))) {
      pathway_mapping[[names(kegg_pathways)[i]]] <- c(
        pathway_mapping[[names(kegg_pathways)[i]]],
        kegg_pathways[i]
      )
    }
  }
  return(list(id = id_mapping, pathway = pathway_mapping))
}

analyse_mapping <- function(maps) {
  # Some uniprot IDs have no pathways
  # (1) Have no KEGG ID (2) Corresponding KEGG ID has no pathways
  upids <- names(maps$id)
  upids_no_keggid <- upids[is.na(maps$id)]
  keggids_no_pathway <- names(Filter(is.null, maps$pathway))
  # some uniprot ids map to the same kegg id -> treat as one protein
  upids_no_pathway <- upids[maps$id %in% keggids_no_pathway]
  upids_remove <- union(upids_no_keggid, upids_no_pathway)
  list_pathways <- Filter(Negate(is.null), maps$pathway)
  pathways <- unique(unname(do.call(c, list_pathways)))
  # Filter pathways that are COVID
  # remove pathways 
  return(pathways)
}

head(prots_lyriks605, 30)
prots1 <- prots_lyriks605[1:25]
keggConv('genes', paste0('up:', prots1))


mapping_605 <- kegg_mapping(prots_lyriks605)

file <- 'tmp/astral/mapping_605.rds'
saveRDS(mapping_605, file)

print(mapping_605$id)
length(unique(c(1,2,NA)))
table(mapping_605$id)
sum(!is.na(mapping_605$id))
print(mapping_605$id)
print(mapping_605$pathway)
names(Filter(Negate(is.null), mapping_605$pathway))
Filter(!is.null, mapping_605$pathway)

analyse_mapping(mapping_605)
