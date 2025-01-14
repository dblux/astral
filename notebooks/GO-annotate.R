library(biomaRt)
library(magrittr)
library(ggplot2)

file <- "data/entropic_genes-002.txt" 
entropic_genes <- readLines(file)

ensembl <- useMart(
  "ensembl", dataset = "hsapiens_gene_ensembl"
  # host = "https://asia.ensembl.org"
)

annots <- c("ensembl_gene_id", "go_id", "name_1006", "namespace_1003") # "definition_1006"
mapping <- getBM(
  attributes = annots,
  filters = "ensembl_gene_id",
  values = entropic_genes,
  mart = ensembl
)
# Some genes do not map to any GO while some genes map to multiple GO
mapping[mapping == ""] <- NA
mapping_omit <- na.omit(mapping)

writeLines(unique(mapping_omit$name_1006), "tmp/go-names.txt")

freq <- head(sort(table(mapping1$go_id), decreasing = TRUE), 20)
go_names <- mapping1$name_1006[match(names(freq), mapping1$go_id)]
go <- data.frame(go_names, count = as.numeric(freq))

ax <- ggplot(go) +
  geom_col(aes(
    x = count,
    y = reorder(go_names, count, sum)
  )) +
  labs(y = "GO")
file <- "fig/GO-20.pdf"
ggsave(file, ax, width = 6, height = 4)

# Annotations
description <- listAttributes(ensembl, what = "description")
description[1:263] # most important attributes
description[44:48] # GO terms
attributes <- listAttributes(ensembl) # returns data frame
attributes[44:48,]
