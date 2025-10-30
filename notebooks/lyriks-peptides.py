import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

from notebooks.getters import get_isoforms, get_exon_coordinates

file = 'data/astral/processed/combat_knn5_lyriks-605_402.csv'
lyriks = pd.read_csv(file, index_col=0, header=0).T

file = 'data/astral/processed/metadata-lyriks407.csv'
md = pd.read_csv(file, index_col=0, header=0)
md = md[md.label != 'QC']
md['period'] = md['period'].astype(int)

# Check metadata
filepath = 'data/astral/metadata/metadata-all.csv'
metadata = pd.read_csv(filepath, index_col=0)
metadata.Study.value_counts()

filepath = 'data/astral/raw/reprocessed-data.csv'
reprocessed = pd.read_csv(filepath, index_col=0)
reprocessed.head()

filepath = 'data/tmp/spliceoforms/venn-69.txt'
with open(filepath, 'r') as f:
    venn69 = [line.strip() for line in f.readlines()] 

filepath = 'data/tmp/spliceoforms/venn-10.txt'
with open(filepath, 'r') as f:
    venn10 = [line.strip() for line in f.readlines()] 

uniprot_symbol_map = dict(zip(reprocessed.index, reprocessed.Gene))
lyriks_symbols = lyriks.columns.map(uniprot_symbol_map)
in_lyriks = [symbol in lyriks_symbols for symbol in venn69]
print(in_lyriks)

# Model 1A: cvt (M0) v.s. non-cvt (M0)
# Prognostic markers
# Only M0 and exclude ctrl samples
md_1a = md[(md.final_label != 'ctrl') & (md.period == 0)]
md_1a.final_label.replace({'rmt': 0, 'mnt': 0, 'cvt': 1}, inplace=True)
lyriks_1a = lyriks.loc[md_1a.index]
print(md_1a.final_label.value_counts()) # imbalanced

# Psychosis prognostic (ANCOVA, BH)
# Dataframe with proteomic and clinical features
# relapse is labelled as mnt
data = pd.concat([lyriks_1a, md_1a[['final_label', 'age', 'gender']]], axis=1)
print(data.final_label.value_counts()) # imbalanced
data.head()

pvalues = []
coeffs = []
for prot in data.columns[:-3]:
    model = ols(
        f'{prot} ~ final_label + age + gender',
        data=data
    ).fit()
    pvalues.append(model.pvalues['final_label'])
    coeffs.append(model.params['final_label'])
    # table = sm.stats.anova_lm(model, typ=2)

_, qvalues, _, _ = multipletests(pvalues, alpha=0.05, method='fdr_bh')
stats = pd.DataFrame(
    {'Coefficient': coeffs, 'p': pvalues, 'q': qvalues},
    index=lyriks_symbols
)
stats.head()

# Divide into two groups
stats.loc[venn69, :].to_csv('tmp/astral/peptides/stats-venn69.csv')
stats.loc[venn10, :].to_csv('tmp/astral/peptides/stats-venn10.csv')

### Compare 69 v.s. 10
filepath = 'data/tmp/spliceoforms/comparison-69_10.csv'
comparison = pd.read_csv(filepath, index_col=0)
comparison.columns
comparison.head()

comparison1 = comparison.merge(
    stats[['q']], left_index=True, right_index=True
)
comparison1.columns
comparison1.avg_qval < 0.05 & comparison1.q > 0.05 

sns.stripplot(
    data=comparison,
    x='group',
    y='entropy',
    hue='has_isoforms',
)
plt.show()


sns.scatterplot(
    data=comparison,
    x='avg_qval',
    y='entropy',
    hue='has_isoforms',
    style='group',
    palette={True: 'red', False: 'grey'},
    alpha=0.7,
    edgecolor=None,
)
filepath = "tmp/astral/peptides/fig/scatter-avg_qval_entropy.png" 
plt.savefig(filepath, dpi=300, bbox_inches="tight")
plt.close()

sns.scatterplot(
    data=comparison1,
    x='nonDE_pct',
    y='q',
    hue='has_isoforms',
    style='group',
    palette={True: 'red', False: 'grey'},
    alpha=0.7,
    edgecolor=None,
)
filepath = "tmp/astral/peptides/fig/scatter-protein_peptide.png" 
plt.savefig(filepath, dpi=300, bbox_inches="tight")
plt.close()

plt.show()

itih1_1_uid = "P19827-1"
itih2_1_uid = 'P19823'
results = get_isoforms(itih1_1_uid)
results

kng1_2_uid = "P01042-2"
exons = get_exon_coordinates(kng1_2_uid)
print(exons)



### Peptide-level analysis

file = "data/tmp/zhihao/gene-entropy_isoform.tsv"
genes = pd.read_table(file, index_col=0)
genes.head()

file = 'data/tmp/biomarkers/biomarkers-ancova.csv'
bm_ancova = pd.read_csv(file, index_col=2)
bm_ancova.head()

file = 'data/tmp/biomarkers/biomarkers-elasticnet.csv'
bm_enet = pd.read_csv(file, index_col=2)
bm_enet.head()

genes['in_ancova'] = genes.index.isin(bm_ancova.index)
genes['in_enet'] = genes.index.isin(bm_enet.index)

file = 'tmp/astral/peptides/bm-ancova.csv'
genes.loc[genes.in_ancova].drop(['in_ancova', 'in_enet'], axis=1).to_csv(file)

file = 'tmp/astral/peptides/bm-enet.csv'
genes.loc[genes.in_enet].drop(['in_ancova', 'in_enet'], axis=1).to_csv(file)

# Retaining proteins with at least n peptides (for entropy)
n_total = 6
genes_supp = genes[genes.Total >= n_total]
genes_supp.has_isoform.value_counts() # T: 98, F: 118

# Plot no. of peptides mapping to proteins
isoforms_20 = genes.loc[genes.has_isoform].sort_values('DE_count', ascending=False).head(20)
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(
    data=isoforms_20,
    x=isoforms_20.index,
    y='DE_count',
    hue='has_isoform',
    ax=ax,
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
ax.set_xlabel('')
ax.set_ylabel('Number of DE peptides')
filepath = f"tmp/astral/peptides/fig/barplot-isoforms20.png"
plt.savefig(filepath, dpi=300, bbox_inches="tight")

sns.histplot(
    data=genes_supp,
    x='Entropy',
    hue='has_isoform',
    # stat='density',
    # common_norm=True,
    bins=15,
    multiple='dodge',
)
plt.title(f'Proteins (>= {n_total} peptides)')
filepath = f"tmp/astral/peptides/fig/hist-entropy.png"
plt.savefig(filepath, dpi=300, bbox_inches="tight")

# Comparing proteins in LYRIKS (ANCOVA) v.s. rest
# bm_ancova.index.isin(genes.index).sum()  # 15
genes_supp['in_ancova'] = genes_supp.index.isin(bm_ancova.index)
ctab = pd.crosstab(genes_supp.has_isoform, genes_supp.in_ancova)
print(ctab)
genes_supp.loc[genes_supp.in_ancova, ["Entropy", "has_isoform"]]
genes_supp.loc[genes_supp.in_ancova]

plt.figure(figsize=(6, 3))
ax = sns.histplot(
    data=genes_supp,
    x='Entropy',
    hue='in_ancova',
    multiple='dodge',
)
ax.get_legend().set_title('In ANCOVA')
plt.title(f'Proteins (>= {n_total} peptides)')
# save fig
filepath = "tmp/astral/peptides/fig/hist_lyriks-entropy.png"
plt.savefig(filepath, dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 3))
ax = sns.kdeplot(
    data=genes_supp,
    x='Entropy',
    hue='in_ancova',
    common_norm=False,
    clip=(0, 1),
    palette={True: 'red', False: 'gray'},
)
ax.get_legend().set_title('In ANCOVA')
plt.title(f'Proteins (>= {n_total} peptides)')
# save fig
filepath = "tmp/astral/peptides/fig/kde_lyriks-entropy.png"
plt.savefig(filepath, dpi=300, bbox_inches="tight")
plt.close()


