import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc 
import statsmodels.api as sm

from itertools import chain
from dataclasses import dataclass, field
from scipy.stats import f, ttest_ind, ttest_rel, rankdata, norm
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, roc_auc_score
)
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from venn import draw_venn, generate_petal_labels, generate_colors, venn
from sklearn.covariance import EmpiricalCovariance, MinCovDet 
from matplotlib.patches import Ellipse


@dataclass
class Result:
    metadata: dict # required init 
    predictions: list = field(default_factory=list) 
    labels: list = field(default_factory=list) 
    probas: list = field(default_factory=list) 
    pvals: list = field(default_factory=list) 
    test_statistics: list = field(default_factory=list) 
    ranks: list = field(default_factory=list) 
    coefficients: list = field(default_factory=list) 
    inner_folds: list = field(default_factory=list) 
    features: list = field(default_factory=list) 

###### PCA #####

def plot_arrows(
    adata, colour=None, shape=None, size=None, head_width=0.05
):
    if 'X_pca' not in adata.obsm:
        raise ValueError('PCA not performed on AnnData yet!')
    adata.obs.period = adata.obs.period.astype(int)
    X_pca = adata.obsm['X_pca']
    # TODO: % var
    column_idx = [f'PC{i + 1}' for i in range(X_pca.shape[1])]
    data = pd.DataFrame(X_pca, index=adata.obs_names, columns=column_idx)
    data['period'] = adata.obs['period']
    data['sn'] = adata.obs['sn']
    if colour:
        data[colour] = adata.obs[colour]
    if shape:
        data[shape] = adata.obs[shape]
    patients = data['sn'].unique()
    get_colour = plt.colormaps['rainbow']
    n = len(patients)
    col_idx = np.linspace(0, 1, n)
    fig, ax = plt.subplots()
    for i, patient in zip(col_idx, patients):
        # Subset patient data and sort by time
        patient_data = data[data['sn'] == patient].sort_values(by='period')
        x = patient_data['PC1'].values
        y = patient_data['PC2'].values
        ax.scatter(
            x, y, color=get_colour(i), label=patient, alpha=0.6)
        for j in range(len(x) - 1):
            ax.arrow(
                x[j], y[j],
                x[j + 1] - x[j], y[j + 1] - y[j],
                color=get_colour(i), alpha=1, head_width=head_width,
                length_includes_head=True
            )
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    return fig

# Function to draw an ellipse from covariance
# matplotlib: Ellipse
def plot_cov_ellipse(
    cov, center, ax, color, scale=2, alpha=0.3
):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(vals)
    ellipse1 = Ellipse(
        xy=center, width=width, height=height, angle=angle,
        edgecolor=color, facecolor=color, alpha=alpha
    )
    ax.add_patch(ellipse1)
    ellipse2 = Ellipse(
        xy=center, width=scale*width, height=scale*height, angle=angle,
        edgecolor=color, facecolor=color, alpha=alpha
    )
    ax.add_patch(ellipse2)

def plot_clusters(adata, cov_estimator):
    pct_var = adata.uns['pca']['variance_ratio']
    x_pca = adata.obsm["X_pca"]
    is_cvt = adata.obs['Label'] == 'Convert' 
    # fit covariance estimator
    cov_estimator.fit(x_pca[is_cvt, :2])
    cvt_center = cov_estimator.location_
    cvt_cov = cov_estimator.covariance_
    cov_estimator.fit(x_pca[~is_cvt, :2])
    noncvt_center = cov_estimator.location_
    noncvt_cov = cov_estimator.covariance_
    # plot
    fig = sc.pl.pca(
        adata,
        color=['Label'],
        size=200,
        return_fig=True,
    )
    ax = fig.axes[0]
    plot_cov_ellipse(cvt_cov, cvt_center, ax, color='tab:blue')
    plot_cov_ellipse(noncvt_cov, noncvt_center, ax, color='tab:orange')
    ax.set_title('')
    ax.set_xlabel('PC1 (%.1f%%)' % (pct_var[0] * 100))
    ax.set_ylabel('PC2 (%.1f%%)' % (pct_var[1] * 100))
    return fig

### Load data

file = 'data/astral/processed/combat_knn5_lyriks-605_402.csv'
lyriks = pd.read_csv(file, index_col=0, header=0).T

# Contains gene symbol annotation
filepath = 'data/astral/raw/reprocessed-data.csv'
reprocessed = pd.read_csv(filepath, index_col=0, header=0)

# Replace UniProt IDs with gene symbols
lyriks_symbol = lyriks.copy()
uniprot_symbol = reprocessed.loc[lyriks.columns, 'Gene']
# Remove proteins that do not map to gene symbols
lyriks_symbol = lyriks_symbol.loc[:, ~uniprot_symbol.isna()]
# Map UniProt IDs to gene symbols using replace
lyriks_symbol.columns = lyriks_symbol.columns.map(uniprot_symbol)
# uid = 'Q9Y6R7'
# uniprot_symbol[uid]
# lyriks[uid]
# lyriks_symbol[uniprot_symbol[uid]]


file = 'data/astral/processed/metadata-lyriks407.csv'
md = pd.read_csv(file, index_col=0, header=0)
md = md[md.label != 'QC']
md['period'] = md['period'].astype(int)

# Model 1A: cvt (M0) v.s. non-cvt (M0)
# Prognostic markers
# Only M0 and exclude ctrl samples
md_1a = md[(md.final_label != 'ctrl') & (md.period == 0)].copy()
lyriks_1a = lyriks.loc[md_1a.index]
# md.period = md.period.astype(str)
md_1a.rename(columns={'final_label': 'Label'}, inplace=True)
md_1a.Label = md_1a.Label.replace({
    'rmt': 'Non-convert',
    'mnt': 'Non-convert',
    'cvt': 'Convert'
})

### Mongan et al.
# P02489 is not in reprocessed data (not detected)
# P43320 is in reprocessed data
filename = 'data/astral/etc/mongan-etable5.csv'
mongan = pd.read_csv(filename, index_col=0)
mongan_prots162 = mongan.index[mongan.index.isin(lyriks.columns)]
mongan_prots35 = mongan.index[mongan.q < 0.05]
in_astral = mongan_prots35.isin(lyriks.columns)
missing_astral = mongan_prots35[~in_astral]
mongan_prots33 = mongan_prots35[in_astral]
mongan_prots10 = pd.Index([
    'P01023', 'P01871', 'P04003', 'P07225', 'P23142',
    'P02766', 'Q96PD5', 'P02774', 'P10909', 'P13671'
])
in_astral10 = mongan_prots10.isin(lyriks.columns)
mongan_prots10 = mongan_prots10[in_astral10]

### Elastic net biomarkers
filepath = "tmp/astral/lyriks402/new/biomarkers/biomarkers-elasticnet.csv"
enet_signature = pd.read_csv(filepath, index_col=0)
bm_enet = enet_signature.index


# TODO: Explore association with time samples were taken

### Protein families ###

# Some gene symbols are duplicated
apo_genes = lyriks_symbol.columns[
    lyriks_symbol.columns.str.startswith('APO', na=False)]
serpin_genes = lyriks_symbol.columns[
    lyriks_symbol.columns.str.startswith('SERPIN', na=False)]
itih_genes = lyriks_symbol.columns[
    lyriks_symbol.columns.str.startswith('ITIH', na=False)
]
# itih_genes = lyriks_symbol.columns[
#     lyriks_symbol.columns.str.startswith('ITIH', na=False) | 
#     lyriks_symbol.columns.isin(['AMBP', 'SPINT2'])
# ]
znf_genes = lyriks_symbol.columns[
    lyriks_symbol.columns.str.startswith('ZNF', na=False)]
complement_proteins = reprocessed.loc[
    reprocessed.Description.str.startswith('Complement', na=False)]
complement_genes = complement_proteins.Gene
complement_genes = complement_genes[complement_genes.isin(lyriks_symbol.columns)]
coagulation_proteins = reprocessed.loc[
    reprocessed.Description.str.startswith('Coagulation', na=False)]
coagulation_genes = coagulation_proteins.Gene
coagulation_genes = coagulation_genes[coagulation_genes.isin(lyriks_symbol.columns)]
print(complement_genes)
print(coagulation_genes)

astral_itih = ['ITIH1', 'ITIH3', 'ITIH4']
mongan_itih = {'ITIH1': "'", 'ITIH3': '"'}
astral_apo = ['APOB']
mongan_apo = {'APOH': "'", 'APOE':  "'"}
astral_znf = ['ZNF607']
mongan_znf = {}
astral_serpin = ['SERPIND1', 'SERPINA4']
mongan_serpin = {'SERPIND1': '"', 'SERPING1': "'"}
astral_complement = ['C1R', 'C1S']
mongan_complement = {
    'C1R': "'", 'C8A': '"', 'C1QC': "'",
    'CFI': "'", 'CFH': "'", 'CFB': "'",
}
astral_coagulation = ['F9']
mongan_coagulation = {'F11': "'"}

# Immunoglobulins and interleukins are too generic
# ig_proteins = reprocessed.loc[
#     reprocessed.Description.str.startswith('Immunoglobulin', na=False)]
# ig_genes = ig_proteins.Gene

family = 'itih'
genes = itih_genes
astral_set = astral_itih
mongan_set = mongan_itih

lyriks_family = lyriks_symbol.loc[md_1a.index, genes]
lyriks_family = lyriks_family.assign(sample_id=lyriks_family.index)
lyriks_family = lyriks_family.join(md_1a[['Label', 'age', 'gender']], how='left')
lyriks_family_long = lyriks_family.melt(
    id_vars=['sample_id', 'Label', 'age', 'gender'],
    var_name='Gene', value_name='Log intensity'
)
lyriks_family

# Order by ANCOVA pvalue
pvalues = []
for gene in genes:
    print(gene)
    model = ols(
        f'{gene} ~ Label + age + gender',
        data=lyriks_family
    ).fit()
    pvalues.append(model.pvalues[1])

# Observe whether q-values are still significant after correction
_, qvalues, _, _ = multipletests(pvalues, alpha=0.05, method='fdr_bh')
sig = pd.DataFrame(
    {'p': pvalues, 'q': qvalues},
    index=genes
).sort_values(by='q', ascending=True)

sig = sig.sort_index()
# filepath = f'tmp/astral/lyriks402/fig/families/{family}-qvalues.csv'
# sig.to_csv(filepath)

# Order genes according to q-values
lyriks_family_long['Gene'] = pd.Categorical(
    lyriks_family_long['Gene'],
    categories=sig.index,
    ordered=True
)
print(len(genes))

fig, axes = plt.subplots(1, 4, figsize=(10, 3))
axes = axes.flatten()
for i, gene in enumerate(sig.index):
    print(gene)
    sns.stripplot(
        data=lyriks_family_long[lyriks_family_long['Gene'] == gene],
        x='Label', y='Log intensity', hue='Label',
        ax=axes[i], jitter=True, legend=False,
        order=['Non-convert', 'Convert']
    )
    # Determine title
    title = gene
    # if gene in astral_set:
    #     title += '*'
    # if gene in mongan_set:
    #     title += mongan_set[gene]
    # title = f'{title} (p = {sig.loc[gene, "p"]:.3f})'
    title = f'{title} (p < 0.001)'
    axes[i].set_title(title)
    axes[i].set_xlabel('')

for j in range(len(genes), len(axes)):
    fig.delaxes(axes[j])  # Remove empty subplot

plt.tight_layout()
filepath = f'tmp/astral/lyriks402/fig/families/jitter-{family}.pdf'
plt.savefig(filepath)

plt.show()


g = sns.FacetGrid(
    data=lyriks_family_long, col='Gene', hue='Label',
    height=2.5, aspect=0.7, col_wrap=8, sharex=True, sharey=False
)
g.map(sns.stripplot, 'Label', 'Expression value', order=None)
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.set_axis_labels('')
for ax in g.axes.flat:
    ax.tick_params(axis='x', labelrotation=30)

filepath = f'tmp/astral/lyriks402/fig/families/{family}-jitter.pdf'

plt.savefig(filepath, bbox_inches='tight')
plt.close()

# TODO: Plot with specified order and titles


# view sample metadata
sc.pl.violin(data, keys=, groupby='Label', rotation=45)

lyriks_1a = lyriks.loc[md_1a.index]
X = lyriks_1a
X.shape
md_1a.shape


# TODO: Plot apolipoproteins (plot baseline comparisons)
# TODO: Plot apolipoproteins (plot progression through time?)
# TODO: Double check for proteins that were supposedly depleted

reprocessed.loc[reprocessed.Gene.str.startswith('APO', na=False)]

data_mongan33 = data[:, mongan_prots33]
data_elasticnet = data[:, bm_enet]
data_mongan33.shape
data_elasticnet.shape

mle_cov = EmpiricalCovariance()
robust_cov = MinCovDet()

sc.pp.pca(data_mongan33)
fig = plot_clusters(data_mongan33, mle_cov)
fig.set_size_inches(3.6, 2.4)
plt.legend(
    loc='upper center', bbox_to_anchor=(0.5, -0.1),
    frameon=False, ncol=2
)
filename = 'tmp/astral/lyriks402/fig/pca-mongan33.pdf'
fig.savefig(filename, bbox_inches='tight')

sc.pp.pca(data_elasticnet)
fig = plot_clusters(data_elasticnet, mle_cov)
fig.set_size_inches(3, 2)
plt.legend(
    loc='upper center', bbox_to_anchor=(0.5, -0.1),
    frameon=False, ncol=2
)
filename = 'tmp/astral/lyriks402/fig/pca-elasticnet.pdf'
fig.savefig(filename, bbox_inches='tight')

# data_mongan = data[:, mongan_prots[in_astral]]
# uhr_mongan = data[data.obs.final_label != 'ctrl', mongan_prots_astral]
# cvt_mongan = data[data.obs.final_label == 'cvt', mongan_prots_astral]
# 
# data_progq = data[:, result_prognostic_ancova.features]
# uhr_progq = data[data.obs.final_label != 'ctrl', result_prognostic_ancova.features]
# 
# uhr_psych = data[data.obs.final_label != 'ctrl', result_psychotic.features]
# cvt_psych = data[data.obs.final_label == 'cvt', result_psychotic.features]
# 
# uhr_psych_abs = data[data.obs.final_label != 'ctrl', psychotic_abs_prots]
# cvt_psych_abs = data[data.obs.final_label == 'cvt', psychotic_abs_prots]

# Features: All
sc.pp.pca(data)
sc.pl.pca(
    data,
    # data[data.obs.final_label == 'cvt', :],
    color=['final_label', 'period'],
    size=300,
)

sc.pp.pca(cvt)

# Features: Mongan
sc.pp.pca(cvt_mongan)

fig = plot_arrows(cvt_mongan, head_width=0.05)
fig.set_size_inches(10, 6)
filename = 'tmp/astral/fig/traj-mongan-cvt.pdf'
fig.savefig(filename)

fig = sc.pl.pca(
    data_mongan,
    # uhr_mongan[uhr_mongan.obs.final_label == 'cvt'],
    color=['final_label', 'period'],
    size=300,
    return_fig=True,
)
filename = 'tmp/astral/fig/pca-mongan-all.pdf'
fig.savefig(filename)

# Features: Prognostic 
sc.pp.pca(uhr_progq)

fig = sc.pl.pca(
    # uhr_progq,
    uhr_progq[uhr_progq.obs.final_label == 'cvt'],
    color=['final_label', 'period'],
    size=300,
    return_fig=True,
)
filename = 'tmp/astral/fig/pca-progq-cvt.pdf'
fig.savefig(filename)

# Features: Psychotic
sc.pp.pca(uhr_psych)
uhr_psych.shape

fig = sc.pl.pca(
    uhr_psych,
    # uhr_psych[uhr_psych.obs.final_label == 'cvt'],
    color=['final_label', 'period'],
    size=300,
    return_fig=True,
)
filename = 'tmp/astral/fig/pca-psych-uhr.pdf'
fig.savefig(filename)

sc.pp.pca(cvt_psych)
fig = sc.pl.pca(
    cvt_psych,
    # uhr_psych[uhr_psych.obs.final_label == 'cvt'],
    color=['sn', 'period'],
    size=300,
    return_fig=True,
)
filename = 'tmp/astral/fig/pca-psych-cvt.pdf'
fig.savefig(filename)

# Features: Psychotic (abs)
sc.pp.pca(uhr_psych_abs)

fig = sc.pl.pca(
    # uhr_psych_abs,
    uhr_psych_abs[uhr_psych_abs.obs.final_label == 'cvt'],
    color=['sn', 'period'],
    size=300,
    return_fig=True,
)
filename = 'tmp/astral/fig/pca-psych_abs-cvtsub.pdf'
fig.savefig(filename)


fig = sc.pl.pca(
    cvt_psych_abs,
    color=['sn', 'period'],
    size=300,
    return_fig=True,
)
filename = 'tmp/astral/fig/pca-psych_abs-cvt.pdf'
fig.savefig(filename)

sc.pp.pca(cvt_psych)
fig = plot_arrows(cvt_psych)
fig.set_size_inches(10, 6)
filename = 'tmp/astral/fig/traj-psych-cvt.pdf'
fig.savefig(filename)

sc.pp.pca(cvt_psych_abs)
fig = plot_arrows(cvt_psych_abs)
fig.set_size_inches(10, 6)
filename = 'tmp/astral/fig/traj-psych_abs-cvt.pdf'
fig.savefig(filename)

cvt_multiple_pids = ['L0325S', 'L0476S', 'L0544S', 'L0561S', 'L0609S', 'L0646S']
sc.pp.pca(cvt)
fig = sc.pl.pca(
    cvt[~cvt.obs.sn.isin(cvt_multiple_pids)],
    color=['sn', 'period'],
    # dimensions=[(0,1), (0,2)],
    # groups=['L0325S', 'L0609S'],
    projection='2d', size=100,
    return_fig=True,
)
filename = 'tmp/astral/fig/pca-cvt_timepoint1.pdf'
fig.savefig(filename)

print(X.shape)
X_pca = pd.DataFrame(
    pca.fit_transform(X),
    columns=['PC' + str(i) for i in range(1, 4)],
    index=X.index
)
X_y = pd.concat([X_pca, y], axis=1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,8))
sns.scatterplot(data=X_y, x='PC1', y='PC2', hue='final_label', ax=ax1)
sns.scatterplot(data=X_y, x='PC1', y='PC3', hue='final_label', ax=ax2)
filepath = 'tmp/astral/fig/pca-1a-59x607.pdf'
plt.savefig(filepath)

X_ttest = X[result_ttest.features]
print(X_ttest.shape)
X_pca = pd.DataFrame(
    pca.fit_transform(X_ttest),
    columns=['PC' + str(i) for i in range(1, 4)],
    index=X.index
)
X_y = pd.concat([X_pca, y], axis=1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,8))
sns.scatterplot(data=X_y, x='PC1', y='PC2', hue='final_label', ax=ax1)
sns.scatterplot(data=X_y, x='PC1', y='PC3', hue='final_label', ax=ax2)
filepath = 'tmp/astral/fig/pca-1a-59x34.pdf'
plt.savefig(filepath)

# Boruta
X_boruta = X[result_boruta.features]
print(X_boruta.shape)
X_pca = pd.DataFrame(
    pca.fit_transform(X_boruta),
    columns=['PC' + str(i) for i in range(1, 4)],
    index=X.index
)
X_y = pd.concat([X_pca, y], axis=1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,8))
sns.scatterplot(data=X_y, x='PC1', y='PC2', hue='final_label', ax=ax1)
sns.scatterplot(data=X_y, x='PC1', y='PC3', hue='final_label', ax=ax2)
filepath = 'tmp/astral/fig/pca-1a-59x22.pdf'
plt.savefig(filepath)

# Mongan 
mongan_present = list(mongan_prots & set(list(X)))
X_mongan = X[mongan_present]
print(X_mongan.shape)
X_pca = pd.DataFrame(
    pca.fit_transform(X_mongan),
    columns=['PC' + str(i) for i in range(1, 4)],
    index=X.index
)
X_y = pd.concat([X_pca, y], axis=1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,8))
sns.scatterplot(data=X_y, x='PC1', y='PC2', hue='final_label', ax=ax1)
sns.scatterplot(data=X_y, x='PC1', y='PC3', hue='final_label', ax=ax2)
filepath = 'tmp/astral/fig/pca-1a-59x33.pdf'
plt.savefig(filepath)


### Evaluation: barh ###

# Permutation experiment
dirpath = 'tmp/astral/lyriks402/new/pickle/random15'
filepaths = os.listdir(dirpath)
filepaths = [os.path.join(dirpath, filepath) for filepath in filepaths]
filepaths = sorted(filepaths)
mean_aucs = []
for filepath in filepaths:
    with open(filepath, 'rb') as file:
        result = pickle.load(file)
    aucs = []
    for i, (labels, probas) in enumerate(zip(result.labels, result.probas)):
        aucs.append(roc_auc_score(labels, probas))
    mean_aucs.append(np.mean(aucs))

rnd_mean_auc = np.mean(mean_aucs)
rnd_std_auc = np.std(mean_aucs)

# AUCs
dirpath = 'tmp/astral/lyriks402/new/pickle/'
filepaths = os.listdir(dirpath)
filepaths = [
    os.path.join(dirpath, filepath) for filepath in filepaths
    if filepath.endswith('.pkl')
]

rows = []
for filepath in filepaths:
    with open(filepath, 'rb') as file:
        result = pickle.load(file)
    for i, (labels, probas) in enumerate(zip(result.labels, result.probas)):
        rows.append({
            'selector': result.metadata['selector'],
            'class_weight': result.metadata['class_weight'],
            'model': result.metadata['model'],
            'fold': i,
            'auc': roc_auc_score(labels, probas),
        })

data = pd.DataFrame(rows)
data

model_aucs = data.groupby(['selector', 'class_weight', 'model',]).agg(
    mean_auc=('auc', 'mean'),
    std_auc=('auc', 'std'),
).reset_index()
model_aucs.sort_values(by='mean_auc', ascending=False, inplace=True)
model_bal_aucs = model_aucs[model_aucs['class_weight'] == 'balanced']
# filepath = 'tmp/astral/lyriks402/new/models-aucs.csv'
# model_bal_aucs.to_csv(filepath, index=False)


# AUC1 
aucs1 = model_bal_aucs[
    model_bal_aucs['selector'].isin(['mongan10', 'mongan33'])
]
aucs1.index = [
    'EU-GEI (ANCOVA)',
    'EU-GEI (SVM)',
]

fig, ax = plt.subplots(1, 1, figsize=(5, 1.5))
ax.barh(
    aucs1.index, aucs1['mean_auc'], xerr=aucs1['std_auc'],
    capsize=5, color='gray'
)
ax.set_xlabel('AUC')
# plt.axvline(x=0.99, color='tab:cyan', linestyle='dashed') # train set
# plt.axvline(x=0.92, color='tab:red', linestyle='dashed') # test set
plt.xlim(0.5, 1.05)
plt.tight_layout()
filepath = 'tmp/astral/lyriks402/fig/barh-auc1.pdf'
plt.savefig(filepath)

# AUC2
aucs2 = model_bal_aucs.iloc[[0, 1, 3], :]
aucs2.loc[len(aucs2)] = {
    'selector': 'random',
    'class_weight': 'balanced',
    'model': 'elasticnet-bal',
    'mean_auc': rnd_mean_auc,
    'std_auc': rnd_std_auc
}
print(aucs2)

aucs2.index = [
    'LYRIKS (elastic net)',
    'LYRIKS (ANCOVA)',
    'LYRIKS (SVM)',
    'LYRIKS (random)',
]

fig, ax = plt.subplots(1, 1, figsize=(4.6, 2.4))
ax.barh(
    aucs2.index, aucs2['mean_auc'], xerr=aucs2['std_auc'],
    capsize=5, color='gray'
)
ax.set_xlabel('AUC')
plt.xlim(0.5, 1.05)
plt.gca().invert_yaxis()
plt.tight_layout()
filepath = 'tmp/astral/lyriks402/fig/barh-auc2.pdf'
plt.savefig(filepath)

# Calculate p-values for AUCs (p = 1 - F(x))
p = 1 - norm.cdf(aucs2.mean_auc, loc=rnd_mean_auc, scale=rnd_std_auc)
p[3] = np.nan 
aucs2['p'] = p
print(aucs2)

# # Perform t-test between models
# data_bal = data[data['class_weight'] == 'balanced']
# 
# tstat, pval = ttest_ind(
#     data_bal.loc[data['selector'] == 'mongan33', 'auc'].values,
#     data_bal.loc[data['selector'] == 'prognostic_ancova', 'auc'].values
# )
# print(tstat, pval)
# 
# tstat, pval = ttest_ind(
#     data_bal.loc[data['selector'] == 'mongan33', 'auc'].values,
#     data_bal.loc[data['selector'] == 'none', 'auc'].values
# )
# print(tstat, pval)
# 
# ttest_rel(
#     data_bal.loc[data['selector'] == 'mongan33', 'auc'].values,
#     data_bal.loc[data['selector'] == 'prognostic_ancova', 'auc'].values
# )
# 
# ttest_rel(
#     data_bal.loc[data['selector'] == 'mongan33', 'auc'].values,
#     data_bal.loc[data['selector'] == 'none', 'auc'].values
# )
# print(tstat, pval)
#
# paired_ttests = [
#     ttest_rel(
#         lyriks.loc[cvt2, prot],
#         lyriks.loc[cvt1, prot]
#     )
#     for prot in lyriks.columns
# ]



##### Venn #####

filepath = 'data/astral/raw/reprocessed-data.csv'
reprocessed = pd.read_csv(filepath, index_col=0, header=0)

filepath = 'tmp/astral/lyriks402/new/biomarkers/biomarkers-ancova.csv'
data = pd.read_csv(filepath, index_col=0)
bm_ancova = data.index

filepath = 'tmp/astral/lyriks402/new/biomarkers/biomarkers-elasticnet.csv'
data = pd.read_csv(filepath, index_col=0)
bm_elasticnet = data.index

filepath = 'tmp/astral/lyriks402/new/pickle/1a-svm-linear-svm-linear-nestedkfold.pkl'
with open(filepath, 'rb') as file:
    result = pickle.load(file)

bm_svm = result.features

# Mongan
filepath = 'tmp/astral/lyriks402/biomarkers/mongan-etable5.csv'
mongan = pd.read_csv(filepath, index_col=0, header=0)
prots_mongan35 = mongan.index[mongan.q < 0.05]

prots_mongan10 = pd.Series([
    'P01023', 'P01871', 'P04003', 'P07225', 'P23142',
    'P02766', 'Q96PD5', 'P02774', 'P10909', 'P13671'
])
len(prots_mongan35)

bm_ancova
prots_mongan35.columns

# TODO 
# Mongan protein expression in LYRIKS
# Check missing proteins with original data

# Feature selection method
bm_colors = generate_colors(n_colors=5, cmap='Dark2', alpha=0.4)

proteins = {
    'LYRIKS (ANCOVA)': set(bm_ancova),
    'LYRIKS (elastic net)': set(bm_elasticnet),
    'EU-GEI (ANCOVA)': set(prots_mongan35),
}

fig, ax = plt.subplots(figsize=(7, 7))
draw_venn(
    petal_labels=generate_petal_labels(proteins.values(), fmt='{size}'),
    dataset_labels=proteins.keys(),
    hint_hidden=False,
    colors=bm_colors[:2] + bm_colors[3:],
    figsize=(6, 6), fontsize=15, legend_loc='upper right', ax=ax
)

# for t in ax.texts:
#     t.set_fontsize(18)
#     if int(t.get_text()) in [1, 3]:
#         t.set_fontweight('bold')

filename = 'tmp/astral/lyriks402/fig/venn3-literature35.pdf'
plt.savefig(filename)

# Analysing intersections
prots_u = set(bm_ancova).union(set(bm_elasticnet))
prots_d = prots_u - set(prots_mongan35)
prots_i1 = set(bm_ancova).intersection(set(bm_elasticnet))
prots_i2 = prots_u.intersection(set(prots_mongan35))
prots_d1 = set(bm_ancova) - set(prots_mongan35)
prots_i3 = set(bm_ancova) & set(prots_mongan35)
all_3 = set(prots_i1) & set(prots_mongan35)
all_3


annot = reprocessed[['Description', 'Gene']]
annot.loc[list(prots_u)]
annot.loc[list(prots_d)]
annot.loc[list(prots_i1)]
annot.loc[list(prots_i2)]

annot.loc[list(prots_d1)]
annot.loc[list(prots_i3)]
annot.loc[list(all_3)]

annot.loc[list(set(prots_mongan35) - {'P02489'})]

set(prots_prognostic_1a).intersection(set(prots_psychotic_1a))
set(prots_uhr_3a).intersection(set(prots_prognostic_2b))

set(prots_mongan35).intersection(set(prots_ancova).union(set(prots_enet)))
a = set(prots_ancova).union(set(prots_enet)) - set(prots_mongan35)
a
# intersection(set(prots_enet))

# ANCOVA, Elastic Net, SVM

proteins = {
    'LYRIKS (ANCOVA)': set(bm_ancova),
    'LYRIKS (elastic net)': set(bm_elasticnet),
    'LYRIKS (SVM)': set(bm_svm),
}

fig, ax = plt.subplots(figsize=(7, 7))
draw_venn(
    petal_labels=generate_petal_labels(proteins.values(), fmt='{size}'),
    dataset_labels=proteins.keys(),
    hint_hidden=False,
    colors=bm_colors[:3],
    figsize=(6,6), fontsize=15, legend_loc="upper right", ax=ax
)
# for t in ax.texts:
#     t.set_fontsize(18)
#     if int(t.get_text()) in [2, 3, 4, 0]:
#         t.set_fontweight('bold')
filename = 'tmp/astral/lyriks402/fig/venn3-feature_selection.pdf'
plt.savefig(filename)


##### Pathway enrichment analysis #####

### STRING ###
## KEGG
file = 'tmp/astral/lyriks402/new/pathway_enrichment/STRING/wx/enrichment-ancova.tsv'
string_ancova = pd.read_csv(file, sep='\t', index_col=0)

string_ancova['-log10(FDR)'] = -np.log10(string_ancova['false discovery rate']) 
string_ancova.sort_values(by='-log10(FDR)', ascending=True, inplace=True)
string_ancova.iloc[:,[0, 6, 8]]

file = 'tmp/astral/lyriks402/new/pathway_enrichment/STRING/mongan/KEGG.tsv'
string_mongan = pd.read_csv(file, sep='\t', index_col=0)
string_mongan['-log10(FDR)'] = -np.log10(string_mongan['false discovery rate']) 
string_mongan.sort_values(by='-log10(FDR)', ascending=True, inplace=True)
string_mongan.iloc[:,[0, 8]]

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(6, 3), sharex=True,
    gridspec_kw={'height_ratios': [1, 5]}
)
ax1.barh(
    string_kegg_ancova['term description'],
    string_kegg_ancova['-log10(FDR)'],
    capsize=5, color='gray'
)
ax1.set_xlabel('-log10(FDR)')
ax1.set_title('LYRIKS (ANCOVA)')
ax2.barh(
    string_kegg_mongan['term description'],
    string_kegg_mongan['-log10(FDR)'],
    capsize=5, color='gray'
)
ax2.set_xlabel('-log10(FDR)')
ax2.set_title('EU-GEI (ANCOVA)')
plt.tight_layout()
filepath = 'tmp/astral/lyriks402/fig/barh-string_kegg.pdf'
plt.savefig(filepath)

## Local network cluster 
file = 'tmp/astral/lyriks402/new/pathway_enrichment/STRING/wx/enrichment-ancova.tsv'
string_ancova = pd.read_csv(file, sep='\t', index_col=0)
string_lcn_ancova = string_ancova.iloc[0]

# string_lcn_ancova.iloc[4, 0] = 'Mixed, incl. COVID-19, thrombosis and anticoagulation, and ITIH C-terminus'
# string_lcn_ancova.iloc[7, 0] = 'Mixed, incl. ITIH C-terminus and AGP'
# string_lcn_ancova.iloc[8, 0] = 'TRL particle remodeling, and Spherical HDL particle'
# string_lcn_ancova['term_id'] = string_lcn_ancova['term description'] + ' (' + string_lcn_ancova.index + ')'
# string_lcn_ancova['-log10(FDR)'] = -np.log10(string_lcn_ancova['false discovery rate']) 
# string_lcn_ancova.sort_values(by='-log10(FDR)', ascending=True, inplace=True)
# string_lcn_ancova.iloc[:,[8,9]]

fig, ax = plt.subplots(1, 1, figsize=(8, 1.2))
ax.barh(
    string_lcn_ancova['term description'],
    string_lcn_ancova['false discovery rate'],
    capsize=5, color='gray'
)
ax.set_xlabel('FDR')
ax.set_title('LYRIKS (ANCOVA)')
plt.tight_layout()
filepath = 'tmp/astral/lyriks402/fig/barh-string_lcn_ancova.pdf'
plt.savefig(filepath)

# file = 'tmp/astral/lyriks402/new/pathway_enrichment/STRING/mongan/local_network_cluster.tsv'
# string_lcn_mongan = pd.read_csv(file, sep='\t', index_col=0)
# string_lcn_mongan.iloc[:,[0,5]]
# string_lcn_mongan.iloc[1, 0] = 'Mixed, incl. ITIH C-terminus and AGP'
# string_lcn_mongan.iloc[9, 0] = 'Mixed, incl. COVID-19, thrombosis and anticoagulation, and ITIH C-terminus'
# string_lcn_mongan['term_id'] = string_lcn_mongan['term description'] + ' (' + string_lcn_mongan.index + ')'
# string_lcn_mongan['-log10(FDR)'] = -np.log10(string_lcn_mongan['false discovery rate']) 
# string_lcn_mongan.sort_values(by='-log10(FDR)', ascending=True, inplace=True)
# string_lcn_mongan.iloc[:,[8,9]]
# fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# ax.barh(
#     string_lcn_mongan['term_id'],
#     string_lcn_mongan['-log10(FDR)'],
#     capsize=5, color='gray'
# )
# ax.set_xlabel('-log10(FDR)')
# ax.set_title('EU-GEI (ANCOVA)')
# plt.tight_layout()
# filepath = 'tmp/astral/lyriks402/fig/barh-string_lcn_mongan.pdf'
# plt.savefig(filepath)


## ClusterProfiler: GSEA
file = 'tmp/astral/lyriks402/new/pathway_enrichment/clusterprofiler/GSEA-KEGG-ANCOVA.csv'
gsea_ancova = pd.read_csv(file, index_col=0)
gsea_ancova.sort_values(by='qvalue', ascending=False, inplace=True)
gsea_ancova.iloc[:,[0, 6]]

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.barh(
    gsea_ancova['Description'],
    gsea_ancova['qvalue'],
    capsize=5, color='gray'
)
ax.set_xlabel('Q-value')
ax.set_title('LYRIKS (ANCOVA)')
plt.tight_layout()
filepath = 'tmp/astral/lyriks402/fig/barh-gsea_kegg_ancova.pdf'
plt.savefig(filepath)


##### Pathway enrichment analysis #####

filepath = 'data/astral/raw/reprocessed-data.csv'
reprocessed = pd.read_csv(filepath, index_col=0, header=0)
reprocessed.shape
reprocessed.columns

symbol = 'P0DOX2'
reprocessed.loc[symbol, 'Gene']
reprocessed[reprocessed.Gene.isna()]

gene_symbols605  = reprocessed.loc[lyriks.columns, 'Gene']
gene_symbols605.to_csv(
    'tmp/astral/lyriks402/new/biomarkers/gene_symbols605.csv', header=False
)


# Convert UniProt IDs to gene symbols
import requests
from pprint import pprint   

def uniprot_to_symbol(uniprot_id):
    url = f'https://www.uniprot.org/uniprot/{uniprot_id}.json'
    response = requests.get(url).json()
    if response['entryType'] == 'Inactive':
        print(f'UniProt entry {uniprot_id} is inactive!')
        switched_id = response['inactiveReason']['mergeDemergeTo'][0]
        assert isinstance(switched_id, str)
        return uniprot_to_symbol(switched_id)
    print(uniprot_id)
    return response['genes'][0]['geneName']['value']

symbol = 'P0CG06'
symbol = 'Q9Y490'
res = uniprot_to_symbol(symbol)

mongan_symbols = mongan.index.map(uniprot_to_symbol)
mongan.insert(0, 'gene_symbol', mongan_symbols) 
mongan.head()
mongan.to_csv('tmp/astral/lyriks402/new/biomarkers/mongan-etable5.csv')


# Formatting of enrichR results
import os

dirpath = 'tmp/astral/lyriks402/new/pathway_enrichment/enrichr/wx'
filepaths = os.listdir(dirpath)
filepaths = [os.path.join(dirpath, filepath) for filepath in filepaths]

for filepath in filepaths:
    data = pd.read_csv(filepath, sep='\t')
    data['Overlap'] = data['Overlap'].str.replace('/', ' of ')
    filepath = filepath.replace('.tsv', '.csv')
    data.to_csv(filepath, index=False)

##### Biomarker analysis #####

# TODO: Check ANCOVA signature is within Mongan's quantified
filepath = 'data/tmp/biomarkers/biomarkers-ancova.csv'
bm_ancova = pd.read_csv(filepath, index_col=0)

filepath = "data/tmp/biomarkers/biomarkers-elasticnet.csv"
bm_enet = pd.read_csv(filepath, index_col=0)

filepath = 'data/astral/etc/mongan-etable5.csv'
mongan = pd.read_csv(filepath, index_col=0)
mongan_prots162 = mongan.index[mongan.index.isin(lyriks.columns)]
mongan_prots35 = mongan.index[mongan.q < 0.05]
in_astral = mongan_prots35.isin(lyriks.columns)
missing_astral = mongan_prots35[~in_astral]
mongan_prots33 = mongan_prots35[in_astral]
mongan_prots10 = pd.Index([
    'P01023', 'P01871', 'P04003', 'P07225', 'P23142',
    'P02766', 'Q96PD5', 'P02774', 'P10909', 'P13671'
])
in_astral10 = mongan_prots10.isin(lyriks.columns)
mongan_prots10 = mongan_prots10[in_astral10]

len(bm_ancova.index)
len(bm_enet.index)

ancova_notin_mongan = bm_ancova.index[~bm_ancova.index.isin(mongan.index)]
enet_notin_mongan = bm_enet.index[~bm_enet.index.isin(mongan.index)]

anc_notin_avg_value = pd.DataFrame({
    'Average log intensity': lyriks_1a[bm_ancova.index].mean(axis=0),
    'Quantified': ~bm_ancova.index.isin(ancova_notin_mongan), 
    'Group': 'ANCOVA'
})
enet_notin_avg_value = pd.DataFrame({
    'Average log intensity': lyriks_1a[bm_enet.index].mean(axis=0),
    'Quantified': ~bm_enet.index.isin(enet_notin_mongan), 
    'Group': 'Elastic Net'
})
mongan_avg_value = pd.DataFrame({
    'Average log intensity': lyriks_1a[mongan_prots33].mean(axis=0),
    'Quantified': True,
    'Group': 'Mongan'
})
avg_value = pd.concat([anc_notin_avg_value, enet_notin_avg_value, mongan_avg_value])

avg_value['Label'] = avg_value['Group'] + ' (' + avg_value['Quantified'].map({True: 'in', False: 'not in'}) + ' Mongan)'

# Investigate percentage of zeros these biomarkers initially had
lyriks_raw = reprocessed.loc[:, reprocessed.columns.str.startswith('L')]
pct_missing = (lyriks_raw == 0).sum(axis=1) / lyriks_raw.shape[1]
avg_value['Missingness'] = pct_missing[avg_value.index].values

plt.figure(figsize=(12,6))
sns.stripplot(
    data=avg_value, x='Label', y='Average log intensity', hue='Group',
    jitter=True, dodge=False, order=None
)
filepath = 'tmp/astral/lyriks402/fig/biomarkers-avg_expr.pdf'
plt.savefig(filepath, bbox_inches='tight')
plt.close()

# TODO: Save bm_svm

# Plot ANCOVA biomarkers of conversion samples over time
cvt_ancova = lyriks.loc[md.final_label == 'cvt', bm_ancova.index]
cvt_ancova1 = cvt_ancova.join(md[['sn', 'period']])

fig, axes = plt.subplots(3, 5, figsize=(16, 10))
axes = axes.flatten()
for i, uid in enumerate(cvt_ancova.columns):
    print(i)
    ax = axes[i]
    # Plot scatter and line plot of each sample
    sns.lineplot(
        data=cvt_ancova1,
        x='period', y=uid, hue='sn',
        marker='o', legend=False,
        ax=ax
    )

plt.suptitle('LYRIKS (ANCOVA) biomarkers')
filepath = 'tmp/astral/lyriks402/fig/biomarkers-ancova-cvt.pdf'
plt.tight_layout()
plt.savefig(filepath)

# Non-converters
noncvt_ancova = lyriks.loc[md.final_label != 'cvt', bm_ancova.index]
noncvt_ancova1 = noncvt_ancova.join(md[['sn', 'period']])

fig, axes = plt.subplots(3, 5, figsize=(16, 10))
axes = axes.flatten()
for i, uid in enumerate(noncvt_ancova.columns):
    print(i)
    ax = axes[i]
    # Plot scatter and line plot of each sample
    sns.lineplot(
        data=noncvt_ancova1,
        x='period', y=uid, hue='sn',
        marker='o', legend=False,
        ax=ax
    )

plt.suptitle('LYRIKS (ANCOVA) biomarkers')
filepath = 'tmp/astral/lyriks402/fig/biomarkers-ancova-noncvt.pdf'
plt.tight_layout()
plt.savefig(filepath)

### LYRIKS (elastic net) biomarkers
# Converters
cvt_enet = lyriks.loc[md.final_label == 'cvt', bm_enet.index]
cvt_enet1 = cvt_enet.join(md[['sn', 'period']])

fig, axes = plt.subplots(3, 4, figsize=(14, 10))
axes = axes.flatten()
for i, uid in enumerate(cvt_enet.columns):
    print(i)
    ax = axes[i]
    # Plot scatter and line plot of each sample
    sns.lineplot(
        data=cvt_enet1,
        x='period', y=uid, hue='sn',
        marker='o', legend=False,
        ax=ax
    )

plt.suptitle('LYRIKS (elastic net) biomarkers')
filepath = 'tmp/astral/lyriks402/fig/biomarkers-enet-cvt.pdf'
plt.tight_layout()
plt.savefig(filepath)

# Non-Converters
noncvt_enet = lyriks.loc[md.final_label != 'cvt', bm_enet.index]
noncvt_enet1 = noncvt_enet.join(md[['sn', 'period']])

fig, axes = plt.subplots(3, 4, figsize=(14, 10))
axes = axes.flatten()
for i, uid in enumerate(noncvt_enet.columns):
    print(i)
    ax = axes[i]
    # Plot scatter and line plot of each sample
    sns.lineplot(
        data=noncvt_enet1,
        x='period', y=uid, hue='sn',
        marker='o', legend=False,
        ax=ax
    )

plt.suptitle('LYRIKS (elastic net) biomarkers')
filepath = 'tmp/astral/lyriks402/fig/biomarkers-enet-noncvt.pdf'
plt.tight_layout()
plt.savefig(filepath)

# TODO: Investigate prediction proba and t2c

t2c = md.loc[~md.month_of_conversion.isna(), ['sn', 'month_of_conversion']]
t2c.index = t2c.index.str.replace('_24', '_0')

filepath = 'tmp/astral/lyriks402/new/pickle/test-sample_ids.csv'
test_ids = pd.read_csv(filepath, index_col=0, header=None).index
test_ids

# Elastic Net 
filepath = 'tmp/astral/lyriks402/new/pickle/1a-elasticnet-elasticnet-bal-nestedkfold.pkl'
with open(filepath, 'rb') as file:
    result = pickle.load(file)

probas = list(chain.from_iterable(result.probas))
probas = pd.Series(probas, index=test_ids, name='proba')
enet_probas = t2c.join(probas)

sns.scatterplot(data=enet_probas, x='month_of_conversion', y='proba')
plt.title('LYRIKS (elastic net)')
plt.xlabel('Time to conversion (months)')
filepath = 'tmp/astral/lyriks402/fig/enet-proba-t2c.pdf'
plt.savefig(filepath)
plt.close()

# ANCOVA
filepath = 'tmp/astral/lyriks402/new/pickle/1a-prognostic_ancova-elasticnet-bal-kfold.pkl'
with open(filepath, 'rb') as file:
    result = pickle.load(file)

probas = list(chain.from_iterable(result.probas))
probas = pd.Series(probas, index=test_ids, name='proba')
ancova_probas = t2c.join(probas)

sns.scatterplot(data=ancova_probas, x='month_of_conversion', y='proba')
plt.title('LYRIKS (ANCOVA)')
plt.xlabel('Time to conversion (months)')
filepath = 'tmp/astral/lyriks402/fig/ancova-proba-t2c.pdf'
plt.savefig(filepath)
plt.close()

enet_probas
ancova_probas

##### P-value #####

# Probability of no overlap between two sets of randomly chosen genes
# Assumption: Choosing one gene does not affect the probability of the rest
# being chosen.

from math import comb

def compute_p(n, a, b):
    return comb(n - b, a) * comb(n - a, b) / (comb(n, a) * comb(n, b))

n = lyriks.shape[1]
compute_p(607, 10, 12)

##### Grid search #####   
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

### DELETE ###
filepath = 'data/astral/metadata/LYRIKS/metadata_73.csv'
md_73 = pd.read_csv(filepath, index_col=0)

md_73.head()
md_73.sn.unique().shape
