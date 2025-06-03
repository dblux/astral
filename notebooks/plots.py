import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc 
import statsmodels.api as sm

from dataclasses import dataclass, field
from scipy.stats import f, ttest_ind, ttest_rel, rankdata
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

file = 'data/astral/processed/metadata-lyriks407.csv'
md = pd.read_csv(file, index_col=0, header=0)
md = md[md.label != 'QC']
md['period'] = md['period'].astype(int)
# Model 1A: cvt (M0) v.s. non-cvt (M0)
# Prognostic markers
# Only M0 and exclude ctrl samples
md_1a = md[(md.final_label != 'ctrl') & (md.period == 0)]
lyriks_1a = lyriks.loc[md_1a.index]
X = lyriks_1a
X.shape
md_1a.shape

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

# Subset
# md.period = md.period.astype(str)
md_1a.rename(columns={'final_label': 'Label'}, inplace=True)
md_1a.Label = md_1a.Label.replace({
    'rmt': 'Non-convert',
    'mnt': 'Non-convert',
    'cvt': 'Convert'
})
data = ad.AnnData(X, md_1a)

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
aucs1
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
plt.axvline(x=0.99, color='tab:cyan', linestyle='dashed') # train set
plt.axvline(x=0.92, color='tab:red', linestyle='dashed') # test set
plt.xlim(0.5, 1.05)
plt.tight_layout()
filepath = 'tmp/astral/lyriks402/fig/barh-auc1.pdf'
plt.savefig(filepath)

# AUC2
aucs2 = model_bal_aucs.iloc[[0, 1, 3], :]
print(aucs2)
aucs2.index = [
    'LYRIKS (elastic net)',
    'LYRIKS (ANCOVA)',
    'LYRIKS (SVM)',
]
eugei_svm_mean_auc = aucs1.iloc[1, 3]

fig, ax = plt.subplots(1, 1, figsize=(4.6, 1.8))
ax.barh(
    aucs2.index, aucs2['mean_auc'], xerr=aucs2['std_auc'],
    capsize=5, color='gray'
)
ax.set_xlabel('AUC')
plt.axvline(x=eugei_svm_mean_auc, color='tab:green', linestyle='dashed') # EU-GEI (SVM)
plt.axvline(x=rnd_mean_auc, color='tab:purple', linestyle='dashed') # negative ctrl
plt.xlim(0.5, 1.05)
plt.tight_layout()
filepath = 'tmp/astral/lyriks402/fig/barh-auc2.pdf'
plt.savefig(filepath)

# Perform t-test between models
data_bal = data[data['class_weight'] == 'balanced']

tstat, pval = ttest_ind(
    data_bal.loc[data['selector'] == 'mongan33', 'auc'].values,
    data_bal.loc[data['selector'] == 'prognostic_ancova', 'auc'].values
)
print(tstat, pval)

tstat, pval = ttest_ind(
    data_bal.loc[data['selector'] == 'mongan33', 'auc'].values,
    data_bal.loc[data['selector'] == 'none', 'auc'].values
)
print(tstat, pval)

ttest_rel(
    data_bal.loc[data['selector'] == 'mongan33', 'auc'].values,
    data_bal.loc[data['selector'] == 'prognostic_ancova', 'auc'].values
)

ttest_rel(
    data_bal.loc[data['selector'] == 'mongan33', 'auc'].values,
    data_bal.loc[data['selector'] == 'none', 'auc'].values
)

print(tstat, pval)
# paired_ttests = [
#     ttest_rel(
#         lyriks.loc[cvt2, prot],
#         lyriks.loc[cvt1, prot]
#     )
#     for prot in lyriks.columns
# ]



##### Venn #####

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

# filepath = 'data/astral/etc/perkins.csv'
# perkins = pd.read_csv(filepath, index_col=0)
# prots_perkins = perkins['UniProt/CAS'][
#     perkins['UniProt/CAS'].str.startswith(('P', 'Q'))
# ]

# filename = 'tmp/astral/lyriks402/biomarkers/signatures-psychosis_prognostic.csv'
# data = pd.read_csv(filename, index_col=0, header=0)
# sig_prognostic = data.index[data.p < 0.01] # q-value
# 
# filename = 'tmp/astral/lyriks402/biomarkers/signatures-psychosis_conversion.csv'
# data = pd.read_csv(filename, index_col=0, header=0)
# sig_conversion = data.index[data.p < 0.01] # q-value
# 
# filename = 'tmp/astral/lyriks402/biomarkers/signatures-remission.csv'
# data = pd.read_csv(filename, index_col=0, header=0)
# sig_remission = data.index[data.p < 0.01] # q-value
# 
# filename = 'tmp/astral/lyriks402/biomarkers/signatures-uhr.csv'
# data = pd.read_csv(filename, index_col=0, header=0)
# sig_uhr = data.index[data.p < 0.01] # q-value

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
prots_u = set(bm_ancova).union(set(bm_enet))
prots_d = prots_u - set(prots_mongan35)
prots_i1 = set(bm_ancova).intersection(set(bm_enet))
prots_i2 = prots_u.intersection(set(prots_mongan35))

prots_d1 = set(bm_ancova) - set(prots_mongan35)
prots_i3 = set(bm_ancova) & set(prots_mongan35)


annot = reprocessed[['Description', 'Gene']]
annot.loc[list(prots_d)]
annot.loc[list(prots_i1)]
annot.loc[list(prots_i2)]

annot.loc[list(prots_d1)]
annot.loc[list(prots_i3)]

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


# proteins = {
#     'ANCOVA (Q < .05)': set(bm_ancova),
#     'Mongan et al. biomarkers (k = 35)': set(prots_mongan35),
#     'Mongan et al. proteins (k = 166)': set(mongan.index),
# }
# 
# # Comparing high abundance proteins
# with open('tmp/astral/prots-slyriks_lt80.txt', 'r') as file:
#     prots_lt80 = file.read().splitlines()
# 
# proteins = {
#     'LYRIKS': set(prots_lt80),
#     'EU-GEI': set(mongan.index),
# }
# venn(proteins)
# filename = 'tmp/astral/lyriks402/fig/venn2-coverage.pdf'
# plt.savefig(filename)


# fig, ax = plt.subplots(figsize=(7, 7))
# draw_venn(
#     petal_labels=generate_petal_labels(proteins.values(), fmt='{size}'),
#     dataset_labels=proteins.keys(),
#     hint_hidden=False,
#     colors = ['skyblue', 'orange'],
#     # colors=bm_colors[:2] + bm_colors[3:],
#     figsize=(6, 6), fontsize=15, legend_loc='upper right', ax=ax
# )


# # Signatures
# proteins = {
#     'Psychosis prognostic signature': set(sig_prognostic),
#     'Psychosis conversion signature': set(sig_conversion),
#     'Remission prognostic signature': set(sig_remission),
#     'UHR signature': set(sig_uhr),
# }
# sig_colors = generate_colors(n_colors=4, cmap='plasma', alpha=0.4)
# 
# fig, ax = plt.subplots(figsize=(7, 7))
# draw_venn(
#     petal_labels=generate_petal_labels(proteins.values(), fmt='{size}'),
#     dataset_labels=proteins.keys(),
#     hint_hidden=False,
#     colors=sig_colors,
#     figsize=(6, 6), fontsize=15, legend_loc='upper right', ax=ax
# )
# for t in ax.texts:
#     t.set_fontsize(18)
# 
# filename = 'tmp/astral/lyriks402/fig/venn4-signatures.pdf'
# fig.savefig(filename)


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
