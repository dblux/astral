import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc 

from boruta import BorutaPy
from dataclasses import dataclass, field
from matplotlib_venn import venn2, venn3
from scipy.stats import f, ttest_ind, ttest_rel, rankdata
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFdr, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, roc_auc_score
)
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from venn import venn

@dataclass
class Result:
    metadata: dict # required init 
    predictions: list = field(default_factory=list) 
    labels: list = field(default_factory=list) 
    probas: list = field(default_factory=list) 
    pvals: list = field(default_factory=list) 
    test_statistics: list = field(default_factory=list) 
    ranks: list = field(default_factory=list) 
    features: list = field(default_factory=list) 

def unpaired_ttest(x, y):
    # Ensure inputs are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    levels = np.unique(y)
    assert len(levels) == 2
    group1 = x[y == levels[0]]
    group2 = x[y == levels[1]]
    t_statistic, p_value = ttest_ind(group1, group2)
    return t_statistic, p_value


file = 'data/astral/processed/combat_knn5_lyriks.csv'
lyriks = pd.read_csv(file, index_col=0, header=0).T
file = 'data/astral/processed/metadata-lyriks.csv'
md = pd.read_csv(file, index_col=0, header=0)

filepath = 'data/astral/raw/report.pg_matrix.tsv'
raw = pd.read_csv(filepath, sep='\t', index_col=0, header=0)

filename = 'data/astral/raw/reprocessed-all.csv'
reprocessed = pd.read_csv(filename, index_col=0, header=0)

# Model 1A: cvt (M0) v.s. non-cvt (M0)
# Prognostic markers
# Only M0 and exclude ctrl samples
md_1a = md[(md.final_label != 'ctrl') & (md.period == 0)]
y = md_1a.final_label.replace({'rmt': 0, 'mnt': 0, 'cvt': 1})
lyriks_1a = lyriks.loc[y.index]
X = lyriks_1a
X.shape
print(y.value_counts()) # imbalanced

# Model 1B: cvt (M0) v.s. maintain (M0)
# prognostic_ancova: p < 0.05
# Prognostic markers
# Only M0 and exclude ctrl samples
md_1b = md[(md.final_label.isin(['cvt', 'mnt'])) & (md.period == 0)]
y = md_1b.final_label.replace({'mnt': 0, 'cvt': 1})
lyriks_1b = lyriks.loc[y.index]
X = lyriks_1b
print(y.value_counts()) # imbalanced

# Model 1C: cvt (M0) v.s. remit (M0)
md_1c = md[(md.final_label.isin(['cvt', 'rmt'])) & (md.period == 0)]
y = md_1c.final_label.replace({'rmt': 0, 'cvt': 1})
lyriks_1c = lyriks.loc[y.index]
X = lyriks_1c
print(y.value_counts()) # imbalanced

# Model 2A: maintain (M0) v.s. remit (M0)
md_2a = md[(md.final_label.isin(['mnt', 'rmt'])) & (md.period == 0)]
y = md_2a.final_label.replace({'mnt': 0, 'rmt': 1})
lyriks_2a = lyriks.loc[y.index]
X = lyriks_2a
print(y.value_counts()) # imbalanced

# Model 2B: maintain (M0) v.s. early remit (M0)
md_2b = md[(md.label.isin(['maintain', 'early_remit'])) & (md.period == 0)]
y = md_2b.final_label.replace({'mnt': 0, 'rmt': 1})
lyriks_2b = lyriks.loc[y.index]
X = lyriks_2b
print(y.value_counts()) # imbalanced
# Late remit patients: 9

##### Biomarkers #####
# UHR biomarkers: control (M0/12/24) v.s. maintain (M0)
# mnt patients are most likely medicated after M0 
md_3 = md[(md.final_label.isin(['ctrl', 'mnt'])) & (md.period == 0)]
y = md_3.final_label.replace({'ctrl': 0, 'mnt': 1})
lyriks_3 = lyriks.loc[y.index]
# Prepare dataframe
lyriks_3.columns = lyriks_3.columns.str.replace(';', '')
lyriks_3['final_label'] = md.loc[lyriks_3.index, 'final_label']
lyriks_3['age'] = md.loc[lyriks_3.index, 'age']
lyriks_3['gender'] = md.loc[lyriks_3.index, 'gender']
print(y.value_counts()) # imbalanced

pvalues = []
for prot in lyriks_3.columns[:-3]:
    model = ols(
        f'{prot} ~ final_label + age + gender',
        data=lyriks_3
    ).fit()
    print(model.pvalues.index[1])
    pvalues.append(model.pvalues[1])
    # table = sm.stats.anova_lm(model, typ=2)

_, qvalues, _, _ = multipletests(pvalues, alpha=0.05, method='fdr_bh')
stats = pd.DataFrame(
    {'p': pvalues, 'q': qvalues},
    index=lyriks.columns
)
prots_p = stats.index[stats.p < 0.05] # p-value
prots_q = stats.index[stats.q < 0.05] # q-value

filename = 'tmp/astral/uhr_3a-ancova.csv'
stats.to_csv(filename)

filename = 'tmp/astral/uhr_3a-ancova.csv'
uhr_biomarkers = pd.read_csv(filename, index_col=0, header=0)

# TODO: Investigate medication effects
# maintain (M0) v.s. maintain (M12/24)

##### Feature selection #####
# Mongan et al.
# P02489 is not in reprocessed data, P43320 is in reprocessed data
filename = 'data/astral/etc/mongan-etable5.csv'
mongan = pd.read_csv(filename, index_col=0)
mongan_prots = mongan.index[mongan.q < 0.05]
in_astral = mongan_prots.isin(lyriks.columns)
missing_astral = mongan_prots[~in_astral]
mongan_prots_astral = mongan_prots[in_astral]
mongan_prot_idx = lyriks.columns.get_indexer(mongan_prots_astral)
len(mongan_prots)
len(mongan_prots_astral)
set(mongan_prots) - set(mongan_prots_astral)
# mongan_prots[~mongan_prots.isin(reprocessed.index)]
# sum(reprocessed.index == 'P43320')
# reprocessed.loc['P43320', :]

# Feature selection: Direction-agnostic changes in psychotic (M0, M24)
cvt_pids = set([
    md.loc[sid, 'sn'] for sid in lyriks.index
    if md.loc[sid, 'final_label'] == 'cvt'
])
### Pairs: (0, 24)
cvt_pairs = [
    (sid + '_0', sid + '_24') for sid in sorted(list(cvt_pids))
    if sid + '_0' in lyriks.index and sid + '_24' in lyriks.index
]
cvt1, cvt2 = zip(*cvt_pairs)
cvt1, cvt2 = list(cvt1), list(cvt2) 
M0 = lyriks.loc[cvt1, ]
M24 = lyriks.loc[cvt2, ]
absdelta = abs(M24 - M0.values)
feat_meanabs = absdelta.mean()
# Top 30 features
psychotic_abs_prots = feat_meanabs.nlargest(30).index

# Feature selection (boruta)
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

# Cross-validation
cross_validators = {
    'loocv': LeaveOneOut(),
    'kfold': StratifiedKFold(n_splits=4),
}

# Run detail
result = Result({
    'version': '1a',
    'selector': 'prognostic_ancova',
    'model': 'elasticnet',
    'validator': 'loocv',
    'snapshot': {}, 
})
cross_validator = cross_validators[result.metadata['validator']]
for i, (train_idx, test_idx) in enumerate(cross_validator.split(X, y)):
    print(f"Fold: {i}")
    print('---------------')
    # Split data into train and test
    train_sids = X.index[train_idx]
    X_train, X_test = X.iloc[train_idx].values, X.iloc[test_idx].values
    y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values
    match result.metadata:
        case {'selector': 'psychotic_ttest'}: 
            # paired t-test
            # TODO: feat sel: psychotic markers
            cvt_pids = [
                md.loc[sid, 'sn'] for sid in X.index[train_idx]
                if md.loc[sid, 'final_label'] == 'cvt'
            ]
            ### Pairs: (0, 24)
            cvt_pairs = [
                (sid + '_0', sid + '_24') for sid in cvt_pids
                if sid + '_24' in lyriks.index
            ]
            cvt1, cvt2 = zip(*cvt_pairs)
            cvt1, cvt2 = list(cvt1), list(cvt2) 
            # ### Pairs: (FEP-1, 24)
            # cvt_pid_fil = [
            #     pid for pid in cvt_pids
            #     if pid + '_24' in lyriks.index
            # ]
            # cvt_samples = md.loc[
            #     md.sn.isin(cvt_pid_fil),
            #     ['sn', 'period', 'final_label']
            # ]
            # cvt_pairs = (
            #     cvt_samples
            #     .sort_values(by='period')
            #     .groupby('sn').tail(2)
            #     .sort_values(by='sn')
            # )
            # cvt1 = cvt_pairs.index[cvt_pairs.period != 24]
            # cvt2 = cvt_pairs.index[cvt_pairs.period == 24]
            # Paired t-test
            pvalues = [
                ttest_rel(
                    lyriks.loc[cvt1, prot],
                    lyriks.loc[cvt2, prot]
                ).pvalue
                for prot in list(X)
            ]
            result.pvals.append(pvalues)
            _, qvalues, _, _ = multipletests(
                pvalues, alpha=0.05, method='fdr_bh'
            )
            statvalues = pd.DataFrame(
                {'p': pvalues, 'q': qvalues},
                index=list(X)
            )
            prots = statvalues.index[statvalues.p < 0.05]
            # prots = statvalues.index[statvalues.q < 0.05]
            idx = X.columns.get_indexer(prots)
            X_train_f = X_train[:, idx]
            X_test_f = X_test[:, idx]
            print(f'No. of features selected = {X_train_f.shape[1]}')
        case {'selector': 'prognostic_ancova'}: 
            data_train = pd.concat([
                X.loc[train_sids],
                md.loc[train_sids, ['age', 'gender']],
            ], axis=1)
            data_train['final_label'] = y_train
            data_train.columns = data_train.columns.str.replace(';', '')
            # ANCOVA
            pvalues = []
            for prot in data_train.columns[:-3]:
                model = ols(
                    f'{prot} ~ final_label + age + gender',
                    data=data_train
                ).fit()
                pvalues.append(model.pvalues['final_label'])
                # table = sm.stats.anova_lm(model, typ=2)
            result.pvals.append(pvalues)
            _, qvalues, _, _ = multipletests(pvalues, alpha=0.05, method='fdr_bh')
            statvalues = pd.DataFrame(
                {'p': pvalues, 'q': qvalues},
                index=X.columns
            )
            # prots = statvalues.index[statvalues.p < 0.05] # p-value
            prots = statvalues.index[statvalues.q < 0.05] # q-value
            idx = X.columns.get_indexer(prots)
            X_train_f = X_train[:, idx]
            X_test_f = X_test[:, idx]
            print(f'No. of features selected = {X_train_f.shape[1]}')
        case {'selector': 'prognostic_ttest'}: 
            selector = SelectFdr(unpaired_ttest, alpha=0.05)
            X_train_f = selector.fit_transform(X_train, y_train)
            result.pvals.append(selector.pvalues_) # p-values from F-test
            result.test_statistics.append(selector.scores_)
            print(f'No. of features selected = {X_train_f.shape[1]}')
            X_test_f = selector.transform(X_test)
        case {'selector': 'boruta'}:
            selector = BorutaPy(
                rf, n_estimators='auto', two_step=False,
                verbose=2, random_state=1
            )
            selector.fit(X_train, y_train)
            X_train_f = selector.transform(X_train)
            result.ranks.append(selector.ranking_)
            print(f'No. of features selected = {selector.n_features_}')
            X_test_f = selector.transform(X_test)
        case {'selector': 'mongan'}:
            X_train_f = X_train[:, mongan_prot_idx]
            X_test_f = X_test[:, mongan_prot_idx]
            print(f'No. of features selected = {X_train_f.shape[1]}')
    match result.metadata:
        case {'model': 'elasticnet'}:
            model = LogisticRegression(
                penalty='elasticnet', l1_ratio=0.5,
                solver='saga', max_iter=5000
            )
        case {'model': 'logreg'}:
            model = LogisticRegression(
                max_iter=5000
            )
        case {'model': 'svm_linear'}:
            model = SVC(
                C=1, kernel='linear',
                class_weight='balanced', probability=True
            )
        case {'model': 'svm_rbf'}:
            model = SVC(
                C=1, kernel='rbf',
                class_weight='balanced', probability=True
            )
    model.fit(X_train_f, y_train)
    # Predict on the test set
    y_pred = model.predict(X_test_f)
    y_prob = model.predict_proba(X_test_f)[:, 1] # Probability of the positive class
    match result.metadata:
        case {'validator': 'loocv'}:
            result.probas.append(y_prob.item())
            result.predictions.append(y_pred[0])
            result.labels.append(y_test[0])
        case {'validator': 'kfold'}:
            result.probas.extend(y_prob.tolist())
            result.predictions.extend(y_pred.tolist())
            result.labels.extend(y_test.tolist())
    print()

result.metadata['snapshot'].update({
    # 'selector': repr(selector),
    'model': repr(model),
    'validator': repr(cross_validator),
})
print(result.metadata)


# TODO: Loop over models
# TODO: ANCOVA feat sel (age, sex as covariates)

##### Feature selection ##### 

# Mongan et al.
result.features = mongan_prots_astral
len(result.features)

# BH correction
qvals = np.array([
    multipletests(p, alpha=0.05, method='fdr_bh')[1]
    for p in result.pvals
])
# Assumption: Order of columns is preserved
avg_q = pd.DataFrame({'q': qvals.mean(axis=0)}, index=list(X))
q_fil = avg_q[avg_q.q < 0.05]
result.features = q_fil.index.tolist()
len(result.features)

# No BH correction
pvals = np.array(result.pvals)
avg_p = pd.DataFrame({'p': pvals.mean(axis=0)}, index=X.columns)
p_fil = avg_p[avg_p.p < 0.05]
result.features = p_fil.index.tolist()
len(result.features)
# np.unique(np.sum(pvals < 0.05, axis=0), return_counts=True) 

# Check saved run 
pvals = np.array(result_prognostic_ancova.pvals)
avg_p = pd.DataFrame({'p': pvals.mean(axis=0)}, index=X.columns)
p_fil = avg_p[avg_p.p < 0.05]
prots = p_fil.index
len(prots)
np.unique(np.sum(pvals < 0.05, axis=0), return_counts=True) 

# Boruta
ranks = np.array(result.ranks)
avg_rank = pd.DataFrame({'Rank': ranks.mean(axis=0)}, index=list(X))
# Assumption: Order of columns is preserved
rank_fil = avg_rank[avg_rank.Rank <= 2]
result.features = rank_fil.index.tolist()
len(result.features)

##### Save results #####

# Save
name = (
    f'{result.metadata["version"]}-'
    f'{result.metadata["selector"]}-'
    f'{result.metadata["model"]}-'
    f'{result.metadata["validator"]}'
)
filename = 'tmp/astral/' + name + '.pkl'
print(filename)
with open(filename, 'wb') as file:
    pickle.dump(result, file)

# Load
filename = 'tmp/astral/1a-mongan-elasticnet-kfold.pkl'
with open(filename, 'rb') as file:
    result_mongan = pickle.load(file)

filename = 'tmp/astral/1a-psychotic_ttest-elasticnet-kfold.pkl'
with open(filename, 'rb') as file:
    result_psychotic = pickle.load(file)

filename = 'tmp/astral/1a-prognostic_ancova-elasticnet-kfold.pkl'
with open(filename, 'rb') as file:
    result_prognostic_ancova = pickle.load(file)

filename = 'tmp/astral/1a-boruta-elasticnet-kfold.pkl'
with open(filename, 'rb') as file:
    result_boruta = pickle.load(file)

filename = 'tmp/astral/2b-prognostic_ancova-elasticnet-kfold.pkl'
with open(filename, 'rb') as file:
    result_remission = pickle.load(file)

len(result_boruta.features)

##### Evaluation ##### 

# Threshold = 0.5
tn, fp, fn, tp = confusion_matrix(result.labels, result.predictions).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
npv = tn / (tn + fn)
accuracy = (tp + tn) / (tn + fp + fn + tp) 
# Calculate AUC
auc = roc_auc_score(result.labels, result.probas)
metric_repr = (
    f'AUC = {auc:.3f}; '
    f'Accuracy = {accuracy:.3f}; '
    f'Sensitivity = {sensitivity:.3f}; '
    f'Specificity = {specificity:.3f}; '
    f'Precision = {precision:.3f}; '
    f'NPV = {npv:.3f};'
)
print(metric_repr)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(result.labels, result.probas)
plt.figure(figsize=(4.5, 4))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC curve')
plt.legend()
filepath = 'tmp/astral/fig/roc-' + name + '.pdf'
print(filepath)
plt.savefig(filepath)

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

# Subset
md.period = md.period.astype(str)
data = ad.AnnData(lyriks, md)
cvt = data[data.obs.final_label == 'cvt',:]

data_mongan = data[:, mongan_prots[in_astral]]
uhr_mongan = data[data.obs.final_label != 'ctrl', mongan_prots_astral]
cvt_mongan = data[data.obs.final_label == 'cvt', mongan_prots_astral]

data_progq = data[:, result_prognostic_ancova.features]
uhr_progq = data[data.obs.final_label != 'ctrl', result_prognostic_ancova.features]

uhr_psych = data[data.obs.final_label != 'ctrl', result_psychotic.features]
cvt_psych = data[data.obs.final_label == 'cvt', result_psychotic.features]

uhr_psych_abs = data[data.obs.final_label != 'ctrl', psychotic_abs_prots]
cvt_psych_abs = data[data.obs.final_label == 'cvt', psychotic_abs_prots]

# Features: All
sc.pp.pca(data)
sc.pl.pca(
    data,
    # data[data.obs.final_label == 'cvt', :],
    color=['final_label', 'period'],
    size=100,
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

##### Features #####

prots_uhr = uhr_biomarkers.index[uhr_biomarkers.p < 0.05]


results = [
    result_prognostic_ancova,
    result_psychotic,
    result_remission,
    # result_boruta,
]

# BH correction
result = result_prognostic_ancova
qvals = np.array([
    multipletests(p, alpha=0.05, method='fdr_bh')[1]
    for p in result.pvals
])
# Assumption: Order of columns is preserved
symbols = reprocessed.loc[lyriks.columns, ['Description', 'Gene']]
symbols['q'] = qvals.mean(axis=0)
symbols_fil = symbols[symbols.q < 0.05]
name = (
    f'{result.metadata["version"]}-'
    f'{result.metadata["selector"]}'
)
filepath = 'tmp/astral/biomarkers-' + name + '.csv'
symbols_fil.sort_values(by='q').to_csv(filepath, float_format='%.3g')
print(filepath)

result = result_remission 
result = result_psychotic
# No BH correction
pvals = np.array(result.pvals)
# Assumption: Order of columns is preserved
symbols = reprocessed.loc[lyriks.columns, ['Description', 'Gene']]
symbols['p'] = pvals.mean(axis=0)
symbols_fil = symbols[symbols.p < 0.05]
name = (
    f'{result.metadata["version"]}-'
    f'{result.metadata["selector"]}'
)
filepath = 'tmp/astral/biomarkers-' + name + '.csv'
symbols_fil.sort_values(by='p').to_csv(filepath, float_format='%.3g')
print(filepath)

# Boruta
result = result_boruta
ranks = np.array(result.ranks)
# Assumption: Order of columns is preserved
symbols = reprocessed.loc[lyriks.columns, ['Description', 'Gene']]
symbols['Average rank'] = ranks.mean(axis=0)
symbols_fil = symbols[symbols['Average rank'] <= 2]
name = (
    f'{result.metadata["version"]}-'
    f'{result.metadata["selector"]}'
)
filepath = 'tmp/astral/biomarkers-' + name + '.csv'
symbols_fil.sort_values(by='Average rank').to_csv(filepath, float_format='%.3g')
print(filepath)

##### Venn #####

# Mongan

filepath = 'data/astral/etc/mongan-etable5.csv'
mongan = pd.read_csv(filepath, index_col=0, header=0)
mongan_prots = mongan.index[mongan.q < 0.05]

# TODO 
# Check missing proteins with original data
# They cannot be detected by astral?
# Any prognostic that was not detected by Mongan?

# No BH correction
pvals = np.array(result_prognostic_ancova.pvals)
avg_p = pd.DataFrame({'p': pvals.mean(axis=0)}, index=lyriks.columns)
p_fil = avg_p[avg_p.p < 0.05]
prots_progp = p_fil.index.tolist()
len(prots_progp)

proteins = {
    'UHR biomarker': set(prots_p),
    'Prognostic (ANCOVA; q < 0.05)': set(result_prognostic_ancova.features),
    'Psychotic (M0, M24)': set(result_psychotic.features),
    'Remission (ANCOVA; p < 0.05)': set(result.features),
}
venn(proteins)
filename = 'tmp/astral/fig/venn4-feats-uhrp_progq_psychotic_remission.pdf'
plt.savefig(filename)

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
