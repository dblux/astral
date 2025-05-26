import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc 
import statsmodels.api as sm

from boruta import BorutaPy
from dataclasses import dataclass, field
from scipy.stats import f, ttest_ind, ttest_rel, rankdata
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFdr, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, roc_auc_score
)
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from venn import draw_venn, generate_petal_labels, generate_colors, venn


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


# file = 'data/astral/processed/combat_knn5_lyriks.csv'
# lyriks = pd.read_csv(file, index_col=0, header=0).T

file = 'data/astral/processed/combat_knn5_lyriks-605_402.csv'
lyriks = pd.read_csv(file, index_col=0, header=0).T

# file = 'data/astral/processed/metadata-lyriks.csv'
# md = pd.read_csv(file, index_col=0, header=0)

file = 'data/astral/processed/metadata-lyriks407.csv'
md = pd.read_csv(file, index_col=0, header=0)
md = md[md.label != 'QC']
md['period'] = md['period'].astype(int)

filepath = 'data/astral/raw/report.pg_matrix.tsv'
raw = pd.read_csv(filepath, sep='\t', index_col=0, header=0)

filepath = 'data/astral/raw/reprocessed-data.csv'
reprocessed = pd.read_csv(filepath, index_col=0, header=0)

# Model 1A: cvt (M0) v.s. non-cvt (M0)
# Prognostic markers
# Only M0 and exclude ctrl samples
md_1a = md[(md.final_label != 'ctrl') & (md.period == 0)]
y = md_1a.final_label.replace({'rmt': 0, 'mnt': 0, 'cvt': 1})
lyriks_1a = lyriks.loc[y.index]
X = lyriks_1a
print(y.value_counts()) # imbalanced

# # Model 1B: cvt (M0) v.s. maintain (M0)
# # prognostic_ancova: p < 0.05
# # Prognostic markers
# # Only M0 and exclude ctrl samples
# md_1b = md[(md.final_label.isin(['cvt', 'mnt'])) & (md.period == 0)]
# y = md_1b.final_label.replace({'mnt': 0, 'cvt': 1})
# lyriks_1b = lyriks.loc[y.index]
# X = lyriks_1b
# print(y.value_counts()) # imbalanced
# 
# # Model 1C: cvt (M0) v.s. remit (M0)
# md_1c = md[(md.final_label.isin(['cvt', 'rmt'])) & (md.period == 0)]
# y = md_1c.final_label.replace({'rmt': 0, 'cvt': 1})
# lyriks_1c = lyriks.loc[y.index]
# X = lyriks_1c
# print(y.value_counts()) # imbalanced
# 
# # Model 2A: maintain (M0) v.s. remit (M0)
# md_2a = md[(md.final_label.isin(['mnt', 'rmt'])) & (md.period == 0)]
# y = md_2a.final_label.replace({'mnt': 0, 'rmt': 1})
# lyriks_2a = lyriks.loc[y.index]
# X = lyriks_2a
# print(y.value_counts()) # imbalanced
# 
# # Model 2B: maintain (M0) v.s. early remit (M0)
# md_2b = md[(md.label.isin(['maintain', 'early_remit'])) & (md.period == 0)]
# y = md_2b.final_label.replace({'mnt': 0, 'rmt': 1})
# lyriks_2b = lyriks.loc[y.index]
# X = lyriks_2b
# print(y.value_counts()) # imbalanced
# # Late remit patients: 9

# ##### Biomarkers (entire data set) #####
# 
# # Psychosis prognostic (ANCOVA, BH)
# 
# # Dataframe with proteomic and clinical features
# # relapse is labelled as mnt
# md_1a = md.loc[(md.final_label != 'ctrl') & (md.period == 0)]
# data = pd.concat([
#     lyriks.loc[md_1a.index],
#     md_1a.final_label.replace({
#         'rmt': 'non-cvt', 'mnt': 'non-cvt'
#     }).astype('category').cat.reorder_categories(['non-cvt', 'cvt']),
#     md_1a[['age', 'gender']]
# ], axis=1)
# print(data.final_label.value_counts()) # imbalanced
# print(data.final_label.cat.categories)
# 
# pvalues = []
# coeffs = []
# for prot in data.columns[:-3]:
#     model = ols(
#         f'{prot} ~ final_label + age + gender',
#         data=data
#     ).fit()
#     pvalues.append(model.pvalues[1])
#     coeffs.append(model.params[1])
#     # table = sm.stats.anova_lm(model, typ=2)
# 
# print(model.pvalues.index[1])
# print(model.pvalues)
# 
# _, qvalues, _, _ = multipletests(pvalues, alpha=0.05, method='fdr_bh')
# stats = pd.DataFrame(
#     {'Coefficient': coeffs, 'p': pvalues, 'q': qvalues},
#     index=lyriks.columns
# )
# filename = 'tmp/astral/lyriks402/1a-prognostic_ancova.csv'
# stats.to_csv(filename)
# 
# # UHR biomarkers: control (M0/12/24) v.s. maintain (M0)
# # mnt patients are most likely medicated after M0 
# md_3 = md[(md.label.isin(['control', 'maintain'])) & (md.period == 0)]
# lyriks_3 = lyriks.loc[md_3.index]
# data = lyriks_3.join(md_3[['label', 'age', 'gender']])
# print(data.label.value_counts()) # imbalanced
# 
# pvalues = []
# coeffs = []
# for prot in data.columns[:-3]:
#     model = ols(
#         f'{prot} ~ label + age + gender',
#         data=data
#     ).fit()
#     pvalues.append(model.pvalues[1])
#     coeffs.append(model.params[1])
# 
# print(model.pvalues.index[1])
# 
# _, qvalues, _, _ = multipletests(pvalues, alpha=0.05, method='fdr_bh')
# stats = pd.DataFrame(
#     {'Coefficient': coeffs, 'p': pvalues, 'q': qvalues},
#     index=lyriks.columns
# )
# filename = 'tmp/astral/lyriks402/uhr_3a-ancova.csv'
# stats.to_csv(filename)
# 
# # mnt v.s. early_remit
# md_2b = md[(md.label.isin(['maintain', 'early_remit'])) & (md.period == 0)]
# md_2b.label = md_2b.label.astype('category').cat.reorder_categories([
#     'maintain', 'early_remit'
# ])
# lyriks_2b = lyriks.loc[md_2b.index]
# data = lyriks_2b.join(md_2b[['label', 'age', 'gender']])
# 
# pvalues = []
# coeffs = []
# for prot in data.columns[:-3]:
#     model = ols(
#         f'{prot} ~ label + age + gender',
#         data=data
#     ).fit()
#     pvalues.append(model.pvalues[1])
#     coeffs.append(model.params[1])
# 
# print(model.pvalues.index[1])
# 
# _, qvalues, _, _ = multipletests(pvalues, alpha=0.05, method='fdr_bh')
# stats = pd.DataFrame(
#     {'Coefficient': coeffs, 'p': pvalues, 'q': qvalues},
#     index=lyriks.columns
# )
# 
# filename = 'tmp/astral/lyriks402/2b-prognostic_ancova.csv'
# stats.to_csv(filename)
# 
# # Paired t-test
# cvt_pids = set([
#     md407.loc[sid, 'sn'] for sid in lyriks.index
#     if md407.loc[sid, 'final_label'] == 'cvt'
# ])
# ### Pairs: (0, 24)
# cvt_pairs = [
#     (sid + '_0', sid + '_24') for sid in cvt_pids
#     if sid + '_24' in lyriks.index
# ]
# cvt1, cvt2 = zip(*cvt_pairs)
# cvt1, cvt2 = list(cvt1), list(cvt2) 
# 
# # t-statistic is np.mean(a - b) / se
# paired_ttests = [
#     ttest_rel(
#         lyriks.loc[cvt2, prot],
#         lyriks.loc[cvt1, prot]
#     )
#     for prot in lyriks.columns
# ]
# 
# res = [(test.statistic, test.pvalue) for test in paired_ttests]
# tstats, pvalues = zip(*res)
# _, qvalues, _, _ = multipletests(pvalues, alpha=0.05, method='fdr_bh')
# stats = pd.DataFrame(
#     {'t': tstats, 'p': pvalues, 'q': qvalues},
#     index=lyriks.columns
# )
# sum(stats.p < 0.01)
# 
# filename = 'tmp/astral/lyriks402/1a-psychotic_ttest.csv'
# stats.to_csv(filename)
# 
# # TODO: Investigate medication effects
# # maintain (M0) v.s. maintain (M12/24)


##### Feature pre-selection #####

### Mongan et al.
# P02489 is not in reprocessed data (not detected)
# P43320 is in reprocessed data
filename = 'data/astral/etc/mongan-etable5.csv'
mongan = pd.read_csv(filename, index_col=0)

mongan_prots162 = mongan.index[mongan.index.isin(lyriks.columns)]
mongan_prots162_idx = lyriks.columns.get_indexer(mongan_prots162)
mongan_prots35 = mongan.index[mongan.q < 0.05]
in_astral = mongan_prots35.isin(lyriks.columns)
missing_astral = mongan_prots35[~in_astral]
mongan_prots33 = mongan_prots35[in_astral]
mongan_prots33_idx = lyriks.columns.get_indexer(mongan_prots33)
# sum(reprocessed.index == 'P43320')
# reprocessed.loc['P43320', :]
mongan_prots10 = pd.Index([
    'P01023', 'P01871', 'P04003', 'P07225', 'P23142',
    'P02766', 'Q96PD5', 'P02774', 'P10909', 'P13671'
])
in_astral10 = mongan_prots10.isin(lyriks.columns)
mongan_prots10 = mongan_prots10[in_astral10]
mongan_prots10_idx = lyriks.columns.get_indexer(mongan_prots10)
len(mongan_prots10_idx)

filepath = 'data/astral/etc/perkins.csv'
perkins = pd.read_csv(filepath, index_col=0)
prots_perkins = perkins['UniProt/CAS'][
    perkins['UniProt/CAS'].str.startswith(('P', 'Q'))]
prots_perkins_idx = lyriks.columns.get_indexer(prots_perkins)
# only 3 proteins present in lyriks dataset

### ANCOVA feature selection
filepath = "tmp/astral/lyriks402/new/biomarkers/biomarkers-ancova.csv"
bm_ancova = pd.read_csv(filepath, index_col=0)

### ANCOVA & Mongan
combined_bm = bm_ancova.index.union(mongan_prots33)
combined_bm_idx = lyriks.columns.get_indexer(combined_bm)
len(combined_bm)

### ANCOVA & Mongan
prots_union = set(mongan_prots_astral) & set(bm_ancova)

### ElasticNet selected features

filename = 'tmp/astral/1a-none-elasticnet-kfold.pkl'
with open(filename, 'rb') as file:
    result_none = pickle.load(file)

coefficients = np.vstack(result_none.coefficients)
coeff = pd.DataFrame({
    'Coefficient': coefficients.mean(axis=0),
    'Dropped': np.sum(coefficients == 0, axis=0),
}, index=lyriks.columns)

prots_elastic10 = coeff.Coefficient.abs().nlargest(10).index
prots_elastic30 = coeff.Coefficient.abs().nlargest(30).index
prots_idx_elastic10 = lyriks.columns.get_indexer(prots_elastic10)
prots_idx_elastic30 = lyriks.columns.get_indexer(prots_elastic30)
coeff.loc[prots_elastic30]

### Direction-agnostic changes in psychotic (M0, M24)
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

##### Prediction modelling #####

# Feature selection (boruta)
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

# Cross-validation
cross_validators = {
    'loocv': LeaveOneOut(),
    'kfold': StratifiedKFold(n_splits=4),
    'inner_kfold': StratifiedKFold(n_splits=3),
}

# Run detail
np.random.seed(0)
n_perms = 1000
for j in range(n_perms):
    result = Result({
        'version': '1a',
        'selector': 'random15',
        'model': 'elasticnet-bal',
        'class_weight': 'balanced',
        'validator': 'kfold',
        'snapshot': {}, 
    })
    rnd_idx = np.random.choice(X.shape[1], 15, replace=False)
    result.features = rnd_idx
    print(result.features)
    cross_validator = cross_validators[result.metadata['validator']]
    for i, (train_idx, test_idx) in enumerate(cross_validator.split(X, y)):
        print(f"Fold: {i}")
        print('---------------')
        # Split data into train and test
        train_sids = X.index[train_idx]
        X_train, X_test = X.iloc[train_idx].values, X.iloc[test_idx].values
        y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values
        match result.metadata:
            case {'selector': 'prognostic_ancova'}: 
                data_train = pd.concat([
                    X.loc[train_sids],
                    md.loc[train_sids, ['age', 'gender']],
                ], axis=1)
                data_train['final_label'] = y_train
                # ANCOVA
                pvalues = []
                coefs = []
                for prot in data_train.columns[:-3]:
                    model = ols(
                        f'{prot} ~ final_label + age + gender',
                        data=data_train
                    ).fit()
                    pvalues.append(model.pvalues['final_label'])
                    coefs.append(model.params['final_label'])
                    # table = sm.stats.anova_lm(model, typ=2)
                result.pvals.append(pvalues)
                result.test_statistics.append(coefs)
                _, qvalues, _, _ = multipletests(
                    pvalues, alpha=0.05, method='fdr_bh'
                )
                statvalues = pd.DataFrame(
                    {'p': pvalues, 'q': qvalues},
                    index=X.columns
                )
                prots = statvalues.index[statvalues.p < 0.01] # p-value
                # prots = statvalues.index[statvalues.q < 0.05] # q-value
                idx = X.columns.get_indexer(prots)
                X_train_f = X_train[:, idx]
                X_test_f = X_test[:, idx]
            case {'selector': 'prognostic_ttest'}: 
                selector = SelectFdr(unpaired_ttest, alpha=0.05)
                X_train_f = selector.fit_transform(X_train, y_train)
                result.pvals.append(selector.pvalues_) # p-values from F-test
                result.test_statistics.append(selector.scores_)
                X_test_f = selector.transform(X_test)
            case {'selector': 'psychotic_ttest'}: 
                # paired t-test
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
                        lyriks.loc[cvt2, prot],
                        lyriks.loc[cvt1, prot]
                    ).pvalue
                    for prot in X.columns 
                ]
                result.pvals.append(pvalues)
                _, qvalues, _, _ = multipletests(
                    pvalues, alpha=0.05, method='fdr_bh'
                )
                statvalues = pd.DataFrame(
                    {'p': pvalues, 'q': qvalues},
                    index=X.columns
                )
                prots = statvalues.index[statvalues.p < 0.05]
                # prots = statvalues.index[statvalues.q < 0.05]
                idx = X.columns.get_indexer(prots)
                X_train_f = X_train[:, idx]
                X_test_f = X_test[:, idx]
            case {'selector': 'boruta'}:
                selector = BorutaPy(
                    rf, n_estimators='auto', two_step=False,
                    verbose=2, random_state=1
                )
                selector.fit(X_train, y_train)
                X_train_f = selector.transform(X_train)
                result.ranks.append(selector.ranking_)
                X_test_f = selector.transform(X_test)
            case {'selector': 'mongan162'}:
                X_train_f = X_train[:, mongan_prots162_idx]
                X_test_f = X_test[:, mongan_prots162_idx]
            case {'selector': 'mongan33'}:
                X_train_f = X_train[:, mongan_prots33_idx]
                X_test_f = X_test[:, mongan_prots33_idx]
            case {'selector': 'mongan10'}:
                X_train_f = X_train[:, mongan_prots10_idx]
                X_test_f = X_test[:, mongan_prots10_idx]
            case {'selector': 'none'}:
                X_train_f = X_train
                X_test_f = X_test
            case {'selector': 'elasticnet10'}:
                X_train_f = X_train[:, prots_idx_elastic10]
                X_test_f = X_test[:, prots_idx_elastic10]
            case {'selector': 'elasticnet30'}:
                X_train_f = X_train[:, prots_idx_elastic30]
                X_test_f = X_test[:, prots_idx_elastic30]
            case {'selector': 'combined'}:
                X_train_f = X_train[:, combined_bm_idx]
                X_test_f = X_test[:, combined_bm_idx]
            case {'selector': 'random15'}:
                X_train_f = X_train[:, rnd_idx]
                X_test_f = X_test[:, rnd_idx]
        print(f'No. of features selected = {X_train_f.shape[1]}')
        match result.metadata:
            case {'model': 'elasticnet'}:
                model = LogisticRegression(
                    penalty='elasticnet', l1_ratio=0.5,
                    solver='saga', max_iter=10000
                )
            case {'model': 'elasticnet-bal'}:
                model = LogisticRegression(
                    penalty='elasticnet', l1_ratio=0.5,
                    class_weight = 'balanced',
                    solver='saga', max_iter=10000
                )
            case {'model': 'logreg'}:
                model = LogisticRegression(
                    max_iter=5000
                )
            case {'model': 'svm-linear'}:
                model = LinearSVC(
                    penalty='l2', C=1, 
                    class_weight='balanced',
                    max_iter=int(1e5)
                )
            case {'model': 'svm-rbf'}:
                model = SVC(
                    C=1, kernel='rbf',
                    class_weight='balanced', probability=True
                )
        model.fit(X_train_f, y_train)
        if hasattr(model, 'coef_'):
            result.coefficients.append(model.coef_)
        # Predict on the test set
        if result.metadata['model'] == 'svm-linear':
            y_pred = model.predict(X_test_f)
            y_prob = model.decision_function(X_test_f)
        else:
            y_pred = model.predict(X_test_f)
            y_prob = model.predict_proba(X_test_f)[:, 1] # Probability of the positive class
        match result.metadata:
            case {'validator': 'loocv'}:
                result.probas.append(y_prob.item())
                result.predictions.append(y_pred[0])
                result.labels.append(y_test[0])
            case {'validator': 'kfold'}:
                result.probas.append(y_prob.tolist())
                result.predictions.append(y_pred.tolist())
                result.labels.append(y_test.tolist())
        print()
    result.metadata['snapshot'].update({
        'model': repr(model),
        'validator': repr(cross_validator),
    })
    # Save
    name = (
        f'{result.metadata["version"]}-'
        f'{result.metadata["selector"]}-'
        f'{result.metadata["model"]}-'
        f'{result.metadata["validator"]}'
    )
    filename = f'tmp/astral/lyriks402/new/pickle/random15/{name}-{j}.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(result, file)
    print(filename)


# Nested k-fold cross-validation for feature selection by elasticnet
result = Result({
    'version': '1a',
    'selector': 'elasticnet-bal',
    'model': 'elasticnet-bal',
    'class_weight': 'balanced',
    'validator': 'nestedkfold',
    'snapshot': {}, 
})
for i, (train_out_idx, test_out_idx) in enumerate(
    cross_validators['kfold'].split(X, y)
):
    print('===============')
    print(f"Outer fold: {i}")
    print('===============')
    match result.metadata:
        case {'model': 'elasticnet'}:
            outer_model = LogisticRegression(
                penalty='elasticnet', l1_ratio=0.5,
                solver='saga', max_iter=10000
            )
        case {'model': 'elasticnet-bal'}:
            outer_model = LogisticRegression(
                penalty='elasticnet', l1_ratio=0.5,
                class_weight = 'balanced',
                solver='saga', max_iter=10000
            )
        case {'model': 'svm-linear'}:
            outer_model = LinearSVC(
                penalty='l2', C=1,
                class_weight='balanced',
                max_iter=int(1e5)
            )
    # Split data into train and test
    X_train_out, X_test_out = X.iloc[train_out_idx], X.iloc[test_out_idx]
    y_train_out, y_test_out = y.iloc[train_out_idx], y.iloc[test_out_idx]
    inner_fold = {
        'probas': [],
        'predictions': [],
        'labels': [],
        'coefficients': [],
    }
    for j, (train_idx, test_idx) in enumerate(
        cross_validators['inner_kfold'].split(X_train_out, y_train_out)
    ):
        print(f"Inner fold: {j}")
        # print('---------------')
        X_train_in, X_test_in = X_train_out.iloc[train_idx], X_train_out.iloc[test_idx]
        y_train_in, y_test_in = y_train_out.iloc[train_idx], y_train_out.iloc[test_idx]
        match result.metadata:
            case {'selector': 'elasticnet'}:
                inner_model = LogisticRegression(
                    penalty='elasticnet', l1_ratio=0.5,
                    solver='saga', max_iter=10000
                )
            case {'selector': 'elasticnet-bal'}:
                inner_model = LogisticRegression(
                    penalty='elasticnet', l1_ratio=0.5,
                    class_weight = 'balanced',
                    solver='saga', max_iter=10000
                )
            case {'selector': 'svm-linear'}:
                inner_model = LinearSVC(
                    penalty='l2', C=1, 
                    class_weight='balanced',
                    max_iter=int(1e5)
                )
        if result.metadata['selector'] == 'svm-linear':
            inner_model.fit(X_train_in, y_train_in)
            y_pred = inner_model.predict(X_test_in)
            y_prob = inner_model.decision_function(X_test_in)
            inner_fold['coefficients'].append(inner_model.coef_)
            inner_fold['probas'].extend(y_prob.tolist())
            inner_fold['predictions'].extend(y_pred.tolist())
            inner_fold['labels'].extend(y_test_in.tolist())
        else:
            inner_model.fit(X_train_in, y_train_in)
            y_pred = inner_model.predict(X_test_in)
            y_prob = inner_model.predict_proba(X_test_in)[:, 1] # Probability of the positive class
            inner_fold['coefficients'].append(inner_model.coef_)
            inner_fold['probas'].extend(y_prob.tolist())
            inner_fold['predictions'].extend(y_pred.tolist())
            inner_fold['labels'].extend(y_test_in.tolist())
        print('-----')
    result.inner_folds.append(inner_fold)
    coefficients = np.vstack(inner_fold['coefficients'])
    result.coefficients.append(coefficients)
    # Fit outer model
    if result.metadata['model'] == 'svm-linear':
        # Average coefficients from models in the internal folds
        mean_coef = pd.DataFrame({
            'coefficient': coefficients.mean(axis=0),
            'dropped': np.sum(coefficients == 0, axis=0),
        }, index=lyriks.columns)
        # Take the top 10 proteins with highest coefficient magnitude!
        prots = mean_coef.coefficient.abs().nlargest(10).index
        X_train_out_f, X_test_out_f = X_train_out[prots], X_test_out[prots] 
        outer_model.fit(X_train_out_f, y_train_out)
        y_pred = outer_model.predict(X_test_out_f)
        y_prob = outer_model.decision_function(X_test_out_f)
        result.probas.append(y_prob.tolist())
        result.predictions.append(y_pred.tolist())
        result.labels.append(y_test_out.tolist())
    else:
        # Average coefficients from models in the internal folds
        mean_coef = pd.DataFrame({
            'coefficient': coefficients.mean(axis=0),
            'dropped': np.sum(coefficients == 0, axis=0),
        }, index=lyriks.columns)
        # filter for coefficients that are all non-zero before taking top 10 magnitude
        prots = mean_coef.coefficient[mean_coef.dropped == 0].abs().nlargest(10).index
        X_train_out_f, X_test_out_f = X_train_out[prots], X_test_out[prots] 
        outer_model.fit(X_train_out_f, y_train_out)
        y_pred = outer_model.predict(X_test_out_f)
        y_prob = outer_model.predict_proba(X_test_out_f)[:, 1] # Probability of the positive class
        result.probas.append(y_prob.tolist())
        result.predictions.append(y_pred.tolist())
        result.labels.append(y_test_out.tolist())

result.metadata['snapshot'].update({
    'model': repr(outer_model),
    'validator': [
        repr(cross_validators['kfold']),
        repr(cross_validators['inner_kfold'])
    ]
})
print(result.metadata)


##### Feature selection ##### 
filepath = 'tmp/astral/lyriks402/new/pickle/1a-elasticnet-elasticnet-bal-nestedkfold.pkl'
with open(filepath, 'rb') as file:
    result = pickle.load(file)

# ANCOVA & Mongan
result.features = combined_bm.tolist()
len(result.features)

# Mongan et al. (m = 162)
result.features = mongan_prots162
print(result.features)
len(result.features)

# Mongan et al. (m = 33)
result.features = mongan_prots162
print(result.features)
len(result.features)

# Mongan et al. (m = 10)
result.features = mongan_prots10
print(result.features)
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

# BH correction and ANCOVA coefficients
qvals = np.array([
    multipletests(p, alpha=0.05, method='fdr_bh')[1]
    for p in result.pvals
])
ancova_coefs  = np.array(result.test_statistics)
stats = pd.DataFrame({
    'coefficient': ancova_coefs.mean(axis=0),
    'q': qvals.mean(axis=0),
}, index=X.columns)
stats_f = stats[stats.q < 0.05]
stats_fs = stats_f.sort_values(by='q')
result.features = stats_fs.index.tolist()
len(result.features)

filepath = 'tmp/astral/lyriks402/new/features-prognostic_ancova-kfold.csv'
stats_fs.to_csv(filepath)

# filepath = 'tmp/astral/lyriks402/new/features_all-ancova.csv'
# stats.to_csv(filepath)

# No BH correction
pvals = np.array(result.pvals)
avg_p = pd.DataFrame({'p': pvals.mean(axis=0)}, index=X.columns)
p_fil = avg_p[avg_p.p < 0.01]
result.features = p_fil.index.tolist()
len(result.features)
# np.unique(np.sum(pvals < 0.05, axis=0), return_counts=True) 

# Boruta
ranks = np.array(result.ranks)
avg_rank = pd.DataFrame({'Rank': ranks.mean(axis=0)}, index=list(X))
# Assumption: Order of columns is preserved
rank_fil = avg_rank[avg_rank.Rank <= 2]
result.features = rank_fil.index.tolist()
len(result.features)

filepath = 'tmp/astral/lyriks402/1a-boruta.csv'
rank_fil.to_csv(filepath)

# No feature selection
result.features = X.columns
len(result.features)

# Logistic regression (elastic) - coefficients
coefficients = np.vstack(result.coefficients)
coeff = pd.DataFrame({
    'Coefficient': coefficients.mean(axis=0),
    'Dropped': np.sum(coefficients == 0, axis=0),
}, index=lyriks.columns)
result.features = coeff.Coefficient.abs().nlargest(10).index
coeff.loc[result.features]
plt.hist(coeff.loc[coeff.Dropped == 0, 'Coefficient'], bins=30)
plt.show()

### Nested k-fold ###
# Elastic net
# Filter for proteins that have always been selected
# Take the top 10 proteins with highest coefficient magnitude
coefs = pd.DataFrame(np.vstack(result.coefficients).T, index=X.columns)
agg_coefs = pd.DataFrame({
    'Mean coefficient': coefs.mean(axis=1),
    'Missing': np.sum(coefs == 0, axis=1),
}, index=coefs.index)
agg_coefs_f = agg_coefs[agg_coefs.Missing == 0]
result.features = agg_coefs_f['Mean coefficient'].abs().nlargest(10).index.tolist()
agg_coefs['Selected'] = agg_coefs.index.isin(result.features)

# filepath = 'tmp/astral/lyriks402/new/biomarkers/features_all-elasticnet.csv'
# agg_coefs.to_csv(filepath)

filepath = 'tmp/astral/lyriks402/new/features-elasticnet-bal-nestedkfold.csv'
coefs.loc[result.features].mean(axis=1).to_csv(filepath)

# SVM (Mongan et al.)
coefs = pd.DataFrame(np.vstack(result.coefficients).T, index=X.columns)
mean_coefs = coefs.mean(axis=1)
result.features = mean_coefs.abs().nlargest(10).index.tolist()
len(result.features)

##### Save results #####

# Save
name = (
    f'{result.metadata["version"]}-'
    f'{result.metadata["selector"]}-'
    f'{result.metadata["model"]}-'
    f'{result.metadata["validator"]}'
)
filename = 'tmp/astral/lyriks402/new/' + name + '.pkl'
with open(filename, 'wb') as file:
    pickle.dump(result, file)

print(filename)

##### Load results #####

# ANCOVA (balanced)
filename = 'tmp/astral/lyriks402/new/1a-prognostic_ancova-elasticnet-bal-kfold.pkl'
with open(filename, 'rb') as file:
    result_ancova_bal = pickle.load(file)

filename = 'tmp/astral/lyriks402/new/1a-none-elasticnet-bal-nestedkfold.pkl'
with open(filename, 'rb') as file:
    result_enet_bal = pickle.load(file)

# 2A
filename = 'tmp/astral/lyriks402/1a-prognostic_ancova-elasticnet-kfold.pkl'
with open(filename, 'rb') as file:
    result_prognostic_ancova = pickle.load(file)

# 2B
filename = 'tmp/astral/lyriks402/1a-none-elasticnet-nestedkfold.pkl'
with open(filename, 'rb') as file:
    result_elasticnet = pickle.load(file)

# 2C
filename = 'tmp/astral/lyriks402/new/1a-mongan33-elasticnet-kfold.pkl'
with open(filename, 'rb') as file:
    result_mongan33 = pickle.load(file)

# 2D
filename = 'tmp/astral/lyriks402/1a-none-elasticnet-kfold.pkl'
with open(filename, 'rb') as file:
    result_605 = pickle.load(file)

# 2E
filename = 'tmp/astral/lyriks402/2b-prognostic_ancova-elasticnet-kfold.pkl'
with open(filename, 'rb') as file:
    result_remission = pickle.load(file)

# e1A
filename = 'tmp/astral/lyriks402/1a-prognostic_ancova-svm_linear-kfold.pkl'
with open(filename, 'rb') as file:
    result_svm = pickle.load(file)

# e1B
filename = 'tmp/astral/lyriks402/1a-boruta-elasticnet-kfold.pkl'
with open(filename, 'rb') as file:
    result_boruta = pickle.load(file)

# e1C
filename = 'tmp/astral/lyriks402/1a-psychotic_ttest-elasticnet-kfold.pkl'
with open(filename, 'rb') as file:
    result_ttest = pickle.load(file)

# e1D
filename = 'tmp/astral/lyriks402/1a-mongan10-elasticnet-kfold.pkl'
with open(filename, 'rb') as file:
    result_mongan10 = pickle.load(file)

# e1E
filename = 'tmp/astral/lyriks402/2b-none-elasticnet-kfold.pkl'
with open(filename, 'rb') as file:
    result_remission_605 = pickle.load(file)


##### Evaluation ##### 

result = result_remission_605
name = (
    f'{result.metadata["version"]}-'
    f'{result.metadata["selector"]}-'
    f'{result.metadata["model"]}-'
    f'{result.metadata["validator"]}'
)
print(name)

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
plt.figure(figsize=(4.2, 4))
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('All proteins (k = 605)')
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.legend()
filepath = 'tmp/astral/lyriks402/fig/roc-' + name + '.pdf'
print(filepath)
plt.savefig(filepath)
plt.close()

# Inner folds
for i, inner_fold in enumerate(result.inner_folds):
    # Threshold = 0.5
    tn, fp, fn, tp = confusion_matrix(
        inner_fold['labels'], inner_fold['predictions']
    ).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    npv = tn / (tn + fn)
    accuracy = (tp + tn) / (tn + fp + fn + tp) 
    # Calculate AUC
    auc = roc_auc_score(inner_fold['labels'], inner_fold['probas'])
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
    fpr, tpr, thresholds = roc_curve(inner_fold['labels'], inner_fold['probas'])
    plt.figure(figsize=(4.5, 4))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC curve')
    plt.legend()
    filepath = f'tmp/astral/fig/roc-{name}-{i}.pdf'
    print(filepath)
    plt.savefig(filepath)


# # Edit result metadata
# dirpath = 'tmp/astral/lyriks402/new/'
# # file = '1a-elasticnet-elasticnet-nestedkfold.pkl'
# file = '1a-elasticnet-elasticnet-bal-nestedkfold.pkl'
# filepath = dirpath + file
# print(filepath)
# 
# with open(filepath, 'rb') as file:
#     result = pickle.load(file)
# 
# result.metadata['selector'] = 'elasticnet'
# with open(filepath, 'wb') as file:
#     pickle.dump(result, file)
# 
# 
# for filepath in filepaths:
#     with open(filepath, 'rb') as file:
#         result = pickle.load(file)
#     if 'bal' in filepath:
#         result.metadata['class_weight'] = 'balanced'
#     else:
#         result.metadata['class_weight'] = 'none'
#         print(filepath)
#     with open(filepath, 'wb') as file:
#         pickle.dump(result, file)
#     
# filepath = dirpath + '1a-svm-linear-svm-linear-nestedkfold.pkl'
# with open(filepath, 'rb') as file:
#     result = pickle.load(file)
# 
# result.metadata['class_weight'] = 'balanced'
# with open(filepath, 'wb') as file:
#     pickle.dump(result, file)

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
plt.axvline(x=0.99, color='tab:blue', linestyle='dashed') # train set
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

fig, ax = plt.subplots(1, 1, figsize=(7, 2.5))
ax.barh(
    aucs2.index, aucs2['mean_auc'], xerr=aucs2['std_auc'],
    capsize=5, color='gray'
)
ax.set_xlabel('AUC')
plt.axvline(x=eugei_svm_mean_auc, color='tab:green', linestyle='dashed') # EU-GEI (SVM)
plt.axvline(x=rnd_mean_auc, color='tab:orange', linestyle='dashed') # negative ctrl
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
# md.period = md.period.astype(str)
md_1a.rename(columns={'final_label': 'Label'}, inplace=True)
md_1a.Label = md_1a.Label.replace({'rmt': 'Non-convert', 'mnt': 'Non-convert', 'cvt': 'Convert'})
data = ad.AnnData(X, md_1a)

data_mongan33 = data[:, mongan_prots33]
data_elasticnet = data[:, bm_elasticnet]
data_mongan33.shape
data_elasticnet.shape

from sklearn.covariance import EmpiricalCovariance, MinCovDet 
from matplotlib.patches import Ellipse


# Function to draw an ellipse from covariance
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

mle_cov = EmpiricalCovariance()
robust_cov = MinCovDet()

sc.pp.pca(data_mongan33)
fig = plot_clusters(data_mongan33, mle_cov)
fig.set_size_inches(4, 3)
filename = 'tmp/astral/lyriks402/fig/pca-mongan33.pdf'
fig.savefig(filename, bbox_inches='tight')

sc.pp.pca(data_elasticnet)
fig = plot_clusters(data_elasticnet, mle_cov)
fig.set_size_inches(5, 3)
filename = 'tmp/astral/lyriks402/fig/pca-elasticnet.pdf'
fig.savefig(filename, bbox_inches='tight')


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

# TODO: PCA (explained var)
# TODO: PCA (ellipses)

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

##### Annotate #####
# Annotate with gene symbols

# ANCOVA
filepath = 'tmp/astral/lyriks402/new/features-prognostic_ancova-kfold.csv'
stats = pd.read_csv(filepath, index_col=0)
symbols = reprocessed.loc[stats.index, ['Description', 'Gene']]
data = symbols.join(stats)
data.sort_values(by='q', inplace=True)

writepath = 'tmp/astral/lyriks402/new/biomarkers-ancova.csv'
data.to_csv(writepath, float_format='%.3g')

# Elastic net
filepath = 'tmp/astral/lyriks402/new/features-elasticnet-bal-nestedkfold.csv'
elasticnet = pd.read_csv(filepath, index_col=0)
# Write gene descriptions of signatures  
symbols = reprocessed.loc[elasticnet.index, ['Description', 'Gene']]
data = symbols.join(elasticnet)
data.sort_values(by='Mean coefficient', key=abs, ascending=False, inplace=True)

writepath = 'tmp/astral/lyriks402/new/biomarkers-elasticnet.csv'
data.to_csv(writepath, float_format='%.3g')

# 4 signatures
filepaths = [
    'tmp/astral/lyriks402/1a-prognostic_ancova.csv',
    'tmp/astral/lyriks402/1a-psychotic_ttest.csv',
    'tmp/astral/lyriks402/2b-prognostic_ancova.csv',
    'tmp/astral/lyriks402/uhr_3a-ancova.csv'
]
for filepath in filepaths:
    signatures = pd.read_csv(filepath, index_col=0) 
    signatures = signatures.loc[signatures.p < 0.01, [True, True, False]]
    signatures.sort_values(by='p', inplace=True)
    symbols_f = symbols.loc[signatures.index]
    data = symbols_f.join(signatures)
    writepath = filepath[:21] + 'signatures-' + filepath[21:]
    print(writepath)
    data.to_csv(writepath, float_format='%.3g')


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
