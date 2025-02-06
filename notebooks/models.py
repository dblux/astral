import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from boruta import BorutaPy
from dataclasses import dataclass, field
from matplotlib_venn import venn2, venn3
from scipy.stats import f, ttest_ind, rankdata
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFdr, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, roc_auc_score
)

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

def adjust_bh(p):
    return p / rankdata(p) * len(p) 


file = 'data/astral/processed/combat_knn5_lyriks.csv'
lyriks = pd.read_csv(file, index_col=0, header=0).T
file = 'data/astral/processed/metadata-lyriks.csv'
md = pd.read_csv(file, index_col=0, header=0)
list(md)

# Model 1A: cvt (M0) v.s. non-cvt (M0)
# Prognostic markers
# Only M0 and exclude ctrl samples
md_1a = md[(md.final_label != 'ctrl') & (md.period == 0)]
y = md_1a.final_label.replace({'rmt': 0, 'mnt': 0, 'cvt': 1})
lyriks_1a = lyriks.loc[y.index]
X = lyriks_1a
print(y.value_counts()) # imbalanced
print(X.shape, y.shape)

# Feature selection
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

# Cross-validation
# cross_validator = LeaveOneOut()
cross_validator = StratifiedKFold(n_splits=4)

# Run detail
result = Result({
    'version': '1a',
    'selector': 'boruta',
    'model': 'logreg',
    'validator': 'kfold',
    'snapshot': {}, 
})
# for i, (train_idx, test_idx) in enumerate(cross_validator.split(X)):
for i, (train_idx, test_idx) in enumerate(cross_validator.split(X, y)):
    print(f"Fold: {i}")
    print('---------------')
    # Split data into train and test
    X_train, X_test = X.iloc[train_idx].values, X.iloc[test_idx].values
    y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values
    match result.metadata:
        case {'selector': 'prognostic_ttest'}: 
            selector = SelectFdr(unpaired_ttest, alpha=0.05)
            X_train_f = selector.fit_transform(X_train, y_train)
            result.pvals.append(selector.pvalues_) # p-values from F-test
            result.test_statistics.append(selector.scores_)
            print(f'No. of features selected = {X_train_f.shape[1]}')
        case {'selector': 'boruta'}:
            selector = BorutaPy(
                rf, n_estimators='auto', two_step=False,
                verbose=2, random_state=1
            )
            selector.fit(X_train, y_train)
            X_train_f = selector.transform(X_train)
            result.ranks.append(selector.ranking_)
            print(f'No. of features selected = {selector.n_features_}')
    X_test_f = selector.transform(X_test)  # Apply the same selection to test
    match result.metadata:
        case {'model': 'logreg'}:
            model = LogisticRegression(max_iter=1000)
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
    'selector': repr(selector),
    'model': repr(model),
    'validator': repr(cross_validator),
})
print(result.metadata)

# TODO: Loop over models

# TODO: PCA on selected features
# TODO: feat sel: psychotic markers

##### Feature selection ##### 

qvals = np.array([adjust_bh(p) for p in result.pvals])
# Assumption: Order of columns is preserved
avg_q = pd.DataFrame({'q': qvals.mean(axis=0)}, index=list(X))
q_fil = avg_q[avg_q.q < 0.05]
print(q_fil.shape[0])
result.features = q_fil.index.tolist()

# Boruta
ranks = np.array(result.ranks)
avg_rank = pd.DataFrame({'Rank': ranks.mean(axis=0)}, index=list(X))
# Assumption: Order of columns is preserved
rank_fil = avg_rank[avg_rank.Rank <= 2]
print(rank_fil.shape)
result.features = rank_fil.index.tolist()

##### Save results #####

# Save
filename = (
    f'{result.metadata["version"]}-'
    f'{result.metadata["selector"]}-'
    f'{result.metadata["model"]}-'
    f'{result.metadata["validator"]}'
)
filepath = 'tmp/astral/' + filename + '.pkl'
print(filepath)
with open(filepath, 'wb') as file:
    pickle.dump(result, file)

# Load
filepath = 'tmp/astral/1a-boruta-logistic_regression.pkl'
with open(filepath, 'rb') as file:
    result_boruta = pickle.load(file)

filepath = 'tmp/astral/1a-prognostic_ttest-logistic_regression.pkl'
with open(filepath, 'rb') as file:
    result_ttest = pickle.load(file)

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
# plt.show()
filepath = 'tmp/astral/fig/roc-' + filename + '.pdf'
print(filepath)
plt.savefig(filepath)
plt.close()

###### PCA #####

# Perform PCA
pca = PCA(n_components=3)

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

# T-test
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

##### Comparison #####


features_ttest = set(result_ttest.features)
features_boruta = set(result_boruta.features)

venn2(subsets=(
    len(features_ttest - features_boruta),
    len(features_boruta - features_ttest),
    len(features_ttest.intersection(features_boruta))
), set_labels=('T-test', 'Boruta'))
filepath = 'tmp/astral/fig/venn-ttest_boruta.pdf'
plt.savefig(filepath)

# TODO: Plot ROC boruta again
# TODO: SVM (other models)

##### Grid search #####   
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Mongan

filepath = 'data/astral/etc/mongan-etable5.csv'
mongan = pd.read_csv(filepath, index_col=0, header=0)
mongan_prots = set(mongan.index[mongan.q < 0.05])
features_ttest = set(result_ttest.features)
features_boruta = set(result_boruta.features)

n1 = len(mongan_prots - features_ttest - features_boruta)
n2 = len(features_ttest - mongan_prots - features_boruta)
n3 = len(features_boruta - mongan_prots - features_ttest)
n12 = len(mongan_prots & features_ttest - features_boruta) 
n13 = len(mongan_prots & features_boruta - features_ttest)
n23 = len(features_ttest & features_boruta - mongan_prots)
n123 = len(mongan_prots & features_ttest & features_boruta)

plt.figure()
venn3(
    subsets=(n1, n2, n12, n3, n13, n23, n123),
    set_labels=('mongan', 'ttest', 'boruta')
)
filepath = 'tmp/astral/fig/venn-mongan_ttest_boruta.pdf'
plt.savefig(filepath)

