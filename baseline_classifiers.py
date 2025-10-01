from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from utils import load_all_recordings
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np
from pyriemann.classification import MDM
from pyriemann.classification import MDM
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import XdawnCovariances
from sklearn.decomposition import PCA
import torch
import joblib


# HDCA20, stepwise linear discriminant analysis23, and Bayesian linear discriminant analysis24.

def csp_lda(X, y, cv_splits=5, return_clf=False):
    """
    X: (n_trials, n_chans, n_times)
    y: (n_trials,)
    """
    clf = Pipeline([
        # ('cov', Covariances(estimator='oas')),
        ('xdawn', XdawnCovariances(nfilter=4, estimator='oas')),
        ('ts', TangentSpace()),
        # ('pca', PCA(n_components=0.95)),
        ('scaler', StandardScaler()),
        # ('logreg', LogisticRegression(solver='liblinear'))
        # ridge classifier
        #('logreg', LogisticRegression(solver='liblinear', C=0.1))
        ('ridge', RidgeClassifier(alpha=0.1))
    ])
    if return_clf:
        # train the clf on the whole data
        clf.fit(X, y)
        return clf
    else:
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, n_jobs=2)
        return scores

def balance_classes(X, y):
    min_class_size = min(np.bincount(y))
    X_balanced = []
    for label in np.unique(y):
        X_balanced.append(X[y == label][:min_class_size])
    X_balanced = np.concatenate(X_balanced, axis=0)
    y_balanced = np.concatenate([np.full(min_class_size, label) for label in np.unique(y)])
    X, y = X_balanced, y_balanced
    return X, y

def loop(X, y):
    means, stds = [], []
    for _ in range(5):
        perm = np.random.permutation(len(y))
        X, y = X[perm], y[perm]
        scores = csp_lda(X, y, cv_splits=8)
        scores = np.array(scores)
        scores = np.round(scores, 2)
        print("CSP+LDA CV scores:", scores, "mean:", scores.mean().round(3), "std:", scores.std().round(3))
        means.append(scores.mean())
        stds.append(scores.std())
    print("Overall mean accuracy: %.3f (+/- %.3f)" % (np.mean(means), np.mean(stds)))


def show_mean_erp_per_class(data_one, data_two):

    mean_one = data_one.mean(axis=0).mean(axis=0)
    mean_two = data_two.mean(axis=0).mean(axis=0)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    n_times = mean_one.shape[0]
    sfreq = 256
    times = np.arange(n_times) / sfreq  # seconds

    plt.plot(times, mean_one, color='blue', alpha=0.1)
    plt.plot(times, mean_two, color='red', alpha=0.1)
    plt.plot(times, mean_one, color='blue', label='Class 0 Mean', linewidth=2)
    plt.plot(times, mean_two, color='red', label='Class 1 Mean', linewidth=2)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('Mean ERP for Each Class')
    plt.legend()
    plt.grid()


    plt.tight_layout()
    plt.show()


def our_dataset():

    # -------- set your root once; wrapper will read them all --------
    root = "eegAcquisitions"  # e.g., contains sub-P001/ses-S001/... etc.

    all_channels = ['AF7', 'Fp1', 'Fp2', 'AF8', 'F3', 'F4', 'P3', 'P4', 'P7', 'O1', 'O2', 'P8', '13', 'eyetracker_open']
    occipital = ['O1', 'O2']
    parietal = ['P3', 'P4', 'P7', 'P8']
    frontal = ['AF7', 'Fp1', 'Fp2', 'AF8', 'F3', 'F4']

    keep = occipital + frontal + parietal

    exclude = [ch for ch in all_channels if ch not in keep]
    X, y, files = load_all_recordings(root, pattern="sub-P001/ses-*/eeg/*.xdf", exclude_ses=["ses-S001"], tmin=-0.2, tmax=0.6, drop_channels=exclude)
    # balance classes
    label_one = np.where(y == 0)[0]
    label_two = np.where(y == 1)[0]

    data_one = X[label_one]
    data_two = X[label_two]
    show_mean_erp_per_class(data_one, data_two)
    return X, y

def erp_dataset():
    data_dict = torch.load("full_data.pt")
    data_dict["data"] = (data_dict["data"] - data_dict["data_mean"]) / data_dict["data_std"]
    print("data_dict keys:", data_dict.keys())
    print("data labels: ", data_dict["labels"])
    # from the tasks
    nan_values = torch.where(torch.isnan(data_dict['tasks']))[0]
    print("nan values:", nan_values)
    # exclude from data_dict['tasks'] all the indices in nan_values 
    if len(nan_values) > 0:
        mask = torch.ones(len(data_dict['tasks']), dtype=bool)
        mask[nan_values] = False
        data_dict['tasks'] = data_dict['tasks'][mask]

    indices = torch.where(((data_dict["tasks"] == 12) | (data_dict["tasks"] == 13)) & (data_dict["subjects"] >=0))[0]
    
    data_dict["data"] = data_dict["data"][indices]
    data_dict["tasks"] = data_dict["tasks"][indices] - 12

    label_one = torch.where(data_dict["tasks"] == 0)[0]
    label_two = torch.where(data_dict["tasks"] == 1)[0]

    data_one = data_dict["data"][label_one]
    data_two = data_dict["data"][label_two]

    show_mean_erp_per_class(data_one.numpy(), data_two.numpy())


    X = data_dict["data"].numpy()
    y = data_dict["tasks"].numpy().astype(np.int64)
    return X, y

def concatenate_datasets(dat_fun1, dat_fun2, indices_erp, indices_ours):
    print("Loading and concatenating datasets...")
    X1, y1 = dat_fun1()
    print("First dataset shape:", X1.shape, y1.shape)
    
    X2, y2 = dat_fun2()
    print("Second dataset shape:", X2.shape, y2.shape)
    # keep only the channels in intersection
    X1 = X1[:, indices_erp, :]
    X2 = X2[:, indices_ours, :-1]
    X = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((y1, y2), axis=0)
    return X, y


# X, y = erp_dataset()
# print("Class distribution: 0 count =", np.sum(y==0), ", 1 count =", np.sum(y==1))
# X, y = balance_classes(X, y)

# list_erp = ["FP1", "F3", "F7", "FC3", "C3", "C5", "P3", "P7", "P9", "PO7", "PO3", "O1", "Oz", "Pz", "CPz", "FP2", "Fz", "F4", "F8", "FC4", "FCz", "Cz", "C4", "C6", "P4", "P8", "P10", "PO8", "PO4", "O2"]
# list_ours = ['AF7', 'Fp1', 'Fp2', 'AF8', 'F3', 'F4', 'P3', 'P4', 'P7', 'O1', 'O2', 'P8', '13', 'eyetracker_open']

# # Transform both lists to uppercase
# list_erp = [ch.upper() for ch in list_erp]
# list_ours = [ch.upper() for ch in list_ours]

# # Get string intersection of list_erp and list_ours both in uppercase
# intersection = list(set(list_erp) & set(list_ours))
# print("Common channels:", intersection)
# print("type of intersection:", type(intersection))

# # Get indices of intersection in list_erp
# indices_erp = [list_erp.index(ch) for ch in intersection]
# print("Indices in erp:", indices_erp)
# # Get indices of intersection in list_ours
# indices_ours = [list_ours.index(ch) for ch in intersection]
# print("Indices in ours:", indices_ours)


X, y = our_dataset()
# store the dataset in a npy file
# np.save("our_dataset.npy", X)
# np.save("our_labels.npy", y)

print("Class distribution: 0 count =", np.sum(y==0), ", 1 count =", np.sum(y==1))
X, y = balance_classes(X, y)
loop(X, y)
exit()
res = input("Press any key if you want to continue with the training on the whole dataset and store the model, or type 'exit' to quit: ")
if res.lower() == 'exit':
    exit()
else:
    # Continue with training
    clf = csp_lda(X, y, return_clf=True)
    joblib.dump(clf, "csp_lda_model.pkl")
    print("Model saved as csp_lda_model.pkl")

# X, y = concatenate_datasets(erp_dataset, our_dataset, indices_erp, indices_ours)
# print("Class distribution: 0 count =", np.sum(y==0), ", 1 count =", np.sum(y==1))
# X, y = balance_classes(X, y)
# print("Final dataset shape:", X.shape, y.shape)
# loop(X, y)