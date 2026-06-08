import os
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, KFold


def get_subject_dependent_splits(n_trials=40, n_splits=5):

    trial_ids = list(range(n_trials))

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    splits = []

    for train_idx, test_idx in kf.split(trial_ids):

        splits.append(
            (train_idx.tolist(), test_idx.tolist())
        )

    return splits


def build_subject_splits(subject_dirs):

    subjects = list(subject_dirs.keys())

    logo = LeaveOneGroupOut()

    X = list(range(len(subjects)))
    groups = subjects

    splits = []

    for train_idx, test_idx in logo.split(X, X, groups):

        train_subjects = [subjects[i] for i in train_idx]
        test_subjects = [subjects[i] for i in test_idx]

        splits.append((train_subjects, test_subjects))

    return splits

def get_loso_splits(subjects):

    logo = LeaveOneGroupOut()

    X = list(range(len(subjects)))
    groups = subjects

    splits = []

    for train_idx, test_idx in logo.split(X, X, groups):

        train_subjects = [subjects[i] for i in train_idx]
        test_subjects = [subjects[i] for i in test_idx]

        splits.append((train_subjects, test_subjects))

    return splits


def get_group_kfold_splits(subjects, n_splits=5):

    n_splits = min(n_splits, len(subjects))

    gkf = GroupKFold(n_splits=n_splits)

    X = list(range(len(subjects)))
    groups = subjects
    splits = []

    for train_idx, test_idx in gkf.split(X, X, groups):
        train_subjects = [subjects[i] for i in train_idx]
        test_subjects = [subjects[i] for i in test_idx]
        splits.append((train_subjects, test_subjects))

    return splits