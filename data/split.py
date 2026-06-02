import os
from sklearn.model_selection import LeaveOneGroupOut


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