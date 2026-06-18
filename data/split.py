import os
import random

import numpy as np
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


def get_loso_splits(subjects, val_ratio=0.15, random_seed=42):
    """
    Δημιουργεί Splits (Train, Validation, Test) με βάση το Subject ID.

    Args:
        subjects (list): Λίστα με τα subject IDs.
        val_ratio (float): Ποσοστό των Train subjects που θα γίνουν Validation.
        random_seed (int): Για αναπαραγωγιμότητα του validation split.

    Returns:
        splits (list): Μια λίστα με tuples (train_subjects, val_subjects, test_subjects)
    """
    # Σταθερότητα για να βγαίνουν ίδια τα val sets σε κάθε run
    random.seed(random_seed)

    logo = LeaveOneGroupOut()

    # Το X δεν έχει σημασία στο LOGO, χρειάζεται μόνο για το μέγεθος
    X = np.zeros(len(subjects))
    groups = np.array(subjects)

    # Βρίσκουμε τα ΜΟΝΑΔΙΚΑ subject IDs (π.χ. 1 έως 25)
    unique_subjects = np.unique(groups)

    splits = []

    for train_idx, test_idx in logo.split(X, y=None, groups=groups):
        # 1. Παίρνουμε τους Test χρήστες (Στο LOSO είναι πάντα 1, αλλά το γράφουμε γενικά)
        test_subjects = np.unique(groups[test_idx]).tolist()

        # 2. Παίρνουμε τους υπόλοιπους (που αρχικά είναι όλοι Train)
        current_train_subjects = np.unique(groups[train_idx]).tolist()

        # 3. Υπολογίζουμε πόσους χρήστες θα πάρουμε για Validation
        num_val_subjects = max(1, int(len(current_train_subjects) * val_ratio))

        # 4. Τραβάμε τυχαία (χωρίς επανατοποθέτηση) τους Val χρήστες
        val_subjects = random.sample(current_train_subjects, num_val_subjects)

        # 5. Οι εναπομείναντες είναι το πραγματικό Train set
        train_subjects = [s for s in current_train_subjects if s not in val_subjects]

        splits.append((train_subjects, val_subjects, test_subjects))

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