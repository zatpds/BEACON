"""
Domain classifier for estimating source-target discrepancy.

Trains a logistic regression on obs-encoder embeddings to distinguish
source from target samples.  Returns **per-source-sample** discrepancy
d_i = P(source | x_i) from the fitted classifier, so source samples
that look target-like get low d while clearly-source samples get high d.

Source embeddings are subsampled to match the target count so the
classifier sees balanced classes.  The classifier is then applied to
ALL source samples to produce per-sample scores.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def compute_d_from_domain_classifier(src_emb, tgt_emb):
    """
    Train a domain classifier and return per-source-sample discrepancy.

    Args:
        src_emb: (m, D) tensor or ndarray — source embeddings.
        tgt_emb: (n, D) tensor or ndarray — target embeddings.

    Returns:
        d_per_sample: (m,) ndarray — P(source | x_i) for each source sample.
        info (dict): auxiliary stats (auc, etc.).
    """
    src = np.asarray(src_emb.detach().cpu() if hasattr(src_emb, "detach") else src_emb)
    tgt = np.asarray(tgt_emb.detach().cpu() if hasattr(tgt_emb, "detach") else tgt_emb)

    m, n = src.shape[0], tgt.shape[0]

    # Subsample source to match target count for balanced training
    if m > n:
        idx = np.random.choice(m, size=n, replace=False)
        src_bal = src[idx]
    else:
        src_bal = src

    n_bal = min(m, n)

    # source=0, target=1
    X_train = np.concatenate([src_bal, tgt], axis=0)
    y_train = np.concatenate([np.zeros(n_bal), np.ones(n)]).astype(np.int32)

    clf = LogisticRegression(solver="lbfgs", max_iter=200, C=1.0)
    clf.fit(X_train, y_train)

    # Per-source-sample d_i = P(source | x_i) = 1 - P(target | x_i)
    # Applied to ALL m source samples (not just the balanced subset)
    src_probs_target = clf.predict_proba(src)[:, 1]
    d_per_sample = 1.0 - src_probs_target  # P(source)

    # AUC on the full data for logging
    X_full = np.concatenate([src, tgt], axis=0)
    y_full = np.concatenate([np.zeros(m), np.ones(n)]).astype(np.int32)
    probs_full = clf.predict_proba(X_full)[:, 1]
    auc = roc_auc_score(y_full, probs_full)

    info = {
        "auc": auc,
        "n_train_per_class": n_bal,
        "m_source": m,
        "n_target": n,
    }
    return d_per_sample.astype(np.float32), info
