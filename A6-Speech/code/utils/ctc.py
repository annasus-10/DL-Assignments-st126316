"""CTC collapsing function and from-scratch forward algorithm."""
import numpy as np

BLANK = '_'  # epsilon
NEG_INF = -1e9


def ctc_collapse(alignment):
    """Merge consecutive duplicates, then remove blanks. alignment: list of chars."""
    merged = []
    for ch in alignment:
        if not merged or ch != merged[-1]:
            merged.append(ch)
    return ''.join(ch for ch in merged if ch != BLANK)


def log_add(a, b):
    """log(exp(a) + exp(b)), numerically stable."""
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    m = max(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def ctc_forward_log_prob(log_probs, labels, blank=0):
    """
    log_probs: (T, V) log-softmax output per frame, V = vocab size (incl. blank)
    labels:    list of label indices (no blanks), length L
    Returns:   log P_CTC(labels | log_probs)
    """
    T, V = log_probs.shape
    ext = [blank]
    for lab in labels:
        ext += [lab, blank]
    S = len(ext)  # extended length = 2L + 1

    alpha = np.full((T, S), NEG_INF)
    alpha[0, 0] = log_probs[0, ext[0]]
    if S > 1:
        alpha[0, 1] = log_probs[0, ext[1]]

    for t in range(1, T):
        for s in range(S):
            stay = alpha[t-1, s]
            prev = alpha[t-1, s-1] if s - 1 >= 0 else NEG_INF
            skip = NEG_INF
            if s - 2 >= 0 and ext[s] != blank and ext[s] != ext[s-2]:
                skip = alpha[t-1, s-2]
            best_prev = log_add(log_add(stay, prev), skip)
            alpha[t, s] = best_prev + log_probs[t, ext[s]]

    if S == 1:
        return alpha[T-1, S-1]
    return log_add(alpha[T-1, S-1], alpha[T-1, S-2])
