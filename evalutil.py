import numpy as np
from sklearn.metrics import fbeta_score


def pick_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 91)
    scores = [fbeta_score(y, p >= t, beta=2) for t in grid]
    return float(grid[int(np.argmax(scores))])
