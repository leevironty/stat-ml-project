import pandas as pd
import numpy as np
import sklearn.linear_model as linear
import sklearn.preprocessing as pre
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import time
import warnings
from typing import overload, Callable
from dataclasses import dataclass

@dataclass
class Curve:
    tps: np.ndarray
    fps: np.ndarray
    ths: np.ndarray
    pi: float
    share: float
    label: str
    def __post_init__(self):
        self.acs = self.tps * self.pi + self.fps * (1 - self.pi)


def get_curves(probs: np.ndarray, class_labels: np.ndarray, label_map, y: np.ndarray, weights: np.ndarray) -> list[Curve]:
    curves = []
    num_class = class_labels.max() + 1
    for c in range(num_class):
        probs_c = probs[class_labels == c]
        y_c = y[class_labels == c]
        weights_c = weights[class_labels == c]
        fp, tp, tr = metrics.roc_curve(y_c, probs_c, sample_weight=weights_c)
        tr = tr.clip(0, 1)  # no max + 1 trickery
        tp, fp, th = convexify(tp, fp, tr)
        curves.append(Curve(
            tps=tp,
            fps=fp,
            ths=th,
            pi=y_c.mean(),
            share=(class_labels == c).mean(),
            label=label_map[c],
        ))
    return curves


def one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode non-numeric features."""
    cats = df.select_dtypes('object')
    nums = df.select_dtypes('int')

    enc = pre.OneHotEncoder(sparse=False)
    oh = enc.fit_transform(cats)

    cols = []
    for main_category, sub_categories in zip(cats.columns, enc.categories_):
        cols += [f'{main_category}_{sub}' for sub in sub_categories]

    enc = pd.DataFrame(data=oh, columns=cols)
    out = pd.concat([nums, enc], axis=1)
    
    return out

@overload
def loss(tp: float, fp: float, pi: float) -> float:
    ...

@overload
def loss(tp: np.ndarray, fp: np.ndarray, pi: float) -> np.ndarray:
    ...

def loss(tp, fp, pi):
    """
    Expected 0-1 loss for binary classifier.
    
    Equal to:
    P(y_hat = 1 | y = 0) P(y = 0) + P(y_hat = 0 | y = 1) P(y = 1)

    Parameters:
    ---
    tp:
        P(y_hat = 1 | y = 1)
    fp:
        P(y_hat = 1 | y = 0)
    pi:
        P(y = 1)
    """
    return fp * (1 - pi) + (1 - tp) * pi


def max_profit(curves: list[Curve]) -> np.ndarray:
    """
    Find max profit thresholds.
    
    ---
    probs:
        Predicted probability of y=1 by the underlying model.
    class_labels:
        Protected class index (0 to c-1) associated with each predicted
        probability. All values from 0 to c - 1 must be present in the array.
    y:
        True value of y.
    ---
    returns:
        Array of thresholds for each class that yields max profit.
    """
    num_class = len(curves)
    thresholds = np.zeros(num_class)
    out_tp = np.zeros(num_class)
    out_fp = np.zeros(num_class)
    for c, curve in enumerate(curves):
        l = loss(curve.tps, curve.fps, curve.pi)
        i = l.argmin()
        thresholds[c] = curve.ths[i]  # choose threshold minimizing loss
        out_tp[c] = curve.tps[i]
        out_fp[c] = curve.fps[i]
    return thresholds, out_tp, out_fp


def convexify(tp: np.ndarray, fp: np.ndarray, tr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    while True:
        slopes = np.diff(tp) / np.diff(fp)
        redundant = slopes[:-1] < slopes[1:]
        mask = np.concatenate([[False], redundant, [False]])
        if mask.sum() == 0:
            break
        tp = tp[~mask]
        fp = fp[~mask]
        tr = tr[~mask]
    return tp, fp, tr


def lookup_to_tp_fp(lk_v: float, lk_domain: np.ndarray, tp: np.ndarray, fp: np.ndarray) -> tuple[float, float]:
    """Find true and false positive rates matching to given lookup value and domain."""
    # binary search for proper interval (np.searchsorted)
    i = np.searchsorted(lk_domain, lk_v)
    if i == 0:
        return tp[0], fp[0]
    # find interpolation ratio
    a = lk_domain[i - 1]
    b = lk_domain[i]
    r = (lk_v - a) / (b - a)
    # interpolate by r for tp and fp
    return (1 - r) * tp[i - 1] + r * tp[i], (1 - r) * fp[i - 1] + r * fp[i]


# stolen / adapted from Wikipedia
def ternary_search(f: Callable[[float], float]) -> float:
    """Find maximum of unimodal function f() within [left, right]."""
    right = 1
    left = 0
    while abs(right - left) >= 1e-6:
        left_third = left + (right - left) / 3
        right_third = right - (right - left) / 3

        if f(left_third) < f(right_third):
            left = left_third
        else:
            right = right_third

    # Left and right are the current bounds; the maximum is between them
    return (left + right) / 2



def equal_demographic(curves: list[Curve]) -> np.ndarray:
    num_class = len(curves)
    thresholds = np.zeros(num_class)
    # tps = []
    # fps = []
    # trs = []
    # acs = []
    # pis = []
    # shares = []

    # # calculate tp, fp, acceptance rates and corresponding thresholds for all
    # for c in range(num_class):
    #     probs_c = probs[class_labels == c]
    #     shares.append((class_labels == c).mean())
    #     y_c = y[class_labels == c]
    #     fp, tp, tr = metrics.roc_curve(y_c, probs_c)
    #     tr = tr.clip(0, 1)  # no max + 1 trickery
    #     tp, fp, tr = convexify(tp, fp, tr)
    #     pi = y_c.mean()
    #     ac = tp * pi + fp * (1 - pi)
    #     tps.append(tp)
    #     fps.append(fp)
    #     trs.append(tr)
    #     acs.append(ac)
    #     pis.append(pi)

    # def neg_loss(v: float) -> float:
    #     l = 0
    #     for tp, fp, ac, pi, share in zip(tps, fps, acs, pis, shares):
    #         v_tp, v_fp = lookup_to_tp_fp(v, ac, tp, fp)
    #         l += loss(v_tp, v_fp, pi) * share
    #     return -l

    def neg_loss(v: float) -> float:
        l = 0
        for curve in curves:
            v_tp, v_fp = lookup_to_tp_fp(v, curve.acs, curve.tps, curve.fps)
            l += loss(v_tp, v_fp, curve.pi) * curve.share
        return -l
    
    ac = ternary_search(neg_loss)

    # generate output from ac
    out_ths = np.zeros((num_class, 2))
    single_ths = np.zeros(num_class)
    out_tp = np.zeros(num_class)
    out_fp = np.zeros(num_class)

    for c, curve in enumerate(curves):
        i = np.searchsorted(curve.acs, ac)
        if i == 0:
            out_ths[c, :] = curve.ths[0]
            single_ths[c] = curve.ths[0]
        else:
            r = (ac - curve.acs[i - 1]) / (curve.acs[i] - curve.acs[i-1])
            single_ths[c] = (1 - r) * curve.ths[i-1] + r * curve.ths[i]
            out_ths[c] = curve.ths[i-1], curve.ths[i]
            out_tp[c], out_fp[c] = lookup_to_tp_fp(ac, curve.acs, curve.tps, curve.fps)
    print(ac)

    return out_ths, out_tp, out_fp, ac, single_ths


def preprocess(fname: str, col_names: list[str], protected: str):
    drops = ['label', 'fnlwgt', 'native-country', 'marital-status'] + [protected]
    df = pd.read_csv(fname, header=None, names=col_names, skipinitialspace=True)
    y = ((df['label'] == '>50K') | (df['label'] == '>50K.')).values * 1
    protected_labels, protected_map = pd.factorize(df[protected])

    weights = df['fnlwgt'].values
    x = one_hot(df.drop(columns=drops))
    return x, y, protected_labels, protected_map, weights

    

def equal_opportunity(curves: list[Curve]) -> np.ndarray:
    num_class = len(curves)
    thresholds = np.zeros(num_class)
    # tps = []
    # fps = []
    # trs = []
    # acs = []
    # pis = []
    # shares = []

    # # calculate tp, fp, acceptance rates and corresponding thresholds for all
    # for c in range(num_class):
    #     probs_c = probs[class_labels == c]
    #     shares.append((class_labels == c).mean())
    #     y_c = y[class_labels == c]
    #     fp, tp, tr = metrics.roc_curve(y_c, probs_c)
    #     tr = tr.clip(0, 1)  # no max + 1 trickery
    #     tp, fp, tr = convexify(tp, fp, tr)
    #     pi = y_c.mean()
    #     ac = tp * pi + fp * (1 - pi)
    #     tps.append(tp)
    #     fps.append(fp)
    #     trs.append(tr)
    #     acs.append(ac)
    #     pis.append(pi)

    def neg_loss(v: float) -> float:
        l = 0
        for curve in curves:
            v_tp, v_fp = lookup_to_tp_fp(v, curve.tps, curve.tps, curve.fps)
            l += loss(v_tp, v_fp, curve.pi) * curve.share
        return -l
    
    tp = ternary_search(neg_loss)

    # generate output from ac
    out_ths = np.zeros((num_class, 2))
    single_ths = np.zeros(num_class)
    out_tp = np.zeros(num_class)
    out_fp = np.zeros(num_class)

    for c, curve in enumerate(curves):
        i = np.searchsorted(curve.tps, tp)
        if i == 0:
            out_ths[c, :] = curve.ths[0]
            single_ths[c] = curve.ths[0]
        else:
            r = (tp - curve.tps[i - 1]) / (curve.tps[i] - curve.tps[i-1])
            single_ths[c] = (1 - r) * curve.ths[i-1] + r * curve.ths[i]
            out_ths[c] = curve.ths[i-1], curve.ths[i]
            out_tp[c], out_fp[c] = lookup_to_tp_fp(tp, curve.tps, curve.tps, curve.fps)
    print(tp)

    return out_ths, out_tp, out_fp, tp, single_ths


def equal_odds(curves: list[Curve]):
    def tp_to_fp(tp: float) -> float:
        """fp corresponding to tp withing the union of roc-curves"""
        fp = 0
        for curve in curves:
            _, fp_curve = lookup_to_tp_fp(tp, curve.tps, curve.tps, curve.fps)
            fp = max(fp, fp_curve)
        return fp
    
    def neg_loss(tp: float) -> float:
        fp = tp_to_fp(tp)
        l = 0
        for curve in curves:
            l += loss(tp, fp, curve.pi) * curve.share
        return -l

    tp = ternary_search(neg_loss)
    fp = tp_to_fp(tp)

    return tp, fp
    

def intersection_threshold(curve: Curve, tp: float, fp: float) -> float:
    def line(x: float) -> float:
        return (1 - tp) / (1 - fp) * (x - 1) + 1
    
    ys = np.array([line(p) for p in curve.fps])
    i = np.argmax(ys - curve.tps < 0)


    mu = (1 - tp) / (1 - fp)
    # x1 = curve.fps[i-1]
    x2 = curve.fps[i]
    y1 = curve.tps[i-1]
    y2 = curve.tps[i]

    l = (1 - y2 + mu * (x2 - 1)) / (y1 - y2 + mu * (x2 - x2))
    # xi = l * x1 + (1 - l) * x2
    # yi = l * y1 + (1 - l) * y2
    th1 = curve.ths[i-1]
    th2 = curve.ths[i]
    return th1 * l + (1 - l) * th2


def main():
    print('preprocessing data')
    with open('adult.names') as file:
        rows = file.read().split('\n')[-15:]
        col_names = [row.split(':')[0] for row in rows][:-1] + ['label']

    x, y, labels, label_map = preprocess(
        fname='adult.data',
        col_names=col_names,
        protected='sex'
    )

    model = linear.LogisticRegression(penalty='none', max_iter=2000, verbose=True)
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            model.fit(x, y)
        except Warning as w:
            print(w)
            exit(1)
    probs = model.predict_proba(x)[:, 1]  # probability of y = 1
    print(f'base model score: {model.score(x, y)}')
    
    curves = get_curves(probs, labels, label_map, y)

    mp = max_profit(curves)
    print(f'max profit: {mp}')

    ed = equal_demographic(curves)
    print(f'equal demographic: {ed}')

    eopp = equal_opportunity(curves)
    print(f'equal opportunity: {eopp}')

    eodd = equal_odds(curves)
    print(f'equal odds: {eodd}')

    fig, ax = plt.subplots()
    for curve in curves:
        ax.plot(curve.fps, curve.tps, label=curve.label, zorder=1)
    for c, curve in enumerate(curves):
        ax.scatter(mp[2][c], mp[1][c], color='k', marker='x', label='max profit' if c == 0 else None)
        ax.scatter(ed[2][c], ed[1][c], color='k', marker='.', label='demographic parity' if c == 0 else None)
        ax.scatter(eopp[2][c], eopp[1][c], color='k', marker='+', label='equal opportunity' if c == 0 else None)
    ax.scatter(eodd[1], eodd[0], color='red', marker='x', label='equal odds')
    ax.legend(loc='lower right')
    ax.set_ylabel('$P(\hat{y}=1|y=1)$')
    ax.set_xlabel('$P(\hat{y}=1|y=0)$')
    ax.set_title('Per group ROC-curve\nconvex hulls')
    fig.show()
    print('done')

if __name__ == '__main__':
    main()
