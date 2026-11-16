"""Draw the figures of the documentation.

Each figure is drawn twice, for the light and the dark theme, into
``sphinx-docs/_static/figures``. Run from the repository root:

    python sphinx-docs/_scripts/make_figures.py [name ...]

The figures are committed, so the documentation build does not train any
model. Numbers quoted in the tutorials are printed by this script.
"""
import pathlib
import sys
import warnings

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap, to_rgb  # noqa: E402

warnings.filterwarnings('ignore')
OUT = pathlib.Path(__file__).resolve().parents[1] / '_static' / 'figures'

THEMES = {
    'light': dict(fg='#1e293b', muted='#64748b', grid='#e2e8f0', bg='#ffffff',
                  classes=['#4f46e5', '#0d9488', '#d97706', '#e11d48']),
    'dark': dict(fg='#e2e8f0', muted='#94a3b8', grid='#334155', bg='#14181e',
                 classes=['#818cf8', '#2dd4bf', '#fbbf24', '#fb7185']),
}


def _style(theme):
    t = THEMES[theme]
    plt.rcParams.update({
        'figure.facecolor': 'none', 'axes.facecolor': 'none', 'savefig.transparent': True,
        'text.color': t['fg'], 'axes.labelcolor': t['fg'], 'axes.edgecolor': t['muted'],
        'xtick.color': t['muted'], 'ytick.color': t['muted'], 'axes.titleweight': 'bold',
        'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10.5,
        'axes.titlesize': 11.5, 'legend.frameon': False, 'axes.grid': False,
    })
    return t


def _save(fig, name, theme):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}-{theme}.png', dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('  wrote', f'{name}-{theme}.png')


def _shade(ax, model, X, t, n=300, pad=0.6):
    """Colour each point of a grid by the predicted class, light where unsure."""
    x0, y0 = np.asarray(X.min(0)) - pad
    x1, y1 = np.asarray(X.max(0)) + pad
    gx, gy = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    grid = jnp.asarray(np.c_[gx.ravel(), gy.ravel()], dtype=jnp.float32)
    proba = np.asarray(model.predict_proba(grid))
    k = proba.argmax(1)
    conf = (proba.max(1) - 1 / proba.shape[1]) / (1 - 1 / proba.shape[1])
    bg = np.array(to_rgb(t['bg']))
    colours = np.array([to_rgb(c) for c in t['classes']])[k]
    alpha = 0.07 + 0.30 * conf[:, None]
    img = (1 - alpha) * bg + alpha * colours
    ax.imshow(img.reshape(n, n, 3), extent=(x0, x1, y0, y1), origin='lower', aspect='auto')
    ax.contour(gx, gy, k.reshape(n, n), levels=np.arange(proba.shape[1]) + 0.5,
               colors=[t['muted']], linewidths=0.8, alpha=0.7)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


# ----------------------------------------------------------------------- hero
def hero():
    from prosemble.datasets import load_moons_jax
    from prosemble.models import GLVQ

    data = load_moons_jax(n_samples=300, noise=0.15, random_state=0)
    X, y = data.input_data, data.labels
    model = GLVQ(n_prototypes_per_class=4, max_iter=300, lr=0.05, random_seed=0)
    model.fit(X, y)
    acc = float(jnp.mean(model.predict(X) == y))
    print(f'  hero: training accuracy {acc:.3f}')
    for theme in THEMES:
        t = _style(theme)
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        _shade(ax, model, X, t)
        cmap = ListedColormap(t['classes'][:2])
        ax.scatter(X[:, 0], X[:, 1], c=np.asarray(y), cmap=cmap, s=14, alpha=0.8, linewidths=0)
        P, L = np.asarray(model.prototypes_), np.asarray(model.prototype_labels_)
        ax.scatter(P[:, 0], P[:, 1], c=[t['classes'][i] for i in L], s=190, marker='o',
                   edgecolors=t['fg'], linewidths=1.8, zorder=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        _save(fig, 'hero', theme)


def _bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ----------------------------------------------- tutorial: interpretable classification
def wine():
    from sklearn.datasets import load_wine
    from prosemble.core.pipeline import StandardScaler
    from prosemble.core.utils import train_test_split_jax
    from prosemble.models import GLVQ, GMLVQ

    data = load_wine()
    X = jnp.asarray(data.data, dtype=jnp.float32)
    y = jnp.asarray(data.target)
    X_train, X_test, y_train, y_test = train_test_split_jax(X, y, test_size=0.3, random_seed=0)
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    glvq = GLVQ(n_prototypes_per_class=1, max_iter=300, lr=0.05, random_seed=0).fit(X_train, y_train)
    model = GMLVQ(n_prototypes_per_class=1, max_iter=300, lr=0.01, random_seed=0).fit(X_train, y_train)
    print(f'  GLVQ test accuracy {float(jnp.mean(glvq.predict(X_test) == y_test)):.3f}')
    print(f'  GMLVQ test accuracy {float(jnp.mean(model.predict(X_test) == y_test)):.3f}')

    lam = np.asarray(model.lambda_matrix)
    relevance = np.diag(lam) / np.trace(lam)
    eigval, eigvec = np.linalg.eigh(lam)
    eigval, eigvec = eigval[::-1], eigvec[:, ::-1]
    print('  eigenvalues of Lambda (share):', np.round(eigval / eigval.sum(), 3)[:4])
    order = np.argsort(relevance)
    names = [n.replace('od280/od315_of_diluted_wines', 'OD280/OD315').replace('_', ' ') for n in data.feature_names]
    for i in order[::-1][:4]:
        print(f'  relevance {names[i]:20s} {relevance[i]:.3f}')

    proj = eigvec[:, :2] * np.sqrt(np.maximum(eigval[:2], 0))
    Z = np.asarray(jnp.concatenate([X_train, X_test])) @ proj
    labels = np.asarray(jnp.concatenate([y_train, y_test]))
    W = np.asarray(model.prototypes_) @ proj
    for theme in THEMES:
        t = _style(theme)
        fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw={'width_ratios': [1, 1.15]})
        a.barh(np.arange(len(order)), relevance[order], color=t['classes'][0], height=0.65)
        a.set_yticks(np.arange(len(order)), [names[i] for i in order])
        a.set_xlabel('relevance (diagonal of Λ, normalised)')
        a.set_title('Which features the distance uses')
        a.tick_params(axis='y', length=0)
        for k in range(3):
            pts = Z[labels == k]
            b.scatter(pts[:, 0], pts[:, 1], s=16, color=t['classes'][k], alpha=0.75, linewidths=0,
                      label=f'cultivar {k + 1}')
        b.scatter(W[:, 0], W[:, 1], s=200, c=[t['classes'][i] for i in np.asarray(model.prototype_labels_)],
                  edgecolors=t['fg'], linewidths=1.8, zorder=5)
        b.set_xlabel('first direction of Λ')
        b.set_ylabel('second direction of Λ')
        b.set_title('The data as the model sees it')
        b.legend(loc='best', fontsize=9, markerscale=1.4)
        fig.tight_layout(w_pad=3)
        _save(fig, 'tutorial-wine', theme)


# ------------------------------------------------------ tutorial: probabilities
def calibration():
    from sklearn.datasets import make_classification
    from prosemble.core.evaluation import expected_calibration_error, selective_accuracy
    from prosemble.core.pipeline import StandardScaler
    from prosemble.core.utils import train_test_split_jax
    from prosemble.models import DKLMRSLVQ

    X, y = make_classification(n_samples=1500, n_features=10, n_informative=5, n_redundant=2,
                               n_classes=3, n_clusters_per_class=2, class_sep=1.0, random_state=0)
    X, y = jnp.asarray(X, dtype=jnp.float32), jnp.asarray(y)
    X_train, X_test, y_train, y_test = train_test_split_jax(X, y, test_size=0.4, random_seed=0)
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    model = DKLMRSLVQ(n_prototypes_per_class=2, max_iter=200, lr=0.01, random_seed=0)
    model.fit(X_train, y_train)
    before = np.asarray(model.predict_proba(X_test))
    model.calibrate(X_train, y_train, cv=3)
    after = np.asarray(model.predict_proba(X_test))
    yt = np.asarray(y_test)

    def log_loss(p):
        return float(-np.mean(np.log(np.clip(p[np.arange(len(yt)), yt], 1e-12, 1))))

    print(f'  accuracy {float(np.mean(after.argmax(1) == yt)):.3f}, temperature {float(model.temperature_):.2f}')
    print(f'  mean confidence {before.max(1).mean():.3f} -> {after.max(1).mean():.3f}')
    print(f'  log-loss {log_loss(before):.3f} -> {log_loss(after):.3f}')
    print(f'  ECE {expected_calibration_error(before, y_test):.3f} -> {expected_calibration_error(after, y_test):.3f}')
    coverages = np.linspace(0.3, 1.0, 36)
    sel = [selective_accuracy(after, y_test, c) for c in coverages]
    for c in (1.0, 0.9, 0.8):
        print(f'  selective accuracy at {c:.0%}: {selective_accuracy(after, y_test, c):.3f}')

    def reliability(p, bins=10):
        conf, correct = p.max(1), p.argmax(1) == yt
        edges = np.linspace(0, 1, bins + 1)
        idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
        rows = [(conf[idx == b].mean(), correct[idx == b].mean(), (idx == b).sum())
                for b in range(bins) if (idx == b).sum() >= 5]
        return np.array(rows)

    for theme in THEMES:
        t = _style(theme)
        fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.1))
        a.plot([0, 1], [0, 1], color=t['muted'], lw=1, ls='--', label='perfect calibration')
        for p, name, colour in [(before, 'fitted', t['classes'][3]), (after, 'calibrated', t['classes'][0])]:
            r = reliability(p)
            a.plot(r[:, 0], r[:, 1], marker='o', color=colour, lw=2, label=name)
        a.set_xlim(0.4, 1.02)
        a.set_ylim(0.4, 1.02)
        a.set_xlabel('confidence (largest class probability)')
        a.set_ylabel('accuracy')
        a.set_title('Reliability on test data')
        a.legend(loc='upper left', fontsize=9)
        b.plot(coverages, sel, color=t['classes'][1], lw=2.2)
        b.axhline(sel[-1], color=t['muted'], lw=1, ls='--')
        b.text(0.99, sel[-1], '  no rejection', color=t['muted'], va='bottom', ha='left', fontsize=9)
        b.set_xlabel('fraction of samples kept (coverage)')
        b.set_ylabel('accuracy on the kept samples')
        b.set_title('Rejecting the least confident samples')
        b.invert_xaxis()
        fig.tight_layout(w_pad=3)
        _save(fig, 'tutorial-calibration', theme)


# --------------------------------------------------- tutorial: anomaly detection
def _ring_data():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 1.5 * np.pi, 300)
    ring = np.c_[2 * np.cos(a), 2 * np.sin(a)] + rng.normal(0, 0.25, (300, 2))
    blob = rng.normal([3.2, -2.2], 0.35, (100, 2))
    normal = np.r_[ring, blob].astype(np.float32)
    test = rng.uniform(-4, 5, (400, 2)).astype(np.float32)
    far = np.min(np.linalg.norm(test[:, None] - normal[None], axis=2), 1) > 0.8
    return normal, test[far][:40]


def anomaly():
    from prosemble.models import OCDKRSLVQ

    normal, outliers = _ring_data()
    model = OCDKRSLVQ(n_prototypes=8, max_iter=300, lr=0.05, random_seed=0)
    model.fit(jnp.asarray(normal))
    accepted = float(np.mean(np.asarray(model.predict(jnp.asarray(normal))) == 0))
    flagged = float(np.mean(np.asarray(model.predict(jnp.asarray(outliers))) != 0))
    print(f'  normal samples accepted {accepted:.3f}; outliers flagged {flagged:.3f} ({len(outliers)})')

    n = 300
    x0, y0, x1, y1 = -4.2, -4.2, 5.2, 4.2
    gx, gy = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    grid = jnp.asarray(np.c_[gx.ravel(), gy.ravel()], dtype=jnp.float32)
    score = np.asarray(model.decision_function(grid)).reshape(n, n)
    for theme in THEMES:
        t = _style(theme)
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        bg = np.array(to_rgb(t['bg']))
        c = np.array(to_rgb(t['classes'][1]))
        alpha = 0.06 + 0.36 * score[..., None]
        ax.imshow((1 - alpha) * bg + alpha * c, extent=(x0, x1, y0, y1), origin='lower', aspect='auto')
        ax.contour(gx, gy, score, levels=[0.5], colors=[t['classes'][1]], linewidths=1.8)
        ax.scatter(normal[:, 0], normal[:, 1], s=10, color=t['classes'][1], alpha=0.7, linewidths=0,
                   label='normal (training)')
        ax.scatter(outliers[:, 0], outliers[:, 1], s=38, marker='x', color=t['classes'][3], linewidths=1.8,
                   label='new outliers')
        P = np.asarray(model.prototypes_)[:8]
        ax.scatter(P[:, 0], P[:, 1], s=170, color=t['classes'][1], edgecolors=t['fg'], linewidths=1.8,
                   zorder=5, label='prototypes')
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        _bare(ax)
        ax.legend(loc='upper right', fontsize=9, facecolor=t['bg'], framealpha=0.85, frameon=True,
                  edgecolor=t['grid'])
        _save(fig, 'tutorial-anomaly', theme)


# ------------------------------------------------------------ tutorial: clustering
def clustering():
    from prosemble.models import FCM

    rng = np.random.default_rng(1)
    centres = np.array([[0, 0], [3.2, 0.6], [1.4, 2.9]])
    X = np.r_[[rng.normal(c, 0.65, (120, 2)) for c in centres]].reshape(-1, 2).astype(np.float32)
    model = FCM(n_clusters=3, fuzzifier=2.0, max_iter=200, random_seed=0).fit(jnp.asarray(X))
    U = np.asarray(model.predict_proba(jnp.asarray(X)))
    print(f'  share of samples with largest membership below 0.6: {np.mean(U.max(1) < 0.6):.3f}')
    n = 280
    x0, y0 = X.min(0) - 0.8
    x1, y1 = X.max(0) + 0.8
    gx, gy = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    G = np.asarray(model.predict_proba(jnp.asarray(np.c_[gx.ravel(), gy.ravel()], dtype=jnp.float32)))
    for theme in THEMES:
        t = _style(theme)
        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        cols = np.array([to_rgb(c) for c in t['classes'][:3]])
        bg = np.array(to_rgb(t['bg']))
        mix = G @ cols                                       # membership-weighted colour
        alpha = 0.10 + 0.40 * ((G.max(1) - 1 / 3) / (2 / 3))[:, None]
        ax.imshow(((1 - alpha) * bg + alpha * mix).reshape(n, n, 3), extent=(x0, x1, y0, y1),
                  origin='lower', aspect='auto')
        ax.scatter(X[:, 0], X[:, 1], s=12, c=U @ cols, linewidths=0, alpha=0.9)
        C = np.asarray(model.centroids_)
        ax.scatter(C[:, 0], C[:, 1], s=220, c=cols, edgecolors=t['fg'], linewidths=2, zorder=5)
        _bare(ax)
        _save(fig, 'tutorial-clustering', theme)


# -------------------------------------------------------------- tutorial: topology
def topology():
    from prosemble.models import GrowingNeuralGas, KohonenSOM

    rng = np.random.default_rng(2)
    t_ = rng.uniform(0.5, 3.6 * np.pi, 900)
    spiral = (np.c_[t_ * np.cos(t_), t_ * np.sin(t_)] / 4 + rng.normal(0, 0.12, (900, 2))).astype(np.float32)
    gng = GrowingNeuralGas(max_nodes=60, max_iter=60, random_seed=0).fit(jnp.asarray(spiral))
    square = rng.uniform(-1, 1, (900, 2)).astype(np.float32)
    som = KohonenSOM(grid_height=12, grid_width=12, max_iter=200, random_seed=0).fit(jnp.asarray(square))
    P = np.asarray(gng.prototypes_)
    E = np.asarray(gng.edges_)
    n_active = int(np.asarray(gng.n_active_)) if np.ndim(gng.n_active_) == 0 else len(P)
    print(f'  GNG nodes {n_active}, edges array {E.shape}')
    for theme in THEMES:
        t = _style(theme)
        fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.8))
        a.scatter(spiral[:, 0], spiral[:, 1], s=6, color=t['muted'], alpha=0.45, linewidths=0)
        if E.ndim == 2 and E.shape[0] == E.shape[1]:
            pairs = np.argwhere(np.triu(E > 0, 1))
        else:
            pairs = E.astype(int)
        for i, j in pairs:
            if i < n_active and j < n_active:
                a.plot(P[[i, j], 0], P[[i, j], 1], color=t['classes'][0], lw=1.4)
        a.scatter(P[:n_active, 0], P[:n_active, 1], s=34, color=t['classes'][0], edgecolors=t['fg'],
                  linewidths=0.8, zorder=5)
        a.set_title('Growing Neural Gas')
        a.set_aspect('equal')
        _bare(a)
        W = np.asarray(som.prototypes_).reshape(12, 12, 2)
        b.scatter(square[:, 0], square[:, 1], s=6, color=t['muted'], alpha=0.45, linewidths=0)
        for r in range(12):
            b.plot(W[r, :, 0], W[r, :, 1], color=t['classes'][1], lw=1.2)
            b.plot(W[:, r, 0], W[:, r, 1], color=t['classes'][1], lw=1.2)
        b.scatter(W[..., 0].ravel(), W[..., 1].ravel(), s=14, color=t['classes'][1], zorder=5)
        b.set_title('Self-organizing map (12 × 12)')
        b.set_aspect('equal')
        _bare(b)
        fig.tight_layout(w_pad=2)
        _save(fig, 'tutorial-topology', theme)


# ------------------------------------------------------------ tutorial: time series
def timeseries():
    import importlib.util
    from prosemble.models import SoftDTWRSLVQ

    run_py = pathlib.Path(__file__).resolve().parents[2] / 'benchmarks' / 'time_series' / 'run.py'
    spec = importlib.util.spec_from_file_location('ucr_benchmark', run_py)
    ucr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ucr)
    X_train, y_train, X_test, y_test = ucr.load('Trace')

    model = SoftDTWRSLVQ(gamma=1.0, n_prototypes_per_class=1, max_iter=150, lr=0.05,
                         batch_size=64, random_seed=0)
    model.fit(X_train, y_train)
    acc = float(np.mean(np.asarray(model.predict(X_test)) == y_test))
    print(f'  Trace: {len(X_train)} training series of length {X_train.shape[1]}; test accuracy {acc:.3f}')
    P, L = np.asarray(model.prototypes_), np.asarray(model.prototype_labels_)
    for theme in THEMES:
        t = _style(theme)
        fig, axes = plt.subplots(1, 4, figsize=(12, 2.9), sharey=True)
        for k, ax in enumerate(axes):
            series = X_train[y_train == k]
            for s in series[:25]:
                ax.plot(s, color=t['muted'], lw=0.6, alpha=0.35)
            ax.plot(series.mean(0), color=t['fg'], lw=1.4, ls='--', label='Euclidean mean')
            ax.plot(P[L == k][0], color=t['classes'][k], lw=2.6, label='soft-DTW prototype')
            ax.set_title(f'class {k + 1}')
            ax.set_xticks([])
            if k:
                ax.spines['left'].set_visible(False)
                ax.tick_params(axis='y', length=0)
        axes[0].legend(loc='upper left', fontsize=8.5)
        fig.tight_layout(w_pad=1)
        _save(fig, 'tutorial-timeseries', theme)


# ------------------------------------------------ tutorial: covariance matrices
def covariance_trials(n_per_class, seed, channels=6, length=200):
    """Covariance matrices of simulated multichannel recordings, 3 classes.

    Six hidden sources are mixed into six channels by a fixed matrix. In
    class k, source k has about e^1.2 = 3.3 times more power. Every
    recording also has a random overall gain, as real recordings do.
    """
    mixing = np.random.default_rng(0).normal(size=(channels, channels))
    log_power = np.zeros((3, channels))
    log_power[[0, 1, 2], [0, 1, 2]] = 1.2
    rng = np.random.default_rng(seed)
    covs, labels = [], []
    for k in range(3):
        for _ in range(n_per_class):
            power = np.exp(log_power[k] + rng.normal(0, 0.4, channels) + rng.normal(0, 0.8))
            sources = rng.normal(size=(channels, length)) * np.sqrt(power)[:, None]
            x = mixing @ sources + 0.1 * rng.normal(size=(channels, length))
            covs.append(x @ x.T / length)
            labels.append(k)
    return np.array(covs, dtype=np.float32), np.array(labels, dtype=np.int32)


def _logm(P):
    w, V = np.linalg.eigh(P)
    return (V * np.log(w)[..., None, :]) @ np.swapaxes(V, -1, -2)


def _plain_log_ticks(ax):
    """Log axes labelled with plain numbers (1, 10, 100 or 30, 40, 60)."""
    for axis in (ax.xaxis, ax.yaxis):
        lo, hi = (ax.get_xlim() if axis is ax.xaxis else ax.get_ylim())
        if hi / lo < 20:
            ticks = [v for v in (20, 30, 40, 60, 80, 100, 150) if lo <= v <= hi]
        else:
            ticks = [10.0 ** e for e in range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
                     if lo <= 10.0 ** e <= hi]
        axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def covariance():
    import jax
    from prosemble.core.manifolds import SPD
    from prosemble.models import GLVQ, RiemannianRSLVQ

    def standardise(a, b):
        mu, sd = a.mean(0), a.std(0) + 1e-8
        return (a - mu) / sd, (b - mu) / sd

    def fit_glvq(train, test, y, seed):
        a, b = standardise(train.reshape(len(train), -1), test.reshape(len(test), -1))
        m = GLVQ(n_prototypes_per_class=1, max_iter=300, lr=0.05, random_seed=seed)
        return m.fit(jnp.asarray(a), jnp.asarray(y)), b

    def fit_riemannian(train, y, seed):
        m = RiemannianRSLVQ(manifold=SPD(6), n_prototypes_per_class=1, sigma='auto',
                            adapt_sigma=True, max_iter=300, lr=0.01, random_seed=seed)
        return m.fit(jnp.asarray(train.reshape(len(train), -1)), jnp.asarray(y))

    P_test, y_test = covariance_trials(200, seed=99)
    P_train, y_train = covariance_trials(40, seed=40000)
    glvq, b = fit_glvq(P_train, P_test, y_train, 0)
    riem = fit_riemannian(P_train, y_train, 0)
    acc_g = float(np.mean(np.asarray(glvq.predict(jnp.asarray(b))) == y_test))
    acc_r = float(np.mean(np.asarray(riem.predict(jnp.asarray(P_test.reshape(len(P_test), -1)))) == y_test))
    print(f'  40 recordings per class: GLVQ on raw matrices {acc_g:.3f}, RiemannianRSLVQ {acc_r:.3f}')

    D_glvq = ((b[:, None, :] - np.asarray(glvq.prototypes_)[None]) ** 2).sum(-1)
    W = jnp.asarray(np.asarray(riem.prototypes_).reshape(-1, 6, 6))
    pairwise = jax.jit(jax.vmap(jax.vmap(SPD(6).distance, in_axes=(None, 0)), in_axes=(0, None)))
    D_riem = np.asarray(pairwise(jnp.asarray(P_test), W)) ** 2

    sizes, seeds = [5, 10, 20, 40, 80], range(5)
    curve = {'raw': [], 'logm': [], 'riem': []}
    for n in sizes:
        rows = {k: [] for k in curve}
        for seed in seeds:
            P, y = covariance_trials(n, seed=1000 * n + seed)
            m, bt = fit_glvq(P, P_test, y, seed)
            rows['raw'].append(np.mean(np.asarray(m.predict(jnp.asarray(bt))) == y_test))
            m, bt = fit_glvq(_logm(P), _logm(P_test), y, seed)
            rows['logm'].append(np.mean(np.asarray(m.predict(jnp.asarray(bt))) == y_test))
            m = fit_riemannian(P, y, seed)
            assert np.isfinite(np.asarray(m.loss_history_)).all()
            rows['riem'].append(np.mean(np.asarray(m.predict(jnp.asarray(P_test.reshape(len(P_test), -1)))) == y_test))
        for k in curve:
            curve[k].append((np.mean(rows[k]), np.std(rows[k])))
        print(f'  n={n:2d}: ' + ', '.join(f'{k} {np.mean(v):.3f}±{np.std(v):.3f}' for k, v in rows.items()))

    def own_other(D, labels, proto_labels):
        own = D[np.arange(len(D)), np.searchsorted(proto_labels, labels)]
        other = np.where(proto_labels[None, :] == labels[:, None], np.inf, D).min(1)
        return own, other

    names = {'raw': 'GLVQ on the raw matrices', 'logm': 'GLVQ on the matrix logarithms',
             'riem': 'RiemannianRSLVQ on SPD(6)'}
    for theme in THEMES:
        t = _style(theme)
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
        for ax, D, model, title in [(axes[0], D_glvq, glvq, 'Euclidean distance, raw matrices'),
                                    (axes[1], D_riem, riem, 'Riemannian distance')]:
            own, other = own_other(D, y_test, np.asarray(model.prototype_labels_))
            for k in range(3):
                ax.scatter(other[y_test == k], own[y_test == k], s=9, alpha=0.7, linewidths=0,
                           color=t['classes'][k], label=f'class {k + 1}')
            lo, hi = min(own.min(), other.min()) * 0.8, max(own.max(), other.max()) * 1.25
            ax.plot([lo, hi], [lo, hi], color=t['muted'], ls='--', lw=1)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            _plain_log_ticks(ax)
            ax.set_xlabel('squared distance to the nearest other prototype')
            ax.set_ylabel('squared distance to own prototype')
            ax.set_title(title)
        axes[0].legend(loc='upper left', fontsize=9, markerscale=1.5)
        fig.tight_layout(w_pad=3)
        _save(fig, 'tutorial-spd-distances', theme)

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for k, colour in [('raw', t['classes'][3]), ('logm', t['classes'][1]), ('riem', t['classes'][0])]:
            mean = np.array([m for m, _ in curve[k]])
            sd = np.array([s for _, s in curve[k]])
            ax.plot(sizes, mean, marker='o', lw=2, color=colour, label=names[k])
            ax.fill_between(sizes, mean - sd, mean + sd, color=colour, alpha=0.15, linewidth=0)
        ax.set_xscale('log')
        ax.set_xticks(sizes, [str(s) for s in sizes])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel('training recordings per class')
        ax.set_ylabel('test accuracy')
        ax.set_title('Accuracy against training-set size')
        ax.legend(loc='lower right', fontsize=9)
        _save(fig, 'tutorial-spd-curve', theme)


# ------------------------------------------------- gallery thumbnails (crops)
def thumbnails():
    """One panel of each multi-panel tutorial figure, for the gallery."""
    from PIL import Image

    crops = {'tutorial-wine': (0.545, 1.0), 'tutorial-calibration': (0.0, 0.5),
             'tutorial-topology': (0.0, 0.5), 'tutorial-timeseries': (0.0, 0.265)}
    for name, (a, b) in crops.items():
        for theme in THEMES:
            img = Image.open(OUT / f'{name}-{theme}.png')
            w, h = img.size
            img.crop((int(a * w), 0, int(b * w), h)).save(OUT / f'thumb-{name[9:]}-{theme}.png')
            print('  wrote', f'thumb-{name[9:]}-{theme}.png')


FIGURES = {'hero': hero, 'wine': wine, 'calibration': calibration, 'anomaly': anomaly,
           'clustering': clustering, 'topology': topology, 'timeseries': timeseries,
           'covariance': covariance, 'thumbnails': thumbnails}

if __name__ == '__main__':
    for name in sys.argv[1:] or FIGURES:
        print(name)
        FIGURES[name]()
