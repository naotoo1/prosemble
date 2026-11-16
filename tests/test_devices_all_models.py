"""Multi-device training and gradient checkpointing across the model families."""

import inspect
import json
import os
import subprocess
import sys
import textwrap

import pytest

import prosemble.models as models
from prosemble.models import SupervisedPrototypeModel


def _supervised_models():
    return sorted(
        name for name in models.__all__
        if inspect.isclass(getattr(models, name))
        and issubclass(getattr(models, name), SupervisedPrototypeModel)
        and name != 'SupervisedPrototypeModel'
    )


@pytest.mark.parametrize('name', _supervised_models())
def test_constructor_accepts_devices_and_checkpointing(name):
    params = inspect.signature(getattr(models, name).__init__).parameters
    forwards_kwargs = any(p.kind is p.VAR_KEYWORD for p in params.values())
    assert forwards_kwargs or {'devices', 'gradient_checkpointing'} <= set(params)


# One model for each way the families store learned quantities besides the
# prototypes (metric matrices, local matrices, widths, thresholds, backbones,
# manifold-valued prototypes, series).
FAMILIES = ['GMLVQ', 'LGMLVQ', 'SMNG', 'MCELVQ_NG', 'LMRSLVQ', 'SiameseGMLVQ',
            'ImageGLVQ', 'DKLMRSLVQ', 'WLMRSLVQ', 'OCGMLVQ_NG', 'SVQOCC_M',
            'RiemannianRSLVQ', 'SoftDTWRSLVQ']

SCRIPT = textwrap.dedent('''
    import json, sys, warnings
    warnings.filterwarnings('ignore')
    import jax, jax.numpy as jnp, numpy as np
    import prosemble.models as models
    from prosemble.core.manifolds import SO

    def setup(name, n=48):
        rng = np.random.default_rng(0)
        kw = dict(max_iter=6, lr=0.01, random_seed=0, optimizer='sgd', use_scan=False)
        if name.startswith('Riemannian'):
            keys = jax.random.split(jax.random.PRNGKey(0), n)
            X = np.asarray(jax.vmap(SO(3).random_point)(keys)).reshape(n, -1)
            y = np.arange(n) % 2
            kw.update(manifold=SO(3), n_prototypes_per_class=1)
        elif name.startswith(('OC', 'SVQOCC')):
            X = rng.normal(size=(n, 4))
            y = (np.arange(n) % 4 == 0).astype(int)
            kw.update(n_prototypes=2, target_label=0)
        elif name.startswith('Image'):
            X = rng.normal(size=(n, 64))
            y = np.arange(n) % 2
            kw.update(input_shape=(8, 8, 1), n_prototypes_per_class=1)
        else:
            X = rng.normal(size=(n, 6))
            X[np.arange(n) % 3 == 1] += 1.5
            y = np.arange(n) % 3
            kw.update(n_prototypes_per_class=1)
        return jnp.asarray(X, dtype=jnp.float32), jnp.asarray(y, dtype=jnp.int32), kw

    out = {'n_devices': len(jax.devices()), 'models': {}}
    for name in sys.argv[1:]:
        X, y, kw = setup(name)
        cls = getattr(models, name)
        single = cls(**kw).fit(X, y)
        multi = cls(devices=jax.devices(), gradient_checkpointing=True, **kw).fit(X, y)
        out['models'][name] = {
            'agree': float(np.mean(np.asarray(single.predict(X)) == np.asarray(multi.predict(X)))),
            'loss_diff': float(np.max(np.abs(np.asarray(single.loss_history_)
                                             - np.asarray(multi.loss_history_)))),
        }
    print(json.dumps(out))
''')


def test_families_train_on_several_devices():
    """After multi-device training, every family predicts on one device.

    Runs in a subprocess, because the number of CPU devices is fixed when JAX
    starts. With plain SGD, single- and multi-device training differ only by
    rounding.
    """
    env = dict(os.environ, XLA_FLAGS='--xla_force_host_platform_device_count=4',
               JAX_PLATFORMS='cpu')
    proc = subprocess.run([sys.executable, '-c', SCRIPT, *FAMILIES], env=env,
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result['n_devices'] == 4
    for name in FAMILIES:
        r = result['models'][name]
        assert r['agree'] == 1.0, (name, r)
        assert r['loss_diff'] < 1e-3, (name, r)
