"""
Unified save/load serialization for all prosemble model families.

Provides :class:`SerializationMixin` — a mixin that consolidates the
save/load logic previously duplicated across ``SupervisedPrototypeModel``,
``UnsupervisedPrototypeModel``, and ``FuzzyClusteringBase``.

Model-family-specific differences are handled via hook methods that
subclasses override.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

#: Current schema version stamped into new save files.
_SCHEMA_VERSION = 1


class SerializationMixin:
    """Mixin providing unified save/load for all model families.

    Subclasses customise behaviour by overriding the hook methods below.
    The ``_get_fitted_arrays`` / ``_set_fitted_arrays`` interface used by
    36+ model subclasses is **not** changed.
    """

    # ------------------------------------------------------------------
    # Hooks — override in subclasses
    # ------------------------------------------------------------------

    def _get_save_metadata(self) -> dict:
        """Return model-family-specific metadata fields.

        Returned dict is merged into the JSON metadata blob.
        Override in each base class (Supervised, Unsupervised, Fuzzy).
        """
        return {}

    def _restore_metadata(self, metadata: dict) -> None:
        """Restore model-family-specific fields from *metadata*.

        Called after the model is constructed and fitted arrays are set.
        """

    def _save_optimizer_state(self, arrays: dict, metadata: dict) -> None:
        """Serialize optimizer state into *arrays* / *metadata*.

        Only ``SupervisedPrototypeModel`` overrides this.
        """

    def _load_optimizer_state(self, data, metadata: dict) -> None:
        """Restore optimizer state from *data* and *metadata*.

        Only ``SupervisedPrototypeModel`` overrides this.
        """

    @classmethod
    def _pre_load_construct(cls, hyperparams: dict, metadata: dict) -> dict:
        """Modify *hyperparams* before model construction.

        Riemannian models override this to reconstruct the manifold object
        and remove manifold-specific keys from the dict.

        Must return the (possibly modified) hyperparams dict.
        """
        return hyperparams

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, path: str, quantize: str | None = None) -> None:
        """Save fitted model to an NPZ file.

        Parameters
        ----------
        path : str
            File path (``.npz`` extension added if not present).
        quantize : str, optional
            Quantize before saving: ``'float16'``, ``'bfloat16'``, or
            ``'int8'``.  The model in memory is unchanged.
        """
        self._check_fitted()

        # Temporarily quantize if requested
        originals: dict = {}
        if quantize is not None and not self.is_quantized:
            for attr in self._get_quantizable_attrs():
                originals[attr] = getattr(self, attr)
            self.quantize(quantize)

        arrays = self._get_fitted_arrays()
        hyperparams = self._get_hyperparams()

        metadata = {
            'schema_version': _SCHEMA_VERSION,
            'class_name': type(self).__name__,
            'module': type(self).__module__,
            'hyperparams': hyperparams,
            'fitted_array_names': list(arrays.keys()),
            'quantized_dtype': self.quantized_dtype,
        }

        # Merge family-specific metadata
        metadata.update(self._get_save_metadata())

        # int8 scale factors
        if self.quantized_dtype == 'int8' and hasattr(self, '_int8_scales'):
            for attr, scale in self._int8_scales.items():
                arrays[f'__scale__{attr}'] = np.asarray(scale)
            metadata['int8_scale_keys'] = list(self._int8_scales.keys())

        # Optimizer state (supervised only)
        self._save_optimizer_state(arrays, metadata)

        save_dict = {'__metadata__': np.array(json.dumps(metadata))}
        save_dict.update(arrays)
        np.savez_compressed(path, **save_dict)

        # Restore originals (no precision loss in memory)
        if originals:
            for attr, val in originals.items():
                setattr(self, attr, val)
            self._quantized_dtype = None
            if hasattr(self, '_int8_scales'):
                self._int8_scales = {}

    @classmethod
    def load(cls, path: str):
        """Load a fitted model from an NPZ file.

        Parameters
        ----------
        path : str
            Path to the ``.npz`` file.

        Returns
        -------
        model
            Reconstructed fitted model.
        """
        from prosemble.models import _MODEL_REGISTRY

        if not path.endswith('.npz'):
            path = path + '.npz'

        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data['__metadata__']))

        class_name = metadata['class_name']
        if class_name not in _MODEL_REGISTRY:
            raise ValueError(f"Unknown model class: {class_name}")

        model_cls = _MODEL_REGISTRY[class_name]
        hyperparams = dict(metadata['hyperparams'])

        # Allow subclasses to modify hyperparams (e.g. manifold reconstruction)
        hyperparams = model_cls._pre_load_construct(hyperparams, metadata)

        model = model_cls(**hyperparams)

        # Restore fitted arrays
        arrays = {name: data[name] for name in metadata['fitted_array_names']}
        model._set_fitted_arrays(arrays)

        # Restore family-specific metadata
        model._restore_metadata(metadata)

        # Restore quantization state
        q_dtype = metadata.get('quantized_dtype')
        if q_dtype:
            model._quantized_dtype = q_dtype
            if q_dtype == 'int8':
                model._int8_scales = {}
                for attr in metadata.get('int8_scale_keys', []):
                    scale_key = f'__scale__{attr}'
                    if scale_key in data:
                        model._int8_scales[attr] = jnp.asarray(data[scale_key])

        # Restore optimizer state (supervised only)
        model._load_optimizer_state(data, metadata)

        return model
