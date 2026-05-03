"""Shared mixins for model base classes."""

import inspect

import jax.numpy as jnp


class MetadataCollectorMixin:
    """Mixin that auto-collects _hyperparams and _fitted_array_names from MRO.

    Any base class that declares per-class ``_hyperparams`` and
    ``_fitted_array_names`` tuples can inherit this mixin to get
    ``_all_hyperparams`` and ``_all_fitted_array_names`` aggregated
    automatically across the entire class hierarchy.
    """

    _hyperparams: tuple[str, ...] = ()
    _fitted_array_names: tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        all_hp: list[str] = []
        all_fa: list[str] = []
        for klass in reversed(cls.__mro__):
            for name in getattr(klass, '_hyperparams', ()):
                if name not in all_hp:
                    all_hp.append(name)
            for name in getattr(klass, '_fitted_array_names', ()):
                if name not in all_fa:
                    all_fa.append(name)
        cls._all_hyperparams = tuple(all_hp)
        cls._all_fitted_array_names = tuple(all_fa)

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Follows the sklearn estimator protocol by inspecting ``__init__``
        signatures across the MRO.

        Parameters
        ----------
        deep : bool, default=True
            Ignored (present for sklearn compatibility).

        Returns
        -------
        dict
            Parameter names mapped to their values.
        """
        params = {}
        for klass in type(self).__mro__:
            init = getattr(klass, '__init__', None)
            if init is None:
                continue
            sig = inspect.signature(init)
            for name, p in sig.parameters.items():
                if name == 'self' or p.kind in (
                    p.VAR_POSITIONAL, p.VAR_KEYWORD,
                ):
                    continue
                if name not in params:
                    params[name] = getattr(self, name, p.default)
        return params

    def set_params(self, **params):
        """Set parameters on this estimator.

        Parameters
        ----------
        **params
            Estimator parameters to set.

        Returns
        -------
        self
        """
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(
                    f"Invalid parameter '{key}' for {type(self).__name__}. "
                    f"Valid parameters: {sorted(valid.keys())}"
                )
            setattr(self, key, value)
        return self


class QuantizationMixin:
    """Mixin for quantizing/dequantizing fitted model parameters.

    Supports float16, bfloat16, and int8 (with per-tensor scale factors).

    Subclasses override ``_get_quantizable_attrs`` to declare which
    fitted attributes are eligible for quantization.
    """

    _VALID_DTYPES = {
        'float16': jnp.float16,
        'bfloat16': jnp.bfloat16,
        'int8': 'int8',
    }

    def _get_quantizable_attrs(self) -> list[str]:
        """Return attribute names of quantizable parameters.

        Subclasses override to return e.g. ``['centroids_']`` or
        ``['prototypes_']``. Default returns empty list.
        """
        return []

    def quantize(self, dtype='float16'):
        """Quantize model parameters to lower precision.

        Post-training quantization for smaller model size and faster inference.

        Parameters
        ----------
        dtype : str
            Target precision: 'float16', 'bfloat16', or 'int8'.

        Returns
        -------
        self
        """
        self._check_fitted()
        if dtype not in self._VALID_DTYPES:
            raise ValueError(
                f"dtype must be one of {list(self._VALID_DTYPES.keys())}, got '{dtype}'"
            )

        if dtype == 'int8':
            self._quantize_int8()
        else:
            target = self._VALID_DTYPES[dtype]
            self._quantize_float(target)

        self._quantized_dtype = dtype
        return self

    def dequantize(self):
        """Restore model parameters to float32.

        Returns
        -------
        self
        """
        self._check_fitted()
        if not hasattr(self, '_quantized_dtype') or self._quantized_dtype is None:
            return self

        if self._quantized_dtype == 'int8':
            self._dequantize_int8()
        else:
            self._dequantize_float()

        self._quantized_dtype = None
        return self

    @property
    def is_quantized(self) -> bool:
        """Whether model parameters are currently quantized."""
        return getattr(self, '_quantized_dtype', None) is not None

    @property
    def quantized_dtype(self) -> str | None:
        """Current quantization dtype, or None if not quantized."""
        return getattr(self, '_quantized_dtype', None)

    def _quantize_float(self, target_dtype):
        """Convert parameters to float16/bfloat16."""
        for attr in self._get_quantizable_attrs():
            val = getattr(self, attr)
            if val is not None:
                setattr(self, attr, val.astype(target_dtype))

    def _quantize_int8(self):
        """Quantize parameters to int8 with per-tensor scale factors."""
        self._int8_scales = {}
        for attr in self._get_quantizable_attrs():
            val = getattr(self, attr)
            if val is not None:
                val_f32 = val.astype(jnp.float32)
                abs_max = jnp.max(jnp.abs(val_f32))
                scale = abs_max / 127.0
                quantized = jnp.round(val_f32 / (scale + 1e-10)).astype(jnp.int8)
                setattr(self, attr, quantized)
                self._int8_scales[attr] = scale

    def _dequantize_float(self):
        """Restore float16/bfloat16 parameters to float32."""
        for attr in self._get_quantizable_attrs():
            val = getattr(self, attr)
            if val is not None:
                setattr(self, attr, val.astype(jnp.float32))

    def _dequantize_int8(self):
        """Restore int8 parameters to float32 using stored scales."""
        scales = getattr(self, '_int8_scales', {})
        for attr in self._get_quantizable_attrs():
            val = getattr(self, attr)
            if val is not None and attr in scales:
                setattr(self, attr, val.astype(jnp.float32) * scales[attr])
        self._int8_scales = {}
