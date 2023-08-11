# Developers guide

A *brief* guide for PyDMD developers on how to leverage GLAPPO for new and old PyDMD codes.

## Guidelines

### Calling `build_linalg_module()`

Be careful on the argument on which you call `build_linalg_module()`. It may happen that some NumPy arrays
are created *en passant* to be used as arguments for more complicated functions. These are not good candidates
for `build_linalg_module()`, as they clearly do not convey information about user preferences on array typing.

### Check aggressively...

Always check that the user is providing appropriate array pairs/triplets in PyDMD entrypoints (e.g. `fit()`).
`linalg.py` provides some utility functions (`is_array(X)`, `assert_same_linalg_type(X,*args)`) to facilitate writing
this kind of checks.

### ... but trust the team

No need to check the output of internal functions like `DMDBase._optimal_dmd_matrices()`. This clutters the
code and provides no additional value, our PRs are carefully reviewed by developers from the core team of
PyDMD.

### Test!

Test new and old code for all new possibilities introduced by GLAPPO, for instance tensorized training. There are
many examples in `tests`.

### Plan for batching

GLAPPO enables batched DMD, namely applying the same DMD operator to multiple datasets in one highly optimized call.
In order to support batching, make sure you index shapes starting from the last one, e.g. `shape[-2]` instead of
`shape[0]` to identify the space dimension, or `shape[-1]` instead of `shape[1]` to identify the time dimension).
Also, all calls to `.T` should be dropped in favor of `.swapaxes(-1, -2)`. You find tons of examples in classes which
already support batched training.

## Example

In this paragraph we're going to look at some snippets from the diff of the GLAPPification of `SubspaceDMD`. 
During the development I recommend to keep the 
[GLAPPO interface](https://github.com/fandreuz/PyDMD/blob/generic-linalg/pydmd/linalg/linalg_base.py)
close to you.

```diff
-import numpy as np
-
 from .dmdbase import DMDBase
 from .dmdoperator import DMDOperator
+from .utils import compute_svd
+from pydmd.linalg import build_linalg_module, is_array
 from .snapshots import Snapshots
 ```

GLAPPO defines its own linear algebra functions, therefore you should not need NumPy. Simple
indexes arrays (for example) generated with `np.arange()` are an exception, but you should
make sure not to mix NumPy and GLAPPO. It's better to try to remove all explicit references to
a linear algebra package (and GLAPPO provides its own `arange` method).

```diff
     def compute_operator(self, Yp, Yf):
         [...]
+        n = Yp.shape[-2] // 2
 
-        n = Yp.shape[0] // 2
```

Using `.shape[-2]` instead of `.shape[0]` is critical, since during tensorized training
`Yp` is expected to have 3 dimensions. The first dimensions is commonly assumed to be the
batch dimension, therefore the other axes (space and time) are shifted one position to the right.

```diff
+        linalg_module = build_linalg_module(Yp)
 
-        Uq, _, _ = reducedsvd(O)
+        O = linalg_module.multi_dot((Yf, Vp, Vp.swapaxes(-1, -2).conj()))
```

`linalg_module` is the implementation of the GLAPPO interface for the array type represented by
the input. It contains all the available methods, in the specific flavor for the desired
linear algebra backend.

```diff
-        Uq1, Uq2 = Uq[:n, :r], Uq[n:, :r]
+        Uq1 = Uq[..., :n, :r]
+        Uq2 = Uq[..., n:, :r]
```

Same as for the `shape`, it's important to use the leading `...` in order to be generic
on the number of axes before the space/time trailing axes. The change above supports 2
axes as well as 3, since the (optional) leading batch dimension is matched by the three
dots.

```diff 
-        M = Uq2.dot(V) * np.reciprocal(S)
+        M = linalg_module.dot(Uq2, V) / (S[:, None] if Yp.ndim == 3 else S)
```

1. `np.reciprocal` is not part of GLAPPO, therefore we use the standard `/` operator;
2. In case we're performing a tensorized training (`Yp.ndim == 3`) we need to make sure
    that the divider is [broadcasted](https://numpy.org/doc/stable/user/basics.broadcasting.html)
    properly against the tensor on the left side of the operator. Tests are your friend here:
    just run a "fake" tensorized training (i.e. batch dimension = 1) and they're going to fail
    if you got the position of `None` wrong.

```diff
+        elif is_array(self._rescale_mode):
```

GLAPPO has its own suite of utility methods to work with generic matrices, you can look it up
[here](https://github.com/fandreuz/PyDMD/blob/generic-linalg/pydmd/linalg/linalg_utils.py).

```diff
+            scaling_factors = linalg_module.to(
+                self.as_array, self._rescale_mode
+            )
```

`linalg_module.to(reference, *args)` takes a `reference` and converts all `args` to the same linear
algebra backend. It also makes sure that all the arrays are on the same device as the `reference`
(e.g. CPU/GPU for PyTorch).

```diff
-        high_dimensional_eigenvectors = M.dot(W) * np.reciprocal(
-            self.eigenvalues
+        high_dimensional_eigenvectors = linalg_module.dot(M, W) / (
+            self.eigenvalues[:, None] if M.ndim == 3 else self.eigenvalues
         )
```

```diff
-    def fit(self, X):
+    def fit(self, X, batch=False):
         [...]
         self._reset()
 
-        self._snapshots_holder = Snapshots(X)
+        self._snapshots_holder = Snapshots(X, batch=batch)
```

Batch training requires a flag to be set to `True`. This is needed because the
default behavior of PyDMD for `ndim > 2` arrays is to flatten all the axes except
the last one.

```diff
-        n_samples = self.snapshots.shape[1]
-        Y0 = self.snapshots[:, :-3]
-        Y1 = self.snapshots[:, 1:-2]
-        Y2 = self.snapshots[:, 2:-1]
-        Y3 = self.snapshots[:, 3:]
+        n_samples = self.snapshots.shape[-1]
+        Y0 = self.snapshots[..., :-3]
+        Y1 = self.snapshots[..., 1:-2]
+        Y2 = self.snapshots[..., 2:-1]
+        Y3 = self.snapshots[..., 3:]
```

Generic dimension indexing, same as above.

```diff
-        Yp = np.vstack((Y0, Y1))
-        Yf = np.vstack((Y2, Y3))
+        linalg_module = build_linalg_module(X)
+        Yp = linalg_module.cat((Y0, Y1), axis=-2)
+        Yf = linalg_module.cat((Y2, Y3), axis=-2)
```

GLAPPO provides its own method `cat()` instead of `h/v/stack`. This method takes
an `axis` parameter which you can use to select the axis to stack on. In this case
we're stacking on the space dimension.

## Additional remarks

Due to the strong requirements of `torch.mul` and `torch.linalg.multi_dot`, the implementation of these
two functions in `pytorch_linalg.py` forces a cast to the biggest **complex** type found in the argumnets.
We decided to take this path instead of placing the burden on user/implementors since for some algorithms
it's hard to control consistently whether the output is complex or real (e.g. `torch.linalg.eig`) and casts
will happen internally quite often. This damages memory efficiency and performance, but ensures correct 
results. It will be subject of investigation if we receive complains from our users.

This kind of casts is logged, in order to get the logs enable the `INFO` logging level:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
