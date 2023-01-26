# Deep-Learning enhanced Dynamic Mode Decomposition
Implementation of some Deep-Learning techniques to optimize DMD performance, based
on PyDMD. 

This project is made possible by mathLab/PyDMD#299, which enables PyTorch in PyDMD.

## Added value
- Enable DMD on GPU;
- Enable backpropagation through DMD on (in parantheses what we can optimize on):
    - Reconstructed data (accuracy, denoisification, ...)
    - Predicted data (prediction accuracy, stability, ...)
    - DMD modes
    - DMD amplitudes (sparsity-promoting, regularization, ...)
    - DMD eigenvalues (stability)
- Enable batched/tensorized DMD for high performance computational campaigns;
    - Non-DL variants can benefit from this as well (e.g. `ParametricDMD`).

## Progress
The project is comprised of different steps, summarized below:
- [ ] **Step 1**: PyDMD support for **backpropagation** and **GPU** architectures: mathLab/PyDMD#299
    - [x] Generic PyDMD linear algebra formulation
    - [x] Adapt NumPy concrete implementation
    - [x] Add PyTorch concrete implementation
    - [ ] Add JAX concrete implementation (see https://jax.readthedocs.io/en/latest/jax.numpy.html)
    - [x] Batched PyDMD formulation 
- [x] **Step 2**: Enhance DMD with Deep Learning techniques
    - [x] DLDMD
    - [ ] Trainable DMD hyperparameters

### Step 1
The first step is the most time consuming, as we need to rewrite the PyDMD framework
in a more general way leveraging linear algebra subroutines supported by all the target
frameworks (NumPy, PyTorch, ...)

This has been achieved by defining a new set of 
[classes](https://github.com/fAndreuzzi/PyDMD/tree/generic-linalg/pydmd/linalg) implementing 
a common linear algebra interface. PyDMD uses this common interface under the hood, and concrete
implementations of the interface are provided for the target frameworks. Below we display a small
sample from those classes:
```python
class LinalgBase:
    @classmethod
    def svd(cls, X, *args, **kwargs):
        raise NotImplementedError

class LinalgNumPy(LinalgBase):
    @classmethod
    def svd(cls, X, *args, **kwargs):
        return np.linalg.svd(X, *args, **kwargs)

class LinalgPyTorch(LinalgBase):
    @classmethod
    def svd(cls, X, *args, **kwargs):
        import torch

        return torch.linalg.svd(X, *args, **kwargs)
```
At the moment we provide support for about 40 functions.

An additional challenge was posed by batched (or *tensorized*) training. Most operators in
PyTorch support a leading *batch* dimension which enables broadcasting the operator to all the
samples in a batch (e.g. `torch.nn.Linear` receives in input a tensor of shape $(*, H_{in})$ ).
This yields both much more idiomatic code and better performance, as we can fully exploit
vectorization or GPU computing power.

The goal here was to support tensorized training avoiding overcomplicated code with tons of
conditional branches (`if X.ndim == 4:`). This was mainly achieved thanks to the `...` operator
in PyTorch and to same careful swapping of tensor axes. As we're going to see in the benchmark
this sub-step is fundamental to fully support Deep Learning on DMD, as the performance toll
imposed otherwise would have made unfeasible any kind of training.

**DMD tensorized fit() performance benchmark**
The following plot represents the performance (in milliseconds) of a batched/tensorized
DMD with different backends on 601 snapshots 3-dimensional snapshots.

![image](https://user-images.githubusercontent.com/8464342/211190539-fc942030-8823-4b91-be3d-631bf66f1e31.png)

**DMD variants to be ported**

We plan support for the DMD variants below:
- [x] CDMD
- [ ] DMD Modes tuner
- [x] DMD
- [x] DMDBase
- [x] DMDC
- [x] DMDOperator
- [x] FbDMD
- [x] HankelDMD
- [ ] HAVOK
- [x] HODMD
- [x] MRDMD
- [ ] OptDMD
- [ ] ParametricDMD
- [x] RDMD
- [ ] SPDMD
- [x] SubspaceDMD
- [ ] Fix second-fit backpropagation

### Step 2
The pair DMD+Deep Learning has been explored a little bit in literature. In `src/` we provide
the implementation for some of the resulting models. It would be interesting to explore the
potential of trainable DMD hyperparameters (e.g. `d` in `HankelDMD`).

## DLDMD
For more results and implementation, see `src/dldmd.py` or `notebooks/dldmd.ipynb`.

### Reconstruction/prediction accuracy (VS standard DMD)
![image](https://user-images.githubusercontent.com/8464342/214721981-01a2e5d4-6e4e-4201-98c3-56955f191d93.png)

### Encoder/decoder pair
![image](https://user-images.githubusercontent.com/8464342/214722370-54621935-1943-4fdb-95ed-6c87b6cda17a.png)


## References
- Alford-Lago, Daniel J., et al. "Deep learning enhanced dynamic mode decomposition." Chaos: An Interdisciplinary Journal of Nonlinear Science 32.3 (2022): 033116.
- Kutz, J. Nathan, et al. Dynamic mode decomposition: data-driven modeling of complex systems. Society for Industrial and Applied Mathematics, 2016.
- Demo, Nicola, Marco Tezzele, and Gianluigi Rozza. "PyDMD: Python dynamic mode decomposition." Journal of Open Source Software 3.22 (2018): 530.
