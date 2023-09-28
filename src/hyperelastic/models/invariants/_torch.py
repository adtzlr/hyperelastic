import torch


class TorchModel:
    r"""Isotropic hyperelastic material formulation based on a given strain energy
    density function ``fun(I1, I2, I3, **kwargs)`` per unit undeformed volume. The
    gradients are carried out by automatic differentiation using PyTorch.

    ..  math::

        \psi = \psi(I_1, I_2, I_3)

    ..  note::
        PyTorch uses single-precision by default. This must be considered in numeric
        simulations, i.e. the error tolerance must not exceed ``tol=1e-8``. For double-
        precision, set ``torch.float64`` as default.

        ..  code-block:: python

            import torch

            torch.set_default_dtype(torch.float64)

    Examples
    --------

    >>> import hyperelastic
    >>>
    >>> def yeoh(I1, I2, I3, C10, C20, C30):
    >>>     "Yeoh isotropic hyperelastic material formulation."
    >>>     return C10 * (I1 - 3) + C20 * (I1 - 3) ** 2 + C30 * (I1 - 3) ** 3
    >>>
    >>> model = hyperelastic.models.invariants.TorchModel(
    >>>     yeoh, C10=0.5, C20=-0.05, C30=0.02
    >>> )
    >>> framework = hyperelastic.InvariantsFramework(model)
    >>> umat = hyperelastic.DistortionalSpace(framework)
    """

    def __init__(self, fun, **kwargs):
        self.fun = fun
        self.kwargs = kwargs

    def _grad(self, fun, x, numpy=False, allow_unused=True, **kwargs):
        "Return the gradient of a sum of a tensor ``fun`` w.r.t. ``x``."

        out = None
        if hasattr(fun, "grad_fn"):
            out = torch.autograd.grad(
                fun.sum(), x, allow_unused=allow_unused, **kwargs
            )[0]
            if numpy:
                if out is not None:
                    return out.detach().numpy()
        return out

    def _astensors(self, *args):
        "Convert a list/tuple of arrays to torch-tensors."

        tensors = [torch.Tensor(arg) for arg in args]
        [tensor.requires_grad_(True) for tensor in tensors]
        return tensors

    def gradient(
        self,
        I1,
        I2,
        I3,
        statevars,
        tensor=False,
        numpy=True,
        create_graph=False,
        retain_graph=False,
    ):
        """The gradient as the partial derivative of the strain energy function w.r.t.
        the invariants."""
        if not tensor:
            I1, I2, I3 = self._astensors(I1, I2, I3)

        f = self.fun(I1, I2, I3, **self.kwargs)
        kwargs = dict(numpy=numpy, create_graph=create_graph, retain_graph=retain_graph)

        return *[self._grad(f, x, **kwargs) for x in [I1, I2, I3]], statevars

    def hessian(self, I1, I2, I3, statevars, numpy=True):
        """The hessian as the second partial derivatives of the strain energy function
        w.r.t. the invariants."""
        I1, I2, I3 = self._astensors(I1, I2, I3)
        dWdI1, dWdI2, dWdI3, statevars = self.gradient(
            I1, I2, I3, statevars, tensor=True, create_graph=True, numpy=False
        )

        return [
            self._grad(f, x, numpy=numpy)
            for f, x in zip(
                [dWdI1, dWdI2, dWdI3, dWdI1, dWdI2, dWdI1], [I1, I2, I3, I2, I3, I3]
            )
        ]
