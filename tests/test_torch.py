import felupe as fem
import torch

import hyperelastic as hel


def fea(umat):
    mesh = fem.Cube(n=3)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    boundaries, loadcase = fem.dof.uniaxial(field, clamped=True)
    solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=5000)
    move = fem.math.linsteps([0, 2], num=3)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )
    job = fem.Job(steps=[step])

    return job


def ogden(stretches, mu, alpha):
    x, y, z = stretches
    return 2 * mu / alpha**2 * (x**alpha + y**alpha + z**alpha - 3)


class TorchMaterialFormulation:
    def __init__(self, strain_energy_function, **kwargs):
        self.W = strain_energy_function

    def gradient(self, stretches, statevars):
        x = torch.Tensor(stretches, requires_grad=True)
        W = self.W(*x, **self.kwargs)
        W.backward()
        return W.grad, statevars

    def hessian(self, stretches, statevars):
        x = torch.Tensor(stretches, requires_grad=True)
        W = self.W(*x, **self.kwargs)
        W.backward(retain_graph=True)
        W.backward()
        return [
            self.d2Wdxdx(*stretches),
            self.d2Wdydy(*stretches),
            self.d2Wdzdz(*stretches),
            self.d2Wdxdy(*stretches),
            self.d2Wdydz(*stretches),
            self.d2Wdxdz(*stretches),
        ]


def test_distortional_stretches_sympy():
    model = TorchMaterialFormulation(ogden, mu=1, alpha=0.436)
    umat = hel.DistortionalSpace(hel.StretchesFramework(model))
    fea(umat).evaluate(verbose=2)


if __name__ == "__main__":
    test_distortional_stretches_sympy()
