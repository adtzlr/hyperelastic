import felupe as fem
import numpy as np
import sympy as sym

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


class SymbolicMaterialFormulation:
    def __init__(self, strain_energy_function, **kwargs):
        self.x, self.y, self.z = sym.symbols("λ_1 λ_2 λ_3")
        self.W = strain_energy_function([self.x, self.y, self.z], **kwargs)

        self.dWdx = sym.lambdify([self.x, self.y, self.z], self.W.diff(self.x))
        self.dWdy = sym.lambdify([self.x, self.y, self.z], self.W.diff(self.y))
        self.dWdz = sym.lambdify([self.x, self.y, self.z], self.W.diff(self.z))

        self.d2Wdxdx = sym.lambdify(
            [self.x, self.y, self.z], self.W.diff(self.x, self.x)
        )
        self.d2Wdydy = sym.lambdify(
            [self.x, self.y, self.z], self.W.diff(self.y, self.y)
        )
        self.d2Wdzdz = sym.lambdify(
            [self.x, self.y, self.z], self.W.diff(self.z, self.z)
        )

        self.d2Wdxdy = lambda x, y, z: np.zeros_like(x)
        self.d2Wdydz = lambda x, y, z: np.zeros_like(x)
        self.d2Wdxdz = lambda x, y, z: np.zeros_like(x)

    def gradient(self, stretches, statevars):
        return [
            self.dWdx(*stretches),
            self.dWdy(*stretches),
            self.dWdz(*stretches),
        ], statevars

    def hessian(self, stretches, statevars):
        return np.array(
            [
                self.d2Wdxdx(*stretches),
                self.d2Wdydy(*stretches),
                self.d2Wdzdz(*stretches),
                self.d2Wdxdy(*stretches),
                self.d2Wdydz(*stretches),
                self.d2Wdxdz(*stretches),
            ]
        )


def test_distortional_stretches_sympy():
    model = SymbolicMaterialFormulation(ogden, mu=1, alpha=0.436)
    umat = hel.spaces.DistortionalSpace(hel.frameworks.StretchesFramework(model))
    fea(umat).evaluate(verbose=2)


if __name__ == "__main__":
    test_distortional_stretches_sympy()
