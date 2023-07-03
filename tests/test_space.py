import felupe as fem

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


def test_distortional_stretches():
    model = hel.models.stretches.Ogden(mu=[1], alpha=[0.436])
    umat = hel.spaces.DistortionalSpace(hel.frameworks.StretchesFramework(model))
    fea(umat).evaluate(verbose=2)


def test_distortional_invariants():
    model = hel.models.invariants.ThirdOrderDeformation(
        C10=0.4, C01=0.1, C11=0.02, C20=-0.05, C30=0.01
    )
    umat = hel.spaces.DistortionalSpace(hel.frameworks.InvariantsFramework(model))
    fea(umat).evaluate(verbose=2)


def test_dilatational_invariants():
    model = hel.models.invariants.ThirdOrderDeformation(C10=0.5)
    umat = hel.spaces.DilatationalSpace(hel.frameworks.InvariantsFramework(model))
    fea(umat).evaluate(verbose=2)


def test_deformation_invariants():
    model = hel.models.invariants.ThirdOrderDeformation(
        C10=0.4, C01=0.1, C11=0.02, C20=-0.05, C30=0.01
    )
    umat = hel.spaces.DeformationSpace(hel.frameworks.InvariantsFramework(model))
    fea(umat).evaluate(verbose=2)


if __name__ == "__main__":
    test_distortional_stretches()
    test_distortional_invariants()
    test_dilatational_invariants()
    test_deformation_invariants()
