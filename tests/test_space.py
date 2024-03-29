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
    umat = hel.DistortionalSpace(hel.StretchesFramework(model))
    fea(umat).evaluate(verbose=2)


def test_distortional_invariants():
    model = hel.models.invariants.ThirdOrderDeformation(
        C10=0.4, C01=0.1, C11=0.02, C20=-0.05, C30=0.01
    )
    umat = hel.DistortionalSpace(hel.InvariantsFramework(model))
    fea(umat).evaluate(verbose=2)


def test_distortional_generalized_strain_invariants():
    tod = hel.models.invariants.ThirdOrderDeformation(
        C10=0.4, C01=0.1, C20=0.1, strain=True
    )
    fun = hel.models.generalized.strain
    framework = hel.GeneralizedInvariantsFramework(tod, fun=fun, exponent=1.4)
    umat = hel.DistortionalSpace(framework)
    fea(umat).evaluate(verbose=2)


def test_distortional_generalized_deformation_invariants():
    tod = hel.models.invariants.ThirdOrderDeformation(
        C10=0.4, C01=0.1, C20=0.1, strain=False
    )
    fun = hel.models.generalized.deformation
    framework = hel.GeneralizedInvariantsFramework(tod, fun=fun, exponent=1.4)
    umat = hel.DistortionalSpace(framework)
    fea(umat).evaluate(verbose=2)


def test_dilatational_invariants():
    model = hel.models.invariants.ThirdOrderDeformation(C10=0.5)
    umat = hel.DilatationalSpace(hel.InvariantsFramework(model))
    fea(umat).evaluate(verbose=2)


def test_deformation_invariants():
    model = hel.models.invariants.ThirdOrderDeformation(
        C10=0.4, C01=0.1, C11=0.02, C20=-0.05, C30=0.01
    )
    umat = hel.DeformationSpace(hel.InvariantsFramework(model))
    fea(umat).evaluate(verbose=2)


def test_deformation_invariants_broadcast():
    model = hel.models.invariants.ThirdOrderDeformation(
        C10=0.5
    )
    umat = hel.DeformationSpace(hel.InvariantsFramework(model))
    fea(umat).evaluate(verbose=2)


if __name__ == "__main__":
    test_distortional_stretches()
    test_distortional_invariants()
    test_dilatational_invariants()
    test_deformation_invariants()
    test_deformation_invariants_broadcast()
    test_distortional_generalized_strain_invariants()
    test_distortional_generalized_deformation_invariants()
