from dolfin import *
import numpy as np

mesh = RectangleMesh(Point(0, 0), Point(1, 1), 2, 2, "left")
V = FunctionSpace(mesh, "N1curl", 1)
bc = [DirichletBC(V, Constant((0.0, 0.0)), DomainBoundary())]

u = TrialFunction(V)
v = TestFunction(V)
a = inner(curl(u), curl(v))*dx
L = inner(Constant((1.0, 1.0)), v)*dx

u = Function(V).vector()
A, b = assemble_system(a, L)
solve(A, u, b)

print np.linalg.det(A.array())

print as_backend_type(A).mat().view()
print as_backend_type(b).vec().view()
print as_backend_type(u).vec().view()
