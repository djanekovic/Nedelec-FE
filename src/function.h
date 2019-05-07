#ifndef FUNCTION_H
#define FUNCTION_H

PetscScalar constant_2D(PetscReal, PetscReal);
PetscScalar constant_3D(PetscReal, PetscReal, PetscReal);

struct function {
	PetscScalar (*value_at_2D)(PetscReal, PetscReal);
	PetscScalar (*value_at_3D)(PetscReal, PetscReal, PetscReal);
};

#endif /* FUNCTION_H */
