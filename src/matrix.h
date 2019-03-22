#ifndef MATRIX_H
#define MATRIX_H

#include "quadrature.h"
#include "nedelec.h"

PetscErrorCode assemble_stiffness(DM dm, struct quadrature q, struct function_space fs, Mat *stiffness);
PetscErrorCode assemble_mass(DM dm, struct quadrature q, struct function_space fs, Mat *mass);
PetscErrorCode assemble_load(DM dm, struct quadrature q, struct function_space fs, Vec *load);

#endif /* MATRIX_H */
