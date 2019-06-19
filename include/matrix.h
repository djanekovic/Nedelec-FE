#ifndef MATRIX_H
#define MATRIX_H

#include <petscsys.h>

#include "nedelec.h"
#include "quadrature.h"

PetscErrorCode PETSC_EXTERN assemble_system_dirichlet(DM, fs_t, Mat, Vec);

#endif /* MATRIX_H */
