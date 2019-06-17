#ifndef MATRIX_H
#define MATRIX_H

#include "nedelec.h"
#include "quadrature.h"

PetscErrorCode assemble_system(DM, fs_t, Mat, Vec);

#endif /* MATRIX_H */
