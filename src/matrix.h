#ifndef MATRIX_H
#define MATRIX_H

#include "quadrature.h"
#include "nedelec.h"

PetscErrorCode assemble_system(DM, struct quadrature, struct function_space,
                               Mat, Vec);

#endif /* MATRIX_H */
