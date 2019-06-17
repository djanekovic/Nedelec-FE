#ifndef NEDELEC_H
#define NEDELEC_H

#include "quadrature.h"

typedef struct {
    PetscScalar *val;
    PetscScalar *cval;
    quadrature_t triangle;
    quadrature_t line;
    PetscInt nbasis;
} fs_t;

PetscErrorCode create_nedelec(fs_t *fs, int);
PetscErrorCode destroy_nedelec(fs_t *fs);

#endif /* NEDELEC_H */
