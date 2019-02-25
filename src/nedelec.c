#include <petscmat.h>

#include "quadrature.h"
#include "nedelec.h"

/**
 * M represents number of quadrature points
 */
PetscErrorCode nedelec_basis(struct quadrature q, struct function_space *fspace) {
    PetscInt M = q.size;

    /* three different M x 2 matrix */
    PetscErrorCode ierr = PetscMalloc1(3 * M * 2, &fspace->val); CHKERRQ(ierr);
    ierr = PetscMalloc1(3 * M * 1, &fspace->cval); CHKERRQ(ierr);

    //TODO: clever define for this ugly addressing
    for (PetscInt i = 0; i < M; i++) {
        fspace->val[0 * (M * 2) + i * 2 + 0] = -q.pw[i * 3 + 1];
        fspace->val[0 * (M * 2) + i * 2 + 1] = q.pw[i * 3 + 0];

        fspace->val[1 * (M * 2) + i * 2 + 0] = -q.pw[i * 3 + 1];
        fspace->val[1 * (M * 2) + i * 2 + 1] = q.pw[i * 3 + 0] - 1;

        fspace->val[2 * (M * 2) + i * 2 + 0] = 1 - q.pw[i * 3 + 1];
        fspace->val[2 * (M * 2) + i * 2 + 1] = q.pw[i * 3 + 0];

        fspace->cval[i * 3 + 0] = 2;
        fspace->cval[i * 3 + 1] = 2;
        fspace->cval[i * 3 + 2] = 2;
    }

    fspace->nbasis = 3;

    return ierr;
}
