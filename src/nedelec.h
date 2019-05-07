#ifndef NEDELEC_H
#define NEDELEC_H

#include "quadrature.h"

struct function_space {
	PetscScalar *val;
	PetscScalar *cval;
	struct quadrature q;
	PetscInt nbasis;
};

PetscErrorCode nedelec_basis(struct function_space *, int);

#endif /* NEDELEC_H */
