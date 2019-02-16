#ifndef UTIL_H
#define UTIL_H

struct ctx {
    PetscInt dim; /* problem dimension */
    PetscInt ref; /* refinement flag */
    PetscInt nelems; /* number of elements */
    PetscInt quad_order; /* quadrature order */
};

PetscErrorCode handle_cli_options(struct ctx *sctx);
PetscErrorCode quad_pw(PetscInt order, Mat *pw);

#endif /* UTIL_H */
