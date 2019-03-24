#ifndef UTIL_H
#define UTIL_H

/* Solver context definition */
struct ctx {
    PetscInt dim;           /* problem dimension        */
    PetscInt ref;           /* refinement flag          */
    PetscInt nelems;        /* number of elements       */
    PetscInt quad_order;    /* quadrature order         */

    Mat G;                  /* discrete gradient matrix */
    // možda overkill...
    int *signs;

    PetscInt cstart;
    PetscInt cend;
    PetscInt estart;
    PetscInt eend;
    PetscInt vstart;
    PetscInt vend;

    //TODO: complex?
    PetscScalar (*stiffness_function_2D)(PetscReal x, PetscReal y);
    PetscScalar (*stiffness_function_3D)(PetscReal x, PetscReal y, PetscReal z);
    PetscScalar (*mass_function_2D)(PetscReal x, PetscReal y);
    PetscScalar (*mass_function_3D)(PetscReal x, PetscReal y, PetscReal z);
    PetscScalar (*load_function_2D)(PetscReal x, PetscReal y);
    PetscScalar (*load_function_3D)(PetscReal x, PetscReal y, PetscReal z);
};

PetscErrorCode handle_cli_options(struct ctx *sctx);

#endif /* UTIL_H */
