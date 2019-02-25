#include <petsc.h>

#include "util.h"


#undef __FUNCT__
#define __FUNCT__ "handle_cli_options"
PetscErrorCode handle_cli_options(struct ctx *sctx) {
    /* defaults */
    sctx->dim = 2;
    sctx->ref = 0;
    sctx->quad_order = 3;

    /* read from cli */
    PetscOptionsGetInt(NULL, NULL, "-dim", &sctx->dim, NULL);
    PetscOptionsGetInt(NULL, NULL, "-ref", &sctx->ref, NULL);

    return (0);
}
