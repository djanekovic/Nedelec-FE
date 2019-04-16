#include <petsc.h>
#include <string.h>

#include "function.h"
#include "util.h"

#undef __FUNCT__
#define __FUNCT__ "handle_cli_options"
PetscErrorCode handle_cli_options(struct ctx *sctx)
{
    memset(sctx, 0, sizeof(*sctx));

    /* defaults */
    sctx->dim = 2;
    sctx->ref = 0;
    sctx->quad_order = 3;

    sctx->stiffness_function_2D = constant_2D;
    sctx->mass_function_2D = constant_2D;
    sctx->load_function_2D = constant_2D;
    /* read from cli */
    PetscOptionsGetInt(NULL, NULL, "-dim", &sctx->dim, NULL);
    PetscOptionsGetInt(NULL, NULL, "-ref", &sctx->ref, NULL);

    return (0);
}
