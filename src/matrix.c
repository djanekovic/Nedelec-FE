#include <petscdmplex.h>
#include <petscmat.h>

#include "util.h"

PetscErrorCode assemble_stiffness(DM dm, struct ctx sctx, Mat signs, Mat *stiffness) {
    PetscInt dim = sctx.dim; PetscInt nelems = sctx.dim;

    return (0);
}
PetscErrorCode assemble_mass(DM dm, struct ctx sctx, Mat signs, Mat mass) {
    return (0);
}
