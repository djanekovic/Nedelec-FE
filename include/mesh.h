#ifndef MESH_H
#define MESH_H

#include <petscsys.h>

#include "util.h"

PETSC_EXTERN PetscErrorCode generate_mesh(struct ctx *sctx, PetscInt **nnz, DM *dm);

#endif /* MESH_H */
