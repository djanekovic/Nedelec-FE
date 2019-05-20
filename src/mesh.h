#ifndef MESH_H
#define MESH_H

#include "util.h"

PetscErrorCode generate_mesh(struct ctx *sctx, PetscInt **nnz, DM *dm);

#endif /* MESH_H */
