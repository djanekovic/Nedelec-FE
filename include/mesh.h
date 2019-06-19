#ifndef MESH_H
#define MESH_H

#include <petscsys.h>

#include "util.h"

PetscErrorCode __attribute__((visibility ("default")))
generate_mesh(struct ctx *sctx, PetscInt **nnz, DM *dm);

#endif /* MESH_H */
