#ifndef MESH_H
#define MESH_H

#include "util.h"

struct mesh_ctx {
    PetscInt *signs;
};

PetscErrorCode generate_mesh(struct ctx *sctx, DM *dm);

#endif /* MESH_H */
