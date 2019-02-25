#include <petscdmplex.h>
#include <petscmat.h>

#include "quadrature.h"
#include "nedelec.h"
#include "matrix.h"
#include "mesh.h"

#undef __FUNCT__
#define __FUNCT__ "assemble_stiffness"
//TODO: refactor and split assembly
PetscErrorCode assemble_stiffness(DM dm, struct quadrature q, struct function_space fs, Mat *stiffness) {
    PetscInt dim, cstart, cend, estart, eend, nedges;
    struct mesh_ctx *mctx;
    const PetscInt *edgelist;

    PetscErrorCode ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, &cstart, &cend); CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, cstart, &nedges); CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 1, &estart, &eend); CHKERRQ(ierr);
    ierr = DMGetApplicationContext(dm, (void **) &mctx); CHKERRQ(ierr);

    ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, eend - estart, eend - estart, 0, NULL, stiffness);
    CHKERRQ(ierr);
    ierr = MatSetOption(*stiffness, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE); CHKERRQ(ierr);

    if (dim == 2) {
        PetscInt row_indices[nedges], col_indices[nedges];
        for (PetscInt c = cstart; c < cend; c++) {
            PetscReal v0, J[9], invJ[9], detJ;
            ierr = DMPlexComputeCellGeometryAffineFEM(dm, c, &v0, (PetscReal *) &J, (PetscReal *) &invJ, &detJ); CHKERRQ(ierr);
            ierr = DMPlexGetCone(dm, c, &edgelist); CHKERRQ(ierr);
            PetscReal local[nedges][nedges];

            for (PetscInt k = 0; k < nedges; k++) {
                row_indices[k] = edgelist[k] - estart;
                for (PetscInt l = 0; l < nedges; l++) {
                    col_indices[l] = edgelist[l] - estart;
                    local[k][l] = 1/PetscAbsReal(detJ);
                    for (PetscInt i = 0; i < q.size; i++) {
                        //define some preprocessor magic for ugly addressing
                        local[k][l] += q.pw[i * 3 + 2] *
                            mctx->signs[(c - cstart) * 3 + k] * fs.cval[i * 3 + k] *
                            mctx->signs[(c - cstart) * 3 + l] * fs.cval[i * 3 + l];
                    }
                }
            }
            ierr = MatSetValues(*stiffness, 3, row_indices, 3, col_indices,
                    *local, ADD_VALUES); CHKERRQ(ierr);
        }
    } else {
        SETERRQ(PETSC_COMM_SELF, 56, "3D is currently not supported\n");
    }

    MatAssemblyBegin(*stiffness, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*stiffness, MAT_FINAL_ASSEMBLY);

    return (0);
}

#undef __FUNCT__
#define __FUNCT__ "assemble_mass"
PetscErrorCode assemble_mass(DM dm, struct quadrature q, struct function_space fs, Mat *mass) {
    PetscInt dim, cstart, cend, estart, eend, nedges;
    struct mesh_ctx *mctx;
    const PetscInt *edgelist;

    PetscErrorCode ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, &cstart, &cend); CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, cstart, &nedges); CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 1, &estart, &eend); CHKERRQ(ierr);
    ierr = DMGetApplicationContext(dm, (void **) &mctx); CHKERRQ(ierr);

    ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, eend - estart, eend - estart, 0, NULL, mass);
    CHKERRQ(ierr);
    ierr = MatSetOption(*mass, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE); CHKERRQ(ierr);

    if (dim == 2) {
        PetscInt row_indices[nedges], col_indices[nedges];
        for (PetscInt c = cstart; c < cend; c++) {
            PetscReal v0, J[9], invJ[9], detJ;
            ierr = DMPlexComputeCellGeometryAffineFEM(dm, c, &v0, (PetscReal *) &J, (PetscReal *) &invJ, &detJ); CHKERRQ(ierr);
            ierr = DMPlexGetCone(dm, c, &edgelist); CHKERRQ(ierr);
            PetscReal local[nedges][nedges];

            for (PetscInt k = 0; k < nedges; k++) {
                row_indices[k] = edgelist[k] - estart;
                for (PetscInt l = 0; l < nedges; l++) {
                    col_indices[l] = edgelist[l] - estart;
                    local[k][l] = PetscAbsReal(detJ);
                    for (PetscInt i = 0; i < q.size; i++) {
                        //define some preprocessor magic for ugly addressing
                        local[k][l] += q.pw[i * 3 + 2] *
                            mctx->signs[(c - cstart) * 3 + k] * mctx->signs[(c - cstart) * 3 + l] *
                            ((invJ[0 * 3 + 0] * fs.val[l * (q.size * 2) + i * 2 + 0] +
                              invJ[0 * 3 + 1] * fs.val[l * (q.size * 2) + i * 2 + 1]) *
                             (invJ[0 * 3 + 0] * fs.val[k * (q.size * 2) + i * 2 + 0] +
                              invJ[0 * 3 + 1] * fs.val[k * (q.size * 2) + i * 2 + 1]) +
                             (invJ[1 * 3 + 0] * fs.val[l * (q.size * 2) + i * 2 + 0] +
                              invJ[1 * 3 + 1] * fs.val[l * (q.size * 2) + i * 2 + 1]) *
                             (invJ[1 * 3 + 0] * fs.val[k * (q.size * 2) + i * 2 + 0] +
                              invJ[1 * 3 + 1] * fs.val[k * (q.size * 2) + i * 2 + 1]));
                    }
                }
            }
            ierr = MatSetValues(*mass, 3, row_indices, 3, col_indices,
                    *local, ADD_VALUES); CHKERRQ(ierr);
        }
    } else {
        SETERRQ(PETSC_COMM_SELF, 56, "3D is currently not supported\n");
    }

    MatAssemblyBegin(*mass, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*mass, MAT_FINAL_ASSEMBLY);

    return ierr;
}
