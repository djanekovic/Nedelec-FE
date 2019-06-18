#include <assert.h>
#include <petsc.h>

#include "matrix.h"
#include "mesh.h"
#include "nedelec.h"

#define F_x 1.0
#define F_y 1.0


/**
 * Compute invBk * invBk^T when matrix is 2x2
 */
static inline void _invBk_invBkT_2D(PetscReal *invBk, PetscReal *res)
{
    res[0] = invBk[0] * invBk[0] + invBk[1] * invBk[1]; /* a^2 + b^2 */
    res[1] = invBk[0] * invBk[2] + invBk[1] * invBk[3]; /* ac + bd */
    res[2] = res[1];
    res[3] = invBk[2] * invBk[2] + invBk[3] * invBk[3]; /* c^2 + d^2 */
}


#undef __FUNCT__
#define __FUNCT__ "assemble_system_neumann"
PetscErrorCode assemble_system_neumann(DM dm, fs_t fs, Mat A, Vec b)
{
    PetscInt dim, cstart, cend, estart, eend, nedges;
    struct ctx *sctx;

    PetscErrorCode ierr = DMGetDimension(dm, &dim);
    CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, &cstart, &cend);
    CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, cstart, &nedges);
    CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 1, &estart, &eend);
    CHKERRQ(ierr);
    ierr = DMGetApplicationContext(dm, (void **) &sctx);
    CHKERRQ(ierr);

    PetscLogEventBegin(sctx->matrix_assembly, 0, 0, 0, 0);

    // TODO: napravi nedelec objekt koji je ispod function_space i koji ima
    // assemble_2D i assemble_3D function pointere
    if (dim == 2) {
        PetscReal local[nedges][nedges], load[nedges];
        PetscInt indices[nedges];
        const PetscInt *edgelist;
        for (PetscInt c = cstart; c < cend; c++) {
            PetscInt offset = (c - cstart) * 3;
            PetscReal v0, Bk[4], invBk[4], detBk, _tmp_matrix[4];
            ierr = DMPlexComputeCellGeometryAffineFEM(
                dm, c, &v0, (PetscReal *) &Bk, (PetscReal *) &invBk, &detBk);
            CHKERRQ(ierr);
            ierr = DMPlexGetCone(dm, c, &edgelist);
            CHKERRQ(ierr);

            _invBk_invBkT_2D(invBk, _tmp_matrix);

            for (PetscInt k = 0; k < nedges; k++) {
                PetscInt edge_neighbours, boundary_edge = -1;
                DMPlexGetSupportSize(dm, edgelist[k], &edge_neighbours);
                if (edge_neighbours == 1) {
                    boundary_edge = k;
                }
                PetscInt sign_k = sctx->signs[offset + k];
                indices[k] = edgelist[k] - estart;
                for (PetscInt l = 0; l < nedges; l++) {
                    PetscInt sign_l = sctx->signs[offset + l];

                    local[k][l] = stiffness_matrix_2D(fs, detBk, sign_k,
                                                      sign_l, k, l);
                    local[k][l] += mass_matrix_2D(_tmp_matrix, fs, detBk,
                                                  sign_k, sign_l, k, l);
                }

                load[k] = load_vector_2D(invBk, fs, detBk, k, sign_k);
            }
            ierr = MatSetValues(A, nedges, indices, nedges, indices,
                                *local, ADD_VALUES);
            CHKERRQ(ierr);
            ierr = VecSetValues(b, nedges, indices, load, ADD_VALUES);
            CHKERRQ(ierr);
        }
    } else {
        SETERRQ(PETSC_COMM_SELF, 56, "3D is currently not supported\n");
    }

    // TODO: pogledaj u dokumentaciji
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    VecAssemblyBegin(b);
    VecAssemblyEnd(b);

    PetscLogEventEnd(sctx->matrix_assembly, 0, 0, 0, 0);

    return (0);
}

#undef __FUNCT__
#define __FUNCT__ "assemble_system_dirichlet"
PetscErrorCode assemble_system_dirichlet(DM dm, fs_t fs,
                                         Mat A, Vec b)
{
    PetscInt dim, cstart, cend, estart, eend, nedges;
    struct ctx *sctx;

    PetscErrorCode ierr = DMGetDimension(dm, &dim);
    CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, &cstart, &cend);
    CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, cstart, &nedges);
    CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 1, &estart, &eend);
    CHKERRQ(ierr);
    ierr = DMGetApplicationContext(dm, (void **) &sctx);
    CHKERRQ(ierr);

    PetscLogEventBegin(sctx->matrix_assembly, 0, 0, 0, 0);

    // TODO: napravi nedelec objekt koji je ispod function_space i koji ima
    // assemble_2D i assemble_3D function pointere
    if (dim == 2) {
        PetscReal local[nedges][nedges], load[nedges];
        PetscInt indices[nedges];
        const PetscInt *edgelist;
        for (PetscInt c = cstart; c < cend; c++) {
            PetscInt offset = (c - cstart) * 3;
            PetscReal v0, Bk[4], invBk[4], detBk, _tmp_matrix[4];
            ierr = DMPlexComputeCellGeometryAffineFEM(
                dm, c, &v0, (PetscReal *) &Bk, (PetscReal *) &invBk, &detBk);
            CHKERRQ(ierr);
            ierr = DMPlexGetCone(dm, c, &edgelist);
            CHKERRQ(ierr);

            _invBk_invBkT_2D(invBk, _tmp_matrix);

            for (PetscInt k = 0; k < nedges; k++) {
                PetscInt edge_neighbours, boundary_edge = -1;
                DMPlexGetSupportSize(dm, edgelist[k], &edge_neighbours);
                if (edge_neighbours == 1) {
                    boundary_edge = k;
                }
                PetscInt sign_k = sctx->signs[offset + k];
                indices[k] = edgelist[k] - estart;
                for (PetscInt l = 0; l < nedges; l++) {
                    PetscInt sign_l = sctx->signs[offset + l];

                    if (k == boundary_edge && l == boundary_edge) {
                        local[k][l] = 1.0;
                        load[k] = 0.0;
                    } else if (k == boundary_edge || l == boundary_edge) {
                        local[k][l] = 0.0;
                    } else {
                        local[k][l] = stiffness_matrix_2D(fs, detBk, sign_k,
                                                          sign_l, k, l);
                        local[k][l] += mass_matrix_2D(_tmp_matrix, fs, detBk,
                                                      sign_k, sign_l, k, l);
                    }
                }
                if (k != boundary_edge) {
                    load[k] = load_vector_2D(invBk, fs, detBk, k, sign_k);
                }
            }
            ierr = MatSetValues(A, nedges, indices, nedges, indices,
                                *local, ADD_VALUES);
            CHKERRQ(ierr);
            ierr = VecSetValues(b, nedges, indices, load, ADD_VALUES);
            CHKERRQ(ierr);
        }
    } else {
        SETERRQ(PETSC_COMM_SELF, 56, "3D is currently not supported\n");
    }

    // TODO: pogledaj u dokumentaciji
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    VecAssemblyBegin(b);
    VecAssemblyEnd(b);

    PetscLogEventEnd(sctx->matrix_assembly, 0, 0, 0, 0);

    return (0);
}
