#include <petsc.h>

#include "matrix.h"
#include "mesh.h"

#define F_x 1.0
#define F_y 1.0

//TODO: inline
static PetscReal stiffness_matrix_2D(struct quadrature q,
                                     struct function_space fs,
                                     struct ctx *sctx, PetscReal detJ,
                                     PetscInt sign_k, PetscInt sign_l,
                                     PetscInt k, PetscInt l)
{
    PetscReal local = 1/PetscAbsReal(detJ);

    for (PetscInt i = 0; i < q.size; i++) {
        local *= q.pw[i * 3 + 2] * fs.cval[i * 3 + k] * fs.cval[i * 3 + l] *
                 sctx->stiffness_function_2D(q.pw[i * 3 + 0], q.pw[i * 3 + 1]) *
                 sign_k * sign_l;
    }

    return local;
}

//TODO: inline ili prouci asm jer je ovo bottleneck?
static PetscReal mass_matrix_2D(struct quadrature q, PetscReal *invJ,
                                struct function_space fs,
                                struct ctx *sctx, PetscReal detJ,
                                PetscInt sign_k, PetscInt sign_l,
                                PetscInt k_ned, PetscInt l_ned)
{
    PetscReal local = PetscAbsReal(detJ);
    PetscInt dim = sctx->dim;

    //TODO: cleanup or maybe call blas, tradeoff?
    for (PetscInt i = 0; i < q.size; i++) {
        for (PetscInt j = 0; j < dim; j++) {
            double x = 0, y = 0;
            for (PetscInt k = 0; j < dim; j++) {
                x += invJ[j * 3 + k] * fs.val[k_ned * (q.size * 2) + i * 2 + j];
                y += invJ[j * 3 + k] * fs.val[l_ned * (q.size * 2) + i * 2 + j];
            }
            local *= x * y;
        }
    }

    return local;
}

static PetscReal load_vector_2D(struct quadrature q, PetscReal *invJ,
                                struct function_space fs,
                                struct ctx *sctx, PetscReal detJ,
                                PetscInt k, PetscInt sign_k)
{
    //TODO: Dario, provjeri matematiku ovdje
    PetscReal local = PetscAbsReal(detJ);
    PetscInt dim = sctx->dim;

    //TODO: provjeri koliko je ovo sporije
    for (PetscInt i = 0; i < q.size; i++) {
        for (PetscInt j = 0; j < dim; j++) {
            for (PetscInt k = 0; k < dim; k++) {
                local += invJ[j * 3 + k] * fs.val[j * (q.size * 2) + i * 2 + k];
            }
        }

        local *= q.pw[i * 3 + 2] * sign_k *
                 sctx->load_function_2D(q.pw[i * 3 + 0], q.pw[i * 3 + 1]);
    }

    return local;
}

#undef __FUNCT__
#define __FUNCT__ "assemble_stiffness"
PetscErrorCode assemble_system(DM dm, struct quadrature q, struct function_space fs,
                               Mat A, Vec b)
{
    PetscInt dim, cstart, cend, estart, eend, nedges;
    struct ctx *sctx;

    PetscErrorCode ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, &cstart, &cend); CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, cstart, &nedges); CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 1, &estart, &eend); CHKERRQ(ierr);
    ierr = DMGetApplicationContext(dm, (void **) &sctx); CHKERRQ(ierr);

    //TODO: napravi nedelec objekt koji je ispod function_space i koji ima
    //assemble_2D i assemble_3D function pointere
    if (dim == 2) {
        PetscReal local[nedges][nedges], load[nedges];
        PetscInt row_indices[nedges], col_indices[nedges];
        const PetscInt *edgelist;
        for (PetscInt c = cstart; c < cend; c++) {
            PetscReal v0, J[9], invJ[9], detJ;

            ierr = DMPlexComputeCellGeometryAffineFEM(dm, c, &v0, (PetscReal *) &J, (PetscReal *) &invJ, &detJ); CHKERRQ(ierr);
            ierr = DMPlexGetCone(dm, c, &edgelist); CHKERRQ(ierr);

            for (PetscInt k = 0; k < nedges; k++) {
                PetscInt sign_k = sctx->signs[(c - cstart) * 3 + k];
                row_indices[k] = edgelist[k] - estart;
                for (PetscInt l = 0; l < nedges; l++) {
                    col_indices[l] = edgelist[l] - estart;
                    PetscInt sign_l = sctx->signs[(c - cstart) * 3 + l];

                    local[k][l] = stiffness_matrix_2D(q, fs, sctx, detJ,
                                                      sign_k, sign_l, k, l);
                    local[k][l] += mass_matrix_2D(q, invJ, fs, sctx, detJ,
                                                  sign_k, sign_l, k, l);
                }
                load[k] = load_vector_2D(q, invJ, fs, sctx, detJ, k, sign_k);
            }
            ierr = MatSetValues(A, 3, row_indices, 3, col_indices, *local, ADD_VALUES); CHKERRQ(ierr);
            ierr = VecSetValues(b, 3, row_indices, load, ADD_VALUES); CHKERRQ(ierr);
        }
    } else {
        SETERRQ(PETSC_COMM_SELF, 56, "3D is currently not supported\n");
    }

    //TODO: pogledaj u dokumentaciji
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    VecAssemblyBegin(b);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    VecAssemblyEnd(b);

    return (0);
}

/*
#undef __FUNCT__
#define __FUNCT__ "assemble_mass"
PetscErrorCode assemble_mass(DM dm, struct quadrature q, struct function_space fs, Mat *mass) {
    PetscInt dim, cstart, cend, estart, eend, nedges;
    struct ctx *mctx;
    const PetscInt *edgelist;

    PetscErrorCode ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, &cstart, &cend); CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, cstart, &nedges); CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 1, &estart, &eend); CHKERRQ(ierr);
    ierr = DMGetApplicationContext(dm, (void **) &mctx); CHKERRQ(ierr);

    ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD, eend - estart, eend - estart, 0, NULL, mass);
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

#undef __FUNCT__
#define __FUNCT__ "assemble_load"
//TODO: attach function eval to mesh_ctx
PetscErrorCode assemble_load(DM dm, struct quadrature q, struct function_space fs, Vec *load) {
    PetscInt dim, cstart, cend, estart, eend, nedges;
    struct ctx *mctx;
    const PetscInt *edgelist;

    PetscErrorCode ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, &cstart, &cend); CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, cstart, &nedges); CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 1, &estart, &eend); CHKERRQ(ierr);
    ierr = DMGetApplicationContext(dm, (void **) &mctx); CHKERRQ(ierr);

    ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, eend - estart, load);
    CHKERRQ(ierr);

    if (dim == 2) {
        PetscInt row_indices[nedges];
        for (PetscInt c = cstart; c < cend; c++) {
            PetscReal v0, J[9], invJ[9], detJ;
            ierr = DMPlexComputeCellGeometryAffineFEM(dm, c, &v0, (PetscReal *) &J, (PetscReal *) &invJ, &detJ); CHKERRQ(ierr);
            ierr = DMPlexGetCone(dm, c, &edgelist); CHKERRQ(ierr);
            PetscReal local[nedges];

            for (PetscInt k = 0; k < nedges; k++) {
                row_indices[k] = edgelist[k] - estart;
                }
            }
            ierr = VecSetValues(*load, 3, row_indices, local, ADD_VALUES); CHKERRQ(ierr);
        }
        VecAssemblyBegin(*load);
        VecAssemblyEnd(*load);
    } else {
        SETERRQ(PETSC_COMM_SELF, 56, "3D is currently not supported\n");
    }


    return ierr;

}

*/
