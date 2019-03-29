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
        int _3i = 3 * i;
        local *= q.pw[_3i + 2] * fs.cval[_3i + k] * fs.cval[_3i + l] *
                 sctx->stiffness_function_2D(q.pw[_3i + 0], q.pw[_3i + 1]) *
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
        int k_offset = k_ned * (q.size * 2) + i * 2;
        int l_offset = l_ned * (q.size * 2) + i * 2;
        for (PetscInt j = 0; j < dim; j++) {
            double x = 0, y = 0;
            int _j3 = j * 3;
            for (PetscInt k = 0; k < dim; j++) {
                x += invJ[_j3 + k] * fs.val[k_offset + j];
                y += invJ[_j3 + k] * fs.val[l_offset + j];
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
        int _2i = i * 2;
        int _3i = i * 3;
        for (PetscInt j = 0; j < dim; j++) {
            int val_offset = j * (q.size * 2) + _2i;
            int _3j = j * 3;
            for (PetscInt k = 0; k < dim; k++) {
                local += invJ[_3j + k] * fs.val[val_offset + k];
            }
        }

        local *= q.pw[_3i + 2] * sign_k *
                 sctx->load_function_2D(q.pw[_3i + 0], q.pw[_3i + 1]);
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
            ierr = DMPlexComputeCellGeometryAffineFEM(dm, c,
                                                      &v0, (PetscReal *) &J,
                                                      (PetscReal *) &invJ,
                                                      &detJ);
            CHKERRQ(ierr);
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
            ierr = MatSetValues(A, 3, row_indices, 3, col_indices,
                                *local, ADD_VALUES);
            CHKERRQ(ierr);
            ierr = VecSetValues(b, 3, row_indices, load, ADD_VALUES);
            CHKERRQ(ierr);
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
