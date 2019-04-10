#include <petsc.h>

#include "matrix.h"
#include "mesh.h"

#define F_x 1.0
#define F_y 1.0

/**
 * Compute stiffness matrix for first order Nedelec element in 2D
 *
 * 1/|det Bk| \int f(x,y) * sign_k * curl_ned_k * sign_l * curl_sign_l dx
 * 1/|det Bk| * sign_k * sign_l \int f(x, y) * curl_ned_k * curl_sign_l dx
 */
static inline PetscReal stiffness_matrix_2D(struct quadrature q,
                                     struct function_space fs,
                                     struct ctx *sctx, PetscReal detJ,
                                     PetscInt sign_k, PetscInt sign_l,
                                     PetscInt k, PetscInt l)
{
    PetscReal local = 1/PetscAbsReal(detJ) * sign_l * sign_k;
    PetscReal sum = 0;

    /** This is used only if f(x, y) is non constant
    for (PetscInt i = 0; i < q.size; i++) {
        int _3i = 3 * i;
        sum += q.pw[_3i + 2] * fs.cval[_3i + k] * fs.cval[_3i + l]
             * sctx->stiffness_function_2D(q.pw[_3i + 0], q.pw[_3i + 1]);
    }
    */
    //TODO: handle stiffness_const()
    return local * 0.5 * fs.cval[k] * fs.cval[l];
}

/**
 * Mass matrix computation
 * |det Bk| \int sign_k * Bk^-T * ned_k * sign_l * Bk^-T * ned_l dx
 * |det Bk| * sign_k * sign_l \int Bk * Bk^-T * ned_k * ned_l dx
 */
static inline PetscReal mass_matrix_2D(struct quadrature q, PetscReal *C,
                                struct function_space fs,
                                struct ctx *sctx, PetscReal detJ,
                                PetscInt sign_k, PetscInt sign_l,
                                PetscInt k_ned, PetscInt l_ned)
{
    PetscReal local = PetscAbsReal(detJ) * sign_k * sign_l;
    PetscReal sum = 0;

    //TODO: cleanup or maybe call blas, tradeoff?
    for (PetscInt i = 0; i < q.size; i++) {
        int k_off = k_ned * (q.size * 2) + i * 2;
        int l_off = l_ned * (q.size * 2) + i * 2;

		PetscReal __x = C[0] * fs.val[k_off + 0] + C[1] * fs.val[k_off + 1];
		PetscReal __y = C[2] * fs.val[k_off + 0] + C[3] * fs.val[k_off + 1];
        PetscReal _mvv = __x * fs.val[l_off + 0] + __y * fs.val[l_off + 1];
        sum += q.pw[i * 3 + 2] * _mvv
             * sctx->mass_function_2D(q.pw[i * 3 + 0], q.pw[i * 3 + 1]);
    }

    return local * sum;
}

static inline PetscReal load_vector_2D(struct quadrature q, PetscReal *invJ,
                                struct function_space fs,
                                struct ctx *sctx, PetscReal detJ,
                                PetscInt k, PetscInt sign_k)
{
    PetscReal local = PetscAbsReal(detJ) * sign_k;
    PetscReal sum = 0;

    PetscReal f_x = 1;
    PetscReal f_y = 1;

    //TODO: implement better handling of vector functions
    //Transpose matrix, multiply with vector and then again with vector
    for (PetscInt i = 0; i < q.size; i++) {
        int k_offset = k * (q.size * 2) + i * 2;
        PetscReal _mvv = invJ[0] * fs.val[k_offset + 0] * f_x
                       + invJ[3] * fs.val[k_offset + 1] * f_x
                       + invJ[2] * fs.val[k_offset + 0] * f_y
                       + invJ[1] * fs.val[k_offset + 1] * f_y;
        sum += q.pw[i * 3 + 2] * _mvv;
    }

    return local * sum;
}

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
            PetscReal v0, Bk[4], invBk[4], detBk, _tmp_matrix[4];
            ierr = DMPlexComputeCellGeometryAffineFEM(dm, c,
                                                      &v0, (PetscReal *) &Bk,
                                                      (PetscReal *) &invBk,
                                                      &detBk);
            CHKERRQ(ierr);
            ierr = DMPlexGetCone(dm, c, &edgelist); CHKERRQ(ierr);

            _invBk_invBkT_2D(invBk, _tmp_matrix);

            for (PetscInt k = 0; k < nedges; k++) {
                PetscInt sign_k = sctx->signs[(c - cstart) * 3 + k];
                row_indices[k] = edgelist[k] - estart;
                for (PetscInt l = 0; l < nedges; l++) {
                    col_indices[l] = edgelist[l] - estart;
                    PetscInt sign_l = sctx->signs[(c - cstart) * 3 + l];

                    local[k][l] = stiffness_matrix_2D(q, fs, sctx, detBk,
                                                      sign_k, sign_l, k, l);
                    local[k][l] += mass_matrix_2D(q, _tmp_matrix, fs, sctx,
                                                  detBk, sign_k, sign_l, k, l);
                }
                load[k] = load_vector_2D(q, invBk, fs, sctx, detBk, k, sign_k);
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
