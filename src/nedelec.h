#ifndef NEDELEC_H
#define NEDELEC_H

#include "quadrature.h"

typedef struct {
    PetscScalar *val;
    PetscScalar *cval;
    quadrature_t triangle;
    quadrature_t line;
    PetscInt nbasis;
} fs_t;

/**
 * Compute stiffness matrix for first order Nedelec element in 2D
 *
 * 1/|det Bk| \int f(x,y) * sign_k * curl_ned_k * sign_l * curl_sign_l dx
 * 1/|det Bk| * sign_k * sign_l \int f(x, y) * curl_ned_k * curl_sign_l dx
 */
static inline PetscReal stiffness_matrix_2D(fs_t fs,
                                            PetscReal detJ, PetscInt sign_k,
                                            PetscInt sign_l, PetscInt k,
                                            PetscInt l)
{
    PetscReal local = 1 / PetscAbsReal(detJ) * sign_l * sign_k;

    /** This is used only if f(x, y) is non constant
    for (PetscInt i = 0; i < fs->triangle.size; i++) {
        int _3i = 3 * i;
        sum += triangle.pw[_3i + 2] * fs.cval[_3i + k] * fs.cval[_3i + l]
             * sctx->stiffness_function_2D(triangle.pw[_3i + 0], triangle.pw[_3i + 1]);
    }
    */
    // TODO: handle stiffness_const()
    return local * 0.5 * fs.cval[k] * fs.cval[l];
}

/**
 * Mass matrix computation
 * |det Bk| \int sign_k * Bk^-T * ned_k * sign_l * Bk^-T * ned_l dx
 * |det Bk| * sign_k * sign_l \int Bk * Bk^-T * ned_k * ned_l dx
 */
static inline PetscReal mass_matrix_2D(PetscReal *C, fs_t fs,
                                       PetscReal detJ, PetscInt sign_k,
                                       PetscInt sign_l, PetscInt k_ned,
                                       PetscInt l_ned)
{
    PetscReal local = PetscAbsReal(detJ) * sign_k * sign_l;
    PetscReal sum = 0;

    for (PetscInt i = 0; i < fs.triangle.size; i++) {
        int k_off = k_ned * (fs.triangle.size * 2) + i * 2;
        int l_off = l_ned * (fs.triangle.size * 2) + i * 2;

        PetscReal __x = C[0] * fs.val[k_off + 0] + C[1] * fs.val[k_off + 1];
        PetscReal __y = C[2] * fs.val[k_off + 0] + C[3] * fs.val[k_off + 1];
        PetscReal _mvv = __x * fs.val[l_off + 0] + __y * fs.val[l_off + 1];
        sum += fs.triangle.pw[i * 3 + 2] * _mvv * 1;
    }

    return local * sum * 0.5;
}

static inline PetscReal load_vector_2D(PetscReal *invJ,
                                       fs_t fs, PetscReal detJ,
                                       PetscInt k, PetscInt sign_k)
{
    PetscReal local = PetscAbsReal(detJ) * sign_k;
    PetscReal sum = 0;

    PetscReal f_x = 1.0;
    PetscReal f_y = 1.0;

    // TODO: implement better handling of vector functions
    // TODO: small matrix lib
    for (PetscInt i = 0; i < fs.triangle.size; i++) {
        int k_off = k * (fs.triangle.size * 2) + i * 2;
        PetscReal _x =
            invJ[0] * fs.val[k_off + 0] + invJ[2] * fs.val[k_off + 1];
        PetscReal _y =
            invJ[1] * fs.val[k_off + 0] + invJ[3] * fs.val[k_off + 1];
        PetscReal _mvv = _x * f_x + _y * f_y;
        sum += fs.triangle.pw[i * 3 + 2] * _mvv;
    }

    return local * sum * 0.5;
}

PetscErrorCode create_nedelec(fs_t *fs, int);
PetscErrorCode destroy_nedelec(fs_t *fs);

#endif /* NEDELEC_H */
