#include <err.h>
#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>

#include <cmocka.h>
#include <petsc.h>

// TODO: ugly hack because methods I want to test are static...
#include "../src/matrix.c"
#include "../src/nedelec.h"
#include "../src/quadrature.h"
#include "../src/util.h"

#define EPS 0.000001
#define assert_double(actual, expected)                                        \
    assert_true(abs(actual - expected) < EPS)

static int init_data(void **data);
static int cleanup(void **data);
static void test_local_stiffness(void **data);
static void test_local_mass(void **data);
static void test_local_load(void **data);

struct test_ctx {
    struct function_space fs;
    struct ctx sctx;
};

int main(int argc, char **argv)
{
    PetscErrorCode ierr = PetscInitialize(&argc, &argv, NULL, NULL);
    if (ierr) {
        return ierr;
    }

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_local_stiffness),
        cmocka_unit_test(test_local_mass),
        cmocka_unit_test(test_local_load),
    };

    return cmocka_run_group_tests(tests, init_data, cleanup);
}

static inline PetscReal one(PetscReal x, PetscReal y)
{
    return 1.0;
}

static int init_data(void **data)
{
    PetscErrorCode ierr;

    struct test_ctx *tctx = malloc(sizeof(struct test_ctx));
    if (tctx == NULL) {
        goto malloc_failed;
    }
    ierr = nedelec_basis(&tctx->fs, 3);
    CHKERRQ(ierr);
    tctx->sctx.dim = 2;
    tctx->sctx.stiffness_function_2D = one;
    tctx->sctx.mass_function_2D = one;

    *data = tctx;
    return 0;

malloc_failed:
    return -1;
}

static int cleanup(void **data)
{
    struct test_ctx *tctx = *data;
    free(tctx);
    PetscFinalize();

    return 0;
}

static void test_local_stiffness(void **data)
{
    struct test_ctx *tctx = *data;
    PetscReal r;

    r = stiffness_matrix_2D(tctx->fs, &tctx->sctx, 0.25, 1, 1, 1, 1);
    assert_double(r, 8.0);
    r = stiffness_matrix_2D(tctx->fs, &tctx->sctx, 0.25, 1, -1, 1, 2);
    assert_double(r, -8.0);
    r = stiffness_matrix_2D(tctx->fs, &tctx->sctx, 0.25, -1, 1, 1, 3);
    assert_double(r, -8.0);
    r = stiffness_matrix_2D(tctx->fs, &tctx->sctx, 0.25, -1, -1, 2, 1);
    assert_double(r, 8.0);
    r = stiffness_matrix_2D(tctx->fs, &tctx->sctx, -0.25, 1, 1, 2, 2);
    assert_double(r, 8.0);
    r = stiffness_matrix_2D(tctx->fs, &tctx->sctx, -0.25, 1, -1, 2, 3);
    assert_double(r, -8.0);
    r = stiffness_matrix_2D(tctx->fs, &tctx->sctx, -0.25, -1, 1, 3, 1);
    assert_double(r, -8.0);
    r = stiffness_matrix_2D(tctx->fs, &tctx->sctx, -0.25, -1, -1, 3, 2);
    assert_double(r, 8.0);
    r = stiffness_matrix_2D(tctx->fs, &tctx->sctx, -0.25, 1, 1, 3, 3);
    assert_double(r, 8.0);

    // TODO: test gaussian and with custom const function
}

static inline PetscReal __compute_mass_0_0(PetscReal *invBk)
{
    PetscReal res;
    PetscReal a = invBk[0], b = invBk[2], c = invBk[1], d = invBk[3];

    res = pow(a, 2) + pow(b, 2) + pow(c, 2) + pow(d, 2);
    res -= (a * b + c * d);
    return 1 / 12.0 * res;
}

static inline PetscReal __compute_mass_0_1(PetscReal *invBk)
{
    PetscReal res;
    PetscReal a = invBk[0], b = invBk[2], c = invBk[1], d = invBk[3];

    res = 1 / 12.0 * (pow(a, 2) + pow(c, 2));
    res += 1 / 8.0 * (a * b + c * d);
    res -= 1 / 24.0 * (a * b + c * d);
    res -= 1 / 12.0 * (pow(b, 2) + pow(d, 2));

    return res;
}

static inline PetscReal __compute_mass_0_2(PetscReal *invBk)
{
    PetscReal res;
    PetscReal a = invBk[0], b = invBk[2], c = invBk[1], d = invBk[3];

    res = -1 / 12.0 * (pow(a, 2) + pow(c, 2));
    res -= 1 / 24.0 * (a * b + c * d);
    res += 1 / 8.0 * (a * b + c * d);
    res += 1 / 12.0 * (pow(b, 2), +pow(d, 2));

    return res;
}

static inline PetscReal __compute_mass_1_1(PetscReal *invBk)
{
    PetscReal res;
    PetscReal a = invBk[0], b = invBk[2], c = invBk[1], d = invBk[3];

    res = 1 / 12.0 * (pow(a, 2) + pow(c, 2));
    res += 1 / 4.0 * (a * b + c * d);
    res += 1 / 4.0 * (pow(b, 2) + pow(d, 2));

    return res;
}

static inline PetscReal __compute_mass_1_2(PetscReal *invBk)
{
    PetscReal res;
    PetscReal a = invBk[0], b = invBk[2], c = invBk[1], d = invBk[3];

    res = -1 / 12.0 * (pow(a, 2) + pow(c, 2));
    res -= 1 / 24.0 * (a * b + c * d);
    res -= 5 / 24.0 * (a * b + c * d);
    res -= 1 / 12.0 * (pow(b, 2) + pow(d, 2));

    return res;
}

static inline PetscReal __compute_mass_2_2(PetscReal *invBk)
{
    PetscReal res;
    PetscReal a = invBk[0], b = invBk[2], c = invBk[1], d = invBk[3];

    res = 1 / 4.0 * (pow(a, 2) + pow(c, 2));
    res += 1 / 4.0 * (a * b + c * d);
    res += 1 / 12.0 * (pow(b, 2) + pow(d, 2));

    return res;
}

static PetscReal _compute_mass_control(PetscReal *invBk, PetscReal detJ,
                                       int sign_k, int sign_l, int k, int l)
{
    PetscReal local = PetscAbsReal(detJ) * sign_k * sign_l;

    if (k == 0 && l == 0) {
        return local * __compute_mass_0_0(invBk);
    } else if ((k == 0 && l == 1) || (l == 0 && k == 1)) {
        return local * __compute_mass_0_1(invBk);
    } else if ((k == 0 && l == 2) || (l == 2 && k == 0)) {
        return local * __compute_mass_0_2(invBk);
    } else if (k == 1 && l == 1) {
        return local * __compute_mass_1_1(invBk);
    } else if ((k == 1 && l == 2) || (k == 2 && l == 1)) {
        return local * __compute_mass_1_2(invBk);
    }

    /* k = 2, l = 2 */
    return local * __compute_mass_2_2(invBk);
}

static void test_local_mass(void **data)
{
    struct test_ctx *tctx = *data;
    int k, l, sign_k, sign_l;
    PetscReal res, control, _tmp_matrix[4], detJ = 0.25;
    PetscReal invBk[4] = { 0, 2, -2, -2 };

    _invBk_invBkT_2D(invBk, _tmp_matrix);

    /* Test k = l = 0 case and all possible detJ, sign_k, sign_l cases */

    k = l = 0;
    sign_k = sign_l = 1;
    res = mass_matrix_2D(_tmp_matrix, tctx->fs, &tctx->sctx, detJ, sign_k,
                         sign_l, k, l);
    control = _compute_mass_control(invBk, detJ, sign_k, sign_l, k, l);
    assert_double(res, control);

    res = mass_matrix_2D(_tmp_matrix, tctx->fs, &tctx->sctx, detJ * -1, sign_k,
                         sign_l, k, l);
    assert_double(res, control);

    sign_k = -1;
    sign_l = 1;
    res = mass_matrix_2D(_tmp_matrix, tctx->fs, &tctx->sctx, detJ, sign_k,
                         sign_l, k, l);
    assert_double(res, control * sign_k);

    sign_l = -1;
    sign_k = 1;
    res = mass_matrix_2D(_tmp_matrix, tctx->fs, &tctx->sctx, detJ, sign_k,
                         sign_l, k, l);
    assert_double(res, control * sign_l);

    sign_l = -1;
    sign_k = -1;
    res = mass_matrix_2D(_tmp_matrix, tctx->fs, &tctx->sctx, detJ, sign_k,
                         sign_l, k, l);
    assert_double(res, control);

    /* Test k=0, l=1 and l=0, k=1 case */

    k = 0;
    l = 1;
    sign_k = sign_l = 1;
    res = mass_matrix_2D(_tmp_matrix, tctx->fs, &tctx->sctx, detJ, sign_k,
                         sign_l, k, l);
    control = _compute_mass_control(invBk, detJ, sign_k, sign_l, k, l);
    assert_double(res, control);
    res = mass_matrix_2D(_tmp_matrix, tctx->fs, &tctx->sctx, detJ, sign_k,
                         sign_l, l, k);
    assert_double(res, control);

    /* Test k=0, l=2 and l=0, k=2 case */

    k = 0;
    l = 2;
    sign_k = sign_l = 1;
    res = mass_matrix_2D(_tmp_matrix, tctx->fs, &tctx->sctx, detJ, sign_k,
                         sign_l, k, l);
    control = _compute_mass_control(invBk, detJ, sign_k, sign_l, k, l);
    assert_double(res, control);
    res = mass_matrix_2D(_tmp_matrix, tctx->fs, &tctx->sctx, detJ, sign_k,
                         sign_l, l, k);
    assert_double(res, control);

    /* Test k=1, l=1 */

    k = 1;
    l = 1;
    sign_k = sign_l = 1;
    res = mass_matrix_2D(_tmp_matrix, tctx->fs, &tctx->sctx, detJ, sign_k,
                         sign_l, k, l);
    control = _compute_mass_control(invBk, detJ, sign_k, sign_l, k, l);
    assert_double(res, control);
    res = mass_matrix_2D(_tmp_matrix, tctx->fs, &tctx->sctx, detJ, sign_k,
                         sign_l, l, k);
    assert_double(res, control);

    /* Test k=1, l=2 and k=2, l=1 */

    k = 1;
    l = 2;
    sign_k = sign_l = 1;
    res = mass_matrix_2D(_tmp_matrix, tctx->fs, &tctx->sctx, detJ, sign_k,
                         sign_l, k, l);
    control = _compute_mass_control(invBk, detJ, sign_k, sign_l, k, l);
    assert_double(res, control);
    res = mass_matrix_2D(_tmp_matrix, tctx->fs, &tctx->sctx, detJ, sign_k,
                         sign_l, l, k);
    assert_double(res, control);

    /* Test k=2, l=2 */

    k = 2;
    l = 2;
    sign_k = sign_l = 1;
    res = mass_matrix_2D(_tmp_matrix, tctx->fs, &tctx->sctx, detJ, sign_k,
                         sign_l, k, l);
    control = _compute_mass_control(invBk, detJ, sign_k, sign_l, k, l);
    assert_double(res, control);
    res = mass_matrix_2D(_tmp_matrix, tctx->fs, &tctx->sctx, detJ, sign_k,
                         sign_l, l, k);
    assert_double(res, control);
}

/**
 * Analytic solution for k = 1:
 *
 *  c - a     d - b
 * ------- + -------
 *    6         6
 */
static inline PetscReal __compute_load_0(PetscReal *invBk)
{
    PetscReal res;
    PetscReal a = invBk[0], b = invBk[1], c = invBk[2], d = invBk[3];

    res = (c - a) + (d - b);

    return res / 6.0;
}

/**
 * Analytic solution for k = 2:
 *
 *    a + 2c    b + 2d
 * - ------- - --------
 *      6         6
 */
static inline PetscReal __compute_load_1(PetscReal *invBk)
{
    PetscReal res;
    PetscReal a = invBk[0], b = invBk[1], c = invBk[2], d = invBk[3];

    res = -(2 * c + a) - (2 * d + b);

    return res / 6.0;
}

/**
 * Analytic solution for k = 3:
 *
 *  2a + c     2b + d
 * -------- + --------
 *     6          6
 */
static inline PetscReal __compute_load_2(PetscReal *invBk)
{
    PetscReal res;
    PetscReal a = invBk[0], b = invBk[1], c = invBk[2], d = invBk[3];

    res = (2 * a + c) + (2 * b + d);

    return res / 6.0;
}

static PetscReal _compute_load_control(PetscReal *invBk, PetscReal detJ,
                                       int sign_k, int k)
{
    PetscReal local = PetscAbsReal(detJ) * sign_k;

    switch (k) {
        case 0:
            return local * __compute_load_0(invBk);
        case 1:
            return local * __compute_load_1(invBk);
        case 2:
        default:
            return local * __compute_load_2(invBk);
    }
}

static void test_local_load(void **data)
{
    struct test_ctx *tctx = *data;
    int k, sign_k;
    PetscReal res, control, detJ = 0.0625;
    PetscReal invBk[4] = { 4, -0, 0, 4 };

    /* Test k = 0 case and all possible detJ, sign_k cases */

    k = 0;
    sign_k = 1;
    res = load_vector_2D(invBk, tctx->fs, &tctx->sctx, detJ, k, sign_k);
    control = _compute_load_control(invBk, detJ, sign_k, k);
    assert_double(res, control);
    printf("%lf == %lf\n", res, control);

    res = load_vector_2D(invBk, tctx->fs, &tctx->sctx, detJ * -1, k, sign_k);
    assert_double(res, control);
    printf("%lf == %lf\n", res, control);

    res = load_vector_2D(invBk, tctx->fs, &tctx->sctx, detJ, k, sign_k * -1);
    assert_double(res, control * -1);
    printf("%lf == %lf\n", res, control * -1);

    /* Test k = 1 */
    k = 1;
    sign_k = 1;
    res = load_vector_2D(invBk, tctx->fs, &tctx->sctx, detJ, k, sign_k);
    control = _compute_load_control(invBk, detJ, sign_k, k);
    printf("%lf == %lf\n", res, control);
    assert_double(res, control);

    /* Test k = 2 */
    k = 2;
    sign_k = 1;
    res = load_vector_2D(invBk, tctx->fs, &tctx->sctx, detJ, k, sign_k);
    control = _compute_load_control(invBk, detJ, sign_k, k);
    printf("%lf == %lf\n", res, control);
    assert_double(res, control);
}
