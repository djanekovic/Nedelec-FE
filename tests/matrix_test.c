#include <petsc.h>
#include <err.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <math.h>
#include <cmocka.h>

//TODO: ugly hack because methods I want to test are static...
#include "../src/matrix.c"
#include "../src/quadrature.h"
#include "../src/nedelec.h"
#include "../src/util.h"

#define EPS 0.000001
#define assert_double(actual, expected) assert_true(abs(actual - expected) < EPS)

static int init_data(void **data);
static int cleanup(void **data);
static void test_local_stiffness(void **data);
static void test_local_mass(void **data);
static void test_local_load(void **data);

struct test_ctx {
    struct quadrature q;
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

static inline PetscReal one(PetscReal x, PetscReal y) {
    return 1.0;
}

static int init_data(void **data)
{
    PetscErrorCode ierr;

    struct test_ctx *tctx = malloc(sizeof(struct test_ctx));
    if (tctx == NULL) {
        goto malloc_failed;
    }
    ierr = generate_quad(2, &tctx->q); CHKERRQ(ierr);
    ierr = nedelec_basis(tctx->q, &tctx->fs); CHKERRQ(ierr);
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

    r = stiffness_matrix_2D(tctx->q, tctx->fs, &tctx->sctx, 0.25, 1, 1, 1, 1);
    assert_double(r, 8.0);
    r = stiffness_matrix_2D(tctx->q, tctx->fs, &tctx->sctx, 0.25, 1, -1, 1, 2);
    assert_double(r, -8.0);
    r = stiffness_matrix_2D(tctx->q, tctx->fs, &tctx->sctx, 0.25, -1, 1, 1, 3);
    assert_double(r, -8.0);
    r = stiffness_matrix_2D(tctx->q, tctx->fs, &tctx->sctx, 0.25, -1, -1, 2, 1);
    assert_double(r, 8.0);
    r = stiffness_matrix_2D(tctx->q, tctx->fs, &tctx->sctx, -0.25, 1, 1, 2, 2);
    assert_double(r, 8.0);
    r = stiffness_matrix_2D(tctx->q, tctx->fs, &tctx->sctx, -0.25, 1, -1, 2, 3);
    assert_double(r, -8.0);
    r = stiffness_matrix_2D(tctx->q, tctx->fs, &tctx->sctx, -0.25, -1, 1, 3, 1);
    assert_double(r, -8.0);
    r = stiffness_matrix_2D(tctx->q, tctx->fs, &tctx->sctx, -0.25, -1, -1, 3, 2);
    assert_double(r, 8.0);
    r = stiffness_matrix_2D(tctx->q, tctx->fs, &tctx->sctx, -0.25, 1, 1, 3, 3);
    assert_double(r, 8.0);

    //TODO: test gaussian and with custom const function
}

static inline PetscReal _compute_control(PetscReal *invBk, PetscReal detJ,
										 int sign_k, int sign_l, int k, int l)
{
	PetscReal res;
	switch (k) {
		case 0:
			if (l == 0) {
				res = pow(invBk[0], 2) + pow(invBk[2], 2) + pow(invBk[1], 2) +
					  pow(invBk[3], 2);
				res = res - invBk[0] * invBk[2] - invBk[1] * invBk[3];
				return detJ * sign_k * sign_l * 1/12 * res;
			}
		default:
			fail_msg("_compute_control wrong argument...");
	}
	return 0;
}

//TODO: smisli testove
static void test_local_mass(void **data)
{
    struct test_ctx *tctx = *data;
	int k, l, sign_k, sign_l;
	PetscReal res, control, _tmp_matrix[4], detJ = 0.25;
	PetscReal invBk[4] = {1, 1, 1, 1};

	_invBk_invBkT_2D(invBk, _tmp_matrix);
	for (int i = 0; i < 4; i++)
		assert_double(_tmp_matrix[i], 2.0);

	/* Test k = l = 0 case and all possible detJ, sign_k, sign_l cases */
	k = l = 0;
	sign_k = sign_l = 1;
	res = mass_matrix_2D(tctx->q, _tmp_matrix, tctx->fs, &tctx->sctx,
						 detJ, sign_k, sign_l, k, l);
	control = _compute_control(invBk, detJ, sign_k, sign_l, k, l);
	assert_double(res, control);

	res = mass_matrix_2D(tctx->q, _tmp_matrix, tctx->fs, &tctx->sctx,
						 detJ * -1, sign_k, sign_l, k, l);
	assert_double(res, control);

	sign_k = -1; sign_l = 1;
	res = mass_matrix_2D(tctx->q, _tmp_matrix, tctx->fs, &tctx->sctx,
						 detJ, sign_k, sign_l, k, l);
	assert_double(res, control * sign_k);

	sign_l = -1; sign_k = 1;
	res = mass_matrix_2D(tctx->q, _tmp_matrix, tctx->fs, &tctx->sctx,
						 detJ, sign_k, sign_l, k, l);
	assert_double(res, control * sign_l);

	sign_l = -1; sign_k = -1;
	res = mass_matrix_2D(tctx->q, _tmp_matrix, tctx->fs, &tctx->sctx,
						 detJ, sign_k, sign_l, k, l);
	assert_double(res, control);


	/* Test k=0, l=1 case */

}

//TODO: smisli testove
static void test_local_load(void **data)
{
    struct test_ctx *tctx = *data;
}
