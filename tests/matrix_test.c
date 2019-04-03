#include <petsc.h>
#include <err.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

//TODO: ugly hack because methods I want to test are static...
#include "../src/matrix.c"
#include "../src/quadrature.h"
#include "../src/nedelec.h"
#include "../src/util.h"

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
    ierr = generate_quad(1, &tctx->q); CHKERRQ(ierr);
    ierr = nedelec_basis(tctx->q, &tctx->fs); CHKERRQ(ierr);
    tctx->sctx.dim = 2;
    tctx->sctx.stiffness_function_2D = one;

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

//TODO: napiši pametnije testove i na papiru provjeri rezultat
static void test_local_stiffness(void **data)
{
    struct test_ctx *tctx = *data;

    PetscReal res = stiffness_matrix_2D(tctx->q, tctx->fs, &tctx->sctx, 0.25, -1, 1, 2, 1);

    printf("%lf\n", res);
}

//TODO: smisli testove
static void test_local_mass(void **data)
{
    struct test_ctx *tctx = *data;
}

//TODO: smisli testove
static void test_local_load(void **data)
{
    struct test_ctx *tctx = *data;
}
