#include <petsc.h>
#include <err.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include "../src/mesh.h"

const char *help = "Unit tests for meshing utility";

static int init_petsc(void **data);
static int finalize_petsc(void **data);
static void test_signs(void **data);
static void test_discrete_gradient(void **data);

struct test_ctx {
    struct ctx *sctx;
    DM *dm;
};

int main(int argc, char **argv)
{
    PetscErrorCode ierr = PetscInitialize(&argc, &argv, NULL, NULL);
    if (ierr) {
        return ierr;
    }

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_signs),
        cmocka_unit_test(test_discrete_gradient),
    };

    cmocka_run_group_tests(tests, init_petsc, finalize_petsc);

    return 0;
}

static int init_petsc(void **data)
{
    PetscErrorCode ierr;

    struct test_ctx *tctx = malloc(sizeof(struct test_ctx));
    if (tctx == NULL) {
        goto malloc_failed;
    }

    struct ctx *sctx = malloc(sizeof(struct ctx));
    if (sctx == NULL) {
        goto malloc_failed;
    }

    DM *dm = malloc(sizeof(DM));
    if (dm == NULL) {
        goto malloc_failed;
    }

    sctx->dim = 2;
    sctx->ref = 0;

    ierr = generate_mesh(sctx, dm); CHKERRQ(ierr);

    tctx->sctx = sctx;
    tctx->dm = dm;
    *data = tctx;

    return 0;

malloc_failed:
    errx(1, "Malloc failed");
}

static void test_signs(void **data)
{
    struct test_ctx *tctx = *data;
    assert_non_null(tctx->sctx->signs);
}

static void test_discrete_gradient(void **data)
{
    struct test_ctx *tctx = *data;
    assert_non_null(tctx->sctx->G);
}

static int finalize_petsc(void **data)
{
    PetscFinalize();

    return 0;
}
