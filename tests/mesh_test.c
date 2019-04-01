#include <petsc.h>
#include <err.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include "../src/mesh.h"

#define EPSILON 0.0001
#define SIGNS_TEST_DATA  1,  1, -1, \
                         1, -1,  1, \
                        -1,  1, -1, \
                        -1,  1, -1, \
                         1, -1,  1, \
                        -1,  1, -1, \
                         1, -1,  1, \
                         1, -1, -1

#define COLS_TEST_DATA  8,  9, \
                        9, 11, \
                        8, 11, \
                       12, 14, \
                       11, 14, \
                       11, 12, \
                        9, 12, \
                       10, 12, \
                       10, 13, \
                       12, 13, \
                        9, 10, \
                       13, 15, \
                       13, 16, \
                       15, 16, \
                       12, 15, \
                       14, 15

#define VALS_TEST_DATA -1.0,  1.0, \
                       -1.0,  1.0, \
                        1.0, -1.0, \
                       -1.0,  1.0, \
                        1.0, -1.0, \
                       -1.0,  1.0, \
                       -1.0,  1.0, \
                        1.0, -1.0, \
                       -1.0,  1.0, \
                        1.0, -1.0, \
                       -1.0,  1.0, \
                        1.0, -1.0, \
                       -1.0,  1.0, \
                        1.0, -1.0, \
                        1.0, -1.0, \
                        1.0, -1.0,


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

    return cmocka_run_group_tests(tests, init_petsc, finalize_petsc);
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
    int test_signs[] = {SIGNS_TEST_DATA};
    struct test_ctx *tctx = *data;
    struct ctx *sctx = tctx->sctx;
    assert_non_null(sctx->signs);

    for (int i = sctx->cstart; i < sctx->cend; i++) {
        for (int j = 0; j < 3; j++) {
            int offset = i * 3 + j;
            assert_int_equal(test_signs[offset], sctx->signs[offset]);
        }
    }
}

static void test_discrete_gradient(void **data)
{
    PetscInt test_cols[] = {COLS_TEST_DATA};
    PetscScalar test_vals[] = {VALS_TEST_DATA};
    struct test_ctx *tctx = *data;
    struct ctx *sctx = tctx->sctx;

    assert_non_null(tctx->sctx->G);

    for (int i = 0; i < sctx->eend - sctx->estart; i++) {
        PetscInt ncols;
        const PetscInt *cols;
        const PetscScalar *vals;
        MatGetRow(sctx->G, i, &ncols, &cols, &vals);
        for (int j = 0; j < ncols; j++) {
            int offset = i * ncols + j;
            assert_int_equal(test_cols[offset] - sctx->vstart, cols[j]);
            assert_true(PetscAbsReal(test_vals[offset] - vals[j]) < EPSILON);
        }
        MatRestoreRow(sctx->G, i, &ncols, &cols, &vals);
    }
}

static int finalize_petsc(void **data)
{
    struct test_ctx *tctx = *data;
    DM *dm = tctx->dm;

    DMDestroy(dm);
    PetscFinalize();

    return 0;
}
