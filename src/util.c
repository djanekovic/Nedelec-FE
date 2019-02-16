#include <petsc.h>

#include "util.h"

#define QUAD_2D_1 0.33333333333333, 0.33333333333333, 1.00000000000000
#define QUAD_2D_2 0.16666666666667, 0.16666666666667, 0.33333333333333, \
                  0.16666666666667, 0.66666666666667, 0.33333333333333, \
                  0.66666666666667, 0.16666666666667, 0.33333333333333
#define QUAD_2D_3 0.33333333333333, 0.33333333333333, -0.56250000000000, \
                  0.20000000000000, 0.20000000000000, 0.52083333333333, \
                  0.20000000000000, 0.60000000000000, 0.52083333333333, \
                  0.60000000000000, 0.20000000000000, 0.52083333333333
#define QUAD_2D_4 0.44594849091597, 0.44594849091597, 0.22338158967801, \
                  0.44594849091597, 0.10810301816807, 0.22338158967801, \
                  0.10810301816807, 0.44594849091597, 0.22338158967801, \
                  0.09157621350977, 0.09157621350977, 0.10995174365532, \
                  0.09157621350977, 0.81684757298046, 0.10995174365532, \
                  0.81684757298046, 0.09157621350977, 0.10995174365532
#define QUAD_2D_5 0.33333333333333, 0.33333333333333, 0.22500000000000, \
                  0.47014206410511, 0.47014206410511, 0.13239415278851, \
                  0.47014206410511, 0.05971587178977, 0.13239415278851, \
                  0.05971587178977, 0.47014206410511, 0.13239415278851, \
                  0.10128650732346, 0.10128650732346, 0.12593918054483, \
                  0.10128650732346, 0.79742698535309, 0.12593918054483, \
                  0.79742698535309, 0.10128650732346, 0.12593918054483
#define QUAD_2D_6 0.24928674517091, 0.24928674517091, 0.11678627572638, \
                  0.24928674517091, 0.50142650965818, 0.11678627572638, \
                  0.50142650965818, 0.24928674517091, 0.11678627572638, \
                  0.06308901449150, 0.06308901449150, 0.05084490637021, \
                  0.06308901449150, 0.87382197101700, 0.05084490637021, \
                  0.87382197101700, 0.06308901449150, 0.05084490637021, \
                  0.31035245103378, 0.63650249912140, 0.08285107561837, \
                  0.63650249912140, 0.05314504984482, 0.08285107561837, \
                  0.05314504984482, 0.31035245103378, 0.08285107561837, \
                  0.63650249912140, 0.31035245103378, 0.08285107561837, \
                  0.31035245103378, 0.05314504984482, 0.08285107561837, \
                  0.05314504984482, 0.63650249912140, 0.08285107561837
#define QUAD_2D_7 0.33333333333333, 0.33333333333333, -0.14957004446768, \
                  0.26034596607904, 0.26034596607904, 0.17561525743321, \
                  0.26034596607904, 0.47930806784192, 0.17561525743321, \
                  0.47930806784192, 0.26034596607904, 0.17561525743321, \
                  0.06513010290222, 0.06513010290222, 0.05334723560884, \
                  0.06513010290222, 0.86973979419557, 0.05334723560884, \
                  0.86973979419557, 0.06513010290222, 0.05334723560884, \
                  0.31286549600487, 0.63844418856981, 0.07711376089026, \
                  0.63844418856981, 0.04869031542532, 0.07711376089026, \
                  0.04869031542532, 0.31286549600487, 0.07711376089026, \
                  0.63844418856981, 0.31286549600487, 0.07711376089026, \
                  0.31286549600487, 0.04869031542532, 0.07711376089026, \
                  0.04869031542532, 0.63844418856981, 0.07711376089026
#define QUAD_2D_8 0.33333333333333, 0.33333333333333, 0.14431560767779, \
                  0.45929258829272, 0.45929258829272, 0.09509163426728, \
                  0.45929258829272, 0.08141482341455, 0.09509163426728, \
                  0.08141482341455, 0.45929258829272, 0.09509163426728, \
                  0.17056930775176, 0.17056930775176, 0.10321737053472, \
                  0.17056930775176, 0.65886138449648, 0.10321737053472, \
                  0.65886138449648, 0.17056930775176, 0.10321737053472, \
                  0.05054722831703, 0.05054722831703, 0.03245849762320, \
                  0.05054722831703, 0.89890554336594, 0.03245849762320, \
                  0.89890554336594, 0.05054722831703, 0.03245849762320, \
                  0.26311282963464, 0.72849239295540, 0.02723031417443, \
                  0.72849239295540, 0.00839477740996, 0.02723031417443, \
                  0.00839477740996, 0.26311282963464, 0.02723031417443, \
                  0.72849239295540, 0.26311282963464, 0.02723031417443, \
                  0.26311282963464, 0.00839477740996, 0.02723031417443, \
                  0.00839477740996, 0.72849239295540, 0.02723031417443

static PetscErrorCode set_order_1(Mat **pw);
static PetscErrorCode set_order_2(Mat **pw);
static PetscErrorCode set_order_3(Mat **pw);
static PetscErrorCode set_order_4(Mat **pw);
static PetscErrorCode set_order_5(Mat **pw);
static PetscErrorCode set_order_6(Mat **pw);
static PetscErrorCode set_order_7(Mat **pw);
static PetscErrorCode set_order_8(Mat **pw);

#undef __FUNCT__
#define __FUNCT__ "handle_cli_options"
PetscErrorCode handle_cli_options(struct ctx *sctx) {
    /* defaults */
    sctx->dim = 2;
    sctx->ref = 0;
    sctx->quad_order = 3;

    /* read from cli */
    PetscOptionsGetInt(NULL, NULL, "-dim", &sctx->dim, NULL);
    PetscOptionsGetInt(NULL, NULL, "-ref", &sctx->ref, NULL);

    return (0);
}

/**
 * Generate quadrature points and weights
 *
 * IN: order - quadrature order
 *
 * OUT: pw - quadrature points and weights matrix. First 2 cols are point tuples
 *           and next column is point weight.
 *
 * NOTE: this routine generates MPI dense matrix. Memory is preallocated and
 *       should be freed afted usage.
 */
#undef __FUNCT__
#define __FUNCT__ "quad_pw"
PetscErrorCode quad_pw(PetscInt order, Mat *pw)
{
    PetscErrorCode ierr;

    switch(order) {
        case 1:
            ierr = set_order_1(&pw);
            break;
        case 2:
            ierr = set_order_2(&pw);
            break;
        case 3:
            ierr = set_order_3(&pw);
            break;
        case 4:
            ierr = set_order_4(&pw);
            break;
        case 5:
            ierr = set_order_5(&pw);
            break;
        case 6:
            ierr = set_order_6(&pw);
            break;
        case 7:
            ierr = set_order_7(&pw);
            break;
        case 8:
            ierr = set_order_8(&pw);
            break;
        default:
            PetscErrorPrintf("Order %d not supported, highest order supported \
                    is %d defaulted to %d\n", order, 8, 8);
            ierr = set_order_8(&pw);
            break;
    }

    return ierr;
}

static PetscErrorCode set_order_1(Mat **pw) {
    PetscScalar tmp[] = {QUAD_2D_1}, *data;

    PetscErrorCode ierr = PetscMalloc1(1 * 3, &data); CHKERRQ(ierr);
    ierr = PetscMemcpy(data, tmp, 1 * 3 * sizeof(PetscScalar)); CHKERRQ(ierr);

    ierr = MatCreateDense(PETSC_COMM_WORLD,
                          PETSC_DECIDE, PETSC_DECIDE, 1, 3, data, *pw);

    return ierr;
}

static PetscErrorCode set_order_2(Mat **pw) {
    PetscScalar tmp[] = {QUAD_2D_2}, *data;
    PetscErrorCode ierr = PetscMalloc1(3* 3, &data); CHKERRQ(ierr);
    ierr = PetscMemcpy(data, tmp, 3 * 3 * sizeof(PetscScalar)); CHKERRQ(ierr);

    ierr = MatCreateDense(PETSC_COMM_WORLD,
                          PETSC_DECIDE, PETSC_DECIDE, 3, 3, data, *pw);

    return ierr;
}

static PetscErrorCode set_order_3(Mat **pw) {
    PetscScalar tmp[] = {QUAD_2D_3}, *data;
    PetscErrorCode ierr = PetscMalloc1(4 * 3, &data); CHKERRQ(ierr);
    ierr = PetscMemcpy(data, tmp, 4 * 3 * sizeof(PetscScalar)); CHKERRQ(ierr);

    ierr = MatCreateDense(PETSC_COMM_WORLD,
                          PETSC_DECIDE, PETSC_DECIDE, 4, 3, data, *pw);

    return ierr;
}

static PetscErrorCode set_order_4(Mat **pw) {
    PetscScalar tmp[] = {QUAD_2D_4}, *data;
    PetscErrorCode ierr = PetscMalloc1(6 * 3, &data); CHKERRQ(ierr);
    ierr = PetscMemcpy(data, tmp, 6 * 3 * sizeof(PetscScalar)); CHKERRQ(ierr);

    ierr = MatCreateDense(PETSC_COMM_WORLD,
                          PETSC_DECIDE, PETSC_DECIDE, 6, 3, data, *pw);

    return ierr;
}

static PetscErrorCode set_order_5(Mat **pw) {
    PetscScalar tmp[] = {QUAD_2D_5}, *data;
    PetscErrorCode ierr = PetscMalloc1(7 * 3, &data); CHKERRQ(ierr);
    ierr = PetscMemcpy(data, tmp, 7 * 3 * sizeof(PetscScalar)); CHKERRQ(ierr);

    ierr = MatCreateDense(PETSC_COMM_WORLD,
                          PETSC_DECIDE, PETSC_DECIDE, 7, 3, data, *pw);

    return ierr;
}

static PetscErrorCode set_order_6(Mat **pw) {
    PetscScalar tmp[] = {QUAD_2D_6}, *data;

    PetscErrorCode ierr = PetscMalloc1(12 * 3, &data); CHKERRQ(ierr);
    ierr = PetscMemcpy(data, tmp, 12 * 3 * sizeof(PetscScalar)); CHKERRQ(ierr);

    ierr = MatCreateDense(PETSC_COMM_WORLD,
                          PETSC_DECIDE, PETSC_DECIDE, 12, 3, data, *pw);

    return ierr;
}

static PetscErrorCode set_order_7(Mat **pw) {
    PetscScalar tmp[] = {QUAD_2D_7}, *data;

    PetscErrorCode ierr = PetscMalloc1(13 * 3, &data); CHKERRQ(ierr);
    ierr = PetscMemcpy(data, tmp, 13 * 3 * sizeof(PetscScalar)); CHKERRQ(ierr);

    ierr = MatCreateDense(PETSC_COMM_WORLD,
                          PETSC_DECIDE, PETSC_DECIDE, 13, 3, data, *pw);
    return ierr;
}

static PetscErrorCode set_order_8(Mat **pw) {
    PetscScalar tmp[] = {QUAD_2D_8}, *data;

    PetscErrorCode ierr = PetscMalloc1(16 * 3, &data); CHKERRQ(ierr);
    ierr = PetscMemcpy(data, tmp, 16 * 3 * sizeof(PetscScalar)); CHKERRQ(ierr);

    ierr = MatCreateDense(PETSC_COMM_WORLD,
                          PETSC_DECIDE, PETSC_DECIDE, 16, 3, data, *pw);

    return ierr;
}
