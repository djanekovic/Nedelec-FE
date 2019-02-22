#ifndef QUADRATURE_H
#define QUADRATURE_H


/* Gaussian quadrature rule preprocessor defines */

#define QUAD_2D_1 0.33333333333333, 0.33333333333333, 1.00000000000000
#define QUAD_2D_1_LEN 3

#define QUAD_2D_2 0.16666666666667, 0.16666666666667, 0.33333333333333, \
                  0.16666666666667, 0.66666666666667, 0.33333333333333, \
                  0.66666666666667, 0.16666666666667, 0.33333333333333
#define QUAD_2D_2_LEN 9

#define QUAD_2D_3 0.33333333333333, 0.33333333333333, -0.56250000000000, \
                  0.20000000000000, 0.20000000000000, 0.52083333333333, \
                  0.20000000000000, 0.60000000000000, 0.52083333333333, \
                  0.60000000000000, 0.20000000000000, 0.52083333333333
#define QUAD_2D_3_LEN 12

#define QUAD_2D_4 0.44594849091597, 0.44594849091597, 0.22338158967801, \
                  0.44594849091597, 0.10810301816807, 0.22338158967801, \
                  0.10810301816807, 0.44594849091597, 0.22338158967801, \
                  0.09157621350977, 0.09157621350977, 0.10995174365532, \
                  0.09157621350977, 0.81684757298046, 0.10995174365532, \
                  0.81684757298046, 0.09157621350977, 0.10995174365532
#define QUAD_2D_4_LEN 18

#define QUAD_2D_5 0.33333333333333, 0.33333333333333, 0.22500000000000, \
                  0.47014206410511, 0.47014206410511, 0.13239415278851, \
                  0.47014206410511, 0.05971587178977, 0.13239415278851, \
                  0.05971587178977, 0.47014206410511, 0.13239415278851, \
                  0.10128650732346, 0.10128650732346, 0.12593918054483, \
                  0.10128650732346, 0.79742698535309, 0.12593918054483, \
                  0.79742698535309, 0.10128650732346, 0.12593918054483
#define QUAD_2D_5_LEN 21

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
#define QUAD_2D_6_LEN 36

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
#define QUAD_2D_7_LEN 39

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
#define QUAD_2D_8_LEN 48

/**
 * .pw in 2D matrix:
 * |  x  |  y  |  w  |
 * where x and y are coordinates and w is point weight. For 3D case, there
 * will be 4 cols.
 *
 * .size is number of rows in pw matrix.
 */
struct quadrature {
    PetscScalar *pw;
    PetscInt order;
    PetscInt size;
};

PetscErrorCode generate_quad(PetscInt order, struct quadrature *q);

#endif /* QUADRATURE_H */
