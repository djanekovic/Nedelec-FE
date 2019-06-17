#include <petsc.h>

#include "quadrature.h"

/**
 * Generate quadrature points and weights for unit triangle
 *
 * IN: order - quadrature order.
 *
 * OUT: 1 - quadrature struct.
 *
 * User has to free q.pw at the end.
 */
#undef __FUNCT__
#define __FUNCT__ "generate_quad_triangle"
PetscErrorCode generate_quad_triangle(quadrature_t *q, PetscInt order)
{
    PetscErrorCode ierr;
    PetscInt size;
    PetscReal *tmp;

    switch (order) {
        case 1:
            tmp = (PetscReal[QUAD_2D_1_LEN]){ QUAD_2D_1 };
            size = QUAD_2D_1_LEN;
            break;
        case 2:
            tmp = (PetscReal[QUAD_2D_2_LEN]){ QUAD_2D_2 };
            size = QUAD_2D_2_LEN;
            break;
        case 3:
            tmp = (PetscReal[QUAD_2D_3_LEN]){ QUAD_2D_3 };
            size = QUAD_2D_3_LEN;
            break;
        case 4:
            tmp = (PetscReal[QUAD_2D_4_LEN]){ QUAD_2D_4 };
            size = QUAD_2D_4_LEN;
            break;
        case 5:
            tmp = (PetscReal[QUAD_2D_5_LEN]){ QUAD_2D_5 };
            size = QUAD_2D_5_LEN;
            break;
        case 6:
            tmp = (PetscReal[QUAD_2D_6_LEN]){ QUAD_2D_6 };
            size = QUAD_2D_6_LEN;
            break;
        case 7:
            tmp = (PetscReal[QUAD_2D_7_LEN]){ QUAD_2D_7 };
            size = QUAD_2D_7_LEN;
            break;
        default:
            PetscErrorPrintf("Order %d not supported, highest order supported \
                    is %d defaulted to %d\n",
                             order, 8, 8);
        case 8:
            tmp = (PetscReal[QUAD_2D_8_LEN]){ QUAD_2D_8 };
            size = QUAD_2D_8_LEN;
            break;
    }

    ierr = PetscMalloc1(size, &q->pw);
    CHKERRQ(ierr);
    ierr = PetscMemcpy(q->pw, tmp, size * sizeof(PetscReal));
    CHKERRQ(ierr);
    q->order = order;
    q->size = size / 3;

    return ierr;
}


PetscErrorCode generate_quad_line(quadrature_t *q, PetscInt order)
{
    PetscInt size, ierr;
    PetscReal *tmp;

    switch(order) {
        case 2:
            size = QUAD_1D_2_LEN;
            tmp = (PetscReal[QUAD_1D_2_LEN]){ QUAD_1D_2 };
            break;
        case 3:
            size = QUAD_1D_3_LEN;
            tmp = (PetscReal[QUAD_1D_3_LEN]){ QUAD_1D_3 };
            break;
        case 4:
            size = QUAD_1D_4_LEN;
            tmp = (PetscReal[QUAD_1D_4_LEN]){ QUAD_1D_4 };
            break;
        case 5:
            size = QUAD_1D_5_LEN;
            tmp = (PetscReal[QUAD_1D_5_LEN]){ QUAD_1D_5 };
            break;
        case 6:
            size = QUAD_1D_6_LEN;
            tmp = (PetscReal[QUAD_1D_6_LEN]){ QUAD_1D_6 };
            break;
        case 7:
            size = QUAD_1D_7_LEN;
            tmp = (PetscReal[QUAD_1D_7_LEN]){ QUAD_1D_7 };
            break;
        case 8:
            size = QUAD_1D_8_LEN;
            tmp = (PetscReal[QUAD_1D_8_LEN]){ QUAD_1D_8 };
            break;
        case 9:
            size = QUAD_1D_9_LEN;
            tmp = (PetscReal[QUAD_1D_9_LEN]){ QUAD_1D_9 };
            break;
        case 10:
            size = QUAD_1D_10_LEN;
            tmp = (PetscReal[QUAD_1D_10_LEN]){ QUAD_1D_10 };
            break;
        case 11:
            size = QUAD_1D_11_LEN;
            tmp = (PetscReal[QUAD_1D_11_LEN]){ QUAD_1D_11 };
            break;
        case 12:
            size = QUAD_1D_12_LEN;
            tmp = (PetscReal[QUAD_1D_12_LEN]){ QUAD_1D_12 };
            break;
        case 13:
            size = QUAD_1D_13_LEN;
            tmp = (PetscReal[QUAD_1D_13_LEN]){ QUAD_1D_13 };
            break;
        case 14:
            size = QUAD_1D_14_LEN;
            tmp = (PetscReal[QUAD_1D_14_LEN]){ QUAD_1D_14 };
            break;
        case 15:
            size = QUAD_1D_15_LEN;
            tmp = (PetscReal[QUAD_1D_15_LEN]){ QUAD_1D_15 };
            break;
        default:
            fprintf(stderr, "Order %d unsupported, defaulting to 16.\n", order);
            order = 16;
        case 16:
            size = QUAD_1D_16_LEN;
            tmp = (PetscReal[QUAD_1D_16_LEN]){ QUAD_1D_16 };
            break;
    }

    ierr = PetscMalloc1(size, &q->pw);
    CHKERRQ(ierr);

    ierr = PetscMemcpy(q->pw, tmp, size * sizeof(PetscReal));
    CHKERRQ(ierr);
    q->size = size;
    q->order = order;

    return (0);
}

PetscErrorCode destroy_quadrature(quadrature_t *q)
{
    PetscFree(q->pw);

    return (0);
}
