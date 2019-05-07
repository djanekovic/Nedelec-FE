#include <petsc.h>

#include "quadrature.h"
/**
 * Generate quadrature points and weights
 *
 * IN: order - quadrature order.
 *
 * OUT: 1 - quadrature struct.
 *
 * User has to free q.pw at the end.
 */
#undef __FUNCT__
#define __FUNCT__ "generate_quad"
PetscErrorCode generate_quad(struct quadrature *q, PetscInt order)
{
	PetscErrorCode ierr;
	PetscInt size;
	PetscScalar *tmp;

	switch (order) {
		case 1:
			tmp = (PetscScalar[QUAD_2D_1_LEN]){ QUAD_2D_1 };
			size = QUAD_2D_1_LEN;
			break;
		case 2:
			tmp = (PetscScalar[QUAD_2D_2_LEN]){ QUAD_2D_2 };
			size = QUAD_2D_2_LEN;
			break;
		case 3:
			tmp = (PetscScalar[QUAD_2D_3_LEN]){ QUAD_2D_3 };
			size = QUAD_2D_3_LEN;
			break;
		case 4:
			tmp = (PetscScalar[QUAD_2D_4_LEN]){ QUAD_2D_4 };
			size = QUAD_2D_4_LEN;
			break;
		case 5:
			tmp = (PetscScalar[QUAD_2D_5_LEN]){ QUAD_2D_5 };
			size = QUAD_2D_5_LEN;
			break;
		case 6:
			tmp = (PetscScalar[QUAD_2D_6_LEN]){ QUAD_2D_6 };
			size = QUAD_2D_6_LEN;
			break;
		case 7:
			tmp = (PetscScalar[QUAD_2D_7_LEN]){ QUAD_2D_7 };
			size = QUAD_2D_7_LEN;
			break;
		default:
			PetscErrorPrintf("Order %d not supported, highest order supported \
                    is %d defaulted to %d\n",
							 order, 8, 8);
		case 8:
			tmp = (PetscScalar[QUAD_2D_8_LEN]){ QUAD_2D_8 };
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
