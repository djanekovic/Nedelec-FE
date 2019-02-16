#include <petscdmplex.h>
#include <petscmat.h>

#include "util.h"
#include "mesh.h"

/**
 * Generate mesh depending with options specified earlier
 *
 * input:
 *  - solver ctx struct
 *
 * output:
 *  - mesh data struct
 */

#undef __FUNCT__
#define __FUNCT__ "generate_mesh"
PetscErrorCode generate_mesh(struct ctx *sctx, DM *dm, Mat *signs) {
    PetscInt cstart, cend, vstart, vend;
    DMLabel label;
    PetscErrorCode ierr;

    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, sctx->dim, PETSC_TRUE, NULL, NULL, NULL, NULL, PETSC_TRUE, dm);
    CHKERRQ(ierr);

    ierr = DMSetFromOptions(*dm); CHKERRQ(ierr);
    ierr = DMViewFromOptions(*dm, NULL, "-dm_view"); CHKERRQ(ierr);

    /* mark boundary faces */
    DMCreateLabel(*dm, "boundary");
    DMGetLabel(*dm, "boundary", &label);
    DMPlexMarkBoundaryFaces(*dm, 1, label);
    DMPlexLabelComplete(*dm, label);

    DMPlexGetHeightStratum(*dm, 0, &cstart, &cend);
    DMPlexGetHeightStratum(*dm, 2, &vstart, &vend);
    cend -= cstart;
    sctx->nelems = cend;

    /* alloc matrix */
    ierr = MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, cend, 3, NULL, signs);
    CHKERRQ(ierr);

    /* generate signs matrix */
    for (PetscInt c = 0; c < cend; c++) {
        PetscInt nclosure;
        PetscInt nodelist[3];
        PetscInt *closure = NULL;
        DMPlexGetTransitiveClosure(*dm, c+cstart, PETSC_TRUE, &nclosure, &closure);
        for (PetscInt i = 0, n = 0; i < nclosure; i++) {
            const PetscInt p = closure[2*i];
            if (p >= vstart && p < vend) {
                nodelist[n++] = p;
            }
        }
        for (PetscInt i = 0; i < 3; i++) {
            if (nodelist[i] < nodelist[(i+1) % 3]) {
                ierr = MatSetValue(*signs, c, i, 1, INSERT_VALUES);
                CHKERRQ(ierr);
            } else {
                ierr = MatSetValue(*signs, c, i, -1, INSERT_VALUES);
                CHKERRQ(ierr);
            }
        }
    }
    MatAssemblyBegin(*signs, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*signs, MAT_FINAL_ASSEMBLY);

//    for (PetscInt c = cstart; c < cend; c++) {
//        ierr = DMPlexComputeCellGeometryAffineFEM(*dm, c, &v0, &J, &invJ, &detJ);
//        CHKERRQ(ierr);
//        PetscInt nclosure;
//        PetscInt *closure = NULL;
//        DMPlexGetTransitiveClosure(*dm, c, PETSC_TRUE, &nclosure, &closure);
//        for (PetscInt i = 0; i < nclosure; i++) {
//            const PetscInt p = closure[2*i];
//            if (p >= vstart && p < vend) {
//                PetscPrintf(PETSC_COMM_WORLD, "cell %d: node %d\n", c, p);
//            }
//            /* edges */
//            if (p >= estart && p < eend) {
//                PetscInt value;
//                PetscPrintf(PETSC_COMM_WORLD, "cell %d: edge %d orient %d\n", c, p, closure[2*i+1]);
//                const PetscInt *cone;
//                DMPlexGetCone(*dm, p, &cone);
//                PetscPrintf(PETSC_COMM_WORLD, "\t %d - %d - %d\n", cone[0], p, cone[1]);
//                DMGetLabelValue(*dm, "boundary", p, &value);
//                PetscPrintf(PETSC_COMM_WORLD, "\t edge %d has value %d\n", p, value);
//            }
//        }
//    }

    return (0);
}
