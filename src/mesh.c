#include <petscdmplex.h>
#include "petscdmshell.h"
#include <petscmat.h>

#include "util.h"
#include "mesh.h"

static PetscErrorCode free_ctx(void **data);

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
/**
 * Note: you have to free signs
 */
PetscErrorCode generate_mesh(struct ctx *sctx, DM *dm) {
    PetscInt cstart, cend, vstart, vend, edgenum, estart, eend;
    DMLabel label;
    //TODO: merge mesh_ctx and global ctx
    struct mesh_ctx *mctx;
    PetscErrorCode ierr = PetscMalloc1(1, &mctx); CHKERRQ(ierr);

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
    DMPlexGetHeightStratum(*dm, 0, &estart, &eend);
    DMPlexGetHeightStratum(*dm, 2, &vstart, &vend);
    DMPlexGetConeSize(*dm, cstart, &edgenum);

    /* alloc signs matrix */
    ierr = PetscMalloc1(edgenum * cend, &mctx->signs); CHKERRQ(ierr);

    /*** Signs matrix generator ***/
    /* First we iterate over all cells in the mesh. For each cell we compute
     * list of edges. Then, we iterate over edges in edgelist.
     * Each edge we view as a line segment with two nodes at its end.
     * If node[0] > node[1] we set 1 in signs matrix, else we set zero.
     */
    for (PetscInt c = cstart; c < cend; c++) {
        const PetscInt *edgelist;
        ierr = DMPlexGetCone(*dm, c, &edgelist); CHKERRQ(ierr);
        for (PetscInt i = 0; i < edgenum; i++) {
            const PetscInt *nodes;
            DMPlexGetCone(*dm, edgelist[i], &nodes);
            if (nodes[0] < nodes[1]) {
                mctx->signs[(c - cstart) * edgenum + i] = 1;
            } else {
                mctx->signs[(c - cstart) * edgenum + i] = -1;
            }
        }
    }

    ierr = DMSetApplicationContext(*dm, mctx); CHKERRQ(ierr);
    ierr = DMSetApplicationContextDestroy(*dm, (PetscErrorCode (*)(void **data)) free_ctx);

    return (0);
}

static PetscErrorCode free_ctx(void **data)
{
    struct mesh_ctx *mctx = (struct mesh_ctx *) *data;
    PetscErrorCode ierr = PetscFree(mctx->signs); CHKERRQ(ierr);
    ierr = PetscFree(mctx); CHKERRQ(ierr);

    return ierr;
}
