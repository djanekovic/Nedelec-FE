#include <petscdmplex.h>
#include <petscmat.h>
#include "petscdmshell.h"

#include "util.h"

static PetscErrorCode mark_boundary_faces(DM dm);
static PetscErrorCode debug_print(DM dm);
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
PetscErrorCode generate_mesh(struct ctx *sctx, PetscInt **nnz, DM *dm)
{
    PetscErrorCode ierr;
    PetscInt cstart, cend, vstart, vend, edgenum, estart, eend;

    PetscLogEventBegin(sctx->mesh_generation, 0, 0, 0, 0);
    const PetscInt faces[2] = {1000, 1000};

    /**
     * Create box mesh
     *
     * MPI commutator - PETSC_COMM_WORLD;
     * dim - sctx->dim;
     * simplex - PETSC_TRUE (for tensor cell, PETSC_FALSE)
     * faces - number of faces per dimension TODO add to solver ctx
     * lower left corner - (0, 0, 0) if NULL
     * upper_right_corner - (1, 1, 1) if NULL
     * periodicity - DM_BOUNDARY_NONE
     * interpolated - (vertices, edges, faces)
     */
    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, sctx->dim, PETSC_TRUE, faces,
                               NULL, NULL, NULL, PETSC_TRUE, dm);
    CHKERRQ(ierr);

    ierr = mark_boundary_faces(*dm);
    CHKERRQ(ierr);

    /**
     * Use this for boundary conditions
     *
    DMSetFromOptions(*dm);
    DMView(*dm, PETSC_VIEWER_STDOUT_WORLD);
    IS points;
    DMGetStratumIS(*dm, "Face Sets", 1, &points);
    ISView(points, PETSC_VIEWER_STDOUT_WORLD);
    */

    DMPlexGetHeightStratum(*dm, 0, &cstart, &cend);
    DMPlexGetHeightStratum(*dm, 1, &estart, &eend);
    DMPlexGetHeightStratum(*dm, 2, &vstart, &vend);
    DMPlexGetConeSize(*dm, cstart, &edgenum);
    sctx->cstart = cstart;
    sctx->cend = cend;
    sctx->estart = estart;
    sctx->eend = eend;
    sctx->vstart = vstart;
    sctx->vend = vend;
    sctx->nelems = (cend - cstart);

    /* alloc signs matrix */
    ierr = PetscMalloc1(edgenum * (cend - cstart), &sctx->signs);
    CHKERRQ(ierr);
    ierr = PetscMalloc1(eend - estart, nnz);
    CHKERRQ(ierr);
    ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD, eend - estart, vend - vstart, 2,
                           NULL, &sctx->G);
    CHKERRQ(ierr);

    /**
     * Signs matrix generator
     *
     * First we iterate over all cells in the mesh. For each cell we compute
     * list of edges. Then, we iterate over edges in edgelist.
     * Each edge we view as a line segment with two nodes at its end.
     * If node[0] > node[1] we set 1 in signs matrix, else we set zero.
     */
    for (PetscInt c = cstart; c < cend; c++) {
        PetscInt offset = (c - cstart) * 3;
        const PetscInt *edgelist;
        const PetscInt *orient;
        ierr = DMPlexGetCone(*dm, c, &edgelist);
        CHKERRQ(ierr);
        ierr = DMPlexGetConeOrientation(*dm, c, &orient);
        CHKERRQ(ierr);
        for (PetscInt i = 0; i < edgenum; i++) {
            const PetscInt *nodes;
            ierr = DMPlexGetCone(*dm, edgelist[i], &nodes);
            CHKERRQ(ierr);

            /* signs matrix generator */
            if (nodes[0] < nodes[1]) {
                sctx->signs[offset + i] = (orient[i] >= 0) ? 1 : -1;
            } else {
                sctx->signs[offset + i] = (orient[i] >= 0) ? -1 : 1;
            }

            /* hypre discrete gradient */
            ierr = MatSetValue(sctx->G, edgelist[i] - estart, nodes[0] - vstart,
                               -1, INSERT_VALUES);
            CHKERRQ(ierr);
            ierr = MatSetValue(sctx->G, edgelist[i] - estart, nodes[1] - vstart, 1,
                        INSERT_VALUES);
            CHKERRQ(ierr);

            /* preallocation only for 2D */
            PetscInt edge_neighbours;
            ierr = DMPlexGetSupportSize(*dm, edgelist[i], &edge_neighbours);
            CHKERRQ(ierr);
            if (edge_neighbours == 1) {
                //this is edge on the boundary, this row will have 3 entries
                (*nnz)[edgelist[i]-estart] = 3;
            } else {
                //this is inner edge
                (*nnz)[edgelist[i]-estart] = 5;
            }
        }
    }

    MatAssemblyBegin(sctx->G, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(sctx->G, MAT_FINAL_ASSEMBLY);

    PetscLogEventEnd(sctx->mesh_generation, 0, 0, 0, 0);

    ierr = DMSetApplicationContext(*dm, sctx);
    CHKERRQ(ierr);
    ierr = DMSetApplicationContextDestroy(
        *dm, (PetscErrorCode(*)(void **data)) free_ctx);

    return (0);
}

#undef __FUNCT__
#define __FUNCT__ "mark_boundary_faces"
static PetscErrorCode mark_boundary_faces(DM dm)
{
    DMLabel label;

    PetscErrorCode ierr = DMCreateLabel(dm, "boundary");
    CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "boundary", &label);
    CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(dm, 1, label);
    CHKERRQ(ierr);
    ierr = DMPlexLabelComplete(dm, label);
    CHKERRQ(ierr);

    return ierr;
}

#undef __FUNCT__
#define __FUNCT__ "free_ctx"
static PetscErrorCode free_ctx(void **data)
{
    PetscErrorCode ierr;
    struct ctx *sctx = (struct ctx *) *data;

    ierr = MatDestroy(&sctx->G);
    CHKERRQ(ierr);
    ierr = PetscFree(sctx->signs);
    CHKERRQ(ierr);

    return ierr;
}

static PetscErrorCode debug_print(DM dm)
{
    Vec c;
    int cstart, cend, vstart, vend, estart, eend;

    DMView(dm, PETSC_VIEWER_STDOUT_WORLD);
    DMGetCoordinates(dm, &c);
    VecView(c, PETSC_VIEWER_STDOUT_WORLD);

    DMPlexGetHeightStratum(dm, 0, &cstart, &cend);
    DMPlexGetHeightStratum(dm, 1, &estart, &eend);
    DMPlexGetHeightStratum(dm, 2, &vstart, &vend);

    /*
    for (int c = cstart; c < cend; c++) {
        PetscInt nclosure, *closure;
        DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &nclosure, &closure);
        for (PetscInt i = 0; i < nclosure; i++) {
            const PetscInt p = closure[2*i];
            if (p >= vstart && p < vend) {
                printf("%d %d\n", c, p);
            }
        }
        DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &nclosure, &closure);
    }
    */

    for (int c = cstart; c < cend; c++) {
        const PetscInt *edges;
        const PetscInt *orient;
        DMPlexGetCone(dm, c, &edges);
        DMPlexGetConeOrientation(dm, c, &orient);
        for (int i = 0; i < 3; i++) {
            const PetscInt *nodes;
            const PetscInt *norient;
            DMPlexGetCone(dm, edges[i], &nodes);
            DMPlexGetConeOrientation(dm, edges[i], &norient);
            printf("%d %d %d: %d - %d\n", c, edges[i], orient[i], nodes[0],
                   nodes[1]);
        }
    }
    return 0;
}
