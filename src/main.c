#include <petsc.h>

#include "matrix.h"
#include "mesh.h"
#include "nedelec.h"
#include "quadrature.h"
#include "util.h"

static const char *help = "Solving eddy currents problems";

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
    struct ctx sctx;
    struct function_space fspace;
    PetscInt estart, eend, *nnz;
    DM dm;
    Mat A;
    Vec load, x;
    KSP ksp;

    PetscErrorCode ierr = PetscInitialize(&argc, &argv, NULL, help);
    if (ierr) {
        return ierr;
    }

    handle_cli_options(&sctx);

    generate_mesh(&sctx, &nnz, &dm);

    ierr = DMPlexGetHeightStratum(dm, 1, &estart, &eend);
    CHKERRQ(ierr);
    PetscInt nedges = eend - estart;

    // TODO: change for parallel
    ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD, nedges, nedges, 0, nnz, &A);
    CHKERRQ(ierr);
    ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, nedges, &load);
    CHKERRQ(ierr);

    nedelec_basis(&fspace, 3);

    // in one function assemble all matrices
    assemble_system(dm, fspace, A, load);

    ierr = VecDuplicate(load, &x);
    CHKERRQ(ierr);

    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);
    CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, A, A);
    ierr = KSPSolve(ksp, load, x);

    MatView(A, PETSC_VIEWER_STDOUT_WORLD);
    VecView(load, PETSC_VIEWER_STDOUT_WORLD);
    VecView(x, PETSC_VIEWER_STDOUT_WORLD);

    ierr = MatDestroy(&A);
    CHKERRQ(ierr);

    ierr = DMDestroy(&dm);
    CHKERRQ(ierr);

    ierr = PetscFree(fspace.cval);
    CHKERRQ(ierr);
    ierr = PetscFree(fspace.val);
    CHKERRQ(ierr);
    ierr = PetscFree(fspace.q.pw);

    PetscFinalize();

    return 0;
}
