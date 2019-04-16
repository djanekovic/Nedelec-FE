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
    struct quadrature q;
    DM dm;
    Mat A;
    Vec load, x;
    KSP ksp;
    PC pc;

    PetscErrorCode ierr = PetscInitialize(&argc, &argv, NULL, help);
    if (ierr) {
        return ierr;
    }

    ierr = handle_cli_options(&sctx);
    CHKERRQ(ierr);
    ierr = generate_mesh(&sctx, &dm);
    CHKERRQ(ierr);

    // TODO: change for parallel
    ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD, sctx.eend - sctx.estart,
                           sctx.eend - sctx.estart, 0, NULL, &A);
    CHKERRQ(ierr);
    // TODO: remove after good preallocation
    ierr = MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    CHKERRQ(ierr);
    ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, sctx.eend - sctx.estart,
                        &load);

    ierr = generate_quad(1, &q);
    CHKERRQ(ierr);
    ierr = nedelec_basis(q, &fspace);
    CHKERRQ(ierr);

    // in one function assemble all matrices
    ierr = assemble_system(dm, q, fspace, A, load);
    CHKERRQ(ierr);

    ierr = VecDuplicate(load, &x);
    CHKERRQ(ierr);

    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);
    CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, A, A);
    ierr = KSPSolve(ksp, load, x);
    VecView(x, PETSC_VIEWER_STDOUT_WORLD);

    ierr = MatDestroy(&A);
    CHKERRQ(ierr);

    ierr = DMDestroy(&dm);
    CHKERRQ(ierr);

    ierr = PetscFree(fspace.cval);
    CHKERRQ(ierr);
    ierr = PetscFree(fspace.val);
    CHKERRQ(ierr);
    ierr = PetscFree(q.pw);
    CHKERRQ(ierr);

    PetscFinalize();

    return 0;
}
