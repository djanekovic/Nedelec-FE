#include <petsc.h>

#include "util.h"
#include "quadrature.h"
#include "nedelec.h"
#include "mesh.h"
#include "matrix.h"

static const char *help = "Solving eddy currents problems";


#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc, char **argv)
{
    struct ctx sctx;
    struct function_space fspace;
    struct quadrature q;
    DM dm;
    Mat mass, stiffness;
    Vec load, x;
    KSP ksp; PC pc;

    PetscErrorCode ierr = PetscInitialize(&argc, &argv, NULL, help);
    if (ierr) {
        return ierr;
    }

    ierr = handle_cli_options(&sctx); CHKERRQ(ierr);
    ierr = generate_mesh(&sctx, &dm); CHKERRQ(ierr);
    ierr = generate_quad(1, &q); CHKERRQ(ierr);
    ierr = nedelec_basis(q, &fspace); CHKERRQ(ierr);

    //in one function assemble all matrices
    ierr = assemble_stiffness(dm, q, fspace, &stiffness); CHKERRQ(ierr);
    MatView(stiffness, PETSC_VIEWER_STDOUT_WORLD);
    ierr = assemble_mass(dm, q, fspace, &mass); CHKERRQ(ierr);
    MatView(mass, PETSC_VIEWER_STDOUT_WORLD);
    ierr = MatAXPY(stiffness, 1, mass, SAME_NONZERO_PATTERN); CHKERRQ(ierr);

    ierr = assemble_load(dm, q, fspace, &load); CHKERRQ(ierr);
    ierr = VecDuplicate(load, &x); CHKERRQ(ierr);

    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, stiffness, stiffness);
    ierr = KSPSolve(ksp, load, x);
    VecView(x, PETSC_VIEWER_STDOUT_WORLD);

    ierr = MatDestroy(&mass); CHKERRQ(ierr);
    ierr = MatDestroy(&stiffness); CHKERRQ(ierr);

    ierr = DMDestroy(&dm); CHKERRQ(ierr);

    ierr = PetscFree(fspace.cval); CHKERRQ(ierr);
    ierr = PetscFree(fspace.val); CHKERRQ(ierr);
    ierr = PetscFree(q.pw); CHKERRQ(ierr);

    PetscFinalize();

    return 0;
}
