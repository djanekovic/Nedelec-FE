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

    PetscErrorCode ierr = PetscInitialize(&argc, &argv, NULL, help);
    if (ierr) {
        return ierr;
    }

    ierr = handle_cli_options(&sctx); CHKERRQ(ierr);
    ierr = generate_mesh(&sctx, &dm); CHKERRQ(ierr);
    ierr = generate_quad(1, &q); CHKERRQ(ierr);
    ierr = nedelec_basis(q, &fspace); CHKERRQ(ierr);
    ierr = assemble_stiffness(dm, q, fspace, &stiffness); CHKERRQ(ierr);
    ierr = assemble_mass(dm, q, fspace, &mass); CHKERRQ(ierr);

    ierr = MatDestroy(&mass); CHKERRQ(ierr);
    ierr = MatDestroy(&stiffness); CHKERRQ(ierr);
    ierr = DMDestroy(&dm); CHKERRQ(ierr);

    ierr = PetscFree(fspace.cval); CHKERRQ(ierr);
    ierr = PetscFree(fspace.val); CHKERRQ(ierr);
    ierr = PetscFree(q.pw); CHKERRQ(ierr);

    PetscFinalize();

    return 0;
}
