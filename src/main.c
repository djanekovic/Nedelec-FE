#include <petsc.h>

#include "util.h"
#include "mesh.h"
#include "matrix.h"

static const char *help = "Solving eddy currents problems";


#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc, char **argv)
{
    struct ctx sctx;
    DM dm;
    Mat signs;
    Mat mass;
    Mat stiffness;
    Mat pw;

    PetscErrorCode ierr = PetscInitialize(&argc, &argv, NULL, help);
    if (ierr) {
        return ierr;
    }

    handle_cli_options(&sctx);
    generate_mesh(&sctx, &dm, &signs);
    quad_pw(3, &pw);
    MatView(pw, PETSC_VIEWER_STDOUT_WORLD);
    assemble_stiffness(dm, sctx, signs, &stiffness);
    assemble_mass(dm, sctx, signs, &mass);
    PetscFinalize();

    return 0;
}
