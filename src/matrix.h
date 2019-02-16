#ifndef MATRIX_H
#define MATRIX_H

PetscErrorCode assemble_stiffness(DM dm, struct ctx sctx, Mat signs, Mat *stiffness);
PetscErrorCode assemble_mass(DM dm, struct ctx sctx, Mat signs, Mat *mass);

#endif /* MATRIX_H */
