#ifndef MATRIX_H
#define MATRIX_H

PetscErrorCode assemble_stiffness(DM dm, struct quadrature q, struct function_space fs, Mat *stiffness);
PetscErrorCode assemble_mass(DM dm, struct quadrature q, struct function_space fs, Mat *mass);

#endif /* MATRIX_H */
