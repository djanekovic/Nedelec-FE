PETSC_DIR = /home/darko/FEM/petsc

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

src = $(wildcard ./src/*.c)
obj = $(src:.c=.o)

.DEFAULT_GOAL := eddy

solve: eddy_currents.o chkopts
	-${CLINKER} eddy_currents.o -g ${PETSC_LIB}
	${RM} eddy_currents.o

eddy: $(obj) chkopts
	-${CLINKER} -o bin/$@ $(obj) -g ${PETSC_LIB}
	${RM} $(obj)
