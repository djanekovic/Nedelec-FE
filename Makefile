PETSC_DIR = /home/darko/FEM/petsc

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

src = $(wildcard ./src/*.c)
obj = $(src:.c=.o)

# Add your tests here
test_files = mesh

# Add object files and source files for compilation here
test_mesh_src = ./src/mesh.c ./tests/mesh_test.c
test_mesh_obj = $(test_mesh_src:.c=.o)

# Append previously defined object files here
test_obj = $(test_mesh_obj)

# Link with CMocka for tests
LDFLAGS_TEST = -lcmocka


.DEFAULT_GOAL := eddy
eddy: $(obj) chkopts
	@mkdir -p bin/
	-${CLINKER} -o bin/$@ $(obj) -g ${PETSC_LIB}

tests: $(test_files)
	@mkdir -p bin/tests

mesh: $(test_mesh_obj) chkopts
	-${CLINKER} -o bin/tests/$@ $(test_mesh_obj) -g ${PETSC_LIB} $(LDFLAGS_TEST)

clean::
	$(RM) $(obj) $(test_obj)
