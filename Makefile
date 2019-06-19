PETSC_DIR = /home/darko/FEM/petsc

include ${PETSC_DIR}/lib/petsc/conf/variables

LIB_NAME = libffn
LIB_FILENAME = $(LIB_NAME).so
SRC_DIR = src/
BUILD_DIR = $(realpath .)/build
INCLUDE_DIR = $(realpath include/)
LIB = $(BUILD_DIR)/$(LIB_FILENAME)

WARNINGS = -Wjump-misses-init -Wall -Wextra -Wlogical-op -Wdouble-promotion

CFLAGS = $(CCPPFLAGS) -I$(INCLUDE_DIR) $(WARNINGS) -g
export

LDFLAGS = -L$(BUILD_DIR)/ -Wl,-rpath,$(BUILD_DIR)/ $(PETSC_LIB) -lffn
NPD = --no-print-directory

all: $(LIB_NAME)

.PHONY: clean cleanall $(LIB_NAME)
$(LIB_NAME): build $(LIB)
	@echo "Code is compiled!"
	@echo "You can link to it by using -L$(BUILD_DIR) -lffn"
	@echo "Don't forget to link against PETSc also"

eddy: main.o $(LIB_NAME)
	@echo "LD $<"
	@$(PCC) $< -o $@ $(CFLAGS) $(LDFLAGS)

main.o: main.c
	@echo "CC $<"
	@$(PCC) -c $< -o $@ $(CFLAGS)

build:
	@echo "Creating $(BUILD_DIR) folder"
	mkdir -p $(BUILD_DIR)

$(LIB):
	$(MAKE) -C $(SRC_DIR) $(LIB_FILENAME)

clean:
	rm -f $(BUILD_DIR)/libffn.so main.o eddy
	$(MAKE) $(NPD) -C $(SRC_DIR) clean

cleanall:
	rm -rf $(BUILD_DIR) main.o eddy
	$(MAKE) -C $(SRC_DIR) clean
