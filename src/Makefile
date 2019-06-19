obj = function.o matrix.o mesh.o nedelec.o quadrature.o util.o
output_obj = $(addprefix $(BUILD_DIR)/, $(obj))


# if called from here go one up, get env variables and come back down
all:
	make -C ../ all

$(LIB_FILENAME): $(output_obj)
	$(CLINKER) -shared $(CFLAGS) $(output_obj) -o $(LIB) $(PETSC_LIB)

$(BUILD_DIR)/%.o: %.c
	$(PCC) $(CFLAGS) -c -fPIC $< -o $@

clean:
	rm -f $(output_obj)
