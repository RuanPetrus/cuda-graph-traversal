CFLAGS = -Drestrict=__restrict__ -O3 -DGRAPH_GENERATOR_MPI -DREUSE_CSR_FOR_VALIDATION
LDFLAGS = -lpthread

CUDA_HOME    ?= /usr/local/cuda
NVSHMEM_HOME ?= /opt/nvshmem
MPI_HOME     ?= /usr/lib/x86_64-linux-gnu/openmpi
NVCC         = $(CUDA_HOME)/bin/nvcc

BUILD_DIR = build
SRC_DIR = src
EXAMPLES_DIR = examples

LIBRARY_SOURCES = \
	$(SRC_DIR)/graph_generation.cu \
	$(SRC_DIR)/base.cu \
	$(SRC_DIR)/traversal.cu \
	$(SRC_DIR)/visualization.cu \
	$(SRC_DIR)/worker.cu

RUNNER_SOURCES = $(LIBRARY_SOURCES) $(EXAMPLES_DIR)/graph500_runner.cu

HEADERS = \
	$(SRC_DIR)/graph_generation.h \
	$(SRC_DIR)/traversal.h \
	$(SRC_DIR)/visualization.h \
	$(SRC_DIR)/worker.h \
	$(SRC_DIR)/base.h

BINARY = $(BUILD_DIR)/graph500_runner

.PHONY: all dots-svg clean-dots-svg clean

all: $(BINARY)

dots-svg:
	@command -v dot >/dev/null 2>&1 || { echo "Graphviz 'dot' not found. Install graphviz to build SVGs."; exit 1; }
	@for file in dot/*.dot; do \
		[ -e "$$file" ] || { echo "No dot/*.dot files found"; exit 1; }; \
		dot -Tsvg "$$file" -o "$${file%.dot}.svg"; \
	done

clean-dots-svg:
	rm -f dot/*.svg

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BINARY): $(RUNNER_SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(NVCC) $(CFLAGS) -DSSSP -rdc=true \
		-I$(SRC_DIR) -I$(NVSHMEM_HOME)/include -I$(MPI_HOME)/include \
	-o $@ \
	$(RUNNER_SOURCES) \
	$(LDFLAGS) -L$(NVSHMEM_HOME)/lib -L$(MPI_HOME)/lib -lnvshmem_host -lnvshmem_device -lmpi -lm

clean:
	rm -rf $(BUILD_DIR)
