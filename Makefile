CFLAGS = -Drestrict=__restrict__ -O3 -DGRAPH_GENERATOR_MPI -DREUSE_CSR_FOR_VALIDATION
LDFLAGS = -lpthread

CUDA_HOME    ?= /usr/local/cuda
NVSHMEM_HOME ?= /opt/nvshmem
MPI_HOME     ?= /usr/lib/x86_64-linux-gnu/openmpi
NVCC         = $(CUDA_HOME)/bin/nvcc

BUILD_DIR = build
SRC_DIR = src-refactor
OLD_SRC_DIR = src

COMMON_SOURCES = \
	$(SRC_DIR)/graph_generation.cu \
	$(SRC_DIR)/base.cu \
	$(SRC_DIR)/mrg_transitions.cu \
	$(SRC_DIR)/main.cu

OLD_SOURCES = \
	$(OLD_SRC_DIR)/graph500_runner.c \
	$(OLD_SRC_DIR)/graph_generator.cu \
	$(OLD_SRC_DIR)/traversal.cu \
	$(OLD_SRC_DIR)/utils.c \
	$(OLD_SRC_DIR)/validate.c \
	$(OLD_SRC_DIR)/csr_reference.c \
	$(OLD_SRC_DIR)/mpi_message.c

HEADERS = \
	$(SRC_DIR)/graph_generation.h \
	$(SRC_DIR)/base.h

BINARY = $(BUILD_DIR)/graph500_runner
OLD_BINARY = $(BUILD_DIR)/graph500_runner_old

.PHONY: all old clean

all: $(BINARY)

old: $(OLD_BINARY)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BINARY): $(COMMON_SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(NVCC) $(CFLAGS) -DSSSP -rdc=true \
		-I$(NVSHMEM_HOME)/include -I$(MPI_HOME)/include \
	-o $@ \
	$(COMMON_SOURCES) \
	$(LDFLAGS) -L$(NVSHMEM_HOME)/lib -L$(MPI_HOME)/lib -lnvshmem_host -lnvshmem_device -lmpi -lm

$(OLD_BINARY): $(OLD_SOURCES) $(OLD_SRC_DIR)/common.h | $(BUILD_DIR)
	$(NVCC) $(CFLAGS) -DSSSP -DUSER_SETTINGS -rdc=true \
		-I$(NVSHMEM_HOME)/include -I$(MPI_HOME)/include \
	-o $@ \
	$(OLD_SOURCES) \
	$(LDFLAGS) -L$(NVSHMEM_HOME)/lib -L$(MPI_HOME)/lib -lnvshmem_host -lnvshmem_device -lmpi -lm

clean:
	rm -rf $(BUILD_DIR)
