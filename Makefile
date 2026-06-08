CFLAGS = -Drestrict=__restrict__ -O3 -DGRAPH_GENERATOR_MPI -DREUSE_CSR_FOR_VALIDATION
LDFLAGS = -lpthread

CUDA_HOME    ?= /usr/local/cuda
NVSHMEM_HOME ?= /opt/nvshmem
MPI_HOME     ?= /usr/lib/x86_64-linux-gnu/openmpi
NVCC         = $(CUDA_HOME)/bin/nvcc

BUILD_DIR = build
SRC_DIR = src

COMMON_SOURCES = \
	$(SRC_DIR)/csr_reference.c \
	$(SRC_DIR)/graph500_runner.c \
	$(SRC_DIR)/utils.c \
	$(SRC_DIR)/validate.c \
	$(SRC_DIR)/mpi_message.c \
	$(SRC_DIR)/graph_generator.cu

HEADERS = \
	$(SRC_DIR)/common.h \
	$(SRC_DIR)/csr_reference.h \
	$(SRC_DIR)/bitmap_reference.h \
	$(SRC_DIR)/mpi_message.h

BINARY = $(BUILD_DIR)/graph500_runner

.PHONY: all clean

all: $(BINARY)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BINARY): $(SRC_DIR)/traversal.cu $(COMMON_SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(NVCC) $(CFLAGS) -DSSSP -rdc=true \
		-I$(NVSHMEM_HOME)/include -I$(MPI_HOME)/include \
	-o $@ \
	$(SRC_DIR)/traversal.cu $(COMMON_SOURCES) \
	$(LDFLAGS) -L$(NVSHMEM_HOME)/lib -L$(MPI_HOME)/lib -lnvshmem_host -lnvshmem_device -lmpi -lm

clean:
	rm -rf $(BUILD_DIR)
