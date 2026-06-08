#include "mpi_message.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct pending_send {
	MPI_Request request;
	void* buffer;
	int pe;
	int tag;
} pending_send;

static pending_send* pending;
static int pending_count;
static int pending_capacity;
static int* send_counts;
static int send_counts_size;

static void ensure_state(void) {
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	if (send_counts_size == size) return;
	free(send_counts);
	send_counts = (int*)calloc((size_t)size, sizeof(int));
	if (send_counts == NULL && size != 0) {
		fprintf(stderr, "MPI message helper: failed to allocate send counts\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	send_counts_size = size;
}

void mpi_message_send(const void* data, int sz, int pe, int tag) {
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	ensure_state();
	if (sz < 0 || pe < 0 || pe >= send_counts_size) {
		fprintf(stderr, "Rank %d: invalid MPI message send sz=%d pe=%d\n", rank, sz, pe);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (pending_count == pending_capacity) {
		int new_capacity = pending_capacity == 0 ? 1024 : pending_capacity * 2;
		pending_send* new_pending = (pending_send*)realloc(pending, (size_t)new_capacity * sizeof(pending_send));
		if (new_pending == NULL) {
			fprintf(stderr, "Rank %d: failed to grow MPI message send list\n", rank);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		pending = new_pending;
		pending_capacity = new_capacity;
	}

	void* copy = NULL;
	if (sz > 0) {
		copy = malloc((size_t)sz);
		if (copy == NULL) {
			fprintf(stderr, "Rank %d: failed to allocate MPI message buffer\n", rank);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		memcpy(copy, data, (size_t)sz);
	}
	MPI_Isend(copy, sz, MPI_BYTE, pe, tag, MPI_COMM_WORLD, &pending[pending_count].request);
	pending[pending_count].buffer = copy;
	pending[pending_count].pe = pe;
	pending[pending_count].tag = tag;
	++pending_count;
	++send_counts[pe];
}

void mpi_message_exchange(int tag, mpi_message_handler_t handler) {
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	ensure_state();
	int* recv_counts = (int*)malloc((size_t)send_counts_size * sizeof(int));
	if (recv_counts == NULL && send_counts_size != 0) {
		fprintf(stderr, "Rank %d: failed to allocate MPI message receive counts\n", rank);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);

	int expected = 0;
	for (int i = 0; i < send_counts_size; ++i) expected += recv_counts[i];
	for (int i = 0; i < expected; ++i) {
		MPI_Status status;
		MPI_Probe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);
		int sz = 0;
		MPI_Get_count(&status, MPI_BYTE, &sz);
		void* buffer = sz > 0 ? malloc((size_t)sz) : NULL;
		if (sz > 0 && buffer == NULL) {
			fprintf(stderr, "Rank %d: failed to allocate MPI message receive buffer\n", rank);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		MPI_Recv(buffer, sz, MPI_BYTE, status.MPI_SOURCE, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		handler(status.MPI_SOURCE, buffer, sz);
		free(buffer);
	}

	for (int i = 0; i < pending_count; ++i) {
		MPI_Wait(&pending[i].request, MPI_STATUS_IGNORE);
		free(pending[i].buffer);
	}
	pending_count = 0;
	memset(send_counts, 0, (size_t)send_counts_size * sizeof(int));
	free(recv_counts);
	MPI_Barrier(MPI_COMM_WORLD);
}

void mpi_message_finalize(void) {
	for (int i = 0; i < pending_count; ++i) {
		MPI_Wait(&pending[i].request, MPI_STATUS_IGNORE);
		free(pending[i].buffer);
	}
	free(pending);
	pending = NULL;
	pending_count = 0;
	pending_capacity = 0;
	free(send_counts);
	send_counts = NULL;
	send_counts_size = 0;
}
