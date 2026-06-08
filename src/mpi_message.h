#ifndef MPI_MESSAGE_H
#define MPI_MESSAGE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*mpi_message_handler_t)(int from, void* data, int sz);

void mpi_message_send(const void* data, int sz, int pe, int tag);
void mpi_message_exchange(int tag, mpi_message_handler_t handler);
void mpi_message_finalize(void);

#ifdef __cplusplus
}
#endif

#endif
