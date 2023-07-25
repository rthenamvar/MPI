from mpi4py import MPI
import numpy as np

def server():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size <= 1:
        print("This program requires at least 2 processes: 1 server and 1 or more clients.")
        return

    num_clients = size - 1

    for i in range (3):

        data_sum = 0

        for _ in range(num_clients):
            data = np.empty(5, dtype=np.float64)  # Array of size 5 (you can adjust the size as needed)
            comm.Recv(data, source=MPI.ANY_SOURCE, tag=1)
            data_sum += data

        data_avg = data_sum / num_clients

        for client_rank in range(1, size):
            comm.Send(data_avg, dest=client_rank, tag=2)

def client():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    for i in range (3):
        if i==0:
            data = np.random.rand(5)  # Generating an array of size 5 (you can adjust the size as needed)

            print(f" iter {i} Client {rank}: Generated array: {data}")

            comm.Send(data, dest=0, tag=1)

            data_avg = np.empty(5, dtype=np.float64)  # Placeholder for the average array
            comm.Recv(data_avg, source=0, tag=2)

            print(f"iter {i} Client {rank}: Received average array: {data_avg}")
        else:

            data = data_avg*rank  # Generating an array of size 5 (you can adjust the size as needed)

            print(f"ter {i} Client {rank}: Generated array: {data}")

            comm.Send(data, dest=0, tag=1)

            data_avg = np.empty(5, dtype=np.float64)  # Placeholder for the average array
            comm.Recv(data_avg, source=0, tag=2)

            print(f"ter {i} Client {rank}: Received average array: {data_avg}")


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        server()
    else:
        client()
