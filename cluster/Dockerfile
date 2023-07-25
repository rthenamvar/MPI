# Use a base Python image with the necessary MPI libraries
FROM python:3.10

# Install MPI libraries
RUN apt-get update && apt-get install -y \
    libopenmpi-dev \
    openmpi-bin    
RUN pip install mpi4py

# Install NumPy
RUN pip install numpy

RUN mkdir /project

#### CREATE WORKING DIRECTORY FOR USER ####
ARG WORKDIR=/project
ENV WORKDIR ${WORKDIR}
RUN mkdir ${WORKDIR}
RUN chown -R ${USER}:${USER} ${WORKDIR}

WORKDIR ${WORKDIR}

CMD ["/bin/bash"]