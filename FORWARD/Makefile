#! bin/sh

MPICH2_HOME=/home/Software/Intel/impi/5.0.1.035/intel64
CUDA_HOME=/usr/local/cuda-9.0

NVCC = $(CUDA_HOME)/bin/nvcc
MPICC = $(MPICH2_HOME)/bin/mpicc

INC = -I$(MPICH2_HOME)/include -I$(CUDA_HOME)/include

LIB = -L$(MPICH2_HOME)/lib -L$(CUDA_HOME)/lib64
LDFLAGS = -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX -lmpi -lrt -lpthread -lfftw3 -lcudart -lcurand -lcufft

CFILES = acousticobs_moffsetC.cpp
CUFILES = acousticobs_moffset.cu
EXECNAME = obs_moffset

all:
	$(NVCC) -w -v -o $(EXECNAME) $(CFILES) $(CUFILES) $(INC) $(LIB) $(LDFLAGS)
	rm -f *.o

