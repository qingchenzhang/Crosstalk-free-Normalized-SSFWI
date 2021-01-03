# Crosstalk-free-Normalized-SSFWI
### Multi-source algorithms can increase the efficiency of full waveform inversion (FWI) dramatically through reducing the number of times of wavefield simulations. However, the multiple sources induce the crosstalk artifacts, which severely contaminate the inversion results. To solve this problem, we provide a crosstalk-free multi-source algorithm with the normalized seismic data. Sine harmonic functions with arbitrary phases are used as different wavelets in a super shot and as the encoding operator. Based on this algorithm, the crosstalk artifacts are eliminated by deblending the multi-source wavefield with little additional computation. Moreover, the estimation or inversion of source wavelets at each iteration for multi-source data, which is crucial for successful FWI but would severely reduce the efficiency of multi-source algorithms, are avoided by normalizing the seismic data with deconvolution. Since the multi-source data are deblended, the proposed algorithm is naturally applicable to the marine mobile streamer seismic data. Furthermore, it is convenient to select the reference traces for deblended data, instead of multi-source data, to eliminate or unify the wavelet information by deconvolution. Finally, we verify the proposed algorithm with the synthetic data.

## In this section, we mainly introduce the structure of our code package and point out some critical process, which including:
(a)	Multi-source wavefield forward simulation: Select harmonic signals with arbitrary phases as wavelets to conduct the multi-source simulation.

(b)	Multi-source wavefield deblending: Define the reference signals for each source within a super shot and deblending the multi-source wavefield based on the orthogonality of trigonometric functions.

(c)	Data residuals calculation: Normalize the deblended data by deconvolution to eliminate or unify the wavelet information for both simulated and observed data. Then calculate the single-source residuals and blend them again for next step.

(d) 	Multi-adjoint wavefield backward simulation: Backward propagate the blended residuals to simulate the multi-adjoint wavefield.

(e)	Multi-adjoint wavefield deblending: Similar to step (b), define the reference signals to deblend the multi-adjoint wavefield.

(f)	Gradient calculation and summation: Calculate the gradient with the deblended source and adjoint wavefields without the crosstalk artifacts. The deblended wavefields can be represented with few frequency snapshots so that the great memory requirement can be reduce to a lot degree.

(g)	Repeat steps (a)-(f) until the termination condition (the number of iterations) is satisfied.

Compared with conventional multi-source algorithm, the main advantage of this approach is that the efficiency can be improved without the inversion quality degradation (crosstalk artifacts). The deblending process helps this multi-source algorithm gets the applicability to mobile marine streamer seismic data, eliminates the crosstalk artifacts and makes it easy to select the reference trace to normalize the data. Therefore, a successful high-resolution multi-source FWI can be implemented without the true wavelet information.

## How to run the code
First, install the softwares of MPICH, CUDA and FFTW on the compute nodes. Then, set environment variables .bashrc or .cshrc for your account. Next, modify the compilers of mpicc, nvcc and library of FFTW paths in the Makefile. Finally, compile to generate executable file and run it by sh run.sh. Note that the hostfile should contain the correct node names.

Please refer to README.pdf to learn more details.

## The main inversion package includes three folders totally:
### 1	INVERSION/
(1) 	acoustic_HLMPIFWIC.cpp is the main code body of FWI which is defined with MPI, including data input and output, node tasks assignment and collection.

(2)	acoustic_HLMPIFWIMulti.cu is the CUDA-guided GPU parallel code. The main two GPU kernel functions are fdtd_2d_GPU_forward and fdtd_2d_GPU_backward. Besides, the data preprocessing and normalization are included in this code including the GPU-device variables memory allocation and release.

(3)	headmulti.h is the declaration header file of different functions and will be called in codes (1) and (2).

(4)	Makefile is used to compile the whole FWI codes. It should be noted that the path of compilers of mpicc and nvcc should be changed according to the user’s environment.

(5) 	hostfile contains a name list of nodes with GPUs.

(6) 	fwiacoustic is the executable file generated after compiling the code with command make.

(7)	run.sh is used to run the executable file on node cluster by sh run.sh.

(8)	nohup.out is a log file of the code running status. This file is created automatically by sh run.sh.

### 2	input/ 
It contains the true velocity (acc_vp.dat) and input parameter file (parameter.txt). The initial velocity is generated automatically by smoothing the true velocity with a 500.0m×500.0m spatial window.

### 3	output/ 
All the output data are written in this file, including the simulated seismic data, seismogram residuals, conjugate gradient at each iteration, and so on.

As for the observed data, if they are not prepared, we have also provided an additional FORWARD module within this package.

##################################################################

For downloading and using this package for your own studies, please cite the following publications:

[1] Zhang Q, Mao W, Fang J. Crosstalk-free simultaneous-source full waveform inversion with normalized seismic data. Computers & Geosciences, 2020, 138, 104460.

[2] Zhang Q, Mao W, Fang J. Elastic full waveform inversion with source-independent crosstalk-free source-encoding algorithm. IEEE Transactions on Geoscience and Remote Sensing, 2020, 58(4), 2915-2927.

[3] Zhang Q, Mao W, Zhou H, Zhang H, Chen Y. Hybrid-domain simultaneous-source full waveform inversion without crosstalk noise. Geophysical Journal International, 2018, 215(3), 1659–1681.
