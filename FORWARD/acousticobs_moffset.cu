#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define BLOCK_SIZE 16

#define OUTPUT_SNAP 0
#define PI 3.1415926

#include "headobs.h"
#include "cufft.h"

//#define BATCH 834

#define reference_window 1 //0--without 1--with

struct Multistream
{
	cudaStream_t stream,stream_back;
};

__global__ void fdtd_cpml_2d_GPU_kernel_vx(
		float *rho, int itmax,
		float *a_x_half, float *a_z, 
		float *b_x_half, float *b_z, 
		float *vx, float *p,
		float *phi_p_x,
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt,
		int it, int pml, int Lc, float *rc
		)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ic;
	int ip=iz*ntx+ix;

	float one_over_dx=1.0/dx;

	float dp_dx,one_over_rho_half_x;

	if(iz>=Lc&&iz<=ntz-Lc&&ix>=Lc-1&&ix<=ntx-Lc-1)
	{
		dp_dx=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dp_dx+=rc[ic]*(p[ip+ic+1]-p[ip-ic])*one_over_dx;
		}

		phi_p_x[ip]=b_x_half[ix]*phi_p_x[ip]+a_x_half[ix]*dp_dx;

		one_over_rho_half_x=1/(0.5*(rho[ip]+rho[ip+1]));

		vx[ip]=dt*one_over_rho_half_x*(dp_dx+phi_p_x[ip])+vx[ip];
	}   

	__syncthreads();

}


__global__ void fdtd_cpml_2d_GPU_kernel_vz(
		float *rho, int itmax,
		float *a_x, float *a_z_half,
		float *b_x, float *b_z_half,
		float *vz, float *p, 
		float *phi_p_z,
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt,
		int it, int pml, int Lc, float *rc
		)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	float one_over_dz=1.0/dz;

	float dp_dz,one_over_rho_half_z;

	int ic;
	int ip=iz*ntx+ix;

	if(iz>=Lc-1&&iz<=ntz-Lc-1&&ix>=Lc&&ix<=ntx-Lc)
	{
		dp_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dp_dz+=rc[ic]*(p[ip+(ic+1)*ntx]-p[ip-ic*ntx])*one_over_dz;
		}

		phi_p_z[ip]=b_z_half[iz]*phi_p_z[ip]+a_z_half[iz]*dp_dz;

		one_over_rho_half_z=1/(0.5*(rho[ip]+rho[ip+ntx]));

		vz[ip]=dt*one_over_rho_half_z*(dp_dz+phi_p_z[ip])+vz[ip];
	}

	__syncthreads();

}


__global__ void fdtd_cpml_2d_GPU_kernel_p(
		float *rick, float *vp, float *rho,
		float *a_x, float *a_z,
		float *b_x, float *b_z,
		float *vx, float *vz, float *p,
		float *phi_vx_x, float *phi_vz_z, 
		int ntp, int ntx, int ntz,
		float *seismogram, int r_iz, int *r_ix, int r_n, int pml, int Lc, float *rc, 
		float dx, float dz, float dt, int s_ix, int s_iz, int it,
		int inv_flag, int itmax,
		float *p_borders_up, float *p_borders_bottom,
		float *p_borders_left, float *p_borders_right
		)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int pmlc=pml+Lc;

	int nx=ntx-2*pmlc;
	int nz=ntz-2*pmlc;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dz;

	float dvx_dx,dvz_dz;
	int ic,ii;
	int ip=iz*ntx+ix;

	if(iz>=Lc&&iz<=ntz-Lc&&ix>=Lc&&ix<=ntx-Lc)
	{
		dvx_dx=0.0;
		dvz_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dvx_dx+=rc[ic]*(vx[ip+ic]-vx[ip-(ic+1)])*one_over_dx;
			dvz_dz+=rc[ic]*(vz[ip+ic*ntx]-vz[ip-(ic+1)*ntx])*one_over_dz;
		}

		phi_vx_x[ip]=b_x[ix]*phi_vx_x[ip]+a_x[ix]*dvx_dx;
		phi_vz_z[ip]=b_z[iz]*phi_vz_z[ip]+a_z[iz]*dvz_dz;

		p[ip]=dt*rho[ip]*vp[ip]*vp[ip]*(dvx_dx+phi_vx_x[ip]+dvz_dz+phi_vz_z[ip])+p[ip];

	}

	if(iz==s_iz&&ix==s_ix)
	{
		p[ip]=p[ip]+rick[it];
	}

	// Seismogram...   
	for(ii=0;ii<r_n;ii++)
	{
		if(ix==r_ix[ii]&&iz==r_iz&&ix>=pmlc&&ix<ntx-pmlc)
		{
			seismogram[ii*itmax+it]=p[ip];
		}
	}

	// Borders...
	if(inv_flag==1)
	{
		if(ix>=pmlc&&ix<=ntx-pmlc-1&&iz>=pmlc&&iz<=pmlc+2*Lc-1)
		{
			p_borders_up[(iz-pmlc)*itmax*nx+it*nx+ix-pmlc]=p[ip];
		}
		if(ix>=pmlc&&ix<=ntx-pmlc-1&&iz>=ntz-pmlc-2*Lc&&iz<=ntz-pmlc-1)
		{
			p_borders_bottom[(iz-ntz+pmlc+2*Lc)*itmax*nx+it*nx+ix-pmlc]=p[ip];
		}

		if(iz>=pmlc+2*Lc&&iz<=ntz-pmlc-2*Lc-1&&ix>=pmlc&&ix<=pmlc+2*Lc-1)
		{
			p_borders_left[(ix-pmlc)*itmax*(nz-4*Lc)+it*(nz-4*Lc)+iz-pmlc-2*Lc]=p[ip];
		}
		if(iz>=pmlc+2*Lc&&iz<=ntz-pmlc-2*Lc-1&&ix>=ntx-pmlc-2*Lc&&ix<=ntx-pmlc-1)
		{
			p_borders_right[(ix-ntx+pmlc+2*Lc)*itmax*(nz-4*Lc)+it*(nz-4*Lc)+iz-pmlc-2*Lc]=p[ip];
		}
	}
	__syncthreads();
}


/*==========================================================

  This subroutine is used for calculating the forward wave 
  field of 2D in time domain.

  1.
  inv_flag==0----Calculate the observed seismograms of 
  Vx and Vz components...
  2.
  inv_flag==1----Calculate the synthetic seismograms of 
  Vx and Vz components and store the 
  borders of Vx and Vz used for constructing 
  the forward wavefields. 
  ===========================================================*/

extern "C"
void fdtd_2d_GPU_forward(int ntx, int ntz, int ntp, int nx, int nz,
		int pml, int Lc, float *rc, float dx, float dz,
		float *rick, int itmax, float dt,
		int is, struct Source ss[], struct MultiGPU plan[], int GPU_N, int rnmax, float *rho,
		float *vp,
		float *k_x, float *k_x_half,
		float *k_z, float *k_z_half,
		float *a_x, float *a_x_half,
		float *a_z, float *a_z_half,
		float *b_x, float *b_x_half,
		float *b_z, float *b_z_half, int inv_flag)
{
	int i,it,ip;
	int ix,iz;
	int pmlc=pml+Lc;

	float *vx,*vz;
	float *p;
	float *phi_vx_x,*phi_vz_z;
	float *phi_p_x,*phi_p_z;

	size_t size_model=sizeof(float)*ntp;

	FILE *fp;
	char filename[40];

	// allocate the memory of vx,vy,vz,sigmaxx,sigmayy,...
	vx=(float*)malloc(sizeof(float)*ntp); 
	vz=(float*)malloc(sizeof(float)*ntp); 
	p=(float*)malloc(sizeof(float)*ntp);

	phi_vx_x      = (float*)malloc(sizeof(float)*ntp);
	phi_vz_z      = (float*)malloc(sizeof(float)*ntp);

	phi_p_x=(float*)malloc(sizeof(float)*ntp);
	phi_p_z=(float*)malloc(sizeof(float)*ntp);

	Multistream plans[GPU_N];

	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);
		cudaStreamCreate(&plans[i].stream);	
	}

	///////////////////////////////
	// initialize the fields........................

	for(ip=0;ip<ntp;ip++)
	{
		vx[ip]=0.0;
		vz[ip]=0.0;

		p[ip]=0.0;

		phi_vx_x[ip]=0.0;
		phi_vz_z[ip]=0.0;

		phi_p_x[ip]=0.0;
		phi_p_z[ip]=0.0;
	}

	// copy the vectors from the host to the device
	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		cudaMemcpyAsync(plan[i].d_r_ix,ss[is+i].r_ix,sizeof(int)*ss[is+i].r_n,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_rick,rick,sizeof(float)*itmax,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_rc,rc,sizeof(float)*Lc,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_vp,vp,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_rho,rho,size_model,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_a_x,a_x,sizeof(float)*ntx,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_a_x_half,a_x_half,sizeof(float)*ntx,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_a_z,a_z,sizeof(float)*ntz,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_a_z_half,a_z_half,sizeof(float)*ntz,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_b_x,b_x,sizeof(float)*ntx,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_b_x_half,b_x_half,sizeof(float)*ntx,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_b_z,b_z,sizeof(float)*ntz,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_b_z_half,b_z_half,sizeof(float)*ntz,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_vx,vx,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_vz,vz,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_p,p,size_model,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_phi_vx_x,phi_vx_x,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_phi_vz_z,phi_vz_z,size_model,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_phi_p_x,phi_p_x,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_phi_p_z,phi_p_z,size_model,cudaMemcpyHostToDevice,plans[i].stream);
	}
	/////////////////////////////////
	// =============================================================================

	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 dimGrid((ntx+dimBlock.x-1)/dimBlock.x,(ntz+dimBlock.y-1)/dimBlock.y);

	//-----------------------------------------------------------------------//
	//=======================================================================//
	//-----------------------------------------------------------------------//
	for(it=0;it<itmax;it++)
	{
		for(i=0;i<GPU_N;i++)
		{
			cudaSetDevice(i);

			fdtd_cpml_2d_GPU_kernel_vx<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_rho, itmax, plan[i].d_a_x_half, plan[i].d_a_z, 
				 plan[i].d_b_x_half, plan[i].d_b_z, 
				 plan[i].d_vx, plan[i].d_p,
				 plan[i].d_phi_p_x, 
				 ntp, ntx, ntz, dx, dz, dt,
				 it, pml, Lc, plan[i].d_rc
				);

			fdtd_cpml_2d_GPU_kernel_vz<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_rho, itmax,
				 plan[i].d_a_x, plan[i].d_a_z_half,
				 plan[i].d_b_x, plan[i].d_b_z_half,
				 plan[i].d_vz, plan[i].d_p, 
				 plan[i].d_phi_p_z,
				 ntp, ntx, ntz, dx, dz, dt,
				 it, pml, Lc, plan[i].d_rc
				);

			fdtd_cpml_2d_GPU_kernel_p<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_rick, plan[i].d_vp, plan[i].d_rho,
				 plan[i].d_a_x, plan[i].d_a_z, plan[i].d_b_x, plan[i].d_b_z,
				 plan[i].d_vx, plan[i].d_vz, plan[i].d_p,
				 plan[i].d_phi_vx_x, plan[i].d_phi_vz_z,
				 ntp, ntx, ntz,
				 plan[i].d_seismogram, ss[is+i].r_iz, plan[i].d_r_ix, ss[is+i].r_n, pml, Lc, plan[i].d_rc, dx, dz, dt,
				 ss[is+i].s_ix, ss[is+i].s_iz, it,
				 inv_flag, itmax,
				 plan[i].d_p_borders_up, plan[i].d_p_borders_bottom,
				 plan[i].d_p_borders_left, plan[i].d_p_borders_right
				);
/*
			if(inv_flag==1&&it%10==0)
			{
				cudaMemcpyAsync(vx,plan[i].d_p,sizeof(float)*ntp,cudaMemcpyDeviceToHost,plans[i].stream);
				cudaStreamSynchronize(plans[i].stream);

				sprintf(filename,"./output/%dvx%d.dat",it,i);     
				fp=fopen(filename,"wb");
				for(ix=pmlc;ix<ntx-pmlc;ix++)
				{
					for(iz=pmlc;iz<ntz-pmlc;iz++)
					{					
						fwrite(&vx[iz*ntx+ix],sizeof(float),1,fp);
					}
				}
				fclose(fp);
			}
*/

		}//end GPU_N
	}//end it

	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		if(inv_flag==0)
		{
			cudaMemcpyAsync(plan[i].seismogram_obs,plan[i].d_seismogram,
					sizeof(float)*ss[is+i].r_n*itmax,cudaMemcpyDeviceToHost,plans[i].stream);
		}
		else
		{
			cudaMemcpyAsync(plan[i].seismogram_syn,plan[i].d_seismogram,
					sizeof(float)*ss[is+i].r_n*itmax,cudaMemcpyDeviceToHost,plans[i].stream);
		}

		if(inv_flag==1)
		{
/*			cudaMemcpyAsync(plan[i].p_borders_up,plan[i].d_p_borders_up,
					sizeof(float)*2*Lc*nx*itmax,cudaMemcpyDeviceToHost,plans[i].stream);
			cudaMemcpyAsync(plan[i].p_borders_bottom,plan[i].d_p_borders_bottom,
					sizeof(float)*2*Lc*nx*itmax,cudaMemcpyDeviceToHost,plans[i].stream);
			cudaMemcpyAsync(plan[i].p_borders_left,plan[i].d_p_borders_left,
					sizeof(float)*2*Lc*(nz-4*Lc)*itmax,cudaMemcpyDeviceToHost,plans[i].stream);
			cudaMemcpyAsync(plan[i].p_borders_right,plan[i].d_p_borders_right,
					sizeof(float)*2*Lc*(nz-4*Lc)*itmax,cudaMemcpyDeviceToHost,plans[i].stream);
*/
			// Output The wavefields when Time=Itmax;

			cudaMemcpyAsync(vx,plan[i].d_vx,size_model,cudaMemcpyDeviceToHost,plans[i].stream);
			cudaMemcpyAsync(vz,plan[i].d_vz,size_model,cudaMemcpyDeviceToHost,plans[i].stream);
			cudaMemcpyAsync(p,plan[i].d_p,size_model,cudaMemcpyDeviceToHost,plans[i].stream);

			cudaStreamSynchronize(plans[i].stream);

			sprintf(filename,"./output/wavefield_itmax%d.dat",i);
			fp=fopen(filename,"wb");
			fwrite(&vx[0],sizeof(float),ntp,fp);
			fwrite(&vz[0],sizeof(float),ntp,fp);

			fwrite(&p[0],sizeof(float),ntp,fp);
			fclose(fp);
		}
	}

	//free the memory of Vx,Vy,Vz,Sigmaxx,Sigmayy,Sigmazz...  
	free(vx);
	free(vz);
	free(p);

	//free the memory of Phi_vx_x....  
	free(phi_vx_x);
	free(phi_vz_z);

	//free the memory of Phi_vx_x....  
	free(phi_p_x);
	free(phi_p_z);

	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}//end GPU

	//free the memory of DEVICE
	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		cudaStreamDestroy(plans[i].stream);
	}
}

/*=============================================
 * Allocate the memory for wavefield simulation
 * ===========================================*/
extern "C"
void variables_malloc(int ntx, int ntz, int ntp, int nx, int nz,
		int pml, int Lc, float dx, float dz, int itmax,
		struct MultiGPU plan[], int GPU_N, int rnmax, int NN
		)
{
	int i;

	size_t size_model=sizeof(float)*ntp;

	// ==========================================================
	// allocate the memory of Vx,Vy,Vz,Sigmaxx,Sigmayy,...
/*
	cudaMallocHost((void **)&rick,sizeof(float)*itmax);  
	cudaMallocHost((void **)&rc,sizeof(float)*Lc);  

	cudaMallocHost((void **)&lambda,sizeof(float)*ntp); 
	cudaMallocHost((void **)&mu,sizeof(float)*ntp); 
	cudaMallocHost((void **)&rho,sizeof(float)*ntp); 
	cudaMallocHost((void **)&lambda_plus_two_mu,sizeof(float)*ntp); 

	cudaMallocHost((void **)&a_x,sizeof(float)*ntx);
	cudaMallocHost((void **)&a_x_half,sizeof(float)*ntx);
	cudaMallocHost((void **)&a_z,sizeof(float)*ntz);
	cudaMallocHost((void **)&a_z_half,sizeof(float)*ntz);

	cudaMallocHost((void **)&b_x,sizeof(float)*ntx);
	cudaMallocHost((void **)&b_x_half,sizeof(float)*ntx);
	cudaMallocHost((void **)&b_z,sizeof(float)*ntz);
	cudaMallocHost((void **)&b_z_half,sizeof(float)*ntz);

	cudaMallocHost((void **)&vx,sizeof(float)*ntp); 
	cudaMallocHost((void **)&vz,sizeof(float)*ntp); 
	cudaMallocHost((void **)&sigmaxx,sizeof(float)*ntp);
	cudaMallocHost((void **)&sigmazz,sizeof(float)*ntp);
	cudaMallocHost((void **)&sigmaxz,sizeof(float)*ntp);
*/

	// allocate the memory for the device
	// allocate the memory for the device
	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		///device//////////
		///device//////////
		///device//////////
		cudaMalloc((void**)&plan[i].d_seismogram,sizeof(float)*itmax*rnmax);
		cudaMalloc((void**)&plan[i].d_seismogram_rms,sizeof(float)*itmax*(rnmax));

		cudaMalloc((void**)&plan[i].d_s_ix,sizeof(int)*NN);
		cudaMalloc((void**)&plan[i].d_r_ix,sizeof(int)*rnmax);

		cudaMalloc((void**)&plan[i].d_rick,sizeof(float)*NN*itmax);        // ricker wave 
		cudaMalloc((void**)&plan[i].d_rc,sizeof(float)*Lc);        // ricker wave 
//		cudaMalloc((void**)&plan[i].d_asr,sizeof(float)*NN);        // ricker wave 

		cudaMalloc((void**)&plan[i].d_vp,size_model);
		cudaMalloc((void**)&plan[i].d_rho,size_model);


		cudaMalloc((void**)&plan[i].d_a_x,sizeof(float)*ntx);
		cudaMalloc((void**)&plan[i].d_a_x_half,sizeof(float)*ntx);
		cudaMalloc((void**)&plan[i].d_a_z,sizeof(float)*ntz);
		cudaMalloc((void**)&plan[i].d_a_z_half,sizeof(float)*ntz);

		cudaMalloc((void**)&plan[i].d_b_x,sizeof(float)*ntx);
		cudaMalloc((void**)&plan[i].d_b_x_half,sizeof(float)*ntx);
		cudaMalloc((void**)&plan[i].d_b_z,sizeof(float)*ntz);
		cudaMalloc((void**)&plan[i].d_b_z_half,sizeof(float)*ntz);      // atten parameters

		cudaMalloc((void**)&plan[i].d_image_vp,size_model);
		cudaMalloc((void**)&plan[i].d_image_rho,size_model);

		cudaMalloc((void**)&plan[i].d_image_sources,size_model);
		cudaMalloc((void**)&plan[i].d_image_receivers,size_model);

		cudaMalloc((void**)&plan[i].d_psdptx,size_model);
		cudaMalloc((void**)&plan[i].d_psdpty,size_model);

		cudaMalloc((void**)&plan[i].d_psdvxx,size_model);
		cudaMalloc((void**)&plan[i].d_psdvxy,size_model);

		cudaMalloc((void**)&plan[i].d_psdvzx,size_model);
		cudaMalloc((void**)&plan[i].d_psdvzy,size_model);

		cudaMalloc((void**)&plan[i].d_psdpx,size_model);
		cudaMalloc((void**)&plan[i].d_psdpy,size_model);

		cudaMalloc((void**)&plan[i].d_vx,size_model);
		cudaMalloc((void**)&plan[i].d_vz,size_model);
		cudaMalloc((void**)&plan[i].d_p,size_model);

		cudaMalloc((void**)&plan[i].dp_dt,size_model);

		cudaMalloc((void**)&plan[i].d_vx_inv,size_model);
		cudaMalloc((void**)&plan[i].d_vz_inv,size_model);
		cudaMalloc((void**)&plan[i].d_p_inv,size_model);

		cudaMalloc((void**)&plan[i].d_phi_vx_x,size_model);
		cudaMalloc((void**)&plan[i].d_phi_vz_z,size_model);

		cudaMalloc((void**)&plan[i].d_phi_p_x,size_model);
		cudaMalloc((void**)&plan[i].d_phi_p_z,size_model);

/*
		cudaMalloc((void**)&plan[i].d_p_borders_up,sizeof(float)*2*Lc*itmax*nx);
		cudaMalloc((void**)&plan[i].d_p_borders_bottom,sizeof(float)*2*Lc*itmax*nx);
		cudaMalloc((void**)&plan[i].d_p_borders_left,sizeof(float)*2*Lc*itmax*(nz-4*Lc));
		cudaMalloc((void**)&plan[i].d_p_borders_right,sizeof(float)*2*Lc*itmax*(nz-4*Lc));
*/
	}
}

/*=============================================
 * Free the memory for wavefield simulation
 * ===========================================*/
extern "C"
void variables_free(int ntx, int ntz, int ntp, int nx, int nz,
		int pml, int Lc, float dx, float dz, int itmax,
		struct MultiGPU plan[], int GPU_N, int rnmax, int NN
		)
{
	int i;
/*
	cudaFreeHost(rick); 
	cudaFreeHost(rc); 
	
	//free the memory of lambda
	cudaFreeHost(lambda); 
	cudaFreeHost(mu); 
	cudaFreeHost(rho); 
	cudaFreeHost(lambda_plus_two_mu); 

	cudaFreeHost(a_x);
	cudaFreeHost(a_x_half);
	cudaFreeHost(a_z);
	cudaFreeHost(a_z_half);

	cudaFreeHost(b_x);
	cudaFreeHost(b_x_half);
	cudaFreeHost(b_z);
	cudaFreeHost(b_z_half);

	cudaFreeHost(vx);
	cudaFreeHost(vz);
	cudaFreeHost(sigmaxx);
	cudaFreeHost(sigmazz);
	cudaFreeHost(sigmaxz);
*/	 
	//free the memory of DEVICE
	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);
		////device////
		////device////
		////device////
		cudaFree(plan[i].d_seismogram);
		cudaFree(plan[i].d_seismogram_rms);

		cudaFree(plan[i].d_s_ix);
		cudaFree(plan[i].d_r_ix);

		cudaFree(plan[i].d_rick);
		cudaFree(plan[i].d_rc);
//		cudaFree(plan[i].d_asr);

		cudaFree(plan[i].d_vp);
		cudaFree(plan[i].d_rho);

		cudaFree(plan[i].d_a_x);
		cudaFree(plan[i].d_a_x_half);
		cudaFree(plan[i].d_a_z);
		cudaFree(plan[i].d_a_z_half);

		cudaFree(plan[i].d_b_x);
		cudaFree(plan[i].d_b_x_half);
		cudaFree(plan[i].d_b_z);
		cudaFree(plan[i].d_b_z_half);

		cudaFree(plan[i].d_image_vp);
		cudaFree(plan[i].d_image_rho);

		cudaFree(plan[i].d_image_sources);
		cudaFree(plan[i].d_image_receivers);

		cudaFree(plan[i].d_psdptx);
		cudaFree(plan[i].d_psdpty);

		cudaFree(plan[i].d_psdvxx);
		cudaFree(plan[i].d_psdvxy);

		cudaFree(plan[i].d_psdvzx);
		cudaFree(plan[i].d_psdvzy);

		cudaFree(plan[i].d_psdpx);
		cudaFree(plan[i].d_psdpy);

		cudaFree(plan[i].d_vx);
		cudaFree(plan[i].d_vz);
		cudaFree(plan[i].d_p);

		cudaFree(plan[i].dp_dt);

		cudaFree(plan[i].d_vx_inv);
		cudaFree(plan[i].d_vz_inv);
		cudaFree(plan[i].d_p_inv);

		cudaFree(plan[i].d_phi_vx_x);
		cudaFree(plan[i].d_phi_vz_z);
		cudaFree(plan[i].d_phi_p_x);
		cudaFree(plan[i].d_phi_p_z);
/*
		cudaFree(plan[i].d_p_borders_up);
		cudaFree(plan[i].d_p_borders_bottom);
		cudaFree(plan[i].d_p_borders_left);
		cudaFree(plan[i].d_p_borders_right);
*/
	}
}

extern "C"
void getdevice(int *GPU_N)
{
	
	cudaGetDeviceCount(GPU_N);	
//	GPU_N=6;//4;//2;//
}

/*=====================================================================
  This function is used for calculating the single frequency of wavelet
  =====================================================================*/

extern "C"
void ricker_fre(float *rick, int is, struct Encode es[], int GPU_N, struct MultiGPU plan[], int ifreq, float *fs, 
		int itmax, float dt, float dx, int nx, int pml)
{ 
	int i;

	int ix,it,itt,K,NX; 

	K=(int)ceil(log(1.0*itmax)/log(2.0));
	NX=(int)pow(2.0,K);	

	float df=1/(NX*dt);
	float rkmax;

	FILE *fp;
	char filename[50];

	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		cufftComplex *rk0,*rk,*temp1;
		cufftHandle plan1;

		cudaMallocHost((void **)&rk0, sizeof(cufftComplex)*NX);
		cudaMallocHost((void **)&rk, sizeof(cufftComplex)*NX);

		cudaMalloc((void **)&temp1,sizeof(cufftComplex)*NX);

		cufftPlan1d(&plan1,NX,CUFFT_C2C,1);

		for(it=0;it<NX;it++)
		{ 
			rk0[it].x=0.0;
			rk0[it].y=0.0; 
		}            

		for(it=0;it<itmax;it++)
		{
			rk0[it].x=rick[it];	
		}

		cudaMemcpy(temp1,rk0,sizeof(cufftComplex)*NX,cudaMemcpyHostToDevice);
		cufftExecC2C(plan1,temp1,temp1,CUFFT_FORWARD);
		cudaMemcpy(rk0,temp1,sizeof(cufftComplex)*NX,cudaMemcpyDeviceToHost);

		for(ix=0;ix<es[is+i].num;ix++)
		{
			for(it=0;it<NX;it++)
			{ 
				rk[it].x=0.0;
				rk[it].y=0.0; 
			}            
			//itt=ceil(fs[ix+ifreq*4]/df);
			itt=ceil(fs[ifreq]/df);

			rk[itt].x=rk0[itt].x;
			rk[itt].y=rk0[itt].y;

			rk[NX-itt].x=rk0[NX-itt].x;
			rk[NX-itt].y=rk0[NX-itt].y;

			// fft(r_real,r_imag,NFFT,-1);

			cudaMemcpy(temp1,rk,sizeof(cufftComplex)*NX,cudaMemcpyHostToDevice);
			cufftExecC2C(plan1,temp1,temp1,CUFFT_INVERSE);
			cudaMemcpy(rk,temp1,sizeof(cufftComplex)*NX,cudaMemcpyDeviceToHost);

			rkmax=0.0;
			for(it=0;it<itmax;it++)
			{
				plan[i].rick[ix*itmax+it]=rk[it].x;//cos(2*PI*2.0*(it*dt-1.0/8.0));//
				if(rkmax<fabs(rk[it].x))
					rkmax=fabs(rk[it].x);
			} 
			for(it=0;it<itmax;it++)
			{
				plan[i].rick[ix*itmax+it]/=rkmax;
			} 
		}

		sprintf(filename,"./output/%drick.dat",is+i+1);
		fp=fopen(filename,"wb");
		fwrite(&plan[i].rick[0],sizeof(float),es[is+i].num*itmax,fp);
		fclose(fp);

		cudaFreeHost(rk);
		cudaFreeHost(rk0);

		cudaFree(temp1);

		cufftDestroy(plan1);
	}

	return;
}

/*=====================================================================
  This function is used for calculating the angular frequency components
  =====================================================================*/

extern "C"
void seismg_fre(float *seismogram_obs, float *seismogram_tmp, int ifreq, float *fs, int i,
		int itmax, float dt, float dx, int ii, int nx, int pml)
{ 
	cudaSetDevice(i);

	int ix,it,itt,K,NX;
	int BATCH=nx;

	K=(int)ceil(log(1.0*itmax)/log(2.0));
	NX=(int)pow(2.0,K);	

	float df=1/(NX*dt);
	float smax;

	FILE *fp;
	char filename[30];

	int NTP=NX*BATCH;

	cufftComplex *d,*obs,*temp;

	cudaMallocHost((void **)&d, sizeof(cufftComplex)*NX*BATCH);
	cudaMallocHost((void **)&obs,sizeof(cufftComplex)*NX*BATCH);

	cudaMalloc((void **)&temp,sizeof(cufftComplex)*NX*BATCH);

	cufftHandle plan2;
	cufftPlan1d(&plan2,NX,CUFFT_C2C,BATCH);

	for(it=0;it<NTP;it++)
	{ 
		d[it].x=0.0;
		d[it].y=0.0;   
		obs[it].x=0.0;
		obs[it].y=0.0;
	}            
	for(ix=0;ix<nx;ix++)
	{
		for(it=0;it<itmax;it++)
		{
			d[ix*NX+it].x=seismogram_obs[it*nx+ix];	
		}
	}   

	cudaMemcpy(temp,d,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
	cufftExecC2C(plan2,temp,temp,CUFFT_FORWARD);
	cudaMemcpy(d,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

	for(ix=0;ix<nx;ix++)
	{
		//itt=ceil(fs[ii+ifreq*4]/df);
		itt=ceil(fs[ifreq]/df);

		obs[ix*NX+itt].x=d[ix*NX+itt].x;
		obs[ix*NX+itt].y=d[ix*NX+itt].y;

		obs[ix*NX+NX-itt].x=d[ix*NX+NX-itt].x;
		obs[ix*NX+NX-itt].y=d[ix*NX+NX-itt].y;
	}   

	// fft(r_real,r_imag,NFFT,-1);

	cudaMemcpy(temp,obs,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
	cufftExecC2C(plan2,temp,temp,CUFFT_INVERSE);
	cudaMemcpy(obs,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

	smax=0.0;
	for(ix=0;ix<nx;ix++)
	{
		for(it=0;it<itmax;it++)
		{
			seismogram_tmp[it*nx+ix]=obs[ix*NX+it].x;
			if(smax<fabs(obs[ix*NX+it].x))
				smax=fabs(obs[ix*NX+it].x);
		}
	}
	for(ix=0;ix<nx;ix++)
	{
		for(it=0;it<itmax;it++)
		{
			seismogram_tmp[it*nx+ix]/=smax;
		}
	}

	cudaFreeHost(d);
	cudaFreeHost(obs); 

	cudaFree(temp);

	cufftDestroy(plan2);

	return;
}

