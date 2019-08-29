#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define BLOCK_SIZE 16

#define OUTPUT_SNAP 0
#define PI 3.1415926
#define EPS 1.0e-40

#include "headmulti.h"
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
		float *psdptx, float *psdpty, int ifreq, int freqintv, float fs0, int *randnum,
		int ntp, int ntx, int ntz,
		float *seismogram, int r_iz, int *r_ix, int r_n, int pml, int Lc, float *rc, 
		float dx, float dz, float dt, int iter, int s_num, int *s_ix, int s_iz, int it,
		int inv_flag, int itmax, float *dp_dt
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
	int is;

	float df=1.0/(itmax*dt);
	float dw=2*PI*df;
	int wn,T0=itmax/2;

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

		dp_dt[ip]=rho[ip]*vp[ip]*vp[ip]*(dvx_dx+dvz_dz);

	}

	for(is=0;is<s_num;is++)
	{
		//wn=(int)((fs0+(is+iter)%s_num*freqintv*df)/df);
		//wn=(int)((fs0+is*freqintv*df)/df);
		wn=(int)((fs0+randnum[is]*freqintv*df)/df);

		if(it>=itmax-T0)
		{
			psdptx[is*ntp+ip]+=dp_dt[ip]*__cosf(wn*dw*it*dt)/T0;
			psdpty[is*ntp+ip]+=dp_dt[ip]*__cosf(wn*dw*it*dt+PI/2)/T0;
		}

		if(iz==s_iz&&ix==s_ix[is])
		{
			p[ip]+=rick[is*itmax+it];
		}
	}

	for(ii=0;ii<r_n;ii++)
	{
		if(ix==r_ix[ii]&&iz==r_iz&&ix>=pmlc&&ix<ntx-pmlc)
		{
			//seismogram[ii*itmax+it]=p[ip];
			seismogram[(r_ix[ii]-pmlc)*itmax+it]=p[ip];
		}
	}
/*
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
*/	
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
		float *rick, int itmax, float dt, int iter, int ifreq, int freqintv, int Nf, float *fs, int *randnum,
		int is, struct Encode es[], int NN, struct MultiGPU plan[], int GPU_N, int rnmax, float *rho,
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

		cudaMemcpyAsync(plan[i].d_s_ix,es[is+i].s_ix,sizeof(int)*es[is+i].num,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_r_ix,es[is+i].r_ix,sizeof(int)*es[is+i].r_n,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_rick,plan[i].rick,sizeof(float)*es[is+i].num*itmax,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_rc,rc,sizeof(float)*Lc,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_randnum,randnum,sizeof(int)*es[is+i].num,cudaMemcpyHostToDevice,plans[i].stream);

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

/*
		cudaMemcpyAsync(plan[i].d_vx,vx,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_vz,vz,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_p,p,size_model,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_phi_vx_x,phi_vx_x,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_phi_vz_z,phi_vz_z,size_model,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_phi_p_x,phi_p_x,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_phi_p_z,phi_p_z,size_model,cudaMemcpyHostToDevice,plans[i].stream);
*/
		//////////// initialization //////////
		cudaMemsetAsync(plan[i].d_seismogram,0,sizeof(float)*es[is+i].r_n*itmax,plans[i].stream);
		//
		cudaMemsetAsync(plan[i].d_vx,0,sizeof(float)*ntp,plans[i].stream);
		cudaMemsetAsync(plan[i].d_vz,0,sizeof(float)*ntp,plans[i].stream);
		cudaMemsetAsync(plan[i].d_p,0,sizeof(float)*ntp,plans[i].stream);

		cudaMemsetAsync(plan[i].dp_dt,0,sizeof(float)*ntp,plans[i].stream);

		cudaMemsetAsync(plan[i].d_phi_vx_x,0,sizeof(float)*ntp,plans[i].stream);
		cudaMemsetAsync(plan[i].d_phi_vz_z,0,sizeof(float)*ntp,plans[i].stream);

		cudaMemsetAsync(plan[i].d_phi_p_x,0,sizeof(float)*ntp,plans[i].stream);
		cudaMemsetAsync(plan[i].d_phi_p_z,0,sizeof(float)*ntp,plans[i].stream);

		cudaMemsetAsync(plan[i].d_psdptx,0,sizeof(float)*ntp*NN,plans[i].stream);
		cudaMemsetAsync(plan[i].d_psdpty,0,sizeof(float)*ntp*NN,plans[i].stream);
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
				 plan[i].d_psdptx, plan[i].d_psdpty, ifreq, freqintv, fs[ifreq], plan[i].d_randnum, 
				 ntp, ntx, ntz,
				 plan[i].d_seismogram, es[is+i].r_iz, plan[i].d_r_ix, es[is+i].r_n, pml, Lc, plan[i].d_rc, dx, dz, dt,
				 iter, es[is+i].num, plan[i].d_s_ix, es[is+i].s_iz, it,
				 inv_flag, itmax, plan[i].dp_dt
				 );
/*
			if(inv_flag==1&&it%10==0)
			{
				cudaMemcpyAsync(vx,plan[i].d_p,sizeof(float)*ntp,cudaMemcpyDeviceToHost,plans[i].stream);
				cudaStreamSynchronize(plans[i].stream);

				sprintf(filename,"../output/%dvx%d.dat",it,i);     
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

		cudaMemcpyAsync(plan[i].psdptx,plan[i].d_psdptx,size_model*NN,cudaMemcpyDeviceToHost,plans[i].stream);
		cudaMemcpyAsync(plan[i].psdpty,plan[i].d_psdpty,size_model*NN,cudaMemcpyDeviceToHost,plans[i].stream);

		if(inv_flag==0)
		{
			cudaMemcpyAsync(plan[i].seismogram_obs,plan[i].d_seismogram,
					sizeof(float)*es[is+i].r_n*itmax,cudaMemcpyDeviceToHost,plans[i].stream);
		}
		else
		{
			cudaMemcpyAsync(plan[i].seismogram_syn,plan[i].d_seismogram,
					sizeof(float)*es[is+i].r_n*itmax,cudaMemcpyDeviceToHost,plans[i].stream);
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

			// Output The wavefields when Time=Itmax;

			cudaMemcpyAsync(vx,plan[i].d_vx,size_model,cudaMemcpyDeviceToHost,plans[i].stream);
			cudaMemcpyAsync(vz,plan[i].d_vz,size_model,cudaMemcpyDeviceToHost,plans[i].stream);
			cudaMemcpyAsync(p,plan[i].d_p,size_model,cudaMemcpyDeviceToHost,plans[i].stream);

			cudaStreamSynchronize(plans[i].stream);

			sprintf(filename,"../output/wavefield_itmax%d.dat",i);
			fp=fopen(filename,"wb");
			fwrite(&vx[0],sizeof(float),ntp,fp);
			fwrite(&vz[0],sizeof(float),ntp,fp);

			fwrite(&p[0],sizeof(float),ntp,fp);
			fclose(fp);
*/
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

__global__ void fdtd_2d_GPU_kernel_p_backward(
		float *rick, float *vp, float *rho,
		float *vx, float *vz, float *p,
		int ntp, int ntx, int ntz, int pml, int Lc, float *rc,
		float dx, float dz, float dt, int s_ix,
		int s_iz, int it, float *dp_dt
		)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int pmlc=pml+Lc;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dz;

	float dvx_dx,dvz_dz;
	int ic,ii;
	int ip=iz*ntx+ix;

	if(iz>=pmlc+Lc&&iz<=ntz-pmlc-Lc-1&&ix>=pmlc+Lc&&ix<=ntx-pmlc-Lc-1)
	{
		dvx_dx=0.0;
		dvz_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dvx_dx+=rc[ic]*(vx[ip+ic]-vx[ip-(ic+1)])*one_over_dx;
			dvz_dz+=rc[ic]*(vz[ip+ic*ntx]-vz[ip-(ic+1)*ntx])*one_over_dz;
		}

		p[ip]=dt*rho[ip]*vp[ip]*vp[ip]*(dvx_dx+dvz_dz)+p[ip];

		dp_dt[ip]=rho[ip]*vp[ip]*vp[ip]*(dvx_dx+dvz_dz);
	}

	if(iz==s_iz&&ix==s_ix)
	{
		p[ip]=p[ip]-rick[it+1];
	}
	
	__syncthreads();

}


__global__ void fdtd_2d_GPU_kernel_vx_backward(
		float *rho,
		float *vx, float *p,
		int ntp, int ntx, int ntz, int pml, int Lc, float *rc,
		float dx, float dz, float dt
		)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int pmlc=pml+Lc;

	int ic;
	int ip=iz*ntx+ix;

	float one_over_dx=1.0/dx;

	float dp_dx,one_over_rho_half_x;

	if(iz>=pmlc+Lc&&iz<=ntz-pmlc-Lc-1&&ix>=pmlc+Lc&&ix<=ntx-pmlc-Lc-1)
	{
		dp_dx=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dp_dx+=rc[ic]*(p[ip+ic+1]-p[ip-ic])*one_over_dx;
		}

		one_over_rho_half_x=1/(0.5*(rho[ip]+rho[ip+1]));

		vx[ip]=dt*one_over_rho_half_x*dp_dx+vx[ip];
	}

	__syncthreads();
}

__global__ void fdtd_2d_GPU_kernel_vz_backward(
		float *rho,
		float *vz, float *p,
		int ntp, int ntx, int ntz, int pml, int Lc, float *rc,
		float dx, float dz, float dt
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

	int pmlc=pml+Lc;

	int ic;
	int ip=iz*ntx+ix;

	if(iz>=pmlc+Lc&&iz<=ntz-pmlc-Lc-1&&ix>=pmlc+Lc&&ix<=ntx-pmlc-Lc-1)
	{
		dp_dz=0.0;

		for(ic=0;ic<Lc;ic++)
		{
			dp_dz+=rc[ic]*(p[ip+(ic+1)*ntx]-p[ip-ic*ntx])*one_over_dz;
		}

		one_over_rho_half_z=1/(0.5*(rho[ip]+rho[ip+ntx]));

		vz[ip]=dt*one_over_rho_half_z*dp_dz+vz[ip];
	}

	__syncthreads();

}

__global__ void fdtd_2d_GPU_kernel_borders_backward
(
 float *p,
 float *p_borders_up, float *p_borders_bottom,
 float *p_borders_left, float *p_borders_right,
 int ntp, int ntx, int ntz, int pml, int Lc, float *rc, int it, int itmax
 )
{


	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int pmlc=pml+Lc;

	int ip=iz*ntx+ix;
	int nx=ntx-2*pmlc;
	int nz=ntz-2*pmlc;

	if(ix>=pmlc&&ix<=ntx-pmlc-1&&iz>=pmlc&&iz<=pmlc+2*Lc-1)
	{
		p[ip]=p_borders_up[(iz-pmlc)*itmax*nx+it*nx+ix-pmlc];
	}
	if(ix>=pmlc&&ix<=ntx-pmlc-1&&iz>=ntz-pmlc-2*Lc&&iz<=ntz-pmlc-1)
	{
		p[ip]=p_borders_bottom[(iz-ntz+pmlc+2*Lc)*itmax*nx+it*nx+ix-pmlc];
	}

	if(iz>=pmlc+2*Lc&&iz<=ntz-pmlc-2*Lc-1&&ix>=pmlc&&ix<=pmlc+2*Lc-1)
	{
		p[ip]=p_borders_left[(ix-pmlc)*itmax*(nz-4*Lc)+it*(nz-4*Lc)+iz-pmlc-2*Lc];
	}
	if(iz>=pmlc+2*Lc&&iz<=ntz-pmlc-2*Lc-1&&ix>=ntx-pmlc-2*Lc&&ix<=ntx-pmlc-1)
	{
		p[ip]=p_borders_right[(ix-ntx+pmlc+2*Lc)*itmax*(nz-4*Lc)+it*(nz-4*Lc)+iz-pmlc-2*Lc];
	}

	__syncthreads();

}


__global__ void fdtd_cpml_2d_GPU_kernel_vx_backward(
		float *rho, int itmax,
		float *a_x_half, float *a_z, 
		float *b_x_half, float *b_z, 
		float *vx, float *p,
		float *phi_p_x,
		float *psdvxx, float *psdvxy, int ifreq, int freqintv, float fs0, int *randnum,
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt,
		int iter, int s_num, int it, int pml, int Lc, float *rc
		)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ic,ip=iz*ntx+ix;

	float df=1.0/(itmax*dt);
	float dw=2*PI*df;
	int wn,T0=itmax/2;
	int is;

	float one_over_dx=1.0/dx;

	float dp_dx,one_over_rho_half_x;

	if(iz>=0&&iz<=ntz-1&&ix>=Lc-1&&ix<=ntx-Lc-1)
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
/*
	for(is=0;is<s_num;is++)
	{
		//wn=(int)((fs0+(is+iter)%s_num*freqintv*df)/df);
		//wn=(int)((fs0+is*freqintv*df)/df);
		wn=(int)((fs0+randnum[is]*freqintv*df)/df);

		//if(it<itmax/2+200&&it>=200)
		if(it<T0)//itmax/2)
		{
			psdvxx[is*ntp+ip]+=vx[ip]*__cosf(wn*dw*it*dt)/T0;
			psdvxy[is*ntp+ip]+=vx[ip]*__cosf(wn*dw*it*dt+PI/2)/T0;
		}
	}
*/
	__syncthreads();

}


__global__ void fdtd_cpml_2d_GPU_kernel_vz_backward(
		float *rho, int itmax,
		float *a_x, float *a_z_half,
		float *b_x, float *b_z_half,
		float *vz, float *p, 
		float *phi_p_z,
		float *psdvzx, float *psdvzy, int ifreq, int freqintv, float fs0, int *randnum,
		int ntp, int ntx, int ntz, 
		float dx, float dz, float dt,
		int iter, int s_num, int it, int pml, int Lc, float *rc
		)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ic,ip=iz*ntx+ix;

	float df=1.0/(itmax*dt);
	float dw=2*PI*df;
	int wn,T0=itmax/2;
	int is;

	float one_over_dz=1.0/dz;
	float dp_dz,one_over_rho_half_z;

	if(iz>=Lc-1&&iz<=ntz-Lc-1&&ix>=0&&ix<=ntx-1)
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
/*
	for(is=0;is<s_num;is++)
	{
		//wn=(int)((fs0+(is+iter)%s_num*freqintv*df)/df);
		//wn=(int)((fs0+is*freqintv*df)/df);
		wn=(int)((fs0+randnum[is]*freqintv*df)/df);

		if(it<T0)//itmax/2)
		{
			psdvzx[is*ntp+ip]+=vz[ip]*__cosf(wn*dw*it*dt)/T0;
			psdvzy[is*ntp+ip]+=vz[ip]*__cosf(wn*dw*it*dt+PI/2)/T0;
		}
	}
*/

	__syncthreads();

}

__global__ void fdtd_cpml_2d_GPU_kernel_p_backward(
		float *vp, float *rho, int itmax,
		float *a_x, float *a_z,
		float *b_x, float *b_z,
		float *vx, float *vz, float *p,
		float *phi_vx_x, float *phi_vz_z, 
		float *psdpx, float *psdpy, int ifreq, int freqintv, float fs0, int *randnum,
		int ntp, int ntx, int ntz, int pml, int Lc, float *rc,
		float *seismogram_rms, int r_iz, int *r_ix, int r_n, 
		float dx, float dz, float dt, int iter, int s_num, int it
		)

{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ic,ii,ip=iz*ntx+ix;

	float df=1.0/(itmax*dt);
	float dw=2*PI*df;
	int wn,T0=itmax/2;
	int is;

	int pmlc=pml+Lc;

	float one_over_dx=1.0/dx;
	float one_over_dz=1.0/dz;

	float dvx_dx,dvz_dz;
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

	for(is=0;is<s_num;is++)
	{
		//wn=(int)((fs0+(is+iter)%s_num*freqintv*df)/df);
		//wn=(int)((fs0+is*freqintv*df)/df);
		wn=(int)((fs0+randnum[is]*freqintv*df)/df);

		if(it<T0)//itmax/2)
		{
			psdpx[is*ntp+ip]+=p[ip]*__cosf(wn*dw*it*dt)/T0;
			psdpy[is*ntp+ip]+=p[ip]*__cosf(wn*dw*it*dt+PI/2)/T0;
		}
	}

	for(ii=0;ii<r_n;ii++)
	{
		//if(ix==r_ix[ii]&&iz==r_iz)
		if(ix==r_ix[ii]&&iz==r_iz&&ix>=pmlc&&ix<ntx-pmlc)
		{
			//p[ip]=seismogram_rms[ii*itmax+it];
			p[ip]=seismogram_rms[(r_ix[ii]-pmlc)*itmax+it];
		}
	}

	__syncthreads();
}


__global__ void sum_image_GPU_kernel_image
(
 float *vp, float *rho,
 float *p_inv,float *dp_dt,float *p, float *vx, float *vz,
 float *image_vp, float *image_rho, float *image_sources, float *image_receivers, 
 int ntx, int ntz, int pml, int Lc, float *rc, float dx, float dz
 )

{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int pmlc=pml+Lc;

	int ip=iz*ntx+ix;

	float dp_dx,dp_dz;

	if(iz>=pmlc&&iz<=ntz-pmlc-1&&ix>=pmlc&&ix<=ntx-pmlc-1)
	{
		dp_dx=0.5*(p_inv[ip+1]-p_inv[ip-1])/dx;
		dp_dz=0.5*(p_inv[ip+ntx]-p_inv[ip-ntx])/dz;

		image_vp[ip]+=2/(rho[ip]*powf(vp[ip],3.0))*dp_dt[ip]*p[ip];
	
		image_rho[ip]+=(1/(rho[ip]*powf(vp[ip],2.0))*dp_dt[ip]*p[ip]-dp_dx*vx[ip]-dp_dz*vz[ip])/rho[ip];

		image_sources[ip]=image_sources[ip]+dp_dt[ip]*dp_dt[ip];//p_inv[ip]*p_inv[ip];
		image_receivers[ip]=image_receivers[ip]+p[ip]*p[ip];        
	}
	__syncthreads();
}

__global__ void laplace_kernel_image
(
 float *image, float *image_rho, 
 int ntx, int ntz, int pml, int Lc, float *rc, float dx, float dz
 )

{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int pmlc=pml+Lc;

	int ip=iz*ntx+ix;
	float diff1=0.0;
	float diff2=0.0;

	if(iz>=pmlc&&iz<=ntz-pmlc-1&&ix>=pmlc&&ix<=ntx-pmlc-1)
	{
		diff1=(image[ip+ntx]-2.0*image[ip]+image[ip-ntx])/(dz*dz);
		diff2=(image[ip+1]-2.0*image[ip]+image[ip-1])/(dx*dx);	
	}

	image_rho[ip]=diff1+diff2;

	__syncthreads();
}

/*==========================================================

  This subroutine is used for calculating wave field in 2D.

  ===========================================================*/

extern "C"
void fdtd_2d_GPU_backward(int ntx, int ntz, int ntp, int nx, int nz,
		int pml, int Lc, float *rc, float dx, float dz,
		float *rick, int itmax, float dt, int iter, int ifreq, int freqintv, int Nf, float *fs, int *randnum,
		int is, struct Encode es[], int NN, struct MultiGPU plan[], int GPU_N, int rnmax, float *rho,
		float *vp,
		float *k_x, float *k_x_half,
		float *k_z, float *k_z_half,
		float *a_x, float *a_x_half,
		float *a_z, float *a_z_half,
		float *b_x, float *b_x_half,
		float *b_z, float *b_z_half)
{
	int i,it,ip;
	int ix,iz;
	int pmlc=pml+Lc;

	FILE *fp;
	char filename[40];

	float *vx,*vz;
	float *p;
	float *phi_vx_x,*phi_vz_z;
	float *phi_p_x,*phi_p_z;

	// vectors for the devices

	size_t size_model=sizeof(float)*ntp;

	//    int iz,ix;
	// allocate the memory of Vx,Vy,Vz,Sigmaxx,Sigmayy,...
	vx=(float*)malloc(sizeof(float)*ntp); 
	vz=(float*)malloc(sizeof(float)*ntp); 
	p=(float*)malloc(sizeof(float)*ntp);

	// allocate the memory of phi_vx_x...
	phi_vx_x      = (float*)malloc(sizeof(float)*ntp);
	phi_vz_z      = (float*)malloc(sizeof(float)*ntp);

	// allocate the memory of phi_p_x...
	phi_p_x=(float*)malloc(sizeof(float)*ntp);
	phi_p_z=(float*)malloc(sizeof(float)*ntp);

	Multistream plans[GPU_N];

	// allocate the memory for the device
	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);
		cudaStreamCreate(&plans[i].stream);	
	}

	// =============================================================================

	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 dimGrid((ntx+dimBlock.x-1)/dimBlock.x,(ntz+dimBlock.y-1)/dimBlock.y);

	//-----------------------------------------------------------------------//
	//=======================================================================//
	//-----------------------------------------------------------------------//
/*
	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		sprintf(filename,"../output/wavefield_itmax%d.dat",i);
		fp=fopen(filename,"rb");
		fread(&vx[0],sizeof(float),ntp,fp);
		fread(&vz[0],sizeof(float),ntp,fp);

		fread(&p[0],sizeof(float),ntp,fp);
		fclose(fp);

		cudaMemcpyAsync(plan[i].d_vx_inv,vx,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_vz_inv,vz,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_p_inv,p,size_model,cudaMemcpyHostToDevice,plans[i].stream);

		cudaStreamSynchronize(plans[i].stream);
	}
	*/

	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		// Copy the vectors from the host to the device

		cudaMemcpyAsync(plan[i].d_seismogram_rms,plan[i].seismogram_rms,
				sizeof(float)*es[is+i].r_n*itmax,cudaMemcpyHostToDevice,plans[i].stream);
/*
		cudaMemcpyAsync(plan[i].d_p_borders_up,plan[i].p_borders_up,
				sizeof(float)*2*Lc*nx*itmax,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_p_borders_bottom,plan[i].p_borders_bottom,
				sizeof(float)*2*Lc*nx*itmax,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_p_borders_left,plan[i].p_borders_left,
				sizeof(float)*2*Lc*(nz-4*Lc)*itmax,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_p_borders_right,plan[i].p_borders_right,
				sizeof(float)*2*Lc*(nz-4*Lc)*itmax,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_rick,rick,sizeof(float)*itmax,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_rc,rc,sizeof(float)*itmax,cudaMemcpyHostToDevice,plans[i].stream);
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
*/
	}

	// Initialize the fields........................

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

	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);
/*
		cudaMemcpyAsync(plan[i].d_image_vp,plan[i].image_vp,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_image_rho,plan[i].image_rho,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_image_sources,plan[i].image_sources,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_image_receivers,plan[i].image_receivers,size_model,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_vx,vx,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_vz,vz,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_p,p,size_model,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].dp_dt,p,size_model,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_phi_vx_x,phi_vx_x,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_phi_vz_z,phi_vz_z,size_model,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_phi_p_x,phi_p_x,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_phi_p_z,phi_p_z,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		*/
		cudaMemsetAsync(plan[i].d_image_vp,0,size_model,plans[i].stream);
		cudaMemsetAsync(plan[i].d_image_rho,0,size_model,plans[i].stream);
		cudaMemsetAsync(plan[i].d_image_sources,0,size_model,plans[i].stream);
		cudaMemsetAsync(plan[i].d_image_receivers,0,size_model,plans[i].stream);

		cudaMemsetAsync(plan[i].d_vx,0,sizeof(float)*ntp,plans[i].stream);
		cudaMemsetAsync(plan[i].d_vz,0,sizeof(float)*ntp,plans[i].stream);
		cudaMemsetAsync(plan[i].d_p,0,sizeof(float)*ntp,plans[i].stream);

		cudaMemsetAsync(plan[i].d_phi_vx_x,0,sizeof(float)*ntp,plans[i].stream);
		cudaMemsetAsync(plan[i].d_phi_vz_z,0,sizeof(float)*ntp,plans[i].stream);

		cudaMemsetAsync(plan[i].d_phi_p_x,0,sizeof(float)*ntp,plans[i].stream);
		cudaMemsetAsync(plan[i].d_phi_p_z,0,sizeof(float)*ntp,plans[i].stream);

		cudaMemsetAsync(plan[i].d_psdpx,0,sizeof(float)*ntp*NN,plans[i].stream);
		cudaMemsetAsync(plan[i].d_psdpy,0,sizeof(float)*ntp*NN,plans[i].stream);
/*
		cudaMemsetAsync(plan[i].d_psdvxx,0,sizeof(float)*ntp*NN,plans[i].stream);
		cudaMemsetAsync(plan[i].d_psdvxy,0,sizeof(float)*ntp*NN,plans[i].stream);

		cudaMemsetAsync(plan[i].d_psdvzx,0,sizeof(float)*ntp*NN,plans[i].stream);
		cudaMemsetAsync(plan[i].d_psdvzy,0,sizeof(float)*ntp*NN,plans[i].stream);
*/

	}

	//==============================================================================
	//  THIS SECTION IS USED TO CONSTRUCT THE FORWARD WAVEFIELDS...           
	//==============================================================================


	for(it=itmax-2;it>=0;it--)
	{
		for(i=0;i<GPU_N;i++)
		{
			cudaSetDevice(i);
/*
			fdtd_2d_GPU_kernel_p_backward<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_rick, plan[i].d_vp, plan[i].d_rho,
				 plan[i].d_vx_inv, plan[i].d_vz_inv, plan[i].d_p_inv,
				 ntp, ntx, ntz, pml, Lc, plan[i].d_rc, dx, dz, -dt,
				 ss[is+i].s_ix, ss[is+i].s_iz, it, plan[i].dp_dt
				);

			fdtd_2d_GPU_kernel_borders_backward<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_p_inv,
				 plan[i].d_p_borders_up, plan[i].d_p_borders_bottom,
				 plan[i].d_p_borders_left, plan[i].d_p_borders_right,
				 ntp, ntx, ntz, pml, Lc, plan[i].d_rc, it, itmax
				);

			fdtd_2d_GPU_kernel_vx_backward<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_rho, 
				 plan[i].d_vx_inv, plan[i].d_p_inv,
				 ntp, ntx, ntz, pml, Lc, plan[i].d_rc, dx, dz, -dt
				);

			fdtd_2d_GPU_kernel_vz_backward<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_rho,
				 plan[i].d_vz_inv, plan[i].d_p_inv, 
				 ntp, ntx, ntz, pml, Lc, plan[i].d_rc, dx, dz, -dt
				);
*/
			///////////////////////////////////////////////////////////////////////

			fdtd_cpml_2d_GPU_kernel_vx_backward<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_rho, itmax,
				 plan[i].d_a_x_half, plan[i].d_a_z,
				 plan[i].d_b_x_half, plan[i].d_b_z,
				 plan[i].d_vx, plan[i].d_p,
				 plan[i].d_phi_p_x,
				 plan[i].d_psdvxx, plan[i].d_psdvxy, ifreq, freqintv, fs[ifreq], plan[i].d_randnum,
				 ntp, ntx, ntz, -dx, -dz, dt,
				 iter, es[is+i].num, it, pml, Lc, plan[i].d_rc
				);

			fdtd_cpml_2d_GPU_kernel_vz_backward<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_rho, itmax,
				 plan[i].d_a_x, plan[i].d_a_z_half,
				 plan[i].d_b_x, plan[i].d_b_z_half,
				 plan[i].d_vz, plan[i].d_p,
				 plan[i].d_phi_p_z,
				 plan[i].d_psdvzx, plan[i].d_psdvzy, ifreq, freqintv, fs[ifreq], plan[i].d_randnum,
				 ntp, ntx, ntz, -dx, -dz, dt,
				 iter, es[is+i].num, it, pml, Lc, plan[i].d_rc
				);

			fdtd_cpml_2d_GPU_kernel_p_backward<<<dimGrid,dimBlock,0,plans[i].stream>>>
				( 
				 plan[i].d_vp, plan[i].d_rho, itmax,
				 plan[i].d_a_x, plan[i].d_a_z, plan[i].d_b_x, plan[i].d_b_z,
				 plan[i].d_vx, plan[i].d_vz, plan[i].d_p,
				 plan[i].d_phi_vx_x, plan[i].d_phi_vz_z,
				 plan[i].d_psdpx, plan[i].d_psdpy, ifreq, freqintv, fs[ifreq], plan[i].d_randnum,
				 ntp, ntx, ntz, pml, Lc, plan[i].d_rc,
				 plan[i].d_seismogram_rms, es[is+i].r_iz, plan[i].d_r_ix, es[is+i].r_n,
				 -dx, -dz, dt, iter, es[is+i].num, it
				);
/*
			sum_image_GPU_kernel_image<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
				 plan[i].d_vp, plan[i].d_rho,
				 plan[i].d_p_inv, plan[i].dp_dt, plan[i].d_p, plan[i].d_vx, plan[i].d_vz,
				 plan[i].d_image_vp, plan[i].d_image_rho, plan[i].d_image_sources, plan[i].d_image_receivers,
				 ntx, ntz, pml, Lc, plan[i].d_rc, dx, dz
				);

			if(it%10==0)
			{
				cudaMemcpyAsync(vx,plan[i].d_p_inv,sizeof(float)*ntp,cudaMemcpyDeviceToHost,plans[i].stream);
				cudaMemcpyAsync(vz,plan[i].d_p,sizeof(float)*ntp,cudaMemcpyDeviceToHost,plans[i].stream);
				cudaStreamSynchronize(plans[i].stream);

				sprintf(filename,"../output/%dvx%d_inv.dat",it,i);     
				fp=fopen(filename,"wb");
				for(ix=pmlc;ix<ntx-pmlc;ix++)
				{
					for(iz=pmlc;iz<ntz-pmlc;iz++)
					{					
						fwrite(&vx[iz*ntx+ix],sizeof(float),1,fp);
					}
				}
				fclose(fp);

				sprintf(filename,"../output/%dvx%d_bak.dat",it,i);     
				fp=fopen(filename,"wb");
				for(ix=pmlc;ix<ntx-pmlc;ix++)
				{
					for(iz=pmlc;iz<ntz-pmlc;iz++)
					{					
						fwrite(&vz[iz*ntx+ix],sizeof(float),1,fp);
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
		
		cudaMemcpyAsync(plan[i].psdpx,plan[i].d_psdpx,size_model*NN,cudaMemcpyDeviceToHost,plans[i].stream);
		cudaMemcpyAsync(plan[i].psdpy,plan[i].d_psdpy,size_model*NN,cudaMemcpyDeviceToHost,plans[i].stream);
/*
		cudaMemcpyAsync(plan[i].psdvxx,plan[i].d_psdvxx,size_model*NN,cudaMemcpyDeviceToHost,plans[i].stream);
		cudaMemcpyAsync(plan[i].psdvxy,plan[i].d_psdvxy,size_model*NN,cudaMemcpyDeviceToHost,plans[i].stream);

		cudaMemcpyAsync(plan[i].psdvzx,plan[i].d_psdvzx,size_model*NN,cudaMemcpyDeviceToHost,plans[i].stream);
		cudaMemcpyAsync(plan[i].psdvzy,plan[i].d_psdvzy,size_model*NN,cudaMemcpyDeviceToHost,plans[i].stream);

		cudaMemcpyAsync(plan[i].image_vp,plan[i].d_image_vp,sizeof(float)*ntp,cudaMemcpyDeviceToHost,plans[i].stream);
		cudaMemcpyAsync(plan[i].image_rho,plan[i].d_image_rho,sizeof(float)*ntp,cudaMemcpyDeviceToHost,plans[i].stream);

		cudaMemcpyAsync(plan[i].image_sources,plan[i].d_image_sources,sizeof(float)*ntp,cudaMemcpyDeviceToHost,plans[i].stream);
		cudaMemcpyAsync(plan[i].image_receivers,plan[i].d_image_receivers,sizeof(float)*ntp,cudaMemcpyDeviceToHost,plans[i].stream);
*/
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


//*******************************************************//
//*******************Laplacian Filter********************//
extern "C"
void laplacegpu_filter(float *pp_pre,float *pp_f,int nxx,
		int nzz,float dx,float dz)
{ 
	int ix,iz,ip,K,NX,NZ;

	float pi=3.1415926;

	K=(int)ceil(log(1.0*nxx)/log(2.0));
	NX=(int)pow(2.0,K);

	K=(int)ceil(log(1.0*nzz)/log(2.0));
	NZ=(int)pow(2.0,K);

	float dkx,dkz;
	float kx,kz;

	dkx=(float)1.0/((NX)*dx);
	dkz=(float)1.0/((NZ)*dz);

	int NTP=NX*NZ;

	cufftComplex *pp,*temp,*tempout;		

	cudaMallocHost((void **)&pp, sizeof(cufftComplex)*NX*NZ);
	cudaMalloc((void **)&temp,sizeof(cufftComplex)*NX*NZ);
	cudaMalloc((void **)&tempout,sizeof(cufftComplex)*NX*NZ);

	cufftHandle plan;
	cufftPlan2d(&plan,NX,NZ,CUFFT_C2C);

	for(ip=0;ip<NTP;ip++)
	{ 
		pp[ip].x=0.0;
		pp[ip].y=0.0; 
	} 

	for(ix=0;ix<nxx;ix++)
	{            
		for(iz=0;iz<nzz;iz++)
		{
			pp[ix*NZ+iz].x=pp_pre[iz*nxx+ix];
		}
	} 

	cudaMemcpy(temp,pp,sizeof(cufftComplex)*NX*NZ,cudaMemcpyHostToDevice);
	cufftExecC2C(plan,temp,tempout,CUFFT_FORWARD);
	cudaMemcpy(pp,tempout,sizeof(cufftComplex)*NX*NZ,cudaMemcpyDeviceToHost);

	for(ix=0;ix<NX;ix++)
	{            
		for(iz=0;iz<NZ;iz++)
		{
			if(ix<NX/2)
			{
				kx=2*pi*ix*dkx;
			}
			if(ix>NX/2)	
			{
				kx=2*pi*(NX-1-ix)*dkx;
			}

			if(iz<NZ/2)
			{
				kz=2*pi*iz*dkz;//2*pi*(NZ/2-1-iz)*dkz;//0.0;//
			}
			if(iz>NZ/2)
			{
				kz=2*pi*(NZ-1-iz)*dkz;//2*pi*(iz-NZ/2)*dkz;//0.0;//
			}

			ip=ix*NZ+iz;

			pp[ip].x=pp[ip].x*(kx*kx+kz*kz);
			pp[ip].y=pp[ip].y*(kx*kx+kz*kz);

		}
	} 

	// fft(r_real,r_imag,NFFT,-1);

	cudaMemcpy(temp,pp,sizeof(cufftComplex)*NX*NZ,cudaMemcpyHostToDevice);
	cufftExecC2C(plan,temp,tempout,CUFFT_INVERSE);
	cudaMemcpy(pp,tempout,sizeof(cufftComplex)*NX*NZ,cudaMemcpyDeviceToHost);

	for(ix=0;ix<nxx;ix++)
	{            
		for(iz=0;iz<nzz;iz++)
		{
			pp_f[iz*nxx+ix]=pp[ix*NZ+iz].x;
		}
	} 

	cudaFreeHost(pp);
	cudaFree(temp);
	cudaFree(tempout);
	cufftDestroy(plan);

	return;
}

/*=============================================
 * Allocate the memory for wavefield simulation
 * ===========================================*/
extern "C"
void variables_malloc(int ntx, int ntz, int ntp, int nx, int nz,
		int pml, int Lc, float dx, float dz, int itmax,
		struct MultiGPU plan[], int GPU_N, int rnmax, int Nf, int NN
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
		//cudaMalloc((void**)&plan[i].d_seismogram,sizeof(float)*itmax*rnmax);
		//cudaMalloc((void**)&plan[i].d_seismogram_rms,sizeof(float)*itmax*rnmax);
		cudaMalloc((void**)&plan[i].d_seismogram,sizeof(float)*itmax*nx);
		cudaMalloc((void**)&plan[i].d_seismogram_rms,sizeof(float)*itmax*nx);

		cudaMalloc((void**)&plan[i].d_s_ix,sizeof(int)*NN);
		//cudaMalloc((void**)&plan[i].d_r_ix,sizeof(int)*rnmax);
		cudaMalloc((void**)&plan[i].d_r_ix,sizeof(int)*nx);

		cudaMalloc((void**)&plan[i].d_rick,sizeof(float)*NN*itmax);        // ricker wave 
		cudaMalloc((void**)&plan[i].d_rc,sizeof(float)*Lc);        // ricker wave 

//		cudaMalloc((void**)&plan[i].d_asr,sizeof(float)*NN);        // ricker wave 
		cudaMalloc((void**)&plan[i].d_randnum,sizeof(int)*NN);        // randnom number 


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

		cudaMalloc((void**)&plan[i].d_psdptx,size_model*NN);
		cudaMalloc((void**)&plan[i].d_psdpty,size_model*NN);
/*
		cudaMalloc((void**)&plan[i].d_psdvxx,size_model*NN);
		cudaMalloc((void**)&plan[i].d_psdvxy,size_model*NN);

		cudaMalloc((void**)&plan[i].d_psdvzx,size_model*NN);
		cudaMalloc((void**)&plan[i].d_psdvzy,size_model*NN);
*/
		cudaMalloc((void**)&plan[i].d_psdpx,size_model*NN);
		cudaMalloc((void**)&plan[i].d_psdpy,size_model*NN);

		cudaMalloc((void**)&plan[i].d_vx,size_model);
		cudaMalloc((void**)&plan[i].d_vz,size_model);
		cudaMalloc((void**)&plan[i].d_p,size_model);

		cudaMalloc((void**)&plan[i].dp_dt,size_model);
/*
		cudaMalloc((void**)&plan[i].d_vx_inv,size_model);
		cudaMalloc((void**)&plan[i].d_vz_inv,size_model);
		cudaMalloc((void**)&plan[i].d_p_inv,size_model);
*/
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
		struct MultiGPU plan[], int GPU_N, int rnmax, int Nf, int NN
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
		cudaFree(plan[i].d_randnum);


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
/*
		cudaFree(plan[i].d_psdvxx);
		cudaFree(plan[i].d_psdvxy);

		cudaFree(plan[i].d_psdvzx);
		cudaFree(plan[i].d_psdvzy);
*/
		cudaFree(plan[i].d_psdpx);
		cudaFree(plan[i].d_psdpy);

		cudaFree(plan[i].d_vx);
		cudaFree(plan[i].d_vz);
		cudaFree(plan[i].d_p);

		cudaFree(plan[i].dp_dt);
/*
		cudaFree(plan[i].d_vx_inv);
		cudaFree(plan[i].d_vz_inv);
		cudaFree(plan[i].d_p_inv);
*/
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
void ricker_fre(float *rick, int is, struct Encode es[], int GPU_N, struct MultiGPU plan[], int ifreq, int freqintv, float *fs,  int *randnum,
		int itmax, float dt, float dx, int nx, int pml, int ricker_flag)
{ 
	int i;

	int ix,it,itt,K,NX,wn; 

	K=(int)ceil(log(1.0*itmax)/log(2.0));
	NX=itmax;//(int)pow(2.0,K);	

	float df=1/(NX*dt);
	float dw=2*PI*df;

	float rkmax;

	FILE *fp;
	char filename[50];

	if(ricker_flag==0||ricker_flag==2)
	{
		for(i=0;i<GPU_N;i++)
		{
			for(ix=0;ix<es[is+i].num;ix++)
			{
				wn=(int)((fs[ifreq]+randnum[ix]*freqintv*df)/df);
				for(it=0;it<itmax;it++)
				{
					plan[i].rick[ix*itmax+it]=sin(wn*dw*it*dt);//+PI/6
				} 
			}

			sprintf(filename,"../output/%drick.dat",is+i+1);
			fp=fopen(filename,"wb");
			fwrite(&plan[i].rick[0],sizeof(float),es[is+i].num*itmax,fp);
			fclose(fp);
		}//end GPU_N
	}//end ricker_flag
	if(ricker_flag==1)
	{
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

				//itt=(int)((fs[ifreq]+ix*freqintv*df)/df);
				itt=(int)((fs[ifreq]+randnum[ix]*freqintv*df)/df);

				rk[itt].x=rk0[itt].x;
				rk[itt].y=rk0[itt].y;

				//rk[NX-itt].x=rk0[NX-itt].x;
				//rk[NX-itt].y=rk0[NX-itt].y;

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

			sprintf(filename,"../output/%drick.dat",is+i+1);
			fp=fopen(filename,"wb");
			fwrite(&plan[i].rick[0],sizeof(float),es[is+i].num*itmax,fp);
			fclose(fp);

			cudaFreeHost(rk);
			cudaFreeHost(rk0);

			cudaFree(temp1);

			cufftDestroy(plan1);
		}
	}//end ricker_flag

	return;
}

/*=====================================================================
  This function is used for calculating the complex conjugate
  =====================================================================*/

extern "C"
void conjugate_fre(float *seismogram_rms, int i,
		int itmax, float dt, float dx, int nx, int pml)
{ 
	cudaSetDevice(i);

	int ix,it,itt,K,NX,ip;
	int BATCH=nx;

	K=(int)ceil(log(1.0*itmax)/log(2.0));
	NX=itmax;//(int)pow(2.0,K);	

	float df=1/(NX*dt);
	float smax;

	FILE *fp;
	char filename[30];

	int NTP=NX*BATCH;

	cufftComplex *d,*rms,*temp;

	cudaMallocHost((void **)&d, sizeof(cufftComplex)*NX*BATCH);
	cudaMallocHost((void **)&rms,sizeof(cufftComplex)*NX*BATCH);

	cudaMalloc((void **)&temp,sizeof(cufftComplex)*NX*BATCH);

	cufftHandle plan2;
	cufftPlan1d(&plan2,NX,CUFFT_C2C,BATCH);

	for(it=0;it<NTP;it++)
	{ 
		d[it].x=0.0;
		d[it].y=0.0;   
		rms[it].x=0.0;
		rms[it].y=0.0;
	}            
	for(ix=0;ix<nx;ix++)
	{
		for(it=0;it<itmax;it++)
		{
			d[ix*NX+it].x=seismogram_rms[ix*itmax+it];	
		}
	}   

	cudaMemcpy(temp,d,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
	cufftExecC2C(plan2,temp,temp,CUFFT_FORWARD);
	cudaMemcpy(d,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

	for(ix=0;ix<nx;ix++)
	{
		for(it=0;it<NX;it++)
		{
			ip=ix*NX+it;

			rms[ip].x=d[ip].x;
			rms[ip].y=-1.0*d[ip].y;
		}
	}   

	// fft(r_real,r_imag,NFFT,-1);

	cudaMemcpy(temp,rms,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
	cufftExecC2C(plan2,temp,temp,CUFFT_INVERSE);
	cudaMemcpy(rms,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

	smax=0.0;
	for(ix=0;ix<nx;ix++)
	{
		for(it=0;it<itmax;it++)
		{
			seismogram_rms[ix*itmax+it]=rms[ix*NX+it].x;
//			if(smax<fabs(rms[ix*NX+it].x))
//				smax=fabs(rms[ix*NX+it].x);
		}
	}
/*	for(ix=0;ix<nx;ix++)
	{
		for(it=0;it<itmax;it++)
		{
			seismogram_rms[ix*itmax+it]/=smax;
		}
	}
*/
	cudaFreeHost(d);
	cudaFreeHost(rms); 

	cudaFree(temp);

	cufftDestroy(plan2);

	return;
}


/*=====================================================================
  This function is used for calculating the angular frequency components
  =====================================================================*/

extern "C"
void seismgobs_fre(float *seismogram_obs, float *seismogram_tmp, int ii, int ifreq, int freqintv, float *fs, int *randnum, int i,
		int itmax, float dt, float dx, int nx, int pml)
{ 
	cudaSetDevice(i);

	int ix,it,itt,K,NX,ifq;
	int BATCH=nx;

	K=(int)ceil(log(1.0*itmax)/log(2.0));
	NX=itmax;//(int)pow(2.0,K);	//

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
			d[ix*NX+it].x=seismogram_obs[ix*itmax+it];	
		}
	}   

	cudaMemcpy(temp,d,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
	cufftExecC2C(plan2,temp,temp,CUFFT_FORWARD);
	cudaMemcpy(d,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

	for(ix=0;ix<nx;ix++)
	{
		//itt=(int)((fs[ifreq]+ii*freqintv*df)/df);
		itt=(int)((fs[ifreq]+randnum[ii]*freqintv*df)/df);

		obs[ix*NX+itt].x=d[ix*NX+itt].x;
		obs[ix*NX+itt].y=d[ix*NX+itt].y;

		//	obs[ix*NX+NX-itt].x=d[ix*NX+NX-itt].x;
		//	obs[ix*NX+NX-itt].y=d[ix*NX+NX-itt].y;
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
			seismogram_tmp[ix*itmax+it]=obs[ix*NX+it].x;
			if(smax<fabs(obs[ix*NX+it].x))
				smax=fabs(obs[ix*NX+it].x);
		}
	}
	for(ix=0;ix<nx;ix++)
	{
		for(it=0;it<itmax;it++)
		{
			seismogram_tmp[ix*itmax+it]/=smax;
		}
	}

	cudaFreeHost(d);
	cudaFreeHost(obs); 

	cudaFree(temp);

	cufftDestroy(plan2);

	return;
}

/*=====================================================================
  This function is used for calculating the angular frequency components
  =====================================================================*/

extern "C"
void seismgsyn_fre(float *seismogram_syn, float *seismogram_tmp, int is, struct Encode es[], int ifreq, int freqintv, float *fs, int *randnum, int i,
		int itmax, float dt, float dx, int nx, int pmlc)
{ 
	cudaSetDevice(i);

	int ix,it,itt,K,NX,ifq;
	int BATCH=nx;

	K=(int)ceil(log(1.0*itmax)/log(2.0));
	NX=itmax;//(int)pow(2.0,K);	

	float df=1/(NX*dt);
	float smax;

	FILE *fp;
	char filename[30];

	int NTP=NX*BATCH;

	cufftComplex *d,*syn,*temp;

	cudaMallocHost((void **)&d, sizeof(cufftComplex)*NX*BATCH);
	cudaMallocHost((void **)&syn,sizeof(cufftComplex)*NX*BATCH);

	cudaMalloc((void **)&temp,sizeof(cufftComplex)*NX*BATCH);

	cufftHandle plan2;
	cufftPlan1d(&plan2,NX,CUFFT_C2C,BATCH);

	for(it=0;it<NTP;it++)
	{ 
		d[it].x=0.0;
		d[it].y=0.0;   
	}            
	for(ix=0;ix<nx;ix++)
	{
		for(it=0;it<itmax;it++)
		{
			d[ix*NX+it].x=seismogram_syn[ix*itmax+it];	
		}
	}   

	cudaMemcpy(temp,d,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
	cufftExecC2C(plan2,temp,temp,CUFFT_FORWARD);
	cudaMemcpy(d,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

	for(ifq=0;ifq<es[is+i].num;ifq++)
	{
		for(it=0;it<NTP;it++)
		{ 
			syn[it].x=0.0;
			syn[it].y=0.0;
		}            
		for(ix=0;ix<nx;ix++)
		{
			//itt=(int)((fs[ifreq]+ifq*freqintv*df)/df);
			itt=(int)((fs[ifreq]+randnum[ifq]*freqintv*df)/df);

			syn[ix*NX+itt].x=d[ix*NX+itt].x;
			syn[ix*NX+itt].y=d[ix*NX+itt].y;

			//	syn[ix*NX+NX-itt].x=d[ix*NX+NX-itt].x;
			//	syn[ix*NX+NX-itt].y=d[ix*NX+NX-itt].y;
		}   

		// fft(r_real,r_imag,NFFT,-1);

		cudaMemcpy(temp,syn,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
		cufftExecC2C(plan2,temp,temp,CUFFT_INVERSE);
		cudaMemcpy(syn,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

		smax=0.0;
		for(ix=0;ix<nx;ix++)
		{
			for(it=0;it<itmax;it++)
			{
				if(smax<fabs(syn[ix*NX+it].x))
					smax=fabs(syn[ix*NX+it].x);
			}
		}
		for(ix=0;ix<nx;ix++)
		{
			for(it=0;it<itmax;it++)
			{
				seismogram_tmp[ix*itmax+it]+=syn[ix*NX+it].x/smax;
			}
		}
	}//end ifq

	cudaFreeHost(d);
	cudaFreeHost(syn); 

	cudaFree(temp);

	cufftDestroy(plan2);

	return;
}

/*=====================================================================
  This function is used for calculating the residuals and misfit
  =====================================================================*/

extern "C"
void decongpu_fre(float *seismogram_syn, float *seismogram_obs, float *seismogram_rms, float *Misfit, int i, 
		int itmax, float dt, float dx, int is, int nx, struct Encode es[], struct Source ss[], int ifreq, int freqintv, float *fs, int *randnum, int pmlc)//float *ref_obs, float *ref_syn, 
{
	cudaSetDevice(i);

	int ix,it,ip,itt,K,NX;
	int ii,snum;
	float epsilon,rms,rmsmax;
	int BATCH=nx;

	K=(int)ceil(log(1.0*itmax)/log(2.0));
	NX=itmax;//(int)pow(2.0,K);	

	float df=1/(NX*dt);
	float h_sum,sh_sum;

	int reft,iref;

	FILE *fp;
	char filename[60];

	int sx,nn;

	int NTP=NX*BATCH;

	cufftComplex *xx,*d,*h,*sh,*r,*ms,*rr,*temp,*temp1,*obs;
	float *tmp_rms;

	cudaMallocHost((void **)&xx, sizeof(cufftComplex)*NX*BATCH);
	cudaMallocHost((void **)&d, sizeof(cufftComplex)*NX*BATCH);
	cudaMallocHost((void **)&h, sizeof(cufftComplex)*NX);
	cudaMallocHost((void **)&sh,sizeof(cufftComplex)*NX);
	cudaMallocHost((void **)&r, sizeof(cufftComplex)*NX*BATCH);
	cudaMallocHost((void **)&ms,sizeof(cufftComplex)*NX*BATCH);
	cudaMallocHost((void **)&rr,sizeof(cufftComplex)*NX*BATCH);
	cudaMallocHost((void **)&obs,sizeof(cufftComplex)*NX*BATCH);

	tmp_rms=(float *)malloc(sizeof(float)*nx*itmax);

	cudaMalloc((void **)&temp1,sizeof(cufftComplex)*NX);
	cudaMalloc((void **)&temp,sizeof(cufftComplex)*NX*BATCH);

	////////////////////////////////////////////////////
	cufftHandle plan1,plan2;
	cufftPlan1d(&plan1,NX,CUFFT_C2C,1);
	cufftPlan1d(&plan2,NX,CUFFT_C2C,BATCH);

	////////////////////////////////////////////////////
	for(ii=0;ii<es[is+i].num;ii++)
	{
		rmsmax=0.0;
		//itt=(int)((fs[ifreq]+ii*freqintv*df)/df);
		itt=(int)((fs[ifreq]+randnum[ii]*freqintv*df)/df);

		snum=es[is+i].offset+ii;

		sx=ss[snum].s_ix-pmlc;//ss[snum].r_ix[0];//
		reft=sx;
		nn=3;
		if(sx+nn/2>(ss[snum].r_ix[ss[snum].r_n-1]-pmlc))
			reft=ss[snum].r_ix[ss[snum].r_n-1]-pmlc-nn;//sx-nn;
		else if(sx-nn/2<0)
			reft=0;//sx+nn;
		else
			reft=sx-nn/2;

		for(it=0;it<NX;it++)
		{ 
			h[it].x=0.0;
			h[it].y=0.0; 

			sh[it].x=0.0;
			sh[it].y=0.0;    
		}

		for(it=0;it<itmax;it++)
		{
			h[it].x =seismogram_obs[reft*itmax+it];//*ref_window[it]/nn;
			sh[it].x=seismogram_syn[reft*itmax+it];//*ref_window[it]/nn;
		}

		sprintf(filename,"../output/referenceobs%d.dat",is+1);
		fp=fopen(filename,"wb");
		for(it=0;it<itmax;it++)
			fwrite(&h[it].x,sizeof(float),1,fp);
		fclose(fp);

		sprintf(filename,"../output/referencesyn%d.dat",is+1);
		fp=fopen(filename,"wb");
		for(it=0;it<itmax;it++)
			fwrite(&sh[it].x,sizeof(float),1,fp);
		fclose(fp);

		cudaMemcpy(temp1,h,sizeof(cufftComplex)*NX,cudaMemcpyHostToDevice);
		cufftExecC2C(plan1,temp1,temp1,CUFFT_FORWARD);
		cudaMemcpy(h,temp1,sizeof(cufftComplex)*NX,cudaMemcpyDeviceToHost);

		cudaMemcpy(temp1,sh,sizeof(cufftComplex)*NX,cudaMemcpyHostToDevice);
		cufftExecC2C(plan1,temp1,temp1,CUFFT_FORWARD);
		cudaMemcpy(sh,temp1,sizeof(cufftComplex)*NX,cudaMemcpyDeviceToHost);  

		h_sum =0.0;
		sh_sum=0.0;
		for(it=0;it<NX;it++)
		{
			if(it!=itt)
			{
				h[it].x=0.0;
				h[it].y=0.0;

				sh[it].x=0.0;
				sh[it].y=0.0;
			}
			h_sum += h[it].x* h[it].x+ h[it].y* h[it].y;
			sh_sum+=sh[it].x*sh[it].x+sh[it].y*sh[it].y;
		}

		for(it=0;it<NTP;it++)
		{ 
			xx[it].x=0.0;
			xx[it].y=0.0; 
			d[it].x=0.0;
			d[it].y=0.0;   

			r[it].x=0.0;
			r[it].y=0.0;

			obs[it].x=0.0;
			obs[it].y=0.0;

			rr[it].x=0.0;
			rr[it].y=0.0;
		}            
		for(ix=0;ix<nx;ix++)
		{
			for(it=0;it<itmax;it++)
			{
				xx[ix*NX+it].x=seismogram_syn[ix*itmax+it];
				 d[ix*NX+it].x=seismogram_obs[ix*itmax+it];	
			}
		}   

		cudaMemcpy(temp,xx,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
		cufftExecC2C(plan2,temp,temp,CUFFT_FORWARD);
		cudaMemcpy(xx,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

		cudaMemcpy(temp,d,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
		cufftExecC2C(plan2,temp,temp,CUFFT_FORWARD);
		cudaMemcpy(d,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

		//-------------------------------------------------------------------//
		for(it=0;it<NTP;it++)
		{
			obs[it].x=(d[it].x*h[it%NX].x+d[it].y*h[it%NX].y)/h_sum;
			obs[it].y=(d[it].y*h[it%NX].x-d[it].x*h[it%NX].y)/h_sum;
			r[it].x=(xx[it].x*sh[it%NX].x+xx[it].y*sh[it%NX].y)/sh_sum-obs[it].x;
			r[it].y=(xx[it].y*sh[it%NX].x-xx[it].x*sh[it%NX].y)/sh_sum-obs[it].y;
		}  
		//-------------------------------------------------------------------//
		for(it=0;it<NTP;it++)
		{
			rr[it].x=(r[it].x*sh[it%NX].x-r[it].y*sh[it%NX].y)/sh_sum;
			rr[it].y=(r[it].y*sh[it%NX].x+r[it].x*sh[it%NX].y)/sh_sum;
		}   
/*
		for(it=0;it<NTP;it++)
		{
			obs[it].x=sh[it%NX].x*d[it].x-sh[it%NX].y*d[it].y;
			obs[it].y=sh[it%NX].x*d[it].y+sh[it%NX].y*d[it].x;
			r[it].x=xx[it].x*h[it%NX].x-xx[it].y*h[it%NX].y-obs[it].x;
			r[it].y=xx[it].x*h[it%NX].y+xx[it].y*h[it%NX].x-obs[it].y;
		}  
	

		// fft(r_real,r_imag,NFFT,-1);

		//-------------------------------------------------------------------//
		//-------------------------------------------------------------------//
		
		cudaMemcpy(temp,r,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
		cufftExecC2C(plan2,temp,temp,CUFFT_INVERSE);
		cudaMemcpy(ms,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

		cudaMemcpy(temp,obs,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
		cufftExecC2C(plan2,temp,temp,CUFFT_INVERSE);
		cudaMemcpy(obs,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

		epsilon=0.0;
		rms=0.0;
		for(ix=0;ix<nx;ix++)
		{
			for(it=0;it<itmax;it++)
			{
				epsilon=epsilon+9*fabs(obs[ix*NX+it].x)/(itmax*nx);
			}
		}
		for(ix=0;ix<nx;ix++)
		{
			for(it=0;it<itmax;it++)
			{
				rms=rms+ms[ix*NX+it].x*ms[ix*NX+it].x/(epsilon*epsilon);
			}
		}
		*Misfit+=sqrt(1+rms)-1;

		// Calculate the r of ( f= rXdref )	Right hide term of adjoint equation!!!
		for(it=0;it<NTP;it++)
		{
			ms[it].x=ms[it].x/(epsilon*epsilon*sqrt(1+ms[it].x*ms[it].x/(epsilon*epsilon)));  //Time domain ms.x==u*dref-vref*d
			ms[it].y=0.0;
		}

		cudaMemcpy(temp,ms,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
		cufftExecC2C(plan2,temp,temp,CUFFT_FORWARD);
		cudaMemcpy(r,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);
		
		//-------------------------------------------------------------------//
		//-------------------------------------------------------------------//

		for(it=0;it<NTP;it++)
		{
			rr[it].x=h[it%NX].x*r[it].x+h[it%NX].y*r[it].y;
			rr[it].y=h[it%NX].x*r[it].y-h[it%NX].y*r[it].x;
		}   
*/
		//fft(rr_real,rr_imag,NX*BATCH,-1);

		cudaMemcpy(temp,rr,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
		cufftExecC2C(plan2,temp,temp,CUFFT_INVERSE);
		cudaMemcpy(rr,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

		for(ix=0;ix<nx;ix++)
		{
			for(it=0;it<itmax;it++)
			{
				ip=ix*itmax+it;

				tmp_rms[ip]=rr[ix*NX+it].x;

				if(rmsmax<fabs(tmp_rms[ip]))
				{
					rmsmax=fabs(tmp_rms[ip]);
				}
			}//end iz
		} //end ix

		for(ip=0;ip<itmax*nx;ip++)
		{
			seismogram_rms[ip]+=tmp_rms[ip]/rmsmax;
			*Misfit+=pow(seismogram_rms[ip],2.0);
		}
	}//end es.num

	cudaFreeHost(xx);
	cudaFreeHost(d);
	cudaFreeHost(h);
	cudaFreeHost(sh);
	cudaFreeHost(r);   
	cudaFreeHost(ms);   
	cudaFreeHost(rr);
	cudaFreeHost(obs); 

	free(tmp_rms); 

	cudaFree(temp);
	cudaFree(temp1);

	cufftDestroy(plan1);
	cufftDestroy(plan2);

	return;
}


extern "C"
void seismgrms_fre(float *seismogram_syn, float *seismogram_obs, float *seismogram_rms, int is, struct Encode es[], struct Source ss[], int ifreq, int freqintv, float *fs, int *randnum, int i,
		int itmax, float dt, float dx, int nx, int pmlc)
{ 
	cudaSetDevice(i);

	int ip,snum;
	int ix,it,itt,K,NX,ifq;
	int BATCH=nx;

	K=(int)ceil(log(1.0*itmax)/log(2.0));
	NX=itmax;//(int)pow(2.0,K);	

	float df=1/(NX*dt);
	float smax;

	FILE *fp;
	char filename[30];

	int NTP=NX*BATCH;

	cufftComplex *obs,*syn,*rms,*temp;

	cudaMallocHost((void **)&obs,sizeof(cufftComplex)*NX*BATCH);
	cudaMallocHost((void **)&syn,sizeof(cufftComplex)*NX*BATCH);
	cudaMallocHost((void **)&rms,sizeof(cufftComplex)*NX*BATCH);

	cudaMalloc((void **)&temp,sizeof(cufftComplex)*NX*BATCH);

	cufftHandle plan2;
	cufftPlan1d(&plan2,NX,CUFFT_C2C,BATCH);

	for(it=0;it<NTP;it++)
	{ 
		obs[it].x=0.0;
		obs[it].y=0.0;   

		syn[it].x=0.0;
		syn[it].y=0.0;

		rms[it].x=0.0;
		rms[it].y=0.0;
	}            
	for(ix=0;ix<nx;ix++)
	{
		for(it=0;it<itmax;it++)
		{
			obs[ix*NX+it].x=seismogram_obs[ix*itmax+it];	
			syn[ix*NX+it].x=seismogram_syn[ix*itmax+it];	
		}
	}   

	cudaMemcpy(temp,obs,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
	cufftExecC2C(plan2,temp,temp,CUFFT_FORWARD);
	cudaMemcpy(obs,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

	cudaMemcpy(temp,syn,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
	cufftExecC2C(plan2,temp,temp,CUFFT_FORWARD);
	cudaMemcpy(syn,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

	for(ifq=0;ifq<es[is+i].num;ifq++)
	{
		//itt=(int)((fs[ifreq]+ifq*freqintv*df)/df);
		itt=(int)((fs[ifreq]+randnum[ifq]*freqintv*df)/df);
		snum=es[is+i].offset+ifq;

		for(it=0;it<NTP;it++)
		{ 
			rms[it].x=0.0;
			rms[it].y=0.0;
		}            

		for(ix=0;ix<ss[snum].r_n;ix++)
		{
			ip=(ss[snum].r_ix[ix]-pmlc)*NX+itt;

			rms[ip].x=syn[ip].x-obs[ip].x;
			rms[ip].y=syn[ip].y-obs[ip].y;

			//syn[ix*NX+itt].x=obs[ix*NX+itt].x;
			//syn[ix*NX+itt].y=obs[ix*NX+itt].y;

			////syn[ix*NX+NX-itt].x=obs[ix*NX+NX-itt].x;
			////syn[ix*NX+NX-itt].y=obs[ix*NX+NX-itt].y;
		}   

		// fft(r_real,r_imag,NFFT,-1);

		cudaMemcpy(temp,rms,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyHostToDevice);
		cufftExecC2C(plan2,temp,temp,CUFFT_INVERSE);
		cudaMemcpy(rms,temp,sizeof(cufftComplex)*NX*BATCH,cudaMemcpyDeviceToHost);

	/*	smax=0.0;
		for(ix=0;ix<nx;ix++)
		{
			for(it=0;it<itmax;it++)
			{
				if(smax<fabs(rms[ix*NX+it].x))
					smax=fabs(rms[ix*NX+it].x);
			}
		}*/
		for(ix=0;ix<nx;ix++)
		{
			for(it=0;it<itmax;it++)
			{
				//seismogram_rms[ix*itmax+it]+=syn[ix*NX+it].x/smax;
				
				seismogram_rms[ix*itmax+it]+=rms[ix*NX+it].x;
			}
		}
	}//end ifq

	cudaFreeHost(obs);
	cudaFreeHost(syn); 
	cudaFreeHost(rms); 

	cudaFree(temp);

	cufftDestroy(plan2);

	return;
}

