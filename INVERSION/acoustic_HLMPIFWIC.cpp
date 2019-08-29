#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define PI 3.1415926

#include "fftw3.h"
#include "mpi.h"
#include "omp.h"

#include "headmulti.h"

//#define ricker_flag 1
//#define noise_flag 0
//#define seislet_regularization 1

int main(int argc, char *argv[])
{
	int myid,numprocs,namelen,index;
	
	MPI_Comm comm=MPI_COMM_WORLD;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(comm,&myid);
	MPI_Comm_size(comm,&numprocs);
	MPI_Get_processor_name(processor_name,&namelen);

	if(myid==0)
		printf("Number of MPI thread is %d\n",numprocs);

	/*=========================================================
	  Parameters of the time of the system...
	  =========================================================*/
	time_t begin_time;
	//  time_t end_time;
	//  time_t last_time;

	clock_t start;
	clock_t end;

	//  float runtime=0.0;
	int nx,nz;
	int pml,Lc;

	int ip,ipp,iz,ix,it;

	float dx,dz;

	float rectime,dt;
	float f0;
	int Nf,freqintv;
	float freq0;

	int ns;
	float sx0,shotdx,shotdep,recdep,moffsetx;

	int itn,iterb,ifreqb;
	int Ns[10]; // Nf < 10
	int i,ii,jj,GPU_N;
	int ricker_flag,noise_flag;

	input_parameters(&nx,&nz,&pml,&Lc,&dx,&dz,&rectime,&dt,&f0,&Nf,&freqintv,&freq0,&ns,&sx0,
			&shotdx,&shotdep,&recdep,&moffsetx,&itn,&iterb,&ifreqb,Ns,&ricker_flag,&noise_flag);//&GPU_N);
	//////////////////////////
	if(Nf>=10&&myid==0)
	{
		printf("Nf < 10!!!\n");
		return(0);
	}
	if(myid==0)
	{
		printf("The super source number is:");
		for(ip=0;ip<Nf;ip++)
			printf("%2d,",Ns[ip]);
		printf("\n");
	}

	/*=========================================================
	  Parameters of Cartesian coordinate...
	  ========================================================*/  

	int pmlc=pml+Lc;

	int ntz=nz+2*pmlc;
	int ntx=nx+2*pmlc;
	int ntp=ntz*ntx;
	int np=nx*nz;

	/*=========================================================
	  Parameters of ricker wave...
	  ========================================================*/

	int   itmax=(int)(rectime/dt);
	if(itmax%10!=0)
		itmax=itmax+1;

	float *rick;
	float t0=1/f0;

	float df=1.0/(itmax*dt);

	if(myid==0)
		printf("The interval frequency is 1/(%d*%f) = %f Hz\n",itmax,dt,df);

	int ifreq;//
	float fs[Nf];
	float amin=nz/sqrt(pow(nx/4,2.0)+pow(nz,2.0));
	//float amin=nz/sqrt(pow(0.5*moffsetx/dx,2.0)+pow(nz,2.0));

	if(myid==0)
		printf("The frequencies are : \n");
	for(ip=0;ip<Nf;ip++)
	{
		if(ip==0)
			fs[ip]=freq0;
		else
			fs[ip]=floor((fs[ip-1]/amin)/df+0.5)*df;

		//if((int)(100*fs[ip])%(int)(200*df)!=0)
		{
			fs[ip]=((int)(100*fs[ip])/(int)(200*df))*2.0*df;
		}
		//fs[ip]=freq0+ip*freqintv*df;

		if(myid==0)
			printf("%f, ",fs[ip]);
	}
	if(myid==0)
		printf("\n");

	/*=========================================================
	  Iteration parameters...
	  ========================================================*/

	int iter;

	/*=========================================================
	  File name....
	 *========================================================*/

	FILE *fp;
	char filename[50];

	/*=========================================================
	  Parameters of ricker wave...
	  ========================================================*/

	float *rc;
	rc=(float*)malloc(sizeof(float)*Lc);
	cal_xishu(Lc,rc);

	int ic;
	float tmprc=0.0;
	for(ic=0;ic<Lc;ic++)
	{
		tmprc+=fabs(rc[ic]);
	}
	if(myid==0)
		printf("Maximum velocity for stability is %f m/s\n",dx/(tmprc*dt*sqrt(2)));

	/*=========================================================
	  Parameters of GPU...
	  ========================================================*/

	getdevice(&GPU_N);

	int GPU_NN=GPU_N;
	printf("The available Device number is %d on %s\n",GPU_N,processor_name);
	MPI_Barrier(comm);

	struct MultiGPU plan[GPU_N];

	/*=========================================================
	  Parameters of Sources and Receivers...
	  ========================================================*/
	int is,si,snum,rnmax=0;

	int nsid,modsr,prcs,prcss;
	int iss,eachsid,offsets;

	struct Source ss[ns];

	///////trace number of each source////////
	for(is=0;is<ns;is++)
	{
		ss[is].s_ix=pmlc+(int)(sx0/dx)+(int)(shotdx/dx)*is;//18+is*16;//29+is*55;//

		ss[is].s_iz=pmlc+(int)(shotdep/dz);
		ss[is].r_iz=pmlc+(int)(recdep/dz);

		//ss[is].r_n=srn[is];//nx;
		i=0;
		for(ix=0;ix<nx;ix++)
		{
			if(fabs((ss[is].s_ix-ix-pmlc)*dx)<=moffsetx)
				i++;
		}
		ss[is].r_n=i;
	}

	for(is=0;is<ns;is++)
	{
		if(rnmax<ss[is].r_n)
			rnmax=ss[is].r_n;
	}

	if(myid==0)
		printf("The maximum trace number for source is %d\n",rnmax);

	for(is=0;is<ns;is++)
	{
		ss[is].r_ix=(int*)malloc(sizeof(int)*ss[is].r_n);
	} 

	for(is=0;is<ns;is++)
	{
		/*for(ip=0;ip<ss[is].r_n;ip++)
		{
			ss[is].r_ix[ip]=pmlc+ip;
		}*/
		i=0;
		for(ix=0;ix<nx;ix++)
		{
			if(fabs((ss[is].s_ix-ix-pmlc)*dx)<=moffsetx)
			{
				ss[is].r_ix[i]=pmlc+ix;
				i++;
			}
		}
		if(i>ss[is].r_n)
		{
			printf("The trace number of %d th source is out of range!\n",is+1);
			return(0);
		}
	}

	/*=========================================================
	  Parameters of the encoded sources...
	  ========================================================*/
	int Nsmax=0;
	for(ip=0;ip<Nf;ip++)
	{
		if(Nsmax<Ns[ip])
			Nsmax=Ns[ip];
	}

	struct Encode es[Nsmax];
	int nn=ns/Ns[Nf-1];
	int NNmax=0;

	for(is=0;is<Nsmax;is++)
	{
		if(is<ns%Nsmax)
		{
			es[is].num=nn+1;
			es[is].offset=is*(nn+1);
		}
		else
		{
			es[is].num=nn;
			es[is].offset=(ns%Nsmax)*(nn+1)+(is-ns%Nsmax)*nn;
		}
		if(NNmax<es[is].num)
			NNmax=es[is].num;

		es[is].r_n=nx;
	}
	for(is=0;is<Nsmax;is++)
	{
		es[is].s_ix=(int *)malloc(sizeof(int)*es[is].num);
		es[is].r_ix=(int *)malloc(sizeof(int)*es[is].r_n);

		memset(es[is].r_ix,ntx,sizeof(int)*es[is].r_n);
	}
/*
	for(is=0;is<Nsmax;is++)
	{
		sprintf(filename,"../output/%dsource.txt",is+1);
		fp=fopen(filename,"wt");
		for(ix=0;ix<es[is].num;ix++)
		{
			fprintf(fp,"%d %d %d\n",es[is].s_ix[ix],es[is].s_iz,es[is].offset);
		}
		fclose(fp);
	}
*/
	if(myid==0)
		printf("Maximum encoded source num is %d\n",NNmax);

	int *rnum=(int *)malloc(sizeof(int)*NNmax);

	/*=========================================================
	  Parameters of model...
	  ========================================================*/

	float *vp,*rho;
	float *vpn,*rhon;
	float vp_max,rho_max;
	float vp_min,rho_min;

	/*=========================================================
	  Parameters of absorbing layers...
	  ========================================================*/

	float *d_x,*d_x_half,*d_z,*d_z_half;
	float *a_x,*a_x_half,*a_z,*a_z_half;
	float *b_x,*b_x_half,*b_z,*b_z_half;
	float *k_x,*k_x_half,*k_z,*k_z_half;

	/*=========================================================
	  Parameters of Seismograms and Borders...
	  ========================================================*/

	float *ref_window,*seis_window;
	float *ref_obs,*ref_syn;

	float synmax,obsmax;

	/*=========================================================
	  Image / gradient ...
	 *========================================================*/

	float *gradient_vp_all,*gradient_rho_all;

	float *gradient_vp,*gradient_rho;
	float *conjugate_vp,*conjugate_rho;

	float *gradient_vp_pre,*gradient_rho_pre;
	float *conjugate_vp_pre,*conjugate_rho_pre;
	float *step_vp,*step_rho;

	float *tmp1,*tmp2;

	step_vp =(float*)malloc(sizeof(float)*1);
	step_rho =(float*)malloc(sizeof(float)*1);

	/*=========================================================
	  Flags ....
	 *========================================================*/

	int flag,inv_flag;

	//#######################################################################
	// NOW THE PROGRAM BEGIN
	//#######################################################################

	time(&begin_time);
	if(myid==0)
		printf("Today's data and time: %s",ctime(&begin_time));

	/*=========================================================
	  Allocate the memory of parameters of ricker wave...
	  ========================================================*/

	rick=(float*)malloc(sizeof(float)*itmax);

	/*=========================================================
	  Allocate the memory of parameters of model...
	  ========================================================*/

	// allocate the memory of model parameters...

	vp                  = (float*)malloc(sizeof(float)*ntp);
	rho                 = (float*)malloc(sizeof(float)*ntp);

	vpn                  = (float*)malloc(sizeof(float)*ntp);
	rhon                 = (float*)malloc(sizeof(float)*ntp);

	/*=========================================================
	  Allocate the memory of parameters of absorbing layer...
	  ========================================================*/

	d_x      = (float*)malloc(ntx*sizeof(float));
	d_x_half = (float*)malloc(ntx*sizeof(float));    
	d_z      = (float*)malloc(ntz*sizeof(float));
	d_z_half = (float*)malloc(ntz*sizeof(float));


	a_x      = (float*)malloc(ntx*sizeof(float));
	a_x_half = (float*)malloc(ntx*sizeof(float));    
	a_z      = (float*)malloc(ntz*sizeof(float));
	a_z_half = (float*)malloc(ntz*sizeof(float));


	b_x      = (float*)malloc(ntx*sizeof(float));
	b_x_half = (float*)malloc(ntx*sizeof(float));
	b_z      = (float*)malloc(ntz*sizeof(float));
	b_z_half = (float*)malloc(ntz*sizeof(float));


	k_x      = (float*)malloc(ntx*sizeof(float));
	k_x_half = (float*)malloc(ntx*sizeof(float));
	k_z      = (float*)malloc(ntz*sizeof(float));
	k_z_half = (float*)malloc(ntz*sizeof(float));  

	/*=========================================================
	  Allocate the memory of Seismograms...
	  ========================================================*/

	ref_window       =(float*)malloc(sizeof(float)*itmax);
	seis_window      =(float*)malloc(sizeof(float)*itmax);
	ref_obs          =(float*)malloc(sizeof(float)*itmax);
	ref_syn          =(float*)malloc(sizeof(float)*itmax);

	fftw_complex dpdx,dpdz;
	float srcmax;

	for(i=0;i<GPU_N;i++)
	{
		plan[i].rick=(float*)malloc(sizeof(float)*NNmax*itmax);

		plan[i].seismogram_obs=(float*)malloc(sizeof(float)*nx*itmax);
		plan[i].seismogram_syn=(float*)malloc(sizeof(float)*nx*itmax);
		plan[i].seismogram_rms=(float*)malloc(sizeof(float)*nx*itmax);

		plan[i].seismogram=(float*)malloc(sizeof(float)*nx*itmax);
		plan[i].seismogram_tmpobs=(float *)malloc(sizeof(float)*nx*itmax);
		plan[i].seismogram_tmpsyn=(float *)malloc(sizeof(float)*nx*itmax);

		/*======================================================
		  Allocate the memory of image / gradient..
		  =====================================================*/

		plan[i].image_vp=(float*)malloc(sizeof(float)*ntp);
		plan[i].image_rho=(float*)malloc(sizeof(float)*ntp);

		plan[i].image_sources=(float*)malloc(sizeof(float)*ntp);
		plan[i].image_receivers=(float*)malloc(sizeof(float)*ntp);

		///////////PSD parameters////////////////
		plan[i].psdptx=(float *)malloc(sizeof(float)*ntp*NNmax);
		plan[i].psdpty=(float *)malloc(sizeof(float)*ntp*NNmax);
		plan[i].psdptamp=(float *)malloc(sizeof(float)*ntp*NNmax);
		plan[i].psdpttheta=(float *)malloc(sizeof(float)*ntp*NNmax);
/*
		plan[i].psdvxx=(float *)malloc(sizeof(float)*ntp*NNmax);
		plan[i].psdvxy=(float *)malloc(sizeof(float)*ntp*NNmax);
		plan[i].psdvxamp=(float *)malloc(sizeof(float)*ntp*NNmax);
		plan[i].psdvxtheta=(float *)malloc(sizeof(float)*ntp*NNmax);

		plan[i].psdvzx=(float *)malloc(sizeof(float)*ntp*NNmax);
		plan[i].psdvzy=(float *)malloc(sizeof(float)*ntp*NNmax);
		plan[i].psdvzamp=(float *)malloc(sizeof(float)*ntp*NNmax);
		plan[i].psdvztheta=(float *)malloc(sizeof(float)*ntp*NNmax);
*/
		plan[i].psdpx=(float *)malloc(sizeof(float)*ntp*NNmax);
		plan[i].psdpy=(float *)malloc(sizeof(float)*ntp*NNmax);
		plan[i].psdpamp=(float *)malloc(sizeof(float)*ntp*NNmax);
		plan[i].psdptheta=(float *)malloc(sizeof(float)*ntp*NNmax);

		plan[i].ptff=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*ntp);
		plan[i].vxff=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*ntp);
		plan[i].vzff=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*ntp);
		plan[i].pff=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*ntp);
	}

	gradient_vp_all=(float*)malloc(sizeof(float)*ntp);
	gradient_rho_all=(float*)malloc(sizeof(float)*ntp);

	tmp1=(float*)malloc(sizeof(float)*ntp);
	tmp2=(float*)malloc(sizeof(float)*ntp);

	gradient_vp=(float*)malloc(sizeof(float)*np);
	gradient_rho=(float*)malloc(sizeof(float)*np);

	conjugate_vp=(float*)malloc(sizeof(float)*np);
	conjugate_rho=(float*)malloc(sizeof(float)*np);

	gradient_vp_pre=(float*)malloc(sizeof(float)*np);
	gradient_rho_pre=(float*)malloc(sizeof(float)*np);

	conjugate_vp_pre=(float*)malloc(sizeof(float)*np);
	conjugate_rho_pre=(float*)malloc(sizeof(float)*np);

	////////============================////////
	variables_malloc(ntx, ntz, ntp, nx, nz,
		pml, Lc, dx, dz, itmax,
		plan, GPU_N, rnmax, Nf, NNmax
		);

	/*=========================================================
	  Calculate the ricker wave...
	  ========================================================*/

	if(myid==0)
	{
		ricker_wave(rick,itmax,f0,t0,dt,2);
		printf("Ricker wave is done\n");
	}

	MPI_Barrier(comm);
	MPI_Bcast(rick,itmax,MPI_FLOAT,0,comm);

	/*=========================================================
	  Calculate the ture model.../Or read in the true model
	  ========================================================*/

	if(myid==0)
	{
		get_acc_model(vp,rho,ntp,ntx,ntz,pmlc);

		fp=fopen("../output/acc_vp.dat","wb");
		for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
		{
			for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
			{
				fwrite(&vp[iz*ntx+ix],sizeof(float),1,fp);

			}
		}
		fclose(fp);

		fp=fopen("../output/acc_rho.dat","wb");
		for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
		{
			for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
			{
				fwrite(&rho[iz*ntx+ix],sizeof(float),1,fp);

			}
		}
		fclose(fp);

		printf("The true model is done\n"); 

	}//end myid

	MPI_Barrier(comm);
	MPI_Bcast(vp,ntp,MPI_FLOAT,0,comm);
	MPI_Bcast(rho,ntp,MPI_FLOAT,0,comm);

	/////////////////////////////////////////////////////
	vp_max=0.0;
	rho_max=0.0;
	vp_min=5000.0;
	rho_min=5000.0;
	for(ip=0;ip<ntp;ip++)
	{     
		if(vp[ip]>=vp_max)
		{
			vp_max=vp[ip];
		}
		if(rho[ip]>=rho_max)
		{
			rho_max=fabs(rho[ip]);
		}
		if(vp[ip]<=vp_min)
		{
			vp_min=vp[ip];
		}
		if(rho[ip]<=rho_min)
		{
			rho_min=fabs(rho[ip]);
		}
	}
	if(myid==0)
	{
		printf("vp_max = %f\n",vp_max); 
		printf("rho_max = %f\n",rho_max);

		printf("vp_min = %f\n",vp_min); 
		printf("rho_min = %f\n",rho_min);
	}

	/*=========================================================
	  Calculate the parameters of absorbing layers...
	  ========================================================*/

	get_absorbing_parameters(
			d_x,d_x_half,d_z,d_z_half,
			a_x,a_x_half,a_z,a_z_half,
			b_x,b_x_half,b_z,b_z_half,
			k_x,k_x_half,k_z,k_z_half,
			ntz,ntx,nz,nx,pmlc,dx,f0,t0,
			dt,vp_max
			);

	if(myid==0)
	{
		printf("ABC parameters are done\n");
		start=clock();
	}

	/////   ini model and accurate reflectivity  //////
	///////////////////////////////////////////////////

	if(myid==0)
	{
		ini_model_mine(vp,vpn,ntp,ntz,ntx,pmlc,1);
		ini_model_mine(rho,rhon,ntp,ntz,ntx,pmlc,2);

		fp=fopen("../output/ini_vp.dat","wb");
		for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
		{
			for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
			{
				fwrite(&vp[iz*ntx+ix],sizeof(float),1,fp);
			}
		}
		fclose(fp);

		fp=fopen("../output/ini_rho.dat","wb");
		for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
		{
			for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
			{
				fwrite(&rho[iz*ntx+ix],sizeof(float),1,fp);
			}
		}
		fclose(fp);
	}//end myid

	MPI_Barrier(comm);
	MPI_Bcast(vp,ntp,MPI_FLOAT,0,comm);
	MPI_Bcast(rho,ntp,MPI_FLOAT,0,comm);

	MPI_Bcast(vpn,ntp,MPI_FLOAT,0,comm);
	MPI_Bcast(rhon,ntp,MPI_FLOAT,0,comm);

	//********************************************//
	//				FWI Parameters
	//********************************************//
	float sum1,sum2,beta;
	float sum1r,sum2r,betar;
	float misfit[itn];
	float Misfit_old,Misfit_new;
	float *Misfit;
	int reftrace;

	Misfit=(float *)malloc(sizeof(float));

	float Pr[nz];
	float Prf[nz];

	Preprocess(nz,nx,dx,dz,Pr,1);
	Preprocess(nz,nx,dx,dz,Prf,2);

	/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	  !	        ITERATION OF FWI IN TIME DOMAIN BEGINS...                      !
	  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

	/*=======================================================
	  Back-propagate the RMS wavefields and Construct 
	  the forward wavefield..Meanwhile the gradients 
	  of lambda and mu are computed... 
	  ========================================================*/
	if(myid==0)
	{
		printf("====================\n");
		printf("    FWI BEGIN\n");
		printf("====================\n");
	}

	inv_flag=1; //     FWI   FLAG

	for(ifreq=ifreqb;ifreq<Nf;ifreq++)
	{
		GPU_N=GPU_NN;

		if(myid==0)
		{
			printf("======================\n");
			printf("FREQUENCY == %d, %.2f Hz\n",ifreq+1,fs[ifreq]);
			printf("======================\n");
		}

		if(ifreqb!=0)
		{
			if(myid==0)
			{
				sprintf(filename,"../output/%difreq_vp.dat",ifreqb);
				fp=fopen(filename,"rb");
				for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
				{
					for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
					{
						fread(&vp[iz*ntx+ix],sizeof(float),1,fp);
					}
				}
				fclose(fp);

				/*sprintf(filename,"../output/%difreq_rho.dat",ifreqb);
				  fp=fopen(filename,"rb");
				  for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
				  {
				  for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
				  {
				  fread(&rho[iz*ntx+ix],sizeof(float),1,fp);
				  }
				  }
				  fclose(fp);*/
			}//end myid
			MPI_Barrier(comm);

			MPI_Bcast(vp,ntp,MPI_FLOAT,0,comm);

			for(ip=0;ip<ntp;ip++)
			{
				vpn[ip]=vp[ip];
			}

			ifreqb=0;
		}//end ifreqb

		/*==================================================================*/
		/*==================================================================*/
		nn=ns/Ns[ifreq];
		for(is=0;is<Ns[ifreq];is++)
		{
			es[is].r_iz=ss[is].r_iz;
			es[is].s_iz=ss[is].s_iz;

			if(is<ns%Ns[ifreq])
			{
				es[is].num=nn+1;
				es[is].offset=is*(nn+1);
			}
			else
			{
				es[is].num=nn;
				es[is].offset=(ns%Ns[ifreq])*(nn+1)+(is-ns%Ns[ifreq])*nn;
			}

			for(ix=0;ix<es[is].num;ix++)
			{
				es[is].s_ix[ix]=ss[es[is].offset+ix].s_ix;
			}
			/*for(ix=0;ix<es[is].r_n;ix++)
			{
				es[is].r_ix[ix]=pmlc+ix;
			}*/
			i=0;
			//es[is].r_ix[0]=ss[es[is].offset].r_ix[0];
			for(ii=0;ii<es[is].num;ii++)
			{
				snum=es[is].offset+ii;

				for(ix=0;ix<ss[snum].r_n;ix++)
				{
					flag=0;
					for(jj=0;jj<i;jj++)
					{
						if(es[is].r_ix[jj]==ss[snum].r_ix[ix])
							flag=1;
					}
					if(flag==0)
					{
						es[is].r_ix[i]=ss[snum].r_ix[ix];
						i++;
					}
				}//end ss.r_n
			}//end num
			//es[is].r_n=i;
			//printf("Super source %d, trace number %d\n",is+1,i);
		}//end is
		/*==================================================================*/
		nsid=Ns[ifreq]/(GPU_N*numprocs);
		modsr=Ns[ifreq]%(GPU_N*numprocs);
		prcs=modsr/GPU_N;
		prcss=modsr%GPU_N;
		if(prcss==0)
		{
			if(myid<prcs)
			{
				eachsid=nsid+1;

				offsets=myid*(nsid+1)*GPU_N;
			}
			else
			{
				eachsid=nsid;
				offsets=prcs*(nsid+1)*GPU_N+(myid-prcs)*nsid*GPU_N;
			}
		}
		else
		{
			if(myid<=prcs)
			{
				eachsid=nsid+1;

				offsets=myid*(nsid+1)*GPU_N;
			}
			else
			{
				eachsid=nsid;
				offsets=prcs*(nsid+1)*GPU_N+prcss*(nsid+1)+(GPU_N-prcss)*nsid+
					(myid-prcs-1)*nsid*GPU_N;
			}
		}
		/*==================================================================*/

		for(iter=iterb;iter<itn;iter++)
		{
			if(myid==0)
				printf(" The Iteration == %d\n",iter+1);

			if(iterb!=0)
			{
				if(myid==0)
				{
					sprintf(filename,"../output/%dgradient_vp.dat",iterb);
					fp=fopen(filename,"rb");
					fread(&gradient_vp_pre[0],sizeof(float),np,fp);
					fclose(fp);

					sprintf(filename,"../output/%dconjugate_vp.dat",iterb);
					fp=fopen(filename,"rb");
					fread(&conjugate_vp_pre[0],sizeof(float),np,fp);
					fclose(fp);

					//sprintf(filename,"../output/%dgradient_rho.dat",iter+1);
					//fp=fopen(filename,"wb");
					//fwrite(&gradient_rho[0],sizeof(float),np,fp);
					//fclose(fp);

					//sprintf(filename,"../output/%dconjugate_rho.dat",iter+1);
					//fp=fopen(filename,"wb");
					//fwrite(&conjugate_rho[0],sizeof(float),np,fp);
					//fclose(fp);

					get_ini_model(vp, rho, ntp, ntx, ntz, pmlc, iterb);
				}//end myid
				MPI_Barrier(comm);

				MPI_Bcast(gradient_vp_pre,ntp,MPI_FLOAT,0,comm);
				MPI_Bcast(conjugate_vp_pre,ntp,MPI_FLOAT,0,comm);
				MPI_Bcast(vp,ntp,MPI_FLOAT,0,comm);

				for(ip=0;ip<ntp;ip++)
				{
					vpn[ip]=vp[ip];
				}

				iterb=0;
			}//end iterb

			for(ip=0;ip<ntp;ip++)
			{
				tmp1[ip]=0.0;
				tmp2[ip]=0.0;
				gradient_vp_all[ip]=0.0;
				//gradient_rho_all[ip]=0.0;
			}

			Misfit_old=0.0;

			*Misfit=0.0;

			/********** FORWARD & BACKWARD ***********/
			for(iss=0;iss<eachsid;iss++)
			{
				is=offsets+iss*GPU_N;

				if(prcss!=0&&myid==prcs&&iss==eachsid-1)
					GPU_N=prcss;
				else
					GPU_N=GPU_NN;

				for(i=0;i<GPU_N;i++)
				{
					memset(plan[i].seismogram_obs,0,sizeof(float)*nx*itmax);
					memset(plan[i].seismogram_syn,0,sizeof(float)*nx*itmax);
#pragma omp parallel for private(ip)
					for(ip=0;ip<ntp;ip++)
					{
						plan[i].image_vp[ip]=0.0;
						//plan[i].image_rho[ip]=0.0;
						plan[i].image_sources[ip]=0.0;
						plan[i].image_receivers[ip]=0.0;
					}
				}

				/////////////////// random frequency selection number /////////////////////
				srand((unsigned)time(NULL));
				for(ip=0;ip<es[is].num;ip++)
				{
					/*
					   rnum[ip]=rand()%es[is].num;
					   for(ii=0;ii<ip;ii++)
					   {
					   if(rnum[ip]==rnum[ii])
					   {
					   rnum[ip]=rand()%es[is].num;
					   ii=0;
					   }
					   }
					 */

					rnum[ip]=rand()%es[is].num;
					int rflag=1;
					while(rflag==1)
					{
						for(ii=0;ii<ip;ii++)
							if(rnum[ip]==rnum[ii])
								break;
						if(ii<ip)
							rnum[ip]=rand()%es[is].num;
						if(ii==ip)
							rflag=0;
					}
					//printf("%d,",rnum[ip]);
				}
				//printf("\n");

				/////////////////// single frequency of source wavelet /////////////////////
				ricker_fre(rick,is,es,GPU_N,plan,ifreq,freqintv,fs,rnum,
						itmax,dt,dx,nx,pml,ricker_flag);
		
				fdtd_2d_GPU_forward(ntx,ntz,ntp,nx,nz,pml, Lc, rc,dx,dz,
						rick,itmax,dt,iter,ifreq,freqintv,Nf,fs,rnum,
						is,es,NNmax,plan,GPU_N,rnmax,
						rho,vp,
						k_x,k_x_half,k_z,k_z_half,
						a_x,a_x_half,a_z,a_z_half,
						b_x,b_x_half,b_z,b_z_half,
						inv_flag
						);

				// READ IN OBSERVED SEISMOGRAMS...  

				for(i=0;i<GPU_N;i++)
				{
				/*	sprintf(filename,"../output/%dsource_seismogram_obs.dat",is+i+1);
					fp=fopen(filename,"rb");
					fread(&plan[i].seismogram_obs[0],sizeof(float),itmax*ss[is+i].r_n,fp);
					fclose(fp);*/

#pragma omp parallel for private(ip)
					for(ip=0;ip<es[is+i].r_n*itmax;ip++)
					{
						plan[i].seismogram_tmpobs[ip]=0.0;
						plan[i].seismogram_tmpsyn[ip]=0.0;
					}
					for(ii=0;ii<es[is+i].num;ii++)
					{
						snum=es[is+i].offset+ii;

						memset(plan[i].seismogram,0,sizeof(float)*nx*itmax);
						if(noise_flag==0)
							sprintf(filename,"../output/%dsource_seismogram_obs.dat",snum+1);
						if(noise_flag==1)
							sprintf(filename,"../output/%dsource_seismogram_obs_noise.dat",snum+1);
						fp=fopen(filename,"rb");
						fread(&plan[i].seismogram[0],sizeof(float),itmax*ss[snum].r_n,fp);
						fclose(fp);

						seismgobs_fre(plan[i].seismogram,plan[i].seismogram_obs,ii,ifreq,freqintv,fs,rnum,i,itmax,dt,dx,ss[snum].r_n,pmlc);

						/*obsmax=0.0;
						for(ip=0;ip<es[is+i].r_n*itmax;ip++)
						{
							if(obsmax<fabs(plan[i].seismogram_obs[ip]))
								obsmax=fabs(plan[i].seismogram_obs[ip]);
						}*/
						//#pragma omp parallel for private(ix,it,ip,ipp)
						for(ix=0;ix<ss[snum].r_n;ix++)
						{
							for(it=0;it<itmax;it++)
							{
								ip=(ss[snum].r_ix[ix]-pmlc)*itmax+it;
								ipp=ix*itmax+it;

								plan[i].seismogram_tmpobs[ip]+=plan[i].seismogram_obs[ipp];///obsmax;
								//plan[i].seismogram_tmpobs[ipp]+=plan[i].seismogram_obs[ipp];///obsmax;
							}
						}
					}//end ii

					////synthetic seismograms////
					seismgsyn_fre(plan[i].seismogram_syn,plan[i].seismogram_tmpsyn,is,es,ifreq,freqintv,fs,rnum,i,itmax,dt,dx,nx,pmlc);

					synmax=0.0;
					obsmax=0.0;
					for(ip=0;ip<nx*itmax;ip++)
					{
						if(synmax<fabs(plan[i].seismogram_tmpsyn[ip]))
							synmax=fabs(plan[i].seismogram_tmpsyn[ip]);

						if(obsmax<fabs(plan[i].seismogram_tmpobs[ip]))
							obsmax=fabs(plan[i].seismogram_tmpobs[ip]);
					}
					for(ip=0;ip<nx*itmax;ip++)
					{
						plan[i].seismogram_tmpsyn[ip]/=synmax;
						plan[i].seismogram_tmpobs[ip]/=obsmax;
					}

					////seismogram residuals////
					memset(plan[i].seismogram_rms,0,sizeof(float)*nx*itmax);
					/*if(ricker_flag!=0)
					{
						seismgrms_fre(plan[i].seismogram_tmpsyn,plan[i].seismogram_tmpobs,plan[i].seismogram_rms,is,es,ss,ifreq,freqintv,fs,rnum,i,itmax,dt,dx,nx,pmlc);

						for(ip=0;ip<es[is+i].r_n*itmax;ip++)
						{
							*Misfit+=pow(plan[i].seismogram_rms[ip],2.0);
						}
					}
					else*/
					{
						decongpu_fre(plan[i].seismogram_tmpsyn,plan[i].seismogram_tmpobs,plan[i].seismogram_rms,Misfit,i,itmax,dt,dx,is,nx,es,ss,ifreq,freqintv,fs,rnum,pmlc);//ref_window,seis_window,
					}//end ricker_flag

					//conjugate_fre(plan[i].seismogram_rms,i,itmax,dt,dx,es[is+i].r_n,pmlc);
					//
					//// output the seismogram ////
					if(iter==0||iter==itn-1||(iter+1)%5==0)
					{
						///output encoded seismogram////
						sprintf(filename,"../output/%dsource_seismogram_%dobs.dat",is+i+1,iter+1);
						fp=fopen(filename,"wb");
						fwrite(&plan[i].seismogram_tmpobs[0],sizeof(float),itmax*es[is+i].r_n,fp);
						fclose(fp);

						sprintf(filename,"../output/%dsource_seismogram_%dsynf.dat",is+i+1,iter+1);
						fp=fopen(filename,"wb");
						fwrite(&plan[i].seismogram_tmpsyn[0],sizeof(float),itmax*es[is+i].r_n,fp);
						fclose(fp);

						sprintf(filename,"../output/%dsource_seismogram_%dsyn.dat",is+i+1,iter+1);
						fp=fopen(filename,"wb");
						fwrite(&plan[i].seismogram_syn[0],sizeof(float),itmax*es[is+i].r_n,fp);
						fclose(fp);

						sprintf(filename,"../output/%dsource_seismogram_%drms.dat",is+i+1,iter+1);
						fp=fopen(filename,"wb");
						fwrite(&plan[i].seismogram_rms[0],sizeof(float),itmax*es[is+i].r_n,fp);
						fclose(fp);
					}
				}//end GPU

				fdtd_2d_GPU_backward(ntx,ntz,ntp,nx,nz,pml,Lc,rc,dx,dz,
						rick,itmax,dt,iter,ifreq,freqintv,Nf,fs,rnum,
						is,es,NNmax,plan,GPU_N,rnmax,
						rho,vp,
						k_x,k_x_half,k_z,k_z_half,
						a_x,a_x_half,a_z,a_z_half,
						b_x,b_x_half,b_z,b_z_half
						);

				for(i=0;i<GPU_N;i++)
				{
					for(si=0;si<es[is+i].num;si++)
					{
						snum=es[is+i].offset+si;

#pragma omp parallel for private(ip)
						for(ip=0;ip<ntp;ip++)
						{
							plan[i].psdptamp[si*ntp+ip]=2*sqrt(pow(plan[i].psdptx[si*ntp+ip],2.0)+pow(plan[i].psdpty[si*ntp+ip],2.0));
							plan[i].psdpttheta[si*ntp+ip]=atan2(plan[i].psdpty[si*ntp+ip],plan[i].psdptx[si*ntp+ip]);
/*
							plan[i].psdvxamp[si*ntp+ip]=2*sqrt(pow(plan[i].psdvxx[si*ntp+ip],2.0)+pow(plan[i].psdvxy[si*ntp+ip],2.0));
							plan[i].psdvxtheta[si*ntp+ip]=atan2(plan[i].psdvxy[si*ntp+ip],plan[i].psdvxx[si*ntp+ip]);

							plan[i].psdvzamp[si*ntp+ip]=2*sqrt(pow(plan[i].psdvzx[si*ntp+ip],2.0)+pow(plan[i].psdvzy[si*ntp+ip],2.0));
							plan[i].psdvztheta[si*ntp+ip]=atan2(plan[i].psdvzy[si*ntp+ip],plan[i].psdvzx[si*ntp+ip]);
*/
							plan[i].psdpamp[si*ntp+ip]=2*sqrt(pow(plan[i].psdpx[si*ntp+ip],2.0)+pow(plan[i].psdpy[si*ntp+ip],2.0));
							plan[i].psdptheta[si*ntp+ip]=atan2(plan[i].psdpy[si*ntp+ip],plan[i].psdpx[si*ntp+ip]);

						}

#pragma omp parallel for private(ip)
						for(ip=0;ip<ntp;ip++)
						{
							plan[i].ptff[ip][0]=plan[i].psdptamp[si*ntp+ip]*cos(plan[i].psdpttheta[si*ntp+ip]);
							plan[i].ptff[ip][1]=plan[i].psdptamp[si*ntp+ip]*sin(plan[i].psdpttheta[si*ntp+ip])*(-1.0);
/*
							plan[i].vxff[ip][0]=plan[i].psdvxamp[si*ntp+ip]*cos(plan[i].psdvxtheta[si*ntp+ip]);
							plan[i].vxff[ip][1]=plan[i].psdvxamp[si*ntp+ip]*sin(plan[i].psdvxtheta[si*ntp+ip])*(-1.0);

							plan[i].vzff[ip][0]=plan[i].psdvzamp[si*ntp+ip]*cos(plan[i].psdvztheta[si*ntp+ip]);
							plan[i].vzff[ip][1]=plan[i].psdvzamp[si*ntp+ip]*sin(plan[i].psdvztheta[si*ntp+ip])*(-1.0);
*/
							plan[i].pff[ip][0]=plan[i].psdpamp[si*ntp+ip]*cos(plan[i].psdptheta[si*ntp+ip]);
							plan[i].pff[ip][1]=plan[i].psdpamp[si*ntp+ip]*sin(plan[i].psdptheta[si*ntp+ip])*(-1.0);
						}

#pragma omp parallel for private(ix,iz,ip)
						for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
						{
							for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
							{
								ip=iz*ntx+ix;

								//	dpdx[0]=(plan[i].pfff[ip][0]-plan[i].pfff[ip-1][0])/dx;
								//	dpdx[1]=(plan[i].pfff[ip][1]-plan[i].pfff[ip-1][1])/dx;

								//	dpdz[0]=(plan[i].pfff[ip][0]-plan[i].pfff[ip-ntx][0])/dz;
								//	dpdz[1]=(plan[i].pfff[ip][1]-plan[i].pfff[ip-ntx][1])/dz;

								plan[i].image_vp[ip]=2.0/(rho[ip]*pow(vp[ip],3.0))*(plan[i].ptff[ip][0]*plan[i].pff[ip][0]+plan[i].ptff[ip][1]*plan[i].pff[ip][1]);
								//plan[i].image_rho[ip]+=1.0/(rho[ip]*pow(vp[ip],2.0))*(plan[i].ptff[ip][0]*plan[i].pff[ip][0]-plan[i].ptff[ip][1]*plan[i].pff[ip][1]-(dp_dx[0]*plan[i].vxff[ip][0]-dp_dx[1]*plan[i].vxff[ip][1])-(dp_dz[0]*plan[i].vzff[ip][0]-dp_dz[1]*plan[i].vzff[ip][1]))/rho[ip];
								//
								plan[i].image_sources[ip]=2.0/(rho[ip]*pow(vp[ip],3.0))*(plan[i].ptff[ip][0]*plan[i].ptff[ip][0]+plan[i].ptff[ip][1]*plan[i].ptff[ip][1]);
							}
						}

						srcmax=0.0;
						for(ip=0;ip<ntp;ip++)
						{
							if(srcmax<fabs(plan[i].image_sources[ip]))
								srcmax=fabs(plan[i].image_sources[ip]);
						}

#pragma omp parallel for private(ip)
						for(ip=0;ip<ntp;ip++)
						{
							tmp1[ip]+=plan[i].image_vp[ip]/(plan[i].image_sources[ip]+1.0e-3*srcmax);
							////tmp2[ip]+=plan[i].image_rho[ip];
							//plan[i].image_vp[ip]/=(plan[i].image_sources[ip]+1.0e-3*srcmax);
							//plan[i].image_rho[ip]/=(plan[i].image_sources[ip]+1.0e-3*srcmax);
						}

						if(iter==0)
						{
							sprintf(filename,"../output/%dimage_source.dat",snum+1);
							fp=fopen(filename,"wb");
							for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
							{
								for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
								{
									fwrite(&plan[i].image_sources[iz*ntx+ix],sizeof(float),1,fp);
								}
							}
							for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
							{
								for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
								{
									fwrite(&plan[i].image_vp[iz*ntx+ix],sizeof(float),1,fp);
								}
							}
							fclose(fp);
							///////////////////////////////////////////////////////////////////////////////
							sprintf(filename,"../output/%dptamp.dat",snum+1);
							fp=fopen(filename,"wb");
							for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
							{
								for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
								{
									fwrite(&plan[i].psdptamp[si*ntp+iz*ntx+ix],sizeof(float),1,fp);
								}
							}
							fclose(fp);

							sprintf(filename,"../output/%dpttheta.dat",snum+1);
							fp=fopen(filename,"wb");
							for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
							{
								for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
								{
									fwrite(&plan[i].psdpttheta[si*ntp+iz*ntx+ix],sizeof(float),1,fp);
								}
							}
							fclose(fp);
							///////////////////////////////////////////////////////////////////////////////
							sprintf(filename,"../output/%dpamp.dat",snum+1);
							fp=fopen(filename,"wb");
							for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
							{
								for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
								{
									fwrite(&plan[i].psdpamp[si*ntp+iz*ntx+ix],sizeof(float),1,fp);
								}
							}
							fclose(fp);

							sprintf(filename,"../output/%dptheta.dat",snum+1);
							fp=fopen(filename,"wb");
							for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
							{
								for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
								{
									fwrite(&plan[i].psdptheta[si*ntp+iz*ntx+ix],sizeof(float),1,fp);
								}
							}
							fclose(fp);
						}
						///////////////////////////////////////////////////////////////////////////////
						/**/
					}//end si
				}//end GPU_N

	/*			for(i=0;i<GPU_N;i++)
				{
					for(ip=0;ip<ntp;ip++)
					{
						tmp1[ip]+=plan[i].image_vp[ip];
						//tmp2[ip]+=plan[i].image_rho[ip];
					}
				}//end GPU
	*/
			}//end is (shotnumbers)

			MPI_Barrier(comm);

			MPI_Allreduce(tmp1,gradient_vp_all,ntp,MPI_FLOAT,MPI_SUM,comm);
	//		MPI_Allreduce(tmp2,gradient_rho_all,ntp,MPI_FLOAT,MPI_SUM,comm);
			MPI_Allreduce(Misfit,&Misfit_old,1,MPI_FLOAT,MPI_SUM,comm);

			misfit[iter]=Misfit_old;

			if(myid==0)
				printf("==    Misfit_old == %e\n",Misfit_old);

#pragma omp parallel for private(ix,iz,ip)
			for(ix=pmlc;ix<ntx-pmlc;ix++)
			{
				for(iz=pmlc;iz<ntz-pmlc;iz++)
				{
					ip=(ix-pmlc)*nz+iz-pmlc;

					gradient_vp[ip]=gradient_vp_all[iz*ntx+ix]*Pr[iz-pmlc];
					//gradient_rho[ip]=gradient_rho_all[iz*ntx+ix]*Pr[iz-pmlc];
				}
			}

#pragma omp parallel for private(ix,iz,ip)
			for(ix=pmlc;ix<ntx-pmlc;ix++)
			{
				for(iz=pmlc;iz<ntz-pmlc;iz++)
				{
					ip=(ix-pmlc)*nz+iz-pmlc;

					if(ix>ntx-pmlc-3)
					{
						gradient_vp[ip]=gradient_vp_all[iz*ntx+ntx-pmlc-3]*Pr[iz-pmlc];
						//gradient_rho[ip]=gradient_rho_all[iz*ntx+ntx-pmlc-3]*Pr[iz-pmlc];
					}

					if(iz>ntz-pmlc-3)
					{
						gradient_vp[ip]=gradient_vp_all[(ntz-pmlc-3)*ntx+ix]*Pr[iz-pmlc];
						//gradient_rho[ip]=gradient_rho_all[(ntz-pmlc-3)*ntx+ix]*Pr[iz-pmlc];
					}
				}
			}

			/*==========================================================
			  Applying the conjugate_vp gradient method...
			  ==========================================================*/

			if(iter==0)
			{
#pragma omp parallel for private(ip)
				for(ip=0;ip<np;ip++)
				{
					conjugate_vp[ip]=-gradient_vp[ip];
					//conjugate_rho[ip]=-gradient_rho[ip];
				}
			}
			else 
			{
				sum1=0.0;
				sum2=0.0;

				sum1r=0.0;
				sum2r=0.0;

				for(ip=0;ip<np;ip++)
				{
					sum1+=gradient_vp[ip]*gradient_vp[ip];
					sum2+=gradient_vp_pre[ip]*gradient_vp_pre[ip];

					//sum1r+=gradient_rho[ip]*gradient_rho[ip];
					//sum2r+=gradient_rho_pre[ip]*gradient_rho_pre[ip];
				}

				beta=sum1/sum2;
				betar=sum1r/sum2r;

#pragma omp parallel for private(ip)
				for(ip=0;ip<np;ip++)
				{
					conjugate_vp[ip]=-gradient_vp[ip]+beta*conjugate_vp_pre[ip];
					//conjugate_rho[ip]=-gradient_rho[ip]+betar*conjugate_rho_pre[ip];
				}
			}

			/*---------------------------------------------------------------*/
			/*---------------------------------------------------------------*/
			/*---------------------------------------------------------------*/

#pragma omp parallel for private(ip)
			for(ip=0;ip<np;ip++)
			{
				gradient_vp_pre[ip]=gradient_vp[ip];
				//gradient_rho_pre[ip]=gradient_rho[ip];

				conjugate_vp_pre[ip]=conjugate_vp[ip];
				//conjugate_rho_pre[ip]=conjugate_rho[ip];
			}

			//	------------calculate the step --------------- //

			ini_step(conjugate_vp,np,step_vp,vp_max);
			//ini_step(conjugate_rho,np,step_rho,rho_max);

			if(myid==0)
				printf("an_vp == %e\n",*step_vp);

			update_model(vp,vpn,conjugate_vp,step_vp,ntp,ntz,ntx,pmlc,vp_min,vp_max);
			//update_model(rho,rhon,conjugate_rho,step_rho,ntp,ntz,ntx,pmlc,rho_min,rho_max);

#pragma omp parallel for private(ip)
			for(ip=0;ip<ntp;ip++)
			{
				if(vp[ip]>vp_max)
					vp[ip]=vp_max;

				vpn[ip]=vp[ip];
				rhon[ip]=rho[ip];
			}

			/*==========================================================
			  Output the updated model such as vp,rho,...
			  ===========================================================*/

			if(myid==0)
			{
				sprintf(filename,"../output/%dgradient_vp.dat",iter+1);
				fp=fopen(filename,"wb");
				fwrite(&gradient_vp[0],sizeof(float),np,fp);
				fclose(fp);

				//sprintf(filename,"../output/%dgradient_rho.dat",iter+1);
				//fp=fopen(filename,"wb");
				//fwrite(&gradient_rho[0],sizeof(float),np,fp);
				//fclose(fp);

				sprintf(filename,"../output/%dconjugate_vp.dat",iter+1);
				fp=fopen(filename,"wb");
				fwrite(&conjugate_vp[0],sizeof(float),np,fp);
				fclose(fp);

				//sprintf(filename,"../output/%dconjugate_rho.dat",iter+1);
				//fp=fopen(filename,"wb");
				//fwrite(&conjugate_rho[0],sizeof(float),np,fp);
				//fclose(fp);

				sprintf(filename,"../output/%dvp.dat",iter+1);
				fp=fopen(filename,"wb");
				for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
				{
					for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
					{
						fwrite(&vp[iz*ntx+ix],sizeof(float),1,fp);
					}
				}
				fclose(fp);

				//sprintf(filename,"../output/%drho.dat",iter+1);
				//fp=fopen(filename,"wb");
				//for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
				//{
				//	for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
				//	{
				//		fwrite(&rho[iz*ntx+ix],sizeof(float),1,fp);
				//	}
				//}
				//fclose(fp);
			}

			MPI_Barrier(comm);
		}//end iteration

		if(myid==0)
		{
			sprintf(filename,"../output/%difreq_vp.dat",ifreq+1);
			fp=fopen(filename,"wb");
			for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
			{
				for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
				{
					fwrite(&vp[iz*ntx+ix],sizeof(float),1,fp);
				}
			}
			fclose(fp);

			/*sprintf(filename,"../output/%difreq_rho.dat",ifreq+1);
			fp=fopen(filename,"wb");
			for(ix=pmlc;ix<=ntx-pmlc-1;ix++)
			{
				for(iz=pmlc;iz<=ntz-pmlc-1;iz++)
				{
					fwrite(&rho[iz*ntx+ix],sizeof(float),1,fp);
				}
			}
			fclose(fp);*/

			sprintf(filename,"../output/misfit_%difreq.txt",ifreq+1);
			fp=fopen(filename,"w");
			for(iter=0;iter<itn;iter++)
			{
				fprintf(fp,"%e\r\n",misfit[iter]);
			}
			fclose(fp);
		}

		MPI_Barrier(comm);
	}//end frequency

	if(myid==0)
	{
		printf("====================\n");
		printf("      THE END\n");
		printf("====================\n");

		end=clock();
		printf("The cost of the run time is %f seconds\n",
				(double)(end-start)/CLOCKS_PER_SEC);
	}
	/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	  !	        ITERATION OF FWI IN TIME DOMAIN ENDS...                        !
	  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

	variables_free(ntx, ntz, ntp, nx, nz,
			pml, Lc, dx, dz, itmax,
			plan, GPU_N, rnmax, Nf, NNmax
			);

	free(rc); 
	
	for(is=0;is<ns;is++)
	{
		free(ss[is].r_ix);
	} 
	for(is=0;is<Nsmax;is++)
	{
		free(es[is].s_ix);
		free(es[is].r_ix);
	}
	free(step_vp);
	free(step_rho);

	free(rnum);

	free(rick);

	//free the memory of P velocity
	free(vp);
	//free the memory of Density
	free(rho); 

	free(vpn); 
	free(rhon);

	free(d_x);
	free(d_x_half);
	free(d_z);
	free(d_z_half);

	free(a_x);
	free(a_x_half);
	free(a_z);
	free(a_z_half);

	free(b_x);
	free(b_x_half);
	free(b_z);
	free(b_z_half);

	free(k_x);
	free(k_x_half);
	free(k_z);
	free(k_z_half);

	free(ref_window);
	free(seis_window);
	free(ref_obs);
	free(ref_syn);

	for(i=0;i<GPU_N;i++)
	{
		free(plan[i].rick);

		free(plan[i].seismogram_obs);
		free(plan[i].seismogram_syn); 
		free(plan[i].seismogram_rms);

		free(plan[i].seismogram);
		free(plan[i].seismogram_tmpobs);
		free(plan[i].seismogram_tmpsyn);
/*
		free(plan[i].p_borders_up);
		free(plan[i].p_borders_bottom);
		free(plan[i].p_borders_left);
		free(plan[i].p_borders_right);
*/
		free(plan[i].image_vp);
		free(plan[i].image_rho);
		free(plan[i].image_sources);
		free(plan[i].image_receivers);

		free(plan[i].psdptx);
		free(plan[i].psdpty);
		free(plan[i].psdptamp);
		free(plan[i].psdpttheta);
/*
		free(plan[i].psdvxx);
		free(plan[i].psdvxy);
		free(plan[i].psdvxamp);
		free(plan[i].psdvxtheta);

		free(plan[i].psdvzx);
		free(plan[i].psdvzy);
		free(plan[i].psdvzamp);
		free(plan[i].psdvztheta);
*/
		free(plan[i].psdpx);
		free(plan[i].psdpy);
		free(plan[i].psdpamp);
		free(plan[i].psdptheta);

		fftw_free(plan[i].ptff);
		fftw_free(plan[i].vxff);
		fftw_free(plan[i].vzff);
		fftw_free(plan[i].pff);
	}

	free(gradient_vp_all);
	free(gradient_rho_all);

	free(tmp1);
	free(tmp2);

	free(gradient_vp);
	free(gradient_rho);

	free(conjugate_vp);
	free(conjugate_rho);

	free(gradient_vp_pre);
	free(gradient_rho_pre);

	free(conjugate_vp_pre);
	free(conjugate_rho_pre);

	free(Misfit);

	MPI_Barrier(comm);
	MPI_Finalize();
}


/*==========================================================
  This subroutine is used for calculating the parameters of 
  absorbing layers
  ===========================================================*/

void get_absorbing_parameters(
		float *d_x, float *d_x_half, 
		float *d_z, float *d_z_half,
		float *a_x, float *a_x_half,
		float *a_z, float *a_z_half,
		float *b_x, float *b_x_half,
		float *b_z, float *b_z_half,
		float *k_x, float *k_x_half,
		float *k_z, float *k_z_half,
		int ntz, int ntx, int nz, int nx,
		int pml, float dx, float f0, float t0, float dt, float vp_max)

{
	int   N=2;
	int   iz,ix;

	float thickness_of_pml;
	float Rc=1.0e-5;

	float d0;
	float pi=3.1415927;
	float alpha_max=pi*15;

	float Vpmax;


	float *alpha_x,*alpha_x_half;
	float *alpha_z,*alpha_z_half;

	float x_start,x_end,delta_x;
	float z_start,z_end,delta_z;
	float x_current,z_current;

	Vpmax=5500;

	thickness_of_pml=pml*dx;

	d0=-(N+1)*Vpmax*log(Rc)/(2.0*thickness_of_pml);

	alpha_x      = (float*)malloc(ntx*sizeof(float));
	alpha_x_half = (float*)malloc(ntx*sizeof(float));

	alpha_z      = (float*)malloc(ntz*sizeof(float));
	alpha_z_half = (float*)malloc(ntz*sizeof(float));

	//--------------------initialize the vectors--------------

	for(ix=0;ix<ntx;ix++)
	{
		a_x[ix]          = 0.0;
		a_x_half[ix]     = 0.0;
		b_x[ix]          = 0.0;
		b_x_half[ix]     = 0.0;
		d_x[ix]          = 0.0;
		d_x_half[ix]     = 0.0;
		k_x[ix]          = 1.0;
		k_x_half[ix]     = 1.0;
		alpha_x[ix]      = 0.0;
		alpha_x_half[ix] = 0.0;
	}

	for(iz=0;iz<ntz;iz++)
	{
		a_z[iz]          = 0.0;
		a_z_half[iz]     = 0.0;
		b_z[iz]          = 0.0;
		b_z_half[iz]     = 0.0;
		d_z[iz]          = 0.0;
		d_z_half[iz]     = 0.0;
		k_z[iz]          = 1.0;
		k_z_half[iz]     = 1.0;

		alpha_z[iz]      = 0.0;
		alpha_z_half[iz] = 0.0;
	}


	// X direction

	x_start=pml*dx;
	x_end=(ntx-pml-1)*dx;

	// Integer points
	for(ix=0;ix<ntx;ix++)
	{ 
		x_current=ix*dx;

		// LEFT EDGE
		if(x_current<=x_start)
		{
			delta_x=x_start-x_current;
			d_x[ix]=d0*pow(delta_x/thickness_of_pml,2);
			k_x[ix]=1.0;
			alpha_x[ix]=alpha_max*(1.0-(delta_x/thickness_of_pml))+0.1*alpha_max;
		}

		// RIGHT EDGE      
		if(x_current>=x_end)
		{
			delta_x=x_current-x_end;
			d_x[ix]=d0*pow(delta_x/thickness_of_pml,2);
			k_x[ix]=1.0;
			alpha_x[ix]=alpha_max*(1.0-(delta_x/thickness_of_pml))+0.1*alpha_max;
		}
	}


	// Half Integer points
	for(ix=0;ix<ntx;ix++)
	{
		x_current=(ix+0.5)*dx;

		if(x_current<=x_start)
		{
			delta_x=x_start-x_current;
			d_x_half[ix]=d0*pow(delta_x/thickness_of_pml,2);
			k_x_half[ix]=1.0;
			alpha_x_half[ix]=alpha_max*(1.0-(delta_x/thickness_of_pml))+0.1*alpha_max;
		}

		if(x_current>=x_end)
		{
			delta_x=x_current-x_end;
			d_x_half[ix]=d0*pow(delta_x/thickness_of_pml,2);
			k_x_half[ix]=1.0;
			alpha_x_half[ix]=alpha_max*(1.0-(delta_x/thickness_of_pml))+0.1*alpha_max;
		}
	}

	for (ix=0;ix<ntx;ix++)
	{
		if(alpha_x[ix]<0.0)
		{
			alpha_x[ix]=0.0;
		}
		if(alpha_x_half[ix]<0.0)
		{
			alpha_x_half[ix]=0.0;
		}

		b_x[ix]=exp(-(d_x[ix]/k_x[ix]+alpha_x[ix])*dt);

		if(d_x[ix] > 1.0e-6)
		{
			a_x[ix]=d_x[ix]/(k_x[ix]*(d_x[ix]+k_x[ix]*alpha_x[ix]))*(b_x[ix]-1.0);
		}

		b_x_half[ix]=exp(-(d_x_half[ix]/k_x_half[ix]+alpha_x_half[ix])*dt);

		if(d_x_half[ix] > 1.0e-6)
		{
			a_x_half[ix]=d_x_half[ix]/(k_x_half[ix]*(d_x_half[ix]+k_x_half[ix]*alpha_x_half[ix]))*(b_x_half[ix]-1.0);
		}
	}

	// Z direction

	z_start=pml*dx;
	z_end=(ntz-pml-1)*dx;

	// Integer points
	for(iz=0;iz<ntz;iz++)
	{ 
		z_current=iz*dx;

		// LEFT EDGE
		if(z_current<=z_start)
		{
			delta_z=z_start-z_current;
			d_z[iz]=d0*pow(delta_z/thickness_of_pml,2);
			k_z[iz]=1.0;
			alpha_z[iz]=alpha_max*(1.0-(delta_z/thickness_of_pml))+0.1*alpha_max;
		}

		// RIGHT EDGE      
		if(z_current>=z_end)
		{
			delta_z=z_current-z_end;
			d_z[iz]=d0*pow(delta_z/thickness_of_pml,2);
			k_z[iz]=1.0;
			alpha_z[iz]=alpha_max*(1.0-(delta_z/thickness_of_pml))+0.1*alpha_max;
		}
	}

	// Half Integer points
	for(iz=0;iz<ntz;iz++)
	{
		z_current=(iz+0.5)*dx;

		if(z_current<=z_start)
		{
			delta_z=z_start-z_current;
			d_z_half[iz]=d0*pow(delta_z/thickness_of_pml,2);
			k_z_half[iz]=1.0;
			alpha_z_half[iz]=alpha_max*(1.0-(delta_z/thickness_of_pml))+0.1*alpha_max;
		}

		if(z_current>=z_end)
		{
			delta_z=z_current-z_end;
			d_z_half[iz]=d0*pow(delta_z/thickness_of_pml,2);
			k_z_half[iz]=1.0;
			alpha_z_half[iz]=alpha_max*(1.0-(delta_z/thickness_of_pml))+0.1*alpha_max;
		}
	}

	for (iz=0;iz<ntz;iz++)
	{
		if(alpha_z[iz]<0.0)
		{
			alpha_z[iz]=0.0;
		}
		if(alpha_z_half[iz]<0.0)
		{
			alpha_z_half[iz]=0.0;
		}

		b_z[iz]=exp(-(d_z[iz]/k_z[iz]+alpha_z[iz])*dt);

		if(d_z[iz]>1.0e-6)
		{
			a_z[iz]=d_z[iz]/(k_z[iz]*(d_z[iz]+k_z[iz]*alpha_z[iz]))*(b_z[iz]-1.0);
		}

		b_z_half[iz]=exp(-(d_z_half[iz]/k_z_half[iz]+alpha_z_half[iz])*dt);

		if(d_z_half[iz]>1.0e-6)
		{
			a_z_half[iz]=d_z_half[iz]/(k_z_half[iz]*(d_z_half[iz]+k_z_half[iz]*alpha_z_half[iz]))*(b_z_half[iz]-1.0);
		}
	}

	free(alpha_x);
	free(alpha_x_half);
	free(alpha_z);
	free(alpha_z_half);

	return;

}


/*==========================================================
  This subroutine is used for initializing the true model...
  ===========================================================*/

void get_acc_model(float *vp, float *rho, int ntp, int ntx, int ntz, int pml)
{
	int ip,ipp,iz,ix;
	// THE MODEL    
	FILE *fp;

	//fp=fopen("../input/acc_vp.dat","rb");
	fp=fopen("../input/acc_vp.dat","rb");
	for(ix=pml;ix<ntx-pml;ix++)
	{
		for(iz=pml;iz<ntz-pml;iz++)
		{
			ip=iz*ntx+ix;
			fread(&vp[ip],sizeof(float),1,fp);           
		}
	}
	fclose(fp);

	for(iz=0;iz<=pml-1;iz++)
	{

		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+pml;

			vp[ip]=vp[ipp];
		}

		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+ix;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+ntx-pml-1;

			vp[ip]=vp[ipp];
		}
	}

	for(iz=pml;iz<=ntz-pml-1;iz++)
	{
		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+pml;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+ntx-pml-1;

			vp[ip]=vp[ipp];
		}

	}

	for(iz=ntz-pml;iz<ntz;iz++)
	{

		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+pml;

			vp[ip]=vp[ipp];
		}

		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+ix;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+ntx-pml-1;

			vp[ip]=vp[ipp];
		}
	}
	///////////
/*	fp=fopen("../input/acc_rho.dat","rb");
	for(ix=pml;ix<ntx-pml;ix++)
	{
		for(iz=pml;iz<ntz-pml;iz++)
		{
			ip=iz*ntx+ix;
			fread(&rho[ip],sizeof(float),1,fp);

			rho[ip]=rho[ip];
		}
	}
	fclose(fp);

	for(iz=0;iz<=pml-1;iz++)
	{

		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+pml;

			rho[ip]=rho[ipp];
		}

		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+ix;

			rho[ip]=rho[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+ntx-pml-1;

			rho[ip]=rho[ipp];
		}
	}

	for(iz=pml;iz<=ntz-pml-1;iz++)
	{
		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+pml;

			rho[ip]=rho[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+ntx-pml-1;

			rho[ip]=rho[ipp];
		}

	}

	for(iz=ntz-pml;iz<ntz;iz++)
	{

		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+pml;

			rho[ip]=rho[ipp];
		}

		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+ix;

			rho[ip]=rho[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+ntx-pml-1;

			rho[ip]=rho[ipp];
		}
	}
*/
	for(ip=0;ip<ntp;ip++)
	{
		rho[ip]=1000.0;
	}

	return;
}


/*==========================================================
  This subroutine is used for finding the maximum value of 
  a vector.
  ===========================================================*/ 
void maximum_vector(float *vector, int n, float *maximum_value)
{
	int i;

	*maximum_value=1.0e-20;
	for(i=0;i<n;i++)
	{
		if(vector[i]>*maximum_value);
		{
			*maximum_value=vector[i];
		}
	}
	printf("maximum_value=%f\n",*maximum_value);
	return;
}


/*==========================================================
  This subroutine is used for calculating the sum of two 
  vectors!
  ===========================================================*/

void add(float *a,float *b,float *c,int n)
{
	int i;
	for(i=0;i<n;i++)
	{
		c[i]=a[i]-b[i];
	}

}

/*==========================================================

  This subroutine is used for calculating the ricker wave

  ===========================================================*/

void ricker_wave(float *rick, int itmax, float f0, float t0, float dt, int flag)
{
	float pi=3.1415927;
	int   it;
	float temp,max=0.0;

	FILE *fp;

	if(flag==3)
	{	
		for(it=0;it<itmax;it++)
		{
			temp=1.5*pi*f0*(it*dt-t0);
			temp=temp*temp;
			rick[it]=exp(-temp);  

			if(max<fabs(rick[it]))
			{
				max=fabs(rick[it]);
			}
		}

		for(it=0;it<itmax;it++)
		{
			rick[it]=rick[it]/max;
		}

		fp=fopen("../output/rick_third_derive.dat","wb");    
		for(it=0;it<itmax;it++)
		{
			fwrite(&rick[it],sizeof(float),1,fp);
		}    
		fclose(fp);
	}

	if(flag==2)
	{
		for(it=0;it<itmax;it++)
		{
			temp=pi*f0*(it*dt-t0);
			temp=temp*temp;
			rick[it]=(1.0-2.0*temp)*exp(-temp);
		}

		fp=fopen("../output/rick_second_derive.dat","wb");    
		for(it=0;it<itmax;it++)
		{
			fwrite(&rick[it],sizeof(float),1,fp);
		}    
		fclose(fp);
	}
	if(flag==1)
	{
		for(it=0;it<itmax;it++)
		{
			temp=pi*f0*(it*dt-t0);
			temp=temp*temp;         
			rick[it]=(it*dt-t0)*exp(-temp);

			if(max<fabs(rick[it]))
			{
				max=fabs(rick[it]);
			}
		}

		for(it=0;it<itmax;it++)
		{
			rick[it]=rick[it]/max;
		}

		fp=fopen("../output/rick_first_derive.dat","wb");    
		for(it=0;it<itmax;it++)
		{
			fwrite(&rick[it],sizeof(float),1,fp);
		}    
		fclose(fp);
	}

	return;
}

//*************************************************************************
//*******un0*cnmax=vmax*0.01
//************************************************************************

void ini_step(float *dn, int np, float *un0, float max)
{
	float dnmax=0.0;
	int ip;

	for(ip=0;ip<np;ip++)
	{
		if(dnmax<fabs(dn[ip]))
		{
			dnmax=fabs(dn[ip]);
		}
	}   

	*un0=max*0.01/dnmax;    

	return;
}


/*=========================================================================
  To calculate the updated model...
  ========================================================================*/

void update_model(float *vp, float *vpn,
		float *dn_vp, float *un_vp,
		int ntp, int ntz, int ntx, int pml, float vpmin, float vpmax)
{
	int ip,ipp;
	int iz,ix;
	int nx=ntx-2*pml;
	int nz=ntz-2*pml;

	for(iz=pml;iz<=ntz-pml-1;iz++)
	{
		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ix-pml)*nz+iz-pml;
			vp[ip]=vpn[ip]+*un_vp*dn_vp[ipp];
		}
	}

	//  Model in PML..............

	for(iz=0;iz<=pml-1;iz++)
	{

		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+pml;

			vp[ip]=vp[ipp];
		}

		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+ix;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+ntx-pml-1;

			vp[ip]=vp[ipp];
		}
	}

	for(iz=pml;iz<=ntz-pml-1;iz++)
	{
		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+pml;
			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+ntx-pml-1;

			vp[ip]=vp[ipp];
		}

	}

	for(iz=ntz-pml;iz<ntz;iz++)
	{

		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+pml;

			vp[ip]=vp[ipp];
		}

		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+ix;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+ntx-pml-1;

			vp[ip]=vp[ipp];
		}
	}
	return;
}


/***********************************************************************
  !                initial model
  !***********************************************************************/
void ini_model_mine(float *vp, float *vpn, int ntp, int ntz, int ntx, int pml, int flag)
{
	/*  flag == 1 :: P velocity
		flag == 2 :: S velocity
		flag == 3 :: Density
		*/
	int window;

	if(flag==1)
	{
		window=100;
	}
	if(flag==2)
	{
		window=100;
	}

	float *vp_old1;

	float sum;
	int number;

	int iz,ix;
	int izw,ixw,iz1,ix1;
	int ip,ipp;

	vp_old1=(float*)malloc(sizeof(float)*ntp);


	for(ip=0;ip<ntp;ip++)
	{
		vp_old1[ip]=vp[ip];
	}

	//-----smooth in the x direction---------

#pragma omp parallel for private(ix,iz,sum,number,ixw,izw,ix1,iz1,ip,ipp)
	for(iz=pml;iz<=ntz-pml-1;iz++)
	{
		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			sum=0.0;
			number=0;

			for(izw=iz-window;izw<iz+window;izw++)
			{
				for(ixw=ix-window;ixw<ix+window;ixw++)
				{
					if(izw<0)
					{
						iz1=0;                		
					}
					else if(izw>ntz-1)
					{
						iz1=ntz-1;
					}
					else
					{
						iz1=izw;
					}

					if(ixw<0)
					{
						ix1=0;
					}
					else if(ixw>ntx-1)
					{
						ix1=ntx-1;
					}
					else
					{
						ix1=ixw;
					}

					ip=iz1*ntx+ix1;
					sum=sum+vp_old1[ip];
					number=number+1;
				}
			}
			ip=iz*ntx+ix;
			vp[ip]=sum/number;

			if(iz<pml+4)
			{
				vp[ip]=vp_old1[ip];
			}
		}
	}    

	//  Model in PML..............

	for(iz=0;iz<=pml-1;iz++)
	{

		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+pml;

			vp[ip]=vp[ipp];
		}

		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+ix;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+ntx-pml-1;

			vp[ip]=vp[ipp];
		}
	}

	for(iz=pml;iz<=ntz-pml-1;iz++)
	{
		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+pml;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+ntx-pml-1;

			vp[ip]=vp[ipp];
		}

	}

	for(iz=ntz-pml;iz<ntz;iz++)
	{

		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+pml;

			vp[ip]=vp[ipp];
		}

		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+ix;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+ntx-pml-1;

			vp[ip]=vp[ipp];
		}
	}
	

	for(ip=0;ip<ntp;ip++)
	{
		vpn[ip]=vp[ip];
	}

	free(vp_old1);

	return;
}

void get_ini_model(float *vp, float *rho, int ntp, int ntx, int ntz, int pml, int iterb)
{
	int ip,ipp,iz,ix;
	// THE MODEL    
	FILE *fp;
	char filename[30];

	///////////
	sprintf(filename,"../output/%dvp.dat",iterb);
	fp=fopen(filename,"rb");
	for(ix=pml;ix<ntx-pml;ix++)
	{
		for(iz=pml;iz<ntz-pml;iz++)
		{
			ip=iz*ntx+ix;
			fread(&vp[ip],sizeof(float),1,fp);
		}
	}
	fclose(fp);

	for(iz=0;iz<=pml-1;iz++)
	{

		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+pml;

			vp[ip]=vp[ipp];
		}

		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+ix;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=pml*ntx+ntx-pml-1;

			vp[ip]=vp[ipp];
		}
	}

	for(iz=pml;iz<=ntz-pml-1;iz++)
	{
		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+pml;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+ntx-pml-1;

			vp[ip]=vp[ipp];
		}

	}

	for(iz=ntz-pml;iz<ntz;iz++)
	{

		for(ix=0;ix<=pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+pml;

			vp[ip]=vp[ipp];
		}

		for(ix=pml;ix<=ntx-pml-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+ix;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-pml;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-pml-1)*ntx+ntx-pml-1;

			vp[ip]=vp[ipp];
		}
	}

	return;
}

/*=======================================================================

  subroutine preprocess(nz,nx,dx,dz,P)

  !=======================================================================*/
// in this program Precondition P is computed

void Preprocess(int nz, int nx, float dx, float dz, float *P, int flag)
{
	int iz,iz_depth_one,iz_depth_two;
	float z,delta1,a,c,temp,z1,z2;

	a=3.0;
	c=1.0;
	iz_depth_one=1;
	iz_depth_two=4;

	delta1=(iz_depth_two-iz_depth_one)*dx;
	z1=(iz_depth_one-1)*dz;
	z2=(iz_depth_two-1)*dz;

	for(iz=0;iz<nz;iz++)
	{ 
		z=iz*dz;
		if(z<=z1)
		{
			P[iz]=0.0;
		}

		if(z>z1&&z<=z2)
		{
			temp=z-z1-delta1;
			temp=a*temp*2/delta1;
			temp=temp*temp;
			P[iz]=c*exp(-0.5*temp);//0.0;//
		}

		if(z>z2)
		{
			if(flag==1)
				P[iz]=c*(float)z/(float)z2;//
			if(flag==2)
				P[iz]=c;//c*(float)z/(float)z2;//
		}
	}
}
/*
void Preprocess(int nz, int nx, float dx, float dz, float *P)
{
	int iz,iz_depth_one,iz_depth_two;
	float z,delta1,a,temp,z1,z2;

	a=3.0;
	iz_depth_one=1;
	iz_depth_two=4;

	delta1=(iz_depth_two-iz_depth_one)*dx;
	z1=(iz_depth_one-1)*dz;
	z2=(iz_depth_two-1)*dz;

	for(iz=0;iz<nz;iz++)
	{ 
		z=iz*dz;
		if(z<=z1)
		{
			P[iz]=0.0;
		}

		if(z>z1&&z<=z2)
		{
			temp=z-z1-delta1;
			temp=a*temp*2/delta1;
			temp=temp*temp;
			P[iz]=exp(-0.5*temp);//0.0;//
		}

		if(z>z2)
		{
			P[iz]=float(z)/float(z2)*1.0;//1.0;//
		}
	}
}*/

/*===========================================================

  This subroutine is used for FFT/IFFT

  ===========================================================*/
void fft(float *xreal,float *ximag,int n,int sign)
{
	int i,j,k,m,temp;
	int h,q,p;
	float t;
	float *a,*b;
	float *at,*bt;
	int *r;

	a=(float*)malloc(n*sizeof(float));
	b=(float*)malloc(n*sizeof(float));
	r=(int*)malloc(n*sizeof(int));
	at=(float*)malloc(n*sizeof(float));
	bt=(float*)malloc(n*sizeof(float));

	m=(int)(log(n-0.5)/log(2.0))+1; //2的幂，2的m次方等于n；
	for(i=0;i<n;i++)
	{
		a[i]=xreal[i];
		b[i]=ximag[i];
		r[i]=i;
	}
	for(i=0,j=0;i<n-1;i++)  //0到n的反序；
	{
		if(i<j)
		{
			temp=r[i];
			r[i]=j;
			r[j]=temp;
		}
		k=n/2;
		while(k<(j+1))
		{
			j=j-k;
			k=k/2;
		}
		j=j+k;
	}

	t=2*PI/n;
	for(h=m-1;h>=0;h--)
	{
		p=(int)pow(2.0,h);
		q=n/p;
		for(k=0;k<n;k++)
		{
			at[k]=a[k];
			bt[k]=b[k];
		}

		for(k=0;k<n;k++)
		{
			if(k%p==k%(2*p))
			{

				a[k]=at[k]+at[k+p];
				b[k]=bt[k]+bt[k+p];
				a[k+p]=(at[k]-at[k+p])*cos(t*(q/2)*(k%p))-(bt[k]-bt[k+p])*sign*sin(t*(q/2)*(k%p));
				b[k+p]=(bt[k]-bt[k+p])*cos(t*(q/2)*(k%p))+(at[k]-at[k+p])*sign*sin(t*(q/2)*(k%p));
			}
		}

	}

	for(i=0;i<n;i++)
	{
		if(sign==1)
		{
			xreal[r[i]]=a[i];
			ximag[r[i]]=b[i];
		}
		else if(sign==-1)
		{
			xreal[r[i]]=a[i]/n;
			ximag[r[i]]=b[i]/n;
		}
	}

	free(a);
	free(b);
	free(r);
	free(at);
	free(bt);
}

void cal_xishu(int Lx,float *rx)
{
	int m,i;
	float s1,s2;
	for(m=1;m<=Lx;m++)
	{
		s1=1.0;s2=1.0;
		for(i=1;i<m;i++)
		{
			s1=s1*(2.0*i-1)*(2.0*i-1);
			s2=s2*((2.0*m-1)*(2.0*m-1)-(2.0*i-1)*(2.0*i-1));
		}
		for(i=m+1;i<=Lx;i++)
		{
			s1=s1*(2.0*i-1)*(2.0*i-1);
			s2=s2*((2.0*m-1)*(2.0*m-1)-(2.0*i-1)*(2.0*i-1));
		}
		s2=fabs(s2);
		rx[m-1]=pow(-1.0,m+1)*s1/(s2*(2.0*m-1));
	}
}

void input_parameters(int *nx,int *nz,int *pml,int *Lc,float *dx,float *dz,float *rectime,float *dt,float *f0, int *Nf,int *freqintv,float *freq0,int *ns,float *sx0,
		float *shotdx,float *shotdep,float *recdep,float *moffsetx,int *itn,int *iterb,int *ifreqb,int *Ns,int *ricker_flag,int *noise_flag
		//,int *GPU_N
		)
{
	int ip;
	char strtmp[256];
	FILE *fp=fopen("../input/parameter.txt","r");
	if(fp==0)
	{
		printf("Cannot open the parameters1 file!\n");
		exit(0);
	}

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",nx);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",nz);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",pml);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",Lc);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%f",dx);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%f",dz);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%f",rectime);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp); 
	fscanf(fp,"\n");
	fscanf(fp,"%f",dt);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp); 
	fscanf(fp,"\n");
	fscanf(fp,"%f",f0);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",Nf);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",freqintv);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%f",freq0);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",ns);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%f",sx0);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%f",shotdx);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%f",shotdep);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%f",recdep);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%f",moffsetx);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",itn);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",iterb);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",ifreqb);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	for(ip=0;ip<*Nf;ip++)
	{
		fscanf(fp,"%d",&Ns[ip]);
	}
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",ricker_flag);
	fscanf(fp,"\n");

	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",noise_flag);
	fscanf(fp,"\n");

	return;
}
