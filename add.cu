/*利用cuda完成两个1024*1024矩阵的加法*/
#include<iostream>
#include<stdlib.h>
#include<sys/time.h>
#include<math.h>
#include"cuda_runtime.h"

#define cols 1024
#define rows 1024

using namespace std;

__global__ void Add(float** Ad,float** Bd,float** Cd)
{
	int x = blockDim.x*blockIdx.x+threadIdx.x;
	int y = blockDim.y*blockIdx.y+threadIdx.y;
	if(x<cols && y<rows)
	{
		Cd[x][y]=Ad[x][y]+Bd[x][y];
	}
}

int main()
{
	struct timeval start, end;
	gettimeofday( &start, NULL);
	float **A,**B,**C,**Ad,**Bd,**Cd;
	float *a,*b,*c,*ad,*bd,*cd;
	int n=rows * cols;

	A=new float* [cols];
	B=new float* [cols];
	C=new float* [cols];
	a=new float [n];
	b=new float [n];
	c=new float [n];

	cudaMalloc((void**)&Ad,sizeof(float*)*cols);
	cudaMalloc((void**)&Bd,sizeof(float*)*cols);
	cudaMalloc((void**)&Cd,sizeof(float*)*cols);
	cudaMalloc((void**)&ad,sizeof(float)*n);
	cudaMalloc((void**)&bd,sizeof(float)*n);
	cudaMalloc((void**)&cd,sizeof(float)*n);

	for(int i=0;i<n;i++)
	{
		a[i]=90.0;
		b[i]=10.0;
	}
	for(int i=0;i<cols;i++)
	{
		//ad, bd, cd是一维向量，如果在gpu上按照二维矩阵进行运算，则需要将其和Ad, Bd, Cd建立对应关系，建立对应关系的过程在cpu上完成
		A[i]=ad+i*rows;
		B[i]=bd+i*rows;
		C[i]=cd+i*rows;
	}

	cudaMemcpy(Ad,A,cols*sizeof(float*),cudaMemcpyHostToDevice);
	cudaMemcpy(Bd,B,cols*sizeof(float*),cudaMemcpyHostToDevice);
	cudaMemcpy(Cd,C,cols*sizeof(float*),cudaMemcpyHostToDevice);
	cudaMemcpy(ad,a,n*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(bd,b,n*sizeof(float),cudaMemcpyHostToDevice);

	dim3 dimBlock(16,16);
	dim3 dimGrid(cols/16+1,rows/16+1);
	Add<<<dimGrid,dimBlock>>>(Ad,Bd,Cd);

	cudaMemcpy(c,cd,n*sizeof(float),cudaMemcpyDeviceToHost);

	float target=100.0;
	float error=0.0;
	for(int i=0;i<n;i++)
	{
		error+=abs(target-c[i]);
	}
	cout<<"total error is "<<error<<endl;
	delete [] a;
	delete [] b;
	delete [] c;
	delete [] A;
	delete [] B;
	delete [] C;
	cudaFree(Ad);
	cudaFree(Bd);
	cudaFree(Cd);
	cudaFree(ad);
	cudaFree(bd);
	cudaFree(cd);
	gettimeofday( &end, NULL );
	int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
	cout << "total time is " << timeuse/1000 << "ms" <<endl;
	return 0;
}
