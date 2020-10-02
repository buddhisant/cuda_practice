/*用gpu实现2个矩阵之间的乘法*/
#include<iostream>
#include<stdlib.h>
#include<sys/time.h>
#include<math.h>
#include"cuda_runtime.h"

using namespace std;

#define cols 1024
#define rows 1024

__global__ void multiply(float**Ad,float**Bd,float**Cd)
{
	int x = blockDim.x*blockIdx.x+threadIdx.x;
	int y = blockDim.y*blockIdx.y+threadIdx.y;
	if(x<rows && y<cols)
	{
		for(int i=0;i<cols;i++)
		{
				Cd[y][x]+=Ad[y][i]*Bd[i][x];
		}
	}
}

int main()
{
	struct timeval start, end;
	int n=cols*rows;
	float **A,**B,**C,**Ad,**Bd,**Cd;
	float *a,*b,*c,*ad,*bd,*cd;
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
		a[i]=2;
		b[i]=2;
	}

	for(int i=0;i<cols;i++)
	{
		A[i]=ad+i*rows;
		B[i]=bd+i*rows;
		C[i]=cd+i*rows;
	}

	gettimeofday( &start, NULL);//以开始向gpu拷贝数据为起点，记录时间
	cudaMemcpy(Ad,A,sizeof(float*)*cols,cudaMemcpyHostToDevice);
	cudaMemcpy(Bd,B,sizeof(float*)*cols,cudaMemcpyHostToDevice);
	cudaMemcpy(Cd,C,sizeof(float*)*cols,cudaMemcpyHostToDevice);
	cudaMemcpy(ad,a,sizeof(float)*n,cudaMemcpyHostToDevice);
	cudaMemcpy(bd,b,sizeof(float)*n,cudaMemcpyHostToDevice);

	dim3 dimBlock(16,16);
	dim3 dimGrid(cols/16+1,rows/16+1);
	multiply<<<dimGrid,dimBlock>>>(Ad,Bd,Cd);
	cudaMemcpy(c,cd,sizeof(float)*n,cudaMemcpyDeviceToHost);
	gettimeofday( &end, NULL );//以从gpu返回计算数据为终点，记录时间

	float target=4096;
	float error=0.0;
	for(int i=0;i<n;i++)
	{
		error+=abs(c[i]-target);
	}
	cout<<"error is "<<error<<endl;

	int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
	cout << "total time is " << timeuse/1000 << "ms" <<endl;
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
	return 0;
}
