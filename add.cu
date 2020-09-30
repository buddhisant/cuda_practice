//#include <torch/torch.h>
//#include <iostream>
//
//#include <ATen/ATen.h>
//#include <ATen/cuda/CUDAContext.h>
//
//#include <THC/THC.h>
//#include <THC/THCDeviceUtils.cuh>
//
//#include <vector>
//#include <iostream>
//#include <cmath>
//
//#include <torch/extension.h>
//
//#define CHECK_CUDA(x) \
//	TORCH_CHECK(x.device().is_cuda(), #x "must be a CUDA tensor")
//
//int main() {
//  at::Tensor tensor = at::rand({2, 3});
//  at::Tensor tensor_gpu = tensor.cuda();
//  CHECK_CUDA(tensor_gpu);
//  std::cout << tensor_gpu << std::endl;
//  return 0;
//}

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
