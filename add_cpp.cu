/*用cpu实现2个矩阵之间的加法*/
#include<iostream>
#include<stdlib.h>
#include<sys/time.h>
#include<math.h>
#include"cuda_runtime.h"

using namespace std;

#define cols 1024
#define rows 1024

int main()
{
	struct timeval start, end;
	int n=cols*rows;
	float **A,**B,**C;
	float *a,*b,*c;
	A=new float* [cols];
	B=new float* [cols];
	C=new float* [cols];
	a=new float [n];
	b=new float [n];
	c=new float [n];

	for(int i=0;i<n;i++)
	{
		a[i]=2;
		b[i]=2;
	}

	for(int i=0;i<cols;i++)
	{
		A[i]=a+i*rows;
		B[i]=b+i*rows;
		C[i]=c+i*rows;
	}

	gettimeofday( &start, NULL);
	for(int i=0;i<rows;i++)
	{
		for(int j=0;j<cols;j++)
		{
			C[i][j]+=A[i][j]+B[i][j];
		}
	}
	gettimeofday( &end, NULL );

	float target=4.0;
	float error=0.0;
	for(int i=0;i<rows;i++)
	{
		for(int j=0;j<cols;j++)
		{
			error+=abs(C[i][j]-target);
		}
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

	return 0;
}
