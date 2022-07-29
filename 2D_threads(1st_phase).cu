#include "cuda.h"
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
//
typedef struct {
	int *A, *B, *C;
	int n, m, p;
} DataSet;
//
void fillDataSet(DataSet *dataSet);
void printDataSet(DataSet dataSet);
void closeDataSet(DataSet dataSet);
//
//
cudaError_t multiplyGPU(DataSet dataSet);
//
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
//
__global__ void mutrixMulKernel(const int *m_d,const int *n_d, int *p_d, int width)
{
	int pvalue = 0;
	int y = threadIdx.y;
	int x = threadIdx.x;
	for (int k = 0; k < width; ++k)
	{
		int melement = m_d[y * width + k];
		int nelement = n_d[k * width + x];
		pvalue += melement * nelement;
	}
	p_d[y * width + x] = pvalue;
}

int main(int argc, char *argv[])
{
	DataSet dataSet;
	if (argc < 4)
	{
		printf("[-] Invalid No. of arguments.\n");
		printf("[-] Try -> <n> <m> <p>\n");
		printf(">>> ");
		scanf_s("%d %d %d", &dataSet.n, &dataSet.m, &dataSet.p);
	}
	else
	{
		dataSet.n = atoi(argv[1]);
		dataSet.m = atoi(argv[2]);
		dataSet.p = atoi(argv[3]);
	}
	fillDataSet(&dataSet);

	



    // Add vectors in parallel.
    cudaError_t cudaStatus = multiplyGPU(dataSet);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	printDataSet(dataSet);
	closeDataSet(dataSet);


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	getchar();
	getchar();
	getchar();

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t multiplyGPU(DataSet dataSet)
{
   
	int width = dataSet.m;
	int size = width * width * sizeof(int);
	int * m_d, *n_d, *p_d;
	int * m, *n, *p;
	m = dataSet.A;
	n = dataSet.B;
	p = dataSet.C;

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&m_d, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&n_d, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&p_d, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(m_d, m, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(n_d, n, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


	dim3 dimgrid(1, 1);
	dim3 dimblock(width, width);
    // Launch a kernel on the GPU with one thread for each element.
	mutrixMulKernel<<<dimgrid, dimblock>>>(m_d, n_d, p_d, width);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(p, p_d, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(m_d);
    cudaFree(n_d);
    cudaFree(p_d);
    
    return cudaStatus;
}



void fillDataSet(DataSet *dataSet){
	int i, j;

	dataSet->A = (int *)malloc(sizeof(int)* dataSet->n * dataSet->m);
	dataSet->B = (int *)malloc(sizeof(int)* dataSet->m * dataSet->p);
	dataSet->C = (int *)malloc(sizeof(int)* dataSet->n * dataSet->p);

	srand(time(NULL));

	for (i = 0; i < dataSet->n; i++){
		for (j = 0; j < dataSet->m; j++){
			dataSet->A[i*dataSet->m + j] = rand() % 100;
		}
	}

	for (i = 0; i < dataSet->m; i++){
		for (j = 0; j < dataSet->p; j++){
			dataSet->B[i*dataSet->p + j] = rand() % 100;
		}
	}
}

void printDataSet(DataSet dataSet){
	int i, j;

	printf("[-] Matrix A\n");
	for (i = 0; i < dataSet.n; i++){
		for (j = 0; j < dataSet.m; j++){
			printf("%-4d", dataSet.A[i*dataSet.m + j]);
		}
		putchar('\n');
	}

	printf("[-] Matrix B\n");
	for (i = 0; i < dataSet.m; i++){
		for (j = 0; j < dataSet.p; j++){
			printf("%-4d", dataSet.B[i*dataSet.p + j]);
		}
		putchar('\n');
	}

	printf("[-] Matrix C\n");
	for (i = 0; i < dataSet.n; i++){
		for (j = 0; j < dataSet.p; j++){
			printf("%-8d", dataSet.C[i*dataSet.p + j]);
		}
		putchar('\n');
	}
}

void closeDataSet(DataSet dataSet){
	free(dataSet.A);
	free(dataSet.B);
	free(dataSet.C);
}