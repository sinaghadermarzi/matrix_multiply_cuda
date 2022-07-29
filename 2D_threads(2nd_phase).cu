#include "cuda.h"
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <math.h>
//
typedef struct {
	int *A, *B, *C;
	int n, m, p;
} DataSet;
//
void fillDataSet(DataSet *dataSet);
void printDataSet(DataSet dataSet);
void closeDataSet(DataSet dataSet);

void multiply_Serial(DataSet dataSet);
//
//
cudaError_t multiplyGPU(DataSet dataSet, int num_threads);
//
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
//
__global__ void mutrixMulKernel(const int *m_d, const int *n_d , int * p_d ,int width)
{
	int pvalue = 0;
	int y = threadIdx.y;
	int x = threadIdx.x;
	int split_size_x = width / blockDim.x;
	int split_size_y = width / blockDim.y;
	int x_offset = x * split_size_x;
	int y_offset = y * split_size_y;
	for (int i = x_offset; i < x_offset + split_size_x; ++i)
	{

		for (int j = y_offset; j < y_offset + split_size_y; ++j)
		{

			for (int k = 0; k < width; ++k)
			{
				int melement = m_d[j * width + k];
				int nelement = n_d[k * width + i];
				pvalue += melement * nelement;
			}

			p_d[j * width + i] = pvalue;
		}
	}
}


int main(int argc, char *argv[]){
	DataSet dataSet;
	int N, num_threads;
	if (argc < 3){
		printf("[-] Invalid No. of arguments.\n");
		printf("[-] Try -> <N> <thrN>\n");
		printf(">>> ");
		scanf_s("%d %d", &N, &num_threads);

	}
	else{
		N = atoi(argv[1]);
		num_threads = atoi(argv[2]);
	}
	dataSet.m = N;
	dataSet.n = N;
	dataSet.p = N;

	fillDataSet(&dataSet);

	time_t start_t, finish_t;


	//printDataSet(dataSet);


	double sum = 0;
	double avg_p = 0;
	double elapsed_time;


	start_t = time(NULL);
	for (int i = 1; i <= 100; i++)
	{

		multiplyGPU(dataSet, num_threads);
		printf("Parallel-Test<%d>\n", i);

	}
	finish_t = time(NULL);

	elapsed_time = difftime(finish_t, start_t);
	avg_p = elapsed_time / 100;

	printf("\nParallel(%d Threads) - Average elapsed time = %f", num_threads, avg_p);

	printf("\n\nEnter 'd' to print Data or something else to continue... : ");


	if (getchar() == 'd')
	{
		printDataSet(dataSet);
	}


	printf("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");

	double avg_s = 0;
	start_t = time(NULL);

	for (int i = 0; i < 100; i++)
	{
		multiply_Serial(dataSet);
		printf("Serial-Test<%d>\n", i);
	}
	finish_t = time(NULL);

	elapsed_time = difftime(finish_t, start_t);
	avg_s = elapsed_time / 100;
	printf("\nSerial - Average elapsed time = %f", avg_s);
	double speedup = avg_s / avg_p;
	printf("\n\nSpeedup = %f", speedup);

	printf("\n\nEnter 'd' to print Data or something else to continue... : ");


	if (getchar() == 'd')
	{
		printDataSet(dataSet);
	}


	closeDataSet(dataSet);
	system("PAUSE");
	return EXIT_SUCCESS;
}
// Helper function for using CUDA to add vectors in parallel.
cudaError_t multiplyGPU(DataSet dataSet, int num_threads)
{

	int N = dataSet.m;
	int size = N * N * sizeof(int);
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

	int sqrtNTH = sqrt(num_threads);
	int blockdim_x = 0;
	if (sqrtNTH > 32)
	{
		blockdim_x = 32;
	}
	else
	{
		blockdim_x = sqrtNTH;
	}

	if ((sqrtNTH*sqrtNTH != num_threads) || (N % sqrtNTH !=0))
	{
		fprintf(stderr, "INAPPROPRIATE NUMBER OF THREADS  / MATRICE SIZE !\n");
		goto Error;
	}


	dim3 dimgrid(1, 1);
	dim3 dimblock(blockdim_x, blockdim_x);

	mutrixMulKernel<<< dimgrid , dimblock >>>(m_d, n_d, p_d, N);

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

void multiply_Serial(DataSet dataSet)
{
	int i, j, k, sum;

	for (i = 0; i < dataSet.n; i++)
	{
		for (j = 0; j < dataSet.p; j++)
		{
			sum = 0;
			for (k = 0; k < dataSet.m; k++)
			{
				sum += dataSet.A[i * dataSet.m + k] * dataSet.B[k * dataSet.p + j];
			}
			dataSet.C[i*dataSet.m +j] = sum;
		}
	}
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