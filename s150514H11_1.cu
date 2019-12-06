
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include <Windows.h>
#include <time.h>
#include <assert.h>

#define CUDA_CALL(x) { const cudaError_t a = (x); if(a != cudaSuccess) { printf("\nCuda Error: %s (err_num=%d) at line:%d\n", cudaGetErrorString(a), a, __LINE__); cudaDeviceReset(); assert(0);}}
typedef float TIMER_T;

__int64 start, freq, end;
#define CHECK_TIME_START { QueryPerformanceFrequency((LARGE_INTEGER*)&freq); QueryPerformanceCounter((LARGE_INTEGER*)&start); }
#define CHECK_TIME_END(a) { QueryPerformanceCounter((LARGE_INTEGER*)&end); a = (float)((float)(end - start) / (freq / 1000.0f)); }

cudaEvent_t cuda_timer_start, cuda_timer_stop;
#define CUDA_STREAM_0 (0)

// CUDA event 객체를 사용하여 커널 실행시간 측정
void create_device_timer()
{
	CUDA_CALL(cudaEventCreate(&cuda_timer_start));
	CUDA_CALL(cudaEventCreate(&cuda_timer_stop));
}

void destroy_device_timer()
{
	CUDA_CALL(cudaEventDestroy(cuda_timer_start));
	CUDA_CALL(cudaEventDestroy(cuda_timer_stop));
}

inline void start_device_timer()
{
	cudaEventRecord(cuda_timer_start, CUDA_STREAM_0);
}

inline TIMER_T stop_device_timer()
{
	TIMER_T ms;
	cudaEventRecord(cuda_timer_stop, CUDA_STREAM_0);
	cudaEventSynchronize(cuda_timer_stop);

	cudaEventElapsedTime(&ms, cuda_timer_start, cuda_timer_stop);
	return ms;
}

#define CHECK_TIME_INIT_GPU() { create_device_timer(); }
#define CHECK_TIME_START_GPU() { start_device_timer(); }
#define CHECK_TIME_END_GPU(a) { a = stop_device_timer(); }
#define CHECK_TIME_DEST_GPU() { destroy_device_timer(); }

TIMER_T compute_time = 0;
TIMER_T device_time = 0;

int n;
#define BLOCK_SIZE 128
#define ELEMENT_SIZE (1<<20)
const int ELEM_PER_VECTOR = 32;
float(*pVecX), (*pVecY), (*pVecY_G);
float(*pMatA);
void init_MatVec(void)
{
	srand((unsigned)time(NULL));
	FILE* fp = fopen("gen.bin", "rb");
	fread(&n, sizeof(float), 1, fp);

	pVecX = new float[n * ELEM_PER_VECTOR];
	pVecY = new float[n * ELEM_PER_VECTOR];
	pVecY_G = new float[n * ELEM_PER_VECTOR];
	pMatA = new float[ELEM_PER_VECTOR * ELEM_PER_VECTOR];

	fread(pVecX, sizeof(float), n * ELEM_PER_VECTOR, fp);
	fread(pMatA, sizeof(float), ELEM_PER_VECTOR * ELEM_PER_VECTOR, fp);
	fclose(fp);
}

void Mat_Vec_Multiply()
{
	int vec_idx, i, j;

	for (vec_idx = 0; vec_idx < ELEMENT_SIZE; vec_idx++) {
		for (i = 0; i < ELEM_PER_VECTOR; i++) {
			float sum = 0;
			for (j = 0; j < ELEM_PER_VECTOR; j++) {
				sum += pMatA[i * ELEM_PER_VECTOR + j] * pVecX[vec_idx * ELEM_PER_VECTOR + j];
			}
			pVecY[vec_idx * ELEM_PER_VECTOR + i] = sum;
		}
	}
}

__global__ void Mat_Vec_Multiply_Kernel(float *d_VecY, float *d_VecX, float *d_MatA, int Vec_Size)
{
	//TODO
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int id = col;

	for (int i = 0; i < ELEM_PER_VECTOR; i++) {
		float sum = 0;
		for (int j = 0; j < ELEM_PER_VECTOR; j++) {
			sum += d_MatA[i * ELEM_PER_VECTOR + j] * d_VecX[id * ELEM_PER_VECTOR + j];
		}
		d_VecY[id*ELEM_PER_VECTOR + i] = sum;
	}

}

void Mat_Vec_Multiply_GPU(float *p_VecX, float *p_MatA, float *p_VecY_G)
{
	float *d_VecY, *d_VecX, *d_MatA;


	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);

	cudaDeviceProp deviceProp;
	CUDA_CALL(cudaGetDeviceProperties(&deviceProp, 0));

	size_t size = ELEM_PER_VECTOR * sizeof(float);
	//GPU 메모리 할당, 데이터를 CPU 메모리에서 GPU 메모리로 전송
	CUDA_CALL(cudaMalloc(&d_VecX, size*n));
	CUDA_CALL(cudaMemcpy(d_VecX, pVecX, n*size, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMalloc(&d_MatA, ELEM_PER_VECTOR*size));
	CUDA_CALL(cudaMemcpy(d_MatA, pMatA, ELEM_PER_VECTOR*size, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMalloc(&d_VecY, size*n));

	//block size, grid size(블록 수) 를 선언
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(ELEMENT_SIZE / dimBlock.x);

	//아래 코드를 사용하여 커널을 호출
	CHECK_TIME_INIT_GPU();
	CHECK_TIME_START_GPU();
	Mat_Vec_Multiply_Kernel << <dimGrid, dimBlock >> > (d_VecY, d_VecX, d_MatA, ELEM_PER_VECTOR);
	CHECK_TIME_END_GPU(device_time);
	CHECK_TIME_DEST_GPU();

	//결과 데이터를 GPU 메모리에서 CPU 메모리로 이동
	CUDA_CALL(cudaDeviceSynchronize());
	CUDA_CALL(cudaMemcpy(pVecY_G, d_VecY, size*n, cudaMemcpyDeviceToHost));

}

void init_data(int size) {
	srand(0);
	FILE *fp = fopen("gen.bin", "wb");
	fwrite(&size, sizeof(int), 1, fp);

	int i, j;
	float x;

	for (i = 0; i < size; i++) {
		for (j = 0; j < ELEM_PER_VECTOR; j++) {
			x = 2.0f*((float)rand() / RAND_MAX) - 1.0f;
			fwrite(&x, sizeof(float), 1, fp);
		}
	}

	for (i = 0; i < ELEM_PER_VECTOR; i++) {
		for (j = 0; j < ELEM_PER_VECTOR; j++) {
			x = 2.0f*((float)rand() / RAND_MAX) - 1.0f;
			fwrite(&x, sizeof(float), 1, fp);
		}
	}

	fclose(fp);

	return;
}

bool check_equal() {
	for (int i = 0; i < ELEMENT_SIZE * ELEM_PER_VECTOR; i++) {
		if (fabs(pVecY[i] - pVecY_G[i]) > 0.001) {
			return false;
		}
	}
	return true;
}

int main()
{
	init_data(ELEMENT_SIZE);
	init_MatVec();
	printf("n = %d  file open ok.\n", n);

	CHECK_TIME_START;
	Mat_Vec_Multiply();
	CHECK_TIME_END(compute_time);
	//printf("\nn : %d, BLOCK_SIZE : %d\n", ELEMENT_SIZE, BLOCK_SIZE);
	printf("***CPU Time taken = %.6fms\n", compute_time);

	Mat_Vec_Multiply_GPU(pVecX, pMatA, pVecY_G);
	printf("***GPU Time taken = %.6fms\n", device_time);

	bool check = check_equal();
	if (check)
		printf("CPU and GPU calculate same\n");
	else
		printf("CPU and GPU calculate difference\n");

	printf("CPU [10] = %f, GPU [10] = %f\n", pVecY[10], pVecY_G[10]);

	free(pVecX);
	free(pVecY);
	free(pVecY_G);
	free(pMatA);

	return 0;
}


