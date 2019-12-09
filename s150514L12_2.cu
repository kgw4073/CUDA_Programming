#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include <Windows.h>
#include <time.h>
#include <assert.h>
#include "CImg.h"

struct uchar4;

void sharpenParallel(uchar4 *arr, int w, int h);
void sharpenParallel_shared(uchar4 *arr, int w, int h);

#define cimg_display 0

#define TX 32
#define TY 32
#define RAD 1

#define CUDA_CALL(x) { const cudaError_t a = (x); if(a != cudaSuccess) { printf("\nCuda Error: %s (err_num=%d) at line:%d\n", cudaGetErrorString(a), a, __LINE__); cudaDeviceReset(); assert(0);}}
typedef float TIMER_T;

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

int divUp(int a, int b) { return (a + b - 1) / b; }

__device__
unsigned char clip(int n) { return n > 255 ? 255 : (n < 0 ? 0 : n); }

__device__
int idxClip(int idx, int idxMax) {
	return idx > (idxMax - 1) ? (idxMax - 1) : (idx < 0 ? 0 : idx);
}

__device__
int flatten(int col, int row, int width, int height) {
	return idxClip(col, width) + idxClip(row, height)*width;
}

__global__
void sharpenKernel(uchar4 *d_out, const uchar4 *d_in, const float *d_filter, int w, int h) {
	const int c = threadIdx.x + blockDim.x * blockIdx.x;
	const int r = threadIdx.y + blockDim.y * blockIdx.y;
	if ((c >= w) || (r >= h)) return;
	const int i = flatten(c, r, w, h);
	const int fltSz = 2 * RAD + 1;
	float rgb[3] = { 0.f, 0.f, 0.f };

	for (int rd = -RAD; rd <= RAD; ++rd) {
		for (int cd = -RAD; cd <= RAD; ++cd) {
			const int imgIdx = flatten(c + cd, r + rd, w, h);
			const int fltIdx = flatten(RAD + cd, RAD + rd, fltSz, fltSz);
			const uchar4 color = d_in[imgIdx];
			const float weight = d_filter[fltIdx];
			rgb[0] += weight * color.x;
			rgb[1] += weight * color.y;
			rgb[2] += weight * color.z;
		}
	}

	d_out[i].x = clip(rgb[0]);
	d_out[i].y = clip(rgb[1]);
	d_out[i].z = clip(rgb[2]);

}

void sharpenParallel(uchar4 *arr, int w, int h) {
	const int fltSz = 2 * RAD + 1;
	const float filter[9] = { -1.0, -1.0, -1.0,
							 -1.0,  9.0, -1.0,
							 -1.0, -1.0, -1.0 };

	uchar4 *d_in = 0, *d_out = 0;
	float *d_filter = 0;

	cudaMalloc(&d_in, w*h * sizeof(uchar4));
	cudaMemcpy(d_in, arr, w*h * sizeof(uchar4), cudaMemcpyHostToDevice);
	cudaMalloc(&d_out, w*h * sizeof(uchar4));
	cudaMalloc(&d_filter, fltSz*fltSz * sizeof(float));
	cudaMemcpy(d_filter, filter, fltSz*fltSz * sizeof(float),
		cudaMemcpyHostToDevice);

	const dim3 blockSize(TX, TY);
	const dim3 gridSize(divUp(w, TX), divUp(h, TY));
	const size_t smSz = ((TX + 2 * RAD)*(TY + 2 * RAD) + (TX*TY)) * sizeof(uchar4);


	CHECK_TIME_INIT_GPU();
	CHECK_TIME_START_GPU();
	sharpenKernel << <gridSize, blockSize >> > (d_out, d_in, d_filter, w, h);

	CHECK_TIME_END_GPU(device_time);
	CHECK_TIME_DEST_GPU();

	printf("***GPU for global Mem Time taken = %.6fms\n", device_time);

	cudaMemcpy(arr, d_out, w*h * sizeof(uchar4), cudaMemcpyDeviceToHost);
	cudaFree(d_in);
	cudaFree(d_out);
	cudaFree(d_filter);
}

__global__
void sharpenKernel_shared(uchar4 *d_out, const uchar4 *d_in, const float *d_filter, int w, int h) {

	//TODO : global mem to shared mem
	const int c = threadIdx.x + blockDim.x * blockIdx.x;
	const int r = threadIdx.y + blockDim.y * blockIdx.y;
	if ((c >= w) || (r >= h)) return;
	const int i = flatten(c, r, w, h);
	const int s_c = threadIdx.x + RAD;	const int s_r = threadIdx.y + RAD;
	const int s_w = blockDim.x + 2 * RAD;	const int s_h = blockDim.y + 2 * RAD;
	const int s_i = flatten(s_c, s_r, s_w, s_h);	const int fltSz = 2 * RAD + 1;
	float rgb[3] = { 0.f, 0.f, 0.f };
	extern __shared__ uchar4 s_in[];
	// Regular cells
	s_in[s_i] = d_in[i];

	// Halo cells
	if (threadIdx.x < RAD && threadIdx.y < RAD) {
		s_in[flatten(s_c - RAD, s_r - RAD, s_w, s_h)] = d_in[flatten(c - RAD, r - RAD, w, h)];
		s_in[flatten(s_c + blockDim.x, s_r - RAD, s_w, s_h)] = d_in[flatten(c + blockDim.x, r - RAD, w, h)];
		s_in[flatten(s_c - RAD, s_r + blockDim.y, s_w, s_h)] = d_in[flatten(c - RAD, r + blockDim.y, w, h)];
		s_in[flatten(s_c + blockDim.x, s_r + blockDim.y, s_w, s_h)] = d_in[flatten(c + blockDim.x, r + blockDim.y, w, h)];
	}
	if (threadIdx.x < RAD) {
		s_in[flatten(s_c - RAD, s_r, s_w, s_h)] = d_in[flatten(c - RAD, r, w, h)];
		s_in[flatten(s_c + blockDim.x, s_r, s_w, s_h)] = d_in[flatten(c + blockDim.x, r, w, h)];
	}
	if (threadIdx.y < RAD) {
		s_in[flatten(s_c, s_r - RAD, s_w, s_h)] = d_in[flatten(c, r - RAD, w, h)];
		s_in[flatten(s_c, s_r + blockDim.y, s_w, s_h)] = d_in[flatten(c, r + blockDim.y, w, h)];
	}
	__syncthreads();
	//TODO : filtering
	for (int rd = -RAD; rd <= RAD; ++rd) {
		for (int cd = -RAD; cd <= RAD; ++cd) {
			const int imgIdx = flatten(s_c + cd, s_r + rd, s_w, s_h);
			const int fltIdx = flatten(RAD + cd, RAD + rd, fltSz, fltSz);
			const uchar4 color = s_in[imgIdx];
			const float weight = d_filter[fltIdx];
			rgb[0] += weight * color.x;
			rgb[1] += weight * color.y;
			rgb[2] += weight * color.z;
		}
	}
	d_out[i].x = clip(rgb[0]);
	d_out[i].y = clip(rgb[1]);
	d_out[i].z = clip(rgb[2]);
}

void sharpenParallel_shared(uchar4 *arr, int w, int h) {
	
	const int fltSz = 2 * RAD + 1;
	const float filter[9] = { -1.0, -1.0, -1.0,
							 -1.0,  9.0, -1.0,
							 -1.0, -1.0, -1.0 };
	uchar4 *d_in = 0, *d_out = 0;
	float *d_filter = 0;
	//TODO
	cudaMalloc(&d_in, w*h * sizeof(uchar4));
	cudaMemcpy(d_in, arr, w*h * sizeof(uchar4),cudaMemcpyHostToDevice);
	cudaMalloc(&d_out, w*h * sizeof(uchar4));
	cudaMalloc(&d_filter, fltSz*fltSz * sizeof(float));
	cudaMemcpy(d_filter, filter, fltSz*fltSz * sizeof(float), cudaMemcpyHostToDevice);
	const dim3 blockSize(TX, TY);
	const dim3 gridSize(divUp(w, TX), divUp(h, TY));
	const size_t smSz = ((TX + 2 * RAD)*(TY + 2 * RAD)) * sizeof(uchar4);
	CHECK_TIME_INIT_GPU();
	CHECK_TIME_START_GPU();
	sharpenKernel_shared << <gridSize, blockSize, smSz >> > (d_out, d_in, d_filter, w, h);

	CHECK_TIME_END_GPU(device_time);
	CHECK_TIME_DEST_GPU();

	printf("***GPU for shared Mem Time taken = %.6fms\n", device_time);

	//TODO
	cudaMemcpy(arr, d_out, w*h * sizeof(uchar4), cudaMemcpyDeviceToHost);
	cudaFree(d_in);
	cudaFree(d_out);
	cudaFree(d_filter);
}

void sharpen(char* inputName, char* outName) {
	cimg_library::CImg<unsigned char>image(inputName);
	printf("\nRun for %s\n", inputName);

	const int w = image.width();
	const int h = image.height();

	// Initialize uchar4 array for image processing
	uchar4 *arr = (uchar4*)malloc(w*h * sizeof(uchar4));

	// Copy CImg data to array
	for (int r = 0; r < h; ++r) {
		for (int c = 0; c < w; ++c) {
			arr[r*w + c].x = image(c, r, 0);
			arr[r*w + c].y = image(c, r, 1);
			arr[r*w + c].z = image(c, r, 2);
		}
	}

	sharpenParallel(arr, w, h);

	// Copy from array to CImg data
	for (int r = 0; r < h; ++r) {
		for (int c = 0; c < w; ++c) {
			image(c, r, 0) = arr[r*w + c].x;
			image(c, r, 1) = arr[r*w + c].y;
			image(c, r, 2) = arr[r*w + c].z;
		}
	}

	image.save_bmp(outName);


	free(arr);
}


void sharpen_shared(char* inputName, char* outName) {
	cimg_library::CImg<unsigned char>image(inputName);
	printf("\nRun for %s\n", inputName);

	const int w = image.width();
	const int h = image.height();

	// Initialize uchar4 array for image processing
	uchar4 *arr = (uchar4*)malloc(w*h * sizeof(uchar4));

	// Copy CImg data to array
	for (int r = 0; r < h; ++r) {
		for (int c = 0; c < w; ++c) {
			arr[r*w + c].x = image(c, r, 0);
			arr[r*w + c].y = image(c, r, 1);
			arr[r*w + c].z = image(c, r, 2);
		}
	}

	sharpenParallel_shared(arr, w, h);

	// Copy from array to CImg data
	for (int r = 0; r < h; ++r) {
		for (int c = 0; c < w; ++c) {
			image(c, r, 0) = arr[r*w + c].x;
			image(c, r, 1) = arr[r*w + c].y;
			image(c, r, 2) = arr[r*w + c].z;
		}
	}

	image.save_bmp(outName);


	free(arr);
}


int main() {

	sharpen("butterfly.bmp", "out1.bmp");
	sharpen("input.bmp", "out2.bmp");

	sharpen_shared("butterfly.bmp", "out3.bmp");
	sharpen_shared("input.bmp", "out4.bmp");

	return 0;
}