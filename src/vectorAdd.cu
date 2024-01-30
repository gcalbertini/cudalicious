#include "./projectsUdacity/utils.h"
#include <memory>

using namespace std;

static const int BLOCK_SIZE = 256;
static const int N = 2'000;

__global__ void vadd(int *a, int*b, int*c, int N){
	int myID = blockDim.x*blockIdx.x + threadIdx.x;
	// Why id < N? Well, grid of 8 blocks but last one has 256 threads, and only have assignment for 2000-7*256 = 208 of them
	if (myID < N){
		c[myID] = a[myID] + b[myID];
	}
}

int main(void)
{
	// Smart pointer for uniq: dereferencing smart ptr will also dereference its underlying ptr
	// smartu_ptr.get() if you need ptr
	// *smartu_ptr if you need reference

	// host and device ptrs
	unique_ptr<int[]> ha, hb, hc;
	int *da, *dc, *db;
	int i;

	// host mem alloc
	ha = make_unique<int[]>(N);
	hb = make_unique<int[]>(N);
	hc = make_unique<int[]>(N);

	// cuda mem alloc
	checkCudaErrors(cudaMalloc((void **)&da, sizeof(int) * N));
	checkCudaErrors(cudaMalloc((void **)&db, sizeof(int) * N));
	checkCudaErrors(cudaMalloc((void **)&dc, sizeof(int) * N));

	for (i = 0; i < N; ++i)
	{
		ha[i] = rand() % 1000;
		hb[i] = rand() % 1000;
	}

	// data xfer from host to device
	checkCudaErrors(cudaMemcpy((void *)da, ha.get(), sizeof(int) * N, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy((void *)db, hb.get(), sizeof(int) * N, cudaMemcpyHostToDevice));

	int grid = ceil(N * 1.0 / BLOCK_SIZE);
	vadd <<< grid, BLOCK_SIZE >>> (da, db, dc, N);

	checkCudaErrors(cudaDeviceSynchronize());
	// Wait for GPU launched work to complete
	checkCudaErrors(cudaGetLastError());

	// now xfer device to host
	cudaMemcpy((void *)hc.get(), dc, sizeof(int) * N, cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaGetLastError());

	// correctness check
	for (i = 0; i < N; ++i)
		if (hc[i] != ha[i] + hb[i])
		{
			printf("Error at index %i : %i VS %i\n", i, hc[i], ha[i] + hb[i]);
		}

	checkCudaErrors(cudaFree((void *)da));
	checkCudaErrors(cudaFree((void *)db));
	checkCudaErrors(cudaFree((void *)dc));
	checkCudaErrors(cudaDeviceReset()); // cleans up all resources associtaed with current device in the current process. Only call before termination!

	return 0;
}