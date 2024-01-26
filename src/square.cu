// https://www.youtube.com/watch?v=iaRs_yJA_js&list=PLAwxTw4SYaPnFKojVQrmyOGFCqHTxfdv2&index=38
#include <iostream>
#include <math.h>

__global__ void square(float * d_out, float * d_in)
{
	int idx = threadIdx.x;
	float f = d_in[idx];
	d_out[idx] = f * f;
}

int main(int argc, char** argv)
{
	constexpr int ARRAY_SIZE {64};
	constexpr int ARRAY_BYTES {64 * sizeof(float)};

	// generate input array on host
	float h_in[ARRAY_SIZE] {}; //  float* h_in = new float[ARRAY_SIZE]{}; in C++ --> std::unique_ptr<float[]> h_in = std::make_unique<float[]>(ARRAY_SIZE);
	for (int i = 0; i < ARRAY_SIZE; ++i){
		h_in[i] = float(i);
	}
	float h_out[ARRAY_SIZE] {};

	// gpu mem ptrs
	float * d_out, * d_in;

	//allocate GPU mem
	cudaMalloc((void**) &d_in, ARRAY_BYTES);
	cudaMalloc((void**) &d_out, ARRAY_BYTES);

	//transfer array to GPU
	cudaMemcpy(d_in, h_in, ARRAY_SIZE, cudaMemcpyHostToDevice);

	//launch Kernel 
	square<<<1, ARRAY_SIZE>>>(d_out, d_in);

	//transfer from GPU to CPU
	cudaMemcpy(h_out, d_out, ARRAY_SIZE, cudaMemcpyDeviceToHost);

	// print out resulting array
	for (int i = 0; i < ARRAY_SIZE; ++i){
		printf("%f", h_out[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");
	}




}