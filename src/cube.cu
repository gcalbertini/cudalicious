// https://www.youtube.com/watch?v=iaRs_yJA_js&list=PLAwxTw4SYaPnFKojVQrmyOGFCqHTxfdv2&index=38
#include <iostream>
#include <math.h>

__global__ void cube(float * d_out, float * d_in){
	int idx = threadIdx.x;
	float f = d_in[idx];
	d_out[idx] = f * f * f;
}

int main(int argc, char ** argv){
	constexpr int ARRAY_SIZE {96};
	constexpr int ARRAY_BYTES {ARRAY_SIZE*sizeof(float)}; 

	float h_in[ARRAY_SIZE];

	for (int i = 0; i < ARRAY_SIZE; ++i){
		h_in[i] = float(i);
	}
	float h_out[ARRAY_SIZE];

	float * d_in, * d_out;

	cudaMalloc((void**) &d_in, ARRAY_BYTES);
	cudaMalloc((void**) &d_out, ARRAY_BYTES);

	cudaMemcpy(d_in, h_in, ARRAY_SIZE, cudaMemcpyHostToDevice);

	cube<<<1, ARRAY_SIZE>>>(d_out, d_in);

	cudaMemcpy(h_out, d_out, ARRAY_SIZE, cudaMemcpyDeviceToHost);

	// print out resulting array
	for (int i = 0; i < ARRAY_SIZE; ++i){
		printf("%f", h_out[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");
	}

	cudaFree(d_in);
	cudaFree(d_out);

	return 0;


}