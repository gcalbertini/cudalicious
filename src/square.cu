// https://www.youtube.com/watch?v=iaRs_yJA_js&list=PLAwxTw4SYaPnFKojVQrmyOGFCqHTxfdv2&index=38
#include <stdio.h>
#include <math.h>

__global__ void square(float *d_out, float *d_in)
{
	int idx = threadIdx.x;
	float f = d_in[idx];
	d_out[idx] = f * f;
}

int main(int argc, char **argv)
{
	const int ARRAY_SIZE = 64;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	// generate input array on host
	float h_in[ARRAY_SIZE]; //  in C++: float* h_in = new float[ARRAY_SIZE]{}; in C++ --> std::unique_ptr<float[]> h_in = std::make_unique<float[]>(ARRAY_SIZE);
	for (int i = 0; i < ARRAY_SIZE; ++i)
	{
		h_in[i] = float(i + 1);
	}
	float h_out[ARRAY_SIZE];

	// gpu mem ptrs
	float *d_out;
	float *d_in;

	// allocate GPU mem
	cudaMalloc((void **)&d_in, ARRAY_BYTES);
	cudaMalloc((void **)&d_out, ARRAY_BYTES);

	// transfer array to GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// kernel<<<dim3(bx,by,bz), dim3(tx,ty,tz), sharedBytesPerBlock>>>(...) to launch bx*by*bz blocks with tx*ty*tz threads/block for tx*ty*tz*bx*by*bz thread total
	square<<<1, ARRAY_SIZE>>>(d_out, d_in);

	// transfer from GPU to CPU
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	// print out resulting array
	for (int i = 0; i < ARRAY_SIZE; ++i)
	{
		printf("%f", h_out[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");
	}

	cudaFree(d_in);
	cudaFree(d_out);

	return 0;
}