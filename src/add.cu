// https://developer.nvidia.com/blog/even-easier-introduction-cuda/

#include <iostream>
#include <math.h>

// function to add elements of the array; CUDA kernel to add
__global__ void add(int n, float *x, float *y)
{
	for (int i = 0; i < n; ++i)
	{
		y[i] = x[i] + y[i]; // same as *(y + i) += *(x + i);
	}

	/*
		// Or like this but note significant added space for stopping
		// can use comparison here as they are comparing ptrs of the
		// same array
		float* stopping = y + n; // what if n is intmax?
		for (; y < stopping; ++x, ++y) {
			*y += *x;
		}
	*/
}

int main(int argc, char **argv)
{
	int N = (argc > 1) ? std::atoi(argv[1]) : 1 << 2; // shifts the binary representation of the number 1 to the left by 20 positions. In terms of decimal values, this is equivalent to 2 raised to the power of 20.
	// float *x = new float[N];						   //{} throws as narrowing possible: float (HW dependent) can only accurately represent 24 bits worth of significand precision, not the whole 32-bit range of int
	// float *y = new float[N];
	float *x, *y;

	// allocate unified meme -- accessible from CPU or GPU (whichever is device)
	cudaMallocManaged(&x, N * sizeof(float));
	cudaMallocManaged(&y, N * sizeof(float));

	// init x and y arrays on the host
	for (int i = 0; i < N; ++i)
	{
		x[i] = 1.0f; // or *(x + i) = 1.0f;
		*(y + i) = 2.0f;
	}

	// run kernel on 2E20 elements on the CPU
	// add(N, x, y);

	// now on GPU
	add<<<1, 1>>>(N, x, y);

	// Wait for GPU to finish before accessing on host to avoid race conditions
	cudaDeviceSynchronize();

	float maxError = 0.0f;
	for (int i = 0; i < N; ++i)
	{
		maxError = fmax(maxError, fabs(y[i] - 3.0f));
	}
	std::cout << "Max error: " << maxError << std::endl;

	// free memory
	// delete[] x;
	// delete[] y;

	cudaFree(x);
	cudaFree(y);

	return 0;
}