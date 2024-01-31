__global__ void use_global_mem_GPU(float *arr)
{
	arr[threadIdx.x] = 2.0f * (float)threadIdx.x;
}

__global__ void use_shared_mem_GPU(float *arr, int size) {
    int i, index = threadIdx.x;
    float avg, sum = 0.0;

    __shared__ float sharr[128];
	//extern __shared__ float sharr[]; if dynamically set

    // Check bounds for shared array
    if (index < size) {
        sharr[index] = arr[index];
    }

    __syncthreads(); // Ensure all writes to shared copy are done

    // Check bounds for both shared array and input array
    if (index < size) {
		// Finding avg of all previous elements 
        for (i = 0; i < index; ++i) {
            sum += sharr[i];
        }
        avg = sum / (index + 1.0f);

		// if arr[idx] > avg of arr[0, idx-1] replace with avg
		// since arr[] is in global mem, change will be seen by host
		// and other blocks if any
        if (arr[index] > avg) {
            arr[index] = avg;
        }
    }
}
int main(int argc, char **argv)
{

	float h_arr[128];
	float *d_arr;

	cudaMalloc((void **)&d_arr, sizeof(float) * 128);
	cudaMemcpy((void *)d_arr, (void *)h_arr, sizeof(float) * 128, cudaMemcpyHostToDevice);
	use_global_mem_GPU<<<1, 128>>>(d_arr);
	cudaMemcpy((void *)h_arr, (void *)d_arr, sizeof(float) * 128, cudaMemcpyDeviceToHost);
	// ...

	use_shared_mem_GPU<<<1, 128>>>(d_arr, 128);

	// Dynamically set the array size for the shared kernel
    //use_shared_mem_GPU<<<1, arraySize, sizeof(float) * arraySize>>>(d_arr, arraySize);

	cudaMemcpy((void *)h_arr, (void*)d_arr, sizeof(float)*128, cudaMemcpyDeviceToHost);


	return 0;
}