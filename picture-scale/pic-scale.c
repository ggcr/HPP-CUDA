__global__ void PictureKernel(float* d_Pin, float* d_Pout, int n, int m) {
     int Row = blockIdx.y * blockDim.y + threadIdx.y;
     int Col = blockIdx.x * blockDim.x + threadIdx.x;

     if ((Row < m) && (Col < n))
	    d_Pout[Row * n + Col] = 2.0 * d_Pin[Row * n + Col];
}

__host__ void PictureScale(float* h_Pin, int n, int m) {
	// Allocate d_Pin, d_Pout
	int size = n * m * sizeof(float);
	cudaMalloc((void**)&d_Pin, size);
	cudaMalloc((void**)&d_Pout, size);
	// Copy h_Pin to d_Pin
	cudaMemcpy(d_Pin, h_Pin, size, cudaMemcpyDeviceToHost);
	// Kernel
	dim3 DimGrid((n-1)/16 + 1, (m-1)/16 + 1, 1);
	dim3 DimBlock(16, 16, 1);
	PictureKernel<<<DimGrid, DimBlock>>>(d_Pin, d_Pout, n, m);
	// Free
	cudaFree(d_Pin); cudaFree(d_Pout);
}
