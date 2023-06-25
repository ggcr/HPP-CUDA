// Compute C = A * B
__global__ void matrixMultiply(float * A, float * B, float * C,
			       int numARows, int numAColumns,
			       int numBRows, int numBColumns,
			       int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
	
	// Check boundaries
	
}

int main(int argc, char ** argv) {
    float * h_A; // The A matrix
    float * h_B; // The B matrix
    float * h_C; // The output C matrix

    float * d_A;
    float * d_B;
    float * d_C;

    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A

    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B

    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    //@@ Set numCRows and numCColumns
    // if A · B then C = (numARows, numBColumns)
    numCRows = numARows;
    numCColumns = numBColumns;
    if (numAColumns != numBRows) 
        return -1;
    
    //@@ Allocate the hostC matrix
    h_A = (float*) malloc(sizeof(float) * numARows * numAColumns);
    // Initialize
    for (int i = 0; i < numARows; i++) {
        for (int j = 0; j < numAColumns; j++) {
            h_A[i * numAColumns + j] = i * numAColumns + j;
        }
    }

    h_B = (float*) malloc(sizeof(float) * numBRows * numBColumns);
    // Initialize
    for (int i = 0; i < numBRows; i++) {
        for (int j = 0; j < numBColumns; j++) {
            h_B[i * numBColumns + j] = i * numBColumns + j;
        }
    }

    h_C = (float*) malloc(sizeof(float) * numCRows * numCColumns);
	
    //@@ Allocate GPU memory here
    cudaMalloc((void**)&d_A, sizeof(float) * numARows * numAColumns);
    cudaMalloc((void**)&d_B, sizeof(float) * numBRows * numBColumns);
    cudaMalloc((void**)&d_C, sizeof(float) * numCRows * numCColumns);

    //@@ Copy memory to the GPU here
    cudaMemcpy(d_A, h_A, sizeof(float) * numARows * numAColumns, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * numBRows * numBColumns, cudaMemcpyHostToDevice);
    
    //@@ Initialize the grid and block dimensions here
    // En DimGrid mete primero numCColumns y luego NumCRows...a
    // similar al m y n que metía en el ejemplo de la Pic...
    dim3 DimGrid = (((numCColumns-1) / 16 + 1, (numCRows-1) / 16 + 1, 1);
    dim3 DimBlock = (16, 16, 1);
	
    //@@ Launch the GPU Kernel here
    matrixMultiply<<<DimGrid, DimBlock>>>(d_A, d_B, d_C,
                    numARows, numAColumns,
                    numBRows, numBColumns,
                    numCRows, numCColumns);
	
    //@@ Copy the GPU memory back to the CPU here	
    cudaMemcpy(h_C, d_C, sizeof(float) * numCRows * numCColumns, cudaMemcpyDeviceToHost);

    //@@ Free the GPU memory here
    cudaFree(d_A);  cudaFree(d_B);  cudaFree(d_C);
    free(h_A);      free(h_B);      free(h_C);

    return 0;
}