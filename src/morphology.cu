#include "morphology.h"
#include <cuda.h>


enum class MorphOpType
{
    ERODE,
    DILATE,
};


template <MorphOpType opType> inline __device__ unsigned char
elementOp(unsigned char lhs, unsigned char rhs);

template <> inline __device__ unsigned char
elementOp<MorphOpType::ERODE>(unsigned char lhs, unsigned char rhs)
{
    return min(lhs, rhs);
}

template <> inline __device__ unsigned char
elementOp<MorphOpType::DILATE>(unsigned char lhs, unsigned char rhs)
{
    return max(lhs, rhs);
}

template <MorphOpType opType> inline __device__ unsigned char
borderValue();

template <> inline __device__ unsigned char
borderValue<MorphOpType::ERODE>()
{
    return 0;
}

template <> inline __device__ unsigned char
borderValue<MorphOpType::DILATE>()
{
    return 255;
}


// NOTE: step-efficient parallel scan
template <MorphOpType opType> __device__ void
reversedScan(unsigned char* __restrict__ buffer,
	     unsigned char* __restrict__ opArray,
	     int selSize, int tid)
{
    opArray[tid] = buffer[tid];
    __syncthreads();

    for (int offset = 1; offset < selSize; offset *= 2)
    {
        if (tid <= selSize - 1 - offset)
            opArray[tid] = elementOp<opType>(opArray[tid],
					     opArray[tid + offset]);
        __syncthreads();
    }
}

// NOTE: step-efficient parallel scan
template <MorphOpType opType> __device__ void
scan(unsigned char* __restrict__ buffer,
     unsigned char* __restrict__ opArray,
     int selSize, int tid)
{
    opArray[tid] = buffer[tid];
    __syncthreads();

    for (int offset = 1; offset < selSize; offset *= 2)
    {
        if (tid >= offset)
            opArray[tid] = elementOp<opType>(opArray[tid],
					     opArray[tid - offset]);
	__syncthreads();
    }
}

// NOTE: step-efficient parallel scan
template <MorphOpType opType> __device__ void
twoWayScan(unsigned char* __restrict__ buffer,
	   unsigned char* __restrict__ opArray,
	   int selSize, int tid)
{
    opArray[tid]	   = buffer[tid];
    opArray[tid + selSize] = buffer[tid + selSize];
    __syncthreads();

    for (int offset = 1; offset < selSize; offset *= 2)
    {
        if (tid >= offset)
            opArray[tid + selSize - 1]
		= elementOp<opType>(opArray[tid + selSize - 1],
				    opArray[tid + selSize - 1 - offset]);
        if (tid <= selSize - 1 - offset)
            opArray[tid] = elementOp<opType>(opArray[tid],
					     opArray[tid + offset]);
        __syncthreads();
    }
}


template <MorphOpType opType> __global__ void
vhgw_horiz(unsigned char* __restrict__ dst, unsigned char* __restrict__ src,
	   int width, int height, int selSize)
{
    extern __shared__ unsigned char	sMem[];
    unsigned char*			buffer  = sMem;
    unsigned char*			opArray = buffer + 2*selSize;

    const int	u = threadIdx.x + blockIdx.x*blockDim.x;
    const int	v = threadIdx.y + blockIdx.y*blockDim.y;
    if (u >= width || v >= height)
        return;

    buffer[threadIdx.x] = src[v*width + u];
    if (u + selSize < width)
        buffer[threadIdx.x + selSize] = src[v*width + u + selSize];
    __syncthreads();

    // scan<opType>(buffer + selSize - 1, opArray + selSize - 1, selSize, threadIdx.x);
    // reversedScan<opType>(buffer, opArray, selSize, threadIdx.x);
    twoWayScan<opType>(buffer, opArray, selSize, threadIdx.x);

    if (u + selSize/2 < width - selSize/2)
        dst[v*width + u + selSize/2]
	    = elementOp<opType>(opArray[threadIdx.x],
				opArray[threadIdx.x + selSize - 1]);
}

template <MorphOpType opType> __global__ void
vhgw_vert(unsigned char* __restrict__ dst, unsigned char* __restrict__ src,
	  int width, int height, int selSize)
{
    extern __shared__ unsigned char	sMem[];
    unsigned char*			buffer  = sMem;
    unsigned char*			opArray = buffer + 2*selSize;

    const int	u = threadIdx.x + blockIdx.x*blockDim.x;
    const int	v = threadIdx.y + blockIdx.y*blockDim.y;
    if (v >= height || u >= width)
        return;

    buffer[threadIdx.y] = src[v*width + u];
    if (v + selSize < height)
        buffer[threadIdx.y + selSize] = src[(v + selSize)*width + u];
    __syncthreads();

    // scan<opType>(buffer + selSize - 1, opArray + selSize - 1, selSize, threadIdx.y);
    // reversedScan<opType>(buffer, opArray, selSize, threadIdx.y);
    twoWayScan<opType>(buffer, opArray, selSize, threadIdx.y);

    if (v + selSize/2 < height - selSize/2)
        dst[(v + selSize/2)*width + u]
	    = elementOp<opType>(opArray[threadIdx.y],
				opArray[threadIdx.y + selSize - 1]);

    if (v < selSize/2 || v >= height - selSize/2)
        dst[v*width + u] = borderValue<opType>();
}


template <MorphOpType opType> void
morphology(unsigned char* img_d, int width, int height, int hsize, int vsize)
{
    const size_t	imageMemSize = width * height * sizeof(unsigned char);
    unsigned char*	tmp_d;
    cudaMalloc((void **) &tmp_d, imageMemSize);

    dim3	blockSize(hsize, 1);
    dim3	gridSize(roundUp(width, blockSize.x),
			 roundUp(height, blockSize.y));
    size_t	sMemSize = 4 * hsize * sizeof(unsigned char);
    vhgw_horiz<opType><<<gridSize, blockSize, sMemSize>>>(tmp_d, img_d,
							  width, height,
							  hsize);

    // cudaMemset(img_d, 0, imageMemSize);

    blockSize.x = 1;
    blockSize.y = vsize;
    gridSize.x  = roundUp(width, blockSize.x);
    gridSize.y  = roundUp(height, blockSize.y);
    sMemSize	= 4 * vsize * sizeof(unsigned char);
    vhgw_vert<opType><<<gridSize, blockSize, sMemSize>>>(img_d, tmp_d,
							 width, height,
							 vsize);

    cudaFree(tmp_d);
}


extern "C" void
erode(unsigned char* img_d, int width, int height, int hsize, int vsize)
{
    morphology<MorphOpType::ERODE>(img_d, width, height, hsize, vsize);
}

extern "C" void
dilate(unsigned char* img_d, int width, int height, int hsize, int vsize)
{
    morphology<MorphOpType::DILATE>(img_d, width, height, hsize, vsize);
}
