#pragma once

#include "densecrf_base.h"
#include "pairwise_gpu.cuh"
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>

namespace dcrf_cuda {

// GPU CUDA Implementation
template<int M>
class DenseCRFGPU : public DenseCRF {

protected:
    void expAndNormalize( float* out, const float* in, float scale = 1.0, float relax = 1.0 ) override;
    void buildMap() override;
    void stepInit() override;

public:

    // Create a dense CRF model of size N with M labels
    explicit DenseCRFGPU( int N ) : DenseCRF(N) {
        cudaMalloc((void**)&unary_, sizeof(float) * N * M);
        cudaMalloc((void**)&current_, sizeof(float) * N * M);
        cudaMalloc((void**)&next_, sizeof(float) * N * M);
        cudaMalloc((void**)&tmp_, sizeof(float) * N * M);
    }

    ~DenseCRFGPU() override {
        cudaFree(unary_);
        cudaFree(current_);
        cudaFree(next_);
        cudaFree(tmp_);
        if (map_) cudaFree(map_);
    }

    DenseCRFGPU( DenseCRFGPU & o ) = delete;

    // Set the unary potential for all variables and labels (memory order is [x0l0 x0l1 x0l2 .. x1l0 x1l1 ...])
    void setUnaryEnergy( const float * unaryGPU ) override {
        cudaMemcpy(unary_, unaryGPU, sizeof(float) * N_ * M, cudaMemcpyDeviceToDevice);
    }

    // Set the unary potential via label. Length of label array should equal to N.
    void setUnaryEnergyFromLabel(const short* labelGPU, float confidence = 0.5) override;
    void setUnaryEnergyFromLabel(const short* labelGPU, float* confidences) override;
};


template<int M>
__global__ static void expNormKernel(int N, float* out, const float* in, float scale, float relax)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N)
        return;
    const float* b = in + idx * M;
    // Find the max and subtract it so that the exp doesn't explode
    float mx = scale * b[0];
    #pragma unroll
    for (int j = 1; j < M; ++j) {
        if (mx < scale * b[j]) {
            mx = scale * b[j];
        }
    }
    float tt = 0.0;
    float V[M]{0};
    #pragma unroll
    for (int j = 0; j < M; ++j) {
        V[j] = __expf(scale * b[j] - mx);
        tt += V[j];
    }
    // Make it a probability
    #pragma unroll
    for (int j = 0; j < M; ++j) {
        V[j] /= tt;
    }
    float *a = out + idx * M;
    #pragma unroll
    for (int j = 0; j < M; ++j) {
        a[j] = (1 - relax) * a[j] + relax * V[j];
    }
}

template<int M>
__global__ static void expNormKernelOnline(int N, float* out, const float* in, float scale, float relax)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N)
        return;
    const float* b = in + idx * M;
    float mx = scale * b[0];
    float d = 1.0;
    float m_new = 0.0;
    for (int j = 1; j < M; ++j) {
        float curr = scale * b[j];
        if (mx < curr) {
            m_new = curr;
        } else {
            m_new = mx;
        }
        d = d * __expf(mx - m_new) + __expf(curr - m_new);
        mx = m_new;
    }
    float *a = out + idx * M;
    float d_inverse = 1.0f/d;
    for (int j = 0; j < M; ++j) {
        a[j] = (1 - relax) * a[j] + relax * __expf(scale * b[j] - mx)*d_inverse;
    }
}

struct __align__(8) MD
{
    float m;
    float d;
};

__device__ __forceinline__ MD reduce_md_op(MD a, MD b)
{
    bool a_bigger = (a.m > b.m);
    MD bigger_m = a_bigger ? a : b;
    MD smaller_m = a_bigger ? b : a;
    MD res;
    res.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
    res.m = bigger_m.m;
    return res;
}

template<int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__ void expNormKernelOnlineReduce(
    int N,
    float* __restrict__ out,
    const float* __restrict__ in,
    float scale,
    float relax)
{
    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x;
    // int THREADS_PER_VECTOR = 21;
    // int VECTORS_PER_BLOCK = THREADBLOCK_SIZE / THREADS_PER_VECTOR;
    // int vector_in_block = thread_id / THREADS_PER_VECTOR;
    // int thread_in_vector = thread_id % THREADS_PER_VECTOR;

    // if (thread_id + THREADBLOCK_SIZE * vector_id >= N) return;

    // extern __shared__ float shmem[]; // if needed, not in this case

    // reposition input and output
    int M = THREADBLOCK_SIZE; // you can also pass M separately if needed
    const float* x = in + vector_id * M;
    float* y = out + vector_id * M;

    typedef cub::BlockReduce<MD, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ MD md_total;

    MD md_partial;
    md_partial.m = -FLT_MAX;
    md_partial.d = 0.0f;

    if (thread_id < M){
        MD new_elem;
        new_elem.m = scale * x[thread_id];
        new_elem.d = 1.0f;
        md_partial = new_elem;
    }

    MD md = BlockReduce(temp_storage).Reduce(md_partial, reduce_md_op);

    if (thread_id == 0)
        md_total = md;
    __syncthreads();

    float d_total_inverse = __fdividef(1.0f, md_total.d);


    if(thread_id < M) {
        float exp_scaled = __expf(scale * x[thread_id] - md_total.m);
        // y[thread_id] = (1 - relax) * y[thread_id] + relax * exp_scaled * d_total_inverse;
        y[thread_id] = exp_scaled * d_total_inverse;
    }
}

template<int M>
void DenseCRFGPU<M>::expAndNormalize( float* out, const float* in, float scale /* = 1.0 */, float relax /* = 1.0 */ ) {
    dim3 blocks((N_ - 1) / BLOCK_SIZE + 1, 1, 1);
    dim3 blockSize(BLOCK_SIZE, 1, 1);
    expNormKernel<M> <<<blocks, blockSize>>> (N_, out, in, scale, relax);
    cudaErrorCheck();
}

// template<int M>
// void DenseCRFGPU<M>::expAndNormalize( float* out, const float* in, float scale /* = 1.0 */, float relax /* = 1.0 */ ) {
//     dim3 blocks((N_ - 1) /BLOCK_SIZE + 1, 1, 1);
//     dim3 blockSize(BLOCK_SIZE, 1, 1);
//     expNormKernelCudnn<M> <<<blocks, BLOCK_SIZE>>> (N_, out, in);
//     cudnnHandle_t cudnnHandle;
//     cudnnCreate(&cudnnHandle);


//     cudaErrorCheck();
// }

// template<int M>
// void DenseCRFGPU<M>::expAndNormalize( float* out, const float* in, float scale /* = 1.0 */, float relax /* = 1.0 */ ) {
    // dim3 blocks((N_ - 1) /BLOCK_SIZE + 1, 1, 1);
    // dim3 blockSize(BLOCK_SIZE, 1, 1);
//     expNormKernelOnline<M> <<<blocks, BLOCK_SIZE>>> (N_, out, in, scale, relax);
//     cudaErrorCheck();
// }


// template<int M>
// void DenseCRFGPU<M>::expAndNormalize( float* out, const float* in, float scale /* = 1.0 */, float relax /* = 1.0 */ ) {
//     dim3 blocks(N_, 1, 1);
//     dim3 blockSize(M, 1, 1);
//     expNormKernelOnlineReduce<M> <<<blocks, M>>> (N_, out, in, scale, relax);
//     cudaErrorCheck();
// }

template<int M>
__global__ static void unaryFromLabel(const short* inLabel, float* outUnary, int N,
                                      float u_energy, float* n_energies, float* p_energies)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N)
        return;
    short label = inLabel[idx];
    // Unknown.
    if (label == -1) {
        for (int m = 0; m < M; ++m) {
            outUnary[idx * M + m] = u_energy;
        }
    } else {
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            outUnary[idx * M + m] = n_energies[label];
        }
        outUnary[idx * M + label] = p_energies[label];
    }
}

template<int M>
void DenseCRFGPU<M>::setUnaryEnergyFromLabel(const short* labelGPU, float confidence /* = 0.5 */) {
    float confidences[M];
    std::fill(confidences, confidences + M, confidence);
    setUnaryEnergyFromLabel(labelGPU, confidences);
}

template<int M>
void DenseCRFGPU<M>::setUnaryEnergyFromLabel(const short* labelGPU, float* confidences) {
    float u_energy = -log( 1.0f / M );
    float np_energies[2 * M];
    for (int i = 0; i < 2 * M; ++i) {
        if (i < M) { np_energies[i] = -log( (1.0f - confidences[i]) / (M-1) ); }
        else { np_energies[i] = -log( confidences[i - M] ); }
    }
    float* np_energies_device = nullptr;
    cudaMalloc((void**)&np_energies_device, sizeof(float) * 2 * M);
    cudaMemcpy(np_energies_device, np_energies, sizeof(float) * 2 * M, cudaMemcpyHostToDevice);
    dim3 blocks((N_ - 1) / BLOCK_SIZE + 1, 1, 1);
    dim3 blockSize(BLOCK_SIZE, 1, 1);
    unaryFromLabel<M> <<<blocks, blockSize>>>(labelGPU, unary_, N_, u_energy, np_energies_device, np_energies_device + M);
    cudaErrorCheck();
    cudaFree(np_energies_device);
}


template<int M>
__global__ void computeMAP(int N, const float* in_prob, short* out_map)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N)
        return;

    const float* p = in_prob + idx * M;
    float mx = p[0];
    short imx = 0;
     #pragma unroll
    for (short m = 1; m < M; ++m) {
        if (mx < p[m]) {
            mx = p[m]; imx = m;
        }
    }
    out_map[idx] = imx;
}

template<int M>
void DenseCRFGPU<M>::buildMap() {
    // Compute the maximum probability as MAP.
    if (!map_) cudaMalloc((void**)&map_, sizeof(short) * N_);
    dim3 blocks((N_ - 1) / BLOCK_SIZE + 1, 1, 1);
    dim3 blockSize(BLOCK_SIZE, 1, 1);
    computeMAP<M> <<<blocks, blockSize>>> (N_, current_, map_);
    cudaErrorCheck();
}


template<int M>
__global__ void invertKernel(int N, const float* in_unary, float* out_next)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N)
        return;
    const float* p = in_unary + idx * M;
    float* n = out_next + idx * M;
    for (int m = 0; m < M; ++m) {
        n[m] = -p[m];
    }
}

template <int M>
void DenseCRFGPU<M>::stepInit() {
    dim3 blocks((N_ - 1) / BLOCK_SIZE + 1, 1, 1);
    dim3 blockSize(BLOCK_SIZE, 1, 1);
    invertKernel<M> <<<blocks, blockSize>>> (N_, unary_, next_);
    cudaErrorCheck();
}

}   // end namespace
