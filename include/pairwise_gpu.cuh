#pragma once

#include "densecrf_base.h"
#include "permutohedral_gpu.cuh"

namespace dcrf_cuda {

// Weight applying kernels for potts potential.
template<int M, int F>
__global__ static void pottsWeight(float* out, const float* in, const int n, const float pw) {
    const int ni = threadIdx.x + blockIdx.x * blockDim.x;
    const int vi = blockIdx.y;
    if (ni >= n) return;
    out[ni * M + vi] += pw * in[ni * M + vi];
}

// Initializing kernels for potts potential.
template<class T, int M, int F>
__global__ static void assembleImageFeature(int w, int h, const T* features, float posdev, float featuredev, float* out) {
    const int wi = threadIdx.x + blockIdx.x * blockDim.x;
    const int hi = threadIdx.y + blockIdx.y * blockDim.y;
    if (wi >= w || hi >= h) return;

    const int idx = hi * w + wi;
    out[idx * F + 0] = (float) wi / posdev;
    out[idx * F + 1] = (float) hi / posdev;
    #pragma unroll
    for (int i = 2; i < F; ++i) {
        out[idx * F + i] = (float) features[idx * (F - 2) + (i - 2)] / featuredev;
    }
}

template<class PT, class FT, int M, int F>
__global__ static void assembleUnorganizedFeature(int N, int pdim, const PT* positions, const FT* features, float posdev, float featuredev, float* out) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) return;

#pragma unroll
    for (int i = 0; i < pdim; ++i) {
        out[idx * F + i] = (float) positions[idx * pdim + i] / posdev;
    }

#pragma unroll
    for (int i = pdim; i < F; ++i) {
        out[idx * F + i] = (float) features[idx * (F - pdim) + (i - pdim)] / featuredev;
    }
}

// Normalizes the data used in kernels.
template<int M, int F>
__global__ static void normalize(float* out, const float* in, const float* norm, const int n, const float pw) {
    const int ni = threadIdx.x + blockIdx.x * blockDim.x;
    const int vi = blockIdx.y;
    if (ni >= n) return;
    out[ni * M + vi] += norm[ni] * in[ni * M + vi];
}

__global__ void fillOnes(float* norm, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    norm[i] = 1.0f;
}

// __global__ void setNormVal(float* norm, int N, NormalizationType ntype) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= N) return;
//     if (ntype == NORMALIZE_SYMMETRIC) {
//         norm[i] = 1.0f / sqrtf(norm[i] + 1e-20f);
//     } else if (ntype == NORMALIZE_AFTER || ntype == NORMALIZE_BEFORE) {
//         norm[i] = 1.0f / (norm[i] + 1e-20f);
//     }
// }

__global__ void inverseSqrtKernel(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        data[i] = 1.0f / sqrtf(data[i] + 1e-20f);
    }
}

template<int M, int F>
class PottsPotentialGPU: public PairwisePotential {
protected:
    PermutohedralLatticeGPU<float, F, M + 1>* lattice_;
    float w_;
    NormalizationType ntype_;
    float *norm_;
public:
    PottsPotentialGPU(const float* features, int N, float w, NormalizationType ntype = NORMALIZE_SYMMETRIC) : PairwisePotential(N), w_(w), ntype_(ntype) {
        lattice_ = new PermutohedralLatticeGPU<float, F, M + 1>(N);
        lattice_->prepare(features);

        PermutohedralLatticeGPU<float, F, 2>* latticeNorm = new PermutohedralLatticeGPU<float, F, 2>(N);
        latticeNorm->prepare(features);

        dim3 blocks((N_ - 1) / BLOCK_SIZE + 1, M, 1);
        dim3 blockSize(BLOCK_SIZE, 1, 1);

        float* ones_input;
        cudaMalloc(&ones_input, N * sizeof(float));
        fillOnes<<<(N + 255) / 256, 256>>>(ones_input, N);

        cudaMalloc(&norm_, N * sizeof(float));
        cudaMemset(norm_, 0, N * sizeof(float));

        latticeNorm->filter(norm_, ones_input);
        inverseSqrtKernel<<<(N + 255) / 256, 256>>>(norm_, N);
    }
    ~PottsPotentialGPU(){
        delete lattice_;
        cudaFree(norm_);
    }
    PottsPotentialGPU( const PottsPotentialGPU&o ) = delete;

    // Factory functions:
    // Build image-based potential: if features is NULL then applying gaussian filter only.
    template<class T = float>
    static PottsPotentialGPU<M, F>* FromImage(int w, int h, float weight, float posdev, const T* features = nullptr, float featuredev = 0.0, NormalizationType ntype = NORMALIZE_SYMMETRIC) {
        // First assemble features:
        float* allFeatures = nullptr;
        cudaMalloc((void**)&allFeatures, sizeof(float) * F * w * h);
        dim3 blocks((w - 1) / 16 + 1, (h - 1) / 16 + 1, 1);
        dim3 blockSize(16, 16, 1);
        assembleImageFeature<T, M, F> <<<blocks, blockSize>>> (w, h, features, posdev, featuredev, allFeatures);
        cudaErrorCheck();
        auto* pt = new PottsPotentialGPU<M, F>(allFeatures, w * h, weight, ntype);
        cudaFree(allFeatures);
        return pt;
    }
    // Build linear potential:
    template<class PT = float, class FT = float>
    static PottsPotentialGPU<M, F>* FromUnorganizedData(int N, float weight, const PT* positions, float posdev, int posdim,
            const FT* features = nullptr, float featuredev = 0.0) {
        float* allFeatures = nullptr;
        cudaMalloc((void**)&allFeatures, sizeof(float) * F * N);
        dim3 blocks((N - 1) / BLOCK_SIZE + 1, 1, 1);
        dim3 blockSize(BLOCK_SIZE, 1, 1);
        assembleUnorganizedFeature<PT, FT, M, F> <<<blocks, blockSize>>> (N, posdim, positions, features, posdev, featuredev, allFeatures);
        cudaErrorCheck();
        auto* pt = new PottsPotentialGPU<M, F>(allFeatures, N, weight);
        cudaFree(allFeatures);
        return pt;
    }

    // tmp should be larger to store normalization values. (N*(M+1))
    // All pointers are device pointers
    void apply(float* out_values, const float* in_values, float* tmp) const {
        // N_ = W * H;
        // blocks = ((W * H - 1) / 256 + 1, M, 1);
        // blocksize is (256, 1, 1)
        dim3 blocks((N_ - 1) / BLOCK_SIZE + 1, M);
        dim3 blockSize(BLOCK_SIZE);
        normalize<M, F> <<<blocks, blockSize>>> (tmp, in_values, norm_, N_, w_);
        float *out_tmp;
        cudaMalloc(&out_tmp, N_ * M * sizeof(float));
        lattice_->filter(out_tmp, tmp, norm_);
        pottsWeight<M, F> <<<blocks, blockSize>>> (out_values, out_tmp, N_, w_);
    }
};

}