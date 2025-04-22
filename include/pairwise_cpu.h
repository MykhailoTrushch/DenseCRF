#pragma once

#include "densecrf_base.h"
#include "permutohedral_cpu.h"

namespace dcrf_cuda
{

template <int M, int F>
class PottsPotentialCPU : public PairwisePotential
{
protected:
    PermutohedralLatticeCPU lattice_;
    float w_;
    float *norm_;
    NormalizationType ntype_;

public:
    PottsPotentialCPU(const float *features, int N, float w, NormalizationType ntype = NORMALIZE_SYMMETRIC) : PairwisePotential(N), w_(w), ntype_(ntype)
    {
        lattice_.init(features, F, N);
        norm_ = new float[N];
        std::fill(norm_, norm_ + N, 1.0f);
        lattice_.compute(norm_, norm_, 1);
        if (ntype == NO_NORMALIZATION)
        {
            float mean_norm = 0;
            for (int i = 0; i < N; i++)
                mean_norm += norm_[i];
            mean_norm = N / mean_norm;
            for (int i = 0; i < N; i++)
                norm_[i] = mean_norm;
        }
        else if (ntype == NORMALIZE_SYMMETRIC)
        {
            for (int i = 0; i < N_; ++i)
            {
                norm_[i] = 1.0f / sqrt(norm_[i] + 1e-20f);
            }
        }
        else
        {
            for (int i = 0; i < N_; ++i)
            {
                norm_[i] = 1.0f / sqrt(norm_[i] + 1e-20f);
            }
        }
    }

    ~PottsPotentialCPU()
    {
        delete[] norm_;
    }

    PottsPotentialCPU(const PottsPotentialCPU &o) = delete;

    //// Factory functions:
    // Build image-based potential: if features is NULL then applying gaussian filter only.
    template <class T = float>
    static PottsPotentialCPU<M, F> *FromImage(int w, int h, float weight, float posdev, const T *features = nullptr, float featuredev = 0.0)
    {
        // First assemble features:
        auto *allFeatures = new float[F * w * h];
        for (int hi = 0; hi < h; ++hi)
        {
            for (int wi = 0; wi < w; ++wi)
            {
                const int idx = hi * w + wi;
                allFeatures[idx * F + 0] = (float)wi / posdev;
                allFeatures[idx * F + 1] = (float)hi / posdev;
                for (int i = 2; i < F; ++i)
                {
                    allFeatures[idx * F + i] = (float)features[idx * (F - 2) + (i - 2)] / featuredev;
                }
            }
        }
        auto *pt = new PottsPotentialCPU<M, F>(allFeatures, w * h, weight);
        delete[] allFeatures;
        return pt;
    }

    void apply(float *out_values, const float *in_values, float *tmp) const
    {
        if (ntype_ == NORMALIZE_SYMMETRIC || ntype_ == NORMALIZE_BEFORE)
        {
            for (int i = 0, k = 0; i < N_; i++)
                for (int j = 0; j < M; j++, k++)
                    tmp[k] = norm_[i] * in_values[k];
        }
        else
        {
            std::memcpy(tmp, in_values, N_ * M);
        }
        lattice_.compute(tmp, tmp, M);
        for (int i = 0, k = 0; i < N_; i++)
        {
            for (int j = 0; j < M; j++, k++)
            {
                // multiplying by w_ is applying the Potts label compatibility
                if (ntype_ == NORMALIZE_SYMMETRIC || ntype_ == NORMALIZE_AFTER)
                {
                    out_values[k] += w_ * norm_[i] * tmp[k];
                }
                else
                {
                    out_values[k] += w_ * tmp[k];
                }
            }
        }
    }
};

}