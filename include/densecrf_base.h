#pragma once

#include <vector>

namespace dcrf_cuda {

enum Device {
    CPU, GPU
};

// The filter in the dense CRF can be normalized in a few different ways
enum NormalizationType {
	NO_NORMALIZATION,    // No normalization whatsoever (will lead to a substantial approximation error)
	NORMALIZE_BEFORE,    // Normalize before filtering (Not used, just there for completeness)
	NORMALIZE_AFTER,     // Normalize after filtering (original normalization in NIPS 11 work)
	NORMALIZE_SYMMETRIC, // Normalize before and after (ICML 2013, low approximation error and preserves the symmetry of CRF)
};
enum KernelType {
	CONST_KERNEL,   // Constant kernel, no parameters
	DIAG_KERNEL,    // Diagonal kernel (scaling features)
	FULL_KERNEL,    // Full kernel matrix (arbitrary squared matrix)
};

// Define various pair wise weight (compatibility) models.
class PairwisePotential {
protected:
    int N_;
public:
    PairwisePotential(int N) : N_(N) {}
    virtual ~PairwisePotential() = default;
    virtual void apply( float * out_values, const float * in_values, float * tmp ) const = 0;
};


class DenseCRF {
protected:
    // Number of variables and labels
    int N_;
    // Pre-allocated host/device memory
    float *unary_, *current_, *next_, *tmp_;
    short *map_;

    // Store all pairwise potentials
    std::vector<PairwisePotential*> pairwise_;

    // Auxillary functions
    virtual void expAndNormalize( float* out, const float* in, float scale = 1.0, float relax = 1.0 ) = 0;
    virtual void buildMap() = 0;
    virtual void stepInit() = 0;

public:

    DenseCRF(int N) : N_(N), unary_(nullptr), current_(nullptr), next_(nullptr), tmp_(nullptr), map_(nullptr) {}
    virtual ~DenseCRF() {
        for (auto* pPairwise : pairwise_) {
            delete pPairwise;
        }
    }

    // Add your own favorite pairwise potential (ownwership will be transfered to this class)
    void addPairwiseEnergy( PairwisePotential* potential ) { pairwise_.push_back(potential); }

    // Set the unary potential for all variables and labels (memory order is [x0l0 x0l1 x0l2 .. x1l0 x1l1 ...])
    virtual void setUnaryEnergy( const float * unary ) = 0;

    // Set the unary potential via label. Length of label array should equal to N.
    virtual void setUnaryEnergyFromLabel(const short* label, float* confidences) = 0;
    virtual void setUnaryEnergyFromLabel(const short* label, float confidence = 0.5) = 0;

    // Run inference and return the probabilities
    // All returned values are managed by class
    virtual void inference( int n_iterations, bool with_map = false, float relax = 1.0 ) {
        // Initializes current result with softmax over unaries
        expAndNormalize( current_, unary_, -1 );
        for (int it = 0; it < n_iterations; ++it) {
            stepInference(relax);
        }
        if (with_map) {
            buildMap();
        }
    }
    short* getMap() const { return map_; }
    float* getProbability() const { return current_; }

    virtual void stepInference( float relax = 1.0 ) {
        // Set next_ to the negative unary
        stepInit();
        // Add up all pairwise potentials
        for(unsigned int i = 0; i < pairwise_.size(); i++) {
            pairwise_[i]->apply(next_, current_, tmp_);
        }
        // Exponentiate and normalize
        expAndNormalize( current_, next_, 1.0, relax );
    }
};

}