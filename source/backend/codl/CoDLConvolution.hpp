#ifndef CODL_CONVOLUTION_HPP
#define CODL_CONVOLUTION_HPP

#include "MNN/ErrorCode.hpp"
#include "backend/codl/CoDLBackend.hpp"
#include "core/Backend.hpp"
#include "core/Execution.hpp"
#include "backend/cpu/CPUBinary.hpp"

namespace MNN {

class CoDLConvolution : public Execution {
public:
    CoDLConvolution(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);

    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ~CoDLConvolution() = default;
    ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    std::vector<float> onProfiling(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    void postprocess(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);

    const Op *mOp;
    CoDLBackend *mBackend;
    std::shared_ptr<Execution> mCPUConvolution;
    std::shared_ptr<Execution> mOCLConvolution;
    std::vector<Tensor *> mCPUInputs, mCPUOutputs;
    std::vector<Tensor *> mOCLInputs, mOCLOutputs;

    CoDLNodePartitionParam::PartDim mPartDim;
    MNNBinaryExecute mProc;
};

}

#endif