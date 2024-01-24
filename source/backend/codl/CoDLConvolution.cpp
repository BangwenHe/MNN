#include "CoDLCPUGPUMemPack.hpp"
#include "CoDLConvolution.hpp"

namespace MNN {

CoDLConvolution::CoDLConvolution(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Execution(b) {
    mBackend = (CoDLBackend *) b;
    mOp = op;

    mCPUConvolution.reset(mBackend->getCPUBackend()->onCreate(inputs, outputs, op));
    mOCLConvolution.reset(mBackend->getOpenCLBackend()->onCreate(inputs, outputs, op));
}

ErrorCode CoDLConvolution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    std::vector<Tensor*> cpuInputs, cpuOutputs;
    std::vector<Tensor*> oclInputs, oclOutputs;
    for (auto input : inputs) {
      CoDLCPUGPUMemPack *mem = (CoDLCPUGPUMemPack *) input->buffer().device;
      cpuInputs.push_back(mem->getCPUTensor());
      oclInputs.push_back(mem->getOCLTensor());
    }

    for (auto output : outputs) {
      CoDLCPUGPUMemPack *mem = (CoDLCPUGPUMemPack *) output->buffer().device;
      cpuOutputs.push_back(mem->getCPUTensor());
      oclOutputs.push_back(mem->getOCLTensor());
    }

    auto ret1 = mCPUConvolution->onResize(cpuInputs, cpuOutputs);
    auto ret2 = mOCLConvolution->onResize(oclInputs, oclOutputs);
    return ret1 == NO_ERROR ? ret2 : ret1;
}

ErrorCode CoDLConvolution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    std::vector<Tensor*> cpuInputs, cpuOutputs;
    std::vector<Tensor*> oclInputs, oclOutputs;
    for (auto input : inputs) {
      CoDLCPUGPUMemPack *mem = (CoDLCPUGPUMemPack *) input->buffer().device;
      cpuInputs.push_back(mem->getCPUTensor());
      oclInputs.push_back(mem->getOCLTensor());
    }

    for (auto output : outputs) {
      CoDLCPUGPUMemPack *mem = (CoDLCPUGPUMemPack *) output->buffer().device;
      cpuOutputs.push_back(mem->getCPUTensor());
      oclOutputs.push_back(mem->getOCLTensor());
    }

    auto ret1 = mCPUConvolution->onExecute(cpuInputs, cpuOutputs);
    auto ret2 = mOCLConvolution->onExecute(oclInputs, oclOutputs);

    return ret1 == NO_ERROR ? ret2 : ret1;
}

CoDLCreatorRegister<TypedCreator<CoDLConvolution>> __convolution_buffer_op(OpType_Convolution, GpuMemObject::BUFFER);

}