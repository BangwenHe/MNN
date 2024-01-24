#include "CoDLRaster.hpp"
#include "CoDLCPUGPUMemPack.hpp"

namespace MNN {

CoDLRaster::CoDLRaster(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Execution(b) {
    mBackend = (CoDLBackend *) b;
    mOP = op;

    mCPURaster.reset(mBackend->getCPUBackend()->onCreate(inputs, outputs, op));
    mOCLRaster.reset(mBackend->getOpenCLBackend()->onCreate(inputs, outputs, op));
}

ErrorCode CoDLRaster::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
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

    auto ret1 = mCPURaster->onResize(cpuInputs, cpuOutputs);
    auto ret2 = mOCLRaster->onResize(oclInputs, oclOutputs);

    return ret1 == NO_ERROR ? ret2 : ret1;
}


ErrorCode CoDLRaster::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
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

    auto ret1 = mCPURaster->onExecute(cpuInputs, cpuOutputs);
    auto ret2 = mOCLRaster->onExecute(oclInputs, oclOutputs);

    return ret1 == NO_ERROR ? ret2 : ret1;
}

CoDLCreatorRegister<TypedCreator<CoDLRaster>> __raster_buffer_op(OpType_Raster, GpuMemObject::BUFFER);

}
