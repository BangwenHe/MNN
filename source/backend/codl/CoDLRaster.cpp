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
    for (auto input : inputs) {
      CoDLCPUGPUMemPack *mem = (CoDLCPUGPUMemPack *) (input->buffer().device);
      mCPUInputs.push_back(mem->getCPUTensor());
      mOCLInputs.push_back(mem->getOCLTensor());
    }

    for (auto output : outputs) {
      CoDLCPUGPUMemPack *mem = (CoDLCPUGPUMemPack *) (output->buffer().device);
      mCPUOutputs.push_back(mem->getCPUTensor());
      mOCLOutputs.push_back(mem->getOCLTensor());
    }

    auto ret1 = mCPURaster->onResize(mCPUInputs, mCPUOutputs);
    auto ret2 = mOCLRaster->onResize(mOCLInputs, mOCLOutputs);

    return ret1 == NO_ERROR ? ret2 : ret1;
}


ErrorCode CoDLRaster::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    ErrorCode ret1 = NO_ERROR, ret2 = NO_ERROR;
    auto future2 = std::async(std::launch::async, [&]() {
      ret2 = mOCLRaster->onExecute(mOCLInputs, mOCLOutputs);
      mBackend->getOpenCLBackend()->getOpenCLRuntime()->commandQueue().finish();
      return 0;
    });

    ret1 = mCPURaster->onExecute(mCPUInputs, mCPUOutputs);
    future2.get();
    return ret1 == NO_ERROR ? ret2 : ret1;
}

CoDLCreatorRegister<TypedCreator<CoDLRaster>> __raster_buffer_op(OpType_Raster, GpuMemObject::BUFFER);

}
