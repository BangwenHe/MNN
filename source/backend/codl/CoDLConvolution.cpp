#include "CoDLConvolution.hpp"
#include "CoDLCPUGPUMemPack.hpp"
#include "CoDLUtils.hpp"
#include "MNN/AutoTime.hpp"

namespace MNN {

CoDLConvolution::CoDLConvolution(Backend *b, const Op *op,
                                 const std::vector<Tensor *> &inputs,
                                 const std::vector<Tensor *> &outputs)
    : Execution(b) {
  mBackend = (CoDLBackend *)b;
  mOp = op;

  mCPUConvolution.reset(
      mBackend->getCPUBackend()->onCreate(inputs, outputs, op));
  mOCLConvolution.reset(
      mBackend->getOpenCLBackend()->onCreate(inputs, outputs, op));
  mCPUInputs.clear();
  mCPUOutputs.clear();
  mOCLInputs.clear();
  mOCLOutputs.clear();
}

ErrorCode CoDLConvolution::onResize(const std::vector<Tensor *> &inputs,
                                    const std::vector<Tensor *> &outputs) {
  mCPUInputs.clear();
  mCPUOutputs.clear();
  mOCLInputs.clear();
  mOCLOutputs.clear();

  float splitRatio = 0.8;
  CoDLNodePartitionParam param = {CoDLNodePartitionParam::PART_DIM_OC,
                                  splitRatio};
  int numTensor = inputs.size();

  for (int i = 0; i < numTensor; i++) {
    auto input = inputs[i];
    auto output = outputs[i];
    std::vector<int> partCPUInputShape, partCPUOutputShape;
    std::vector<int> partOCLInputShape, partOCLOutputShape;
    CoDLUtils::partConv2dShape(
        input->shape(), output->shape(),
        TensorUtils::getDescribe(input)->dimensionFormat,
        TensorUtils::getDescribe(output)->dimensionFormat, partCPUInputShape,
        partCPUOutputShape, partOCLInputShape, partOCLOutputShape, param);
    CoDLCPUGPUMemPack::resizeMempack(input, partCPUInputShape, partOCLInputShape);
    CoDLCPUGPUMemPack::resizeMempack(output, partCPUOutputShape, partOCLOutputShape);

#ifdef CODL_DEBUG
    MNN_PRINT("tensor %d: \n", i);
    MNN_PRINT("input:  ");
    CoDLUtils::printCoDLTensorShape(input);
    MNN_PRINT("output: ");
    CoDLUtils::printCoDLTensorShape(output);
#endif
  }

  for (auto input : inputs) {
    auto *mem = (CoDLCPUGPUMemPack *)(input->buffer().device);
    mCPUInputs.push_back(mem->getCPUTensor());
    mOCLInputs.push_back(mem->getOCLTensor());
  }

  for (auto output : outputs) {
    auto *mem = (CoDLCPUGPUMemPack *)(output->buffer().device);
    mCPUOutputs.push_back(mem->getCPUTensor());
    mOCLOutputs.push_back(mem->getOCLTensor());
  }

  auto ret1 = mCPUConvolution->onResize(mCPUInputs, mCPUOutputs);
  auto ret2 = mOCLConvolution->onResize(mOCLInputs, mOCLOutputs);
  return ret1 == NO_ERROR ? ret2 : ret1;
}

ErrorCode CoDLConvolution::onExecute(const std::vector<Tensor *> &inputs,
                                     const std::vector<Tensor *> &outputs) {
  ErrorCode ret1 = NO_ERROR, ret2 = NO_ERROR;
  auto future2 = std::async(std::launch::async, [&]() {
    ret2 = mOCLConvolution->onExecute(mOCLInputs, mOCLOutputs);
    // 因为 OpenCL 的执行是异步的, 所以这里需要等待 OpenCL 执行完毕
    mBackend->getOpenCLBackend()->getOpenCLRuntime()->commandQueue().finish();
    return 0;
  });

  ret1 = mCPUConvolution->onExecute(mCPUInputs, mCPUOutputs);
  future2.get();
  return ret1 == NO_ERROR ? ret2 : ret1;
}

CoDLCreatorRegister<TypedCreator<CoDLConvolution>>
    __convolution_buffer_op(OpType_Convolution, GpuMemObject::BUFFER);

CoDLCreatorRegister<TypedCreator<CoDLConvolution>> 
    __convolution_image_op(OpType_Convolution, GpuMemObject::IMAGE);

} // namespace MNN