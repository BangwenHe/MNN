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

  mPartDim = CoDLNodePartitionParam::PART_DIM_IC;
  mProc = CPUBinary::selectForFloat(BinaryOpOperation_ADD);
}

ErrorCode CoDLConvolution::onResize(const std::vector<Tensor *> &inputs,
                                    const std::vector<Tensor *> &outputs) {
  mCPUInputs.clear();
  mCPUOutputs.clear();
  mOCLInputs.clear();
  mOCLOutputs.clear();

  int n = inputs[0]->batch();
  int m = inputs[0]->channel();
  int k = outputs[0]->channel();
  auto param = mBackend->getPartitionParam(n, m, k);
  // CoDLNodePartitionParam param{CoDLNodePartitionParam::PART_DIM_IC, 0.6};
  int numTensor = inputs.size();
  mPartDim = param.mPartDim;

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

#ifdef MNN_CODL_DEBUG
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
#ifdef MNN_CODL_DEBUG
  MNN_PRINT("\n");
  AutoTime _t(__LINE__, __func__);
#endif
  ErrorCode ret1 = NO_ERROR, ret2 = NO_ERROR;
  auto future2 = std::async(std::launch::async, [&]() {
#ifdef MNN_CODL_DEBUG
    AutoTime _t(__LINE__, __func__);
#endif
    ret2 = mOCLConvolution->onExecute(mOCLInputs, mOCLOutputs);
    // 因为 OpenCL 的执行是异步的, 所以这里需要等待 OpenCL 执行完毕
    mBackend->getOpenCLBackend()->getOpenCLRuntime()->commandQueue().finish();
    return 0;
  });

#ifdef MNN_CODL_DEBUG
  {
    AutoTime _t(__LINE__, __func__);
#endif
    ret1 = mCPUConvolution->onExecute(mCPUInputs, mCPUOutputs);
#ifdef MNN_CODL_DEBUG
  }
  {
    AutoTime _t(__LINE__, __func__);
#endif
    future2.wait();
#ifdef MNN_CODL_DEBUG
  }
#endif

#ifdef MNN_CODL_DEBUG
  {
    AutoTime _t(__LINE__, __func__);
#endif

  if (mPartDim == CoDLNodePartitionParam::PART_DIM_IC) {
    if (mProc != nullptr) {
      int n = mCPUOutputs.size();
      for (int i = 0; i < n; i++) {
        // 1. 将 GPU 上的输出数据拷贝到 CPU 上
        auto *gpuTensor = mOCLOutputs[i];
        auto *mapPtr = gpuTensor->map(Tensor::MAP_TENSOR_WRITE, gpuTensor->getDimensionType());
        auto *cpuTensor = mCPUOutputs[i];
        auto *cpuPtr = cpuTensor->host<float>();
        auto *originPtr = outputs[i]->host<float>();

        // 2. 将 CPU 上的数据与 GPU 上的数据相加
        mProc(originPtr, cpuPtr, mapPtr, cpuTensor->elementSize(), 0);

        // 3. 将相加后的数据拷贝到 GPU 上
        std::memcpy(cpuPtr, originPtr, cpuTensor->size());
        std::memcpy(mapPtr, originPtr, cpuTensor->size());

        gpuTensor->unmap(Tensor::MAP_TENSOR_WRITE, gpuTensor->getDimensionType(), mapPtr);
      }
    } else {
      MNN_PRINT("%s:%d: Unsupported binary operation\n", __FILE__, __LINE__);
    }
  } else if (mPartDim == CoDLNodePartitionParam::PART_DIM_OC) {
    int n = mCPUOutputs.size();
    for (int i = 0; i < n; i++) {
      auto *gpuTensor = mOCLOutputs[i];
      auto *mapPtr = static_cast<float*>(gpuTensor->map(Tensor::MAP_TENSOR_WRITE, gpuTensor->getDimensionType()));
      auto *cpuTensor = mCPUOutputs[i];
      auto *cpuPtr = cpuTensor->host<float>();
      auto *originPtr = outputs[i]->host<float>();

      int batch = outputs[i]->batch();
      int k = outputs[i]->channel();
      int cpuK = mCPUOutputs[i]->channel();
      int gpuK = mOCLOutputs[i]->channel();
      for (int j = 0; j < batch; j++) {
        std::memcpy(originPtr + j * k, cpuPtr + j * cpuK, cpuK * sizeof(float));
        std::memcpy(originPtr + j * k + cpuK, mapPtr + j * gpuK, gpuK * sizeof(float));
      }

      gpuTensor->unmap(Tensor::MAP_TENSOR_WRITE, gpuTensor->getDimensionType(), mapPtr);
    }
  } else if (mPartDim == CoDLNodePartitionParam::PART_DIM_N) {
    int n = mCPUOutputs.size();
    for (int i = 0; i < n; i++) {
      auto *gpuTensor = mOCLOutputs[i];
      auto *mapPtr = gpuTensor->map(Tensor::MAP_TENSOR_WRITE, gpuTensor->getDimensionType());
      auto *cpuTensor = mCPUOutputs[i];
      auto *cpuPtr = cpuTensor->host<float>();
      auto *originPtr = outputs[i]->host<float>();

      std::memcpy(originPtr, cpuPtr, cpuTensor->size());
      std::memcpy(originPtr + cpuTensor->size(), mapPtr, gpuTensor->size());

      gpuTensor->unmap(Tensor::MAP_TENSOR_WRITE, gpuTensor->getDimensionType(), mapPtr);
    }
  } else {
    MNN_ERROR("Unsupported partition dimension\n");
  }

#ifdef MNN_CODL_DEBUG
  }
#endif

  return ret1 == NO_ERROR ? ret2 : ret1;
}

CoDLCreatorRegister<TypedCreator<CoDLConvolution>>
    __convolution_buffer_op(OpType_Convolution, GpuMemObject::BUFFER);

CoDLCreatorRegister<TypedCreator<CoDLConvolution>> 
    __convolution_image_op(OpType_Convolution, GpuMemObject::IMAGE);

} // namespace MNN
