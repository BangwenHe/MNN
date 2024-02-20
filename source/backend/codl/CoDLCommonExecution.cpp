#include "CoDLCommonExecution.hpp"
#include "CoDLCPUGPUMemPack.hpp"


namespace MNN {

CoDLCPUOnlyCommonExecution::CoDLCPUOnlyCommonExecution(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs)
    : Execution(b), mOp(op), mBackend(static_cast<CoDLBackend *>(b)) {
  auto *execution = mBackend->getCPUBackend()->onCreate(inputs, outputs, op);
  if (execution == nullptr) {
    execution = mBackend->getBackupCPUBackend()->onCreate(inputs, outputs, op);
  }
  mCPUExecution.reset(execution);
}

ErrorCode CoDLCPUOnlyCommonExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
  mCPUInputs.clear();
  mCPUOutputs.clear();
  mOCLInputs.clear();
  mOCLOutputs.clear();

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

  return mCPUExecution->onResize(mCPUInputs, mCPUOutputs);
}

ErrorCode CoDLCPUOnlyCommonExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
  auto ret = mCPUExecution->onExecute(mCPUInputs, mCPUOutputs);
  if (ret != NO_ERROR) {
    return ret;
  }

  for (auto *output : outputs) {
    auto *mem = (CoDLCPUGPUMemPack *)(output->buffer().device);
    mem->getOCLTensor()->copyFromHostTensor(mem->getCPUTensor());
  }

  return NO_ERROR;
}


CoDLCreatorRegister<TypedCreator<CoDLCPUOnlyCommonExecution>> __while_buffer_op(OpType_While, GpuMemObject::BUFFER);
CoDLCreatorRegister<TypedCreator<CoDLCPUOnlyCommonExecution>> __softmax_buffer_op(OpType_Softmax, GpuMemObject::BUFFER);
CoDLCreatorRegister<TypedCreator<CoDLCPUOnlyCommonExecution>> __layernorm_buffer_op(OpType_LayerNorm, GpuMemObject::BUFFER);
CoDLCreatorRegister<TypedCreator<CoDLCPUOnlyCommonExecution>> __binaryop_buffer_op(OpType_BinaryOp, GpuMemObject::BUFFER);
CoDLCreatorRegister<TypedCreator<CoDLCPUOnlyCommonExecution>> __unaryop_buffer_op(OpType_UnaryOp, GpuMemObject::BUFFER);

CoDLCreatorRegister<TypedCreator<CoDLCPUOnlyCommonExecution>> __while_image_op(OpType_While, GpuMemObject::IMAGE);
CoDLCreatorRegister<TypedCreator<CoDLCPUOnlyCommonExecution>> __softmax_image_op(OpType_Softmax, GpuMemObject::IMAGE);
CoDLCreatorRegister<TypedCreator<CoDLCPUOnlyCommonExecution>> __layernorm_image_op(OpType_LayerNorm, GpuMemObject::IMAGE);
CoDLCreatorRegister<TypedCreator<CoDLCPUOnlyCommonExecution>> __binaryop_image_op(OpType_BinaryOp, GpuMemObject::IMAGE);
CoDLCreatorRegister<TypedCreator<CoDLCPUOnlyCommonExecution>> __unaryop_image_op(OpType_UnaryOp, GpuMemObject::IMAGE);
}