#include "CoDLCPUGPUMemPack.hpp"

namespace MNN {

CoDLCPUGPUMemPack::CoDLCPUGPUMemPack(Backend *cpuBackend, Backend *oclBackend, const Tensor *srcTensor, Backend::StorageType storageType) {
  mCPUTensor.reset(Tensor::create(srcTensor->shape(), srcTensor->getType(), nullptr, srcTensor->getDimensionType()));
  mOCLTensor.reset(Tensor::create(srcTensor->shape(), srcTensor->getType(), nullptr, srcTensor->getDimensionType()));
  cpuBackend->onAcquireBuffer(mCPUTensor.get(), storageType);
  oclBackend->onAcquireBuffer(mOCLTensor.get(), storageType);
}

}