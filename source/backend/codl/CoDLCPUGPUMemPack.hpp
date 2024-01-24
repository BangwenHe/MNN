#ifndef CODL_MEMOBJ_HPP
#define CODL_MEMOBJ_HPP

#include "core/Backend.hpp"
#include "MNN/Tensor.hpp"

namespace MNN {

class CoDLCPUGPUMemPack {
public:
  CoDLCPUGPUMemPack(Backend *cpuBackend, Backend *oclBackend, const Tensor *srcTensor, Backend::StorageType storageType);

  Tensor* getCPUTensor() const {
    return mCPUTensor.get();
  }

  Tensor* getOCLTensor() const {
    return mOCLTensor.get();
  }

private:
  std::shared_ptr<Tensor> mCPUTensor;
  std::shared_ptr<Tensor> mOCLTensor;
};

}

#endif
