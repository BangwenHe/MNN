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

  /**
   * @brief 修改 mempack 中 tensor 的 shape, 但是不修改 tensor 的数据, 也不申请和释放内存
   * 
   * @param tensor 待修改的 tensor
   * @param shape 形状
   */
  static void resizeMempack(Tensor *tensor, const std::vector<int> &cpuNewShape, const std::vector<int> &oclNewShape);

private:
  std::shared_ptr<Tensor> mCPUTensor;
  std::shared_ptr<Tensor> mOCLTensor;
};

}

#endif
