#include "CoDLCPUGPUMemPack.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {

CoDLCPUGPUMemPack::CoDLCPUGPUMemPack(Backend *cpuBackend, Backend *oclBackend, const Tensor *srcTensor, Backend::StorageType storageType) {
  mCPUTensor.reset(Tensor::create(srcTensor->shape(), srcTensor->getType(), nullptr, srcTensor->getDimensionType()));
  mOCLTensor.reset(Tensor::create(srcTensor->shape(), srcTensor->getType(), nullptr, srcTensor->getDimensionType()));
  cpuBackend->onAcquireBuffer(mCPUTensor.get(), storageType);
  oclBackend->onAcquireBuffer(mOCLTensor.get(), storageType);
}

void CoDLCPUGPUMemPack::resizeMempack(Tensor *tensor, const std::vector<int> &cpuNewShape, const std::vector<int> &oclNewShape) {
  auto *mempack = (CoDLCPUGPUMemPack *) (tensor->buffer().device);
  auto *cpuTensor = mempack->getCPUTensor();
  auto *oclTensor = mempack->getOCLTensor();
  auto format = TensorUtils::getDescribe(tensor)->dimensionFormat;

  // TODO: 释放内存并重新创建, 或者添加 offset

  {
    int n = cpuNewShape[0];
    int c = cpuNewShape[1];
    int h = cpuNewShape[2];
    int w = cpuNewShape[3];

    if (format == MNN_DATA_FORMAT_NCHW) {
      cpuTensor->setLength(0, n);
      cpuTensor->setStride(0, c * h * w);
      cpuTensor->setLength(1, c);
      cpuTensor->setStride(1, h * w);
      cpuTensor->setLength(2, h);
      cpuTensor->setStride(2, w);
      cpuTensor->setLength(3, w);
      cpuTensor->setStride(3, 1);
    } else if (format == MNN_DATA_FORMAT_NHWC) {
      std::swap(c, h);
      std::swap(c, w);
      cpuTensor->setLength(0, n);
      cpuTensor->setStride(0, h * w * c);
      cpuTensor->setLength(1, h);
      cpuTensor->setStride(1, w * c);
      cpuTensor->setLength(2, w);
      cpuTensor->setStride(2, c);
      cpuTensor->setLength(3, c);
      cpuTensor->setStride(3, 1);
    } else if (format == MNN_DATA_FORMAT_NC4HW4) {
      // TODO: 这里只对 H W 为 1 的情况设计了, 但是实际上 H W 可能不为 1
      cpuTensor->setLength(0, n);
      cpuTensor->setStride(0, c * h * w);
      cpuTensor->setLength(1, c);
      cpuTensor->setStride(1, h * w);
      cpuTensor->setLength(2, h);
      cpuTensor->setStride(2, w);
      cpuTensor->setLength(3, w);
      cpuTensor->setStride(3, 1);
    }
  }

  {
    int n = oclNewShape[0];
    int c = oclNewShape[1];
    int h = oclNewShape[2];
    int w = oclNewShape[3];

    if (format == MNN_DATA_FORMAT_NCHW) {
      oclTensor->setLength(0, n);
      oclTensor->setStride(0, c * h * w);
      oclTensor->setLength(1, c);
      oclTensor->setStride(1, h * w);
      oclTensor->setLength(2, h);
      oclTensor->setStride(2, w);
      oclTensor->setLength(3, w);
      oclTensor->setStride(3, 1);
    } else if (format == MNN_DATA_FORMAT_NHWC) {
      std::swap(c, h);
      std::swap(c, w);
      oclTensor->setLength(0, n);
      oclTensor->setStride(0, h * w * c);
      oclTensor->setLength(1, h);
      oclTensor->setStride(1, w * c);
      oclTensor->setLength(2, w);
      oclTensor->setStride(2, c);
      oclTensor->setLength(3, c);
      oclTensor->setStride(3, 1);
    } else if (format == MNN_DATA_FORMAT_NC4HW4) {
      oclTensor->setLength(0, n);
      oclTensor->setStride(0, c * h * w);
      oclTensor->setLength(1, c);
      oclTensor->setStride(1, h * w);
      oclTensor->setLength(2, h);
      oclTensor->setStride(2, w);
      oclTensor->setLength(3, w);
      oclTensor->setStride(3, 1);
    }
  }
}

}