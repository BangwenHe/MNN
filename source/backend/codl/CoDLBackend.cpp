#include "CoDLBackend.hpp"
#include "CoDLCPUGPUMemPack.hpp"
#include "MNN/ErrorCode.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/MNNForwardType.h"
#include "MNN/Tensor.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/opencl/core/runtime/OpenCLRuntime.hpp"
#include "core/Backend.hpp"
#include "core/Execution.hpp"
#include "core/TensorUtils.hpp"
#include "core/RuntimeFactory.hpp"
#include <memory>

namespace MNN {



CoDLRuntime::CoDLRuntime(const Backend::Info& info) { 
  Backend::Info cpuInfo;
  cpuInfo.numThread = info.numThread;
  cpuInfo.type = MNN_FORWARD_CPU;
  mCPURuntime.reset((CPURuntime *) RuntimeFactory::create(cpuInfo));

  Backend::Info oclInfo;
  oclInfo.numThread = info.numThread;
  oclInfo.type = MNN_FORWARD_OPENCL;
  mCLRuntime.reset((OpenCL::CLRuntime *) RuntimeFactory::create(oclInfo));
}

CoDLRuntime::~CoDLRuntime() { }

int CoDLRuntime::onGetRuntimeStatus(RuntimeStatus statusEnum) const {
  switch (statusEnum) {
    case STATUS_SUPPORT_FP16: {
      return mCPURuntime->onGetRuntimeStatus(statusEnum) && mCLRuntime->onGetRuntimeStatus(statusEnum);
    }
    case STATUS_SUPPORT_DOT_PRODUCT: {
      return mCPURuntime->onGetRuntimeStatus(statusEnum) && mCLRuntime->onGetRuntimeStatus(statusEnum);
    }
    case STATUS_SUPPORT_POWER_LOW: {
      return mCLRuntime->onGetRuntimeStatus(statusEnum);
    }
    default: {
      MNN_ERROR("unsupported interface");
      break;
    }
  }

  return 0;
}

Backend* CoDLRuntime::onCreate(const BackendConfig* config) const {
  return new CoDLBackend(this, config);
}

void CoDLRuntime::onGabageCollect(int level) {
  mCPURuntime->onGabageCollect(level);
  mCLRuntime->onGabageCollect(level);
}

float CoDLRuntime::onGetMemoryInMB() {
  return mCPURuntime->onGetMemoryInMB() + mCLRuntime->onGetMemoryInMB();
}

void CoDLRuntime::onConcurrencyBegin() const {
  mCPURuntime->onConcurrencyBegin();
}

void CoDLRuntime::onConcurrencyEnd() const {
  mCPURuntime->onConcurrencyEnd();
}

bool CoDLRuntime::onCheckInfo(Backend::Info& info) const {
  return mCPURuntime->onCheckInfo(info) && mCLRuntime->onCheckInfo(info);
}

bool CoDLRuntime::onSetCache(const void* buffer, size_t size) {
  // OpenCL 需要设置 cache, 否则没有编译的 kernel
  return mCLRuntime->onSetCache(buffer, size);
}

std::pair<const void*, size_t> CoDLRuntime::onGetCache() {
  // Only get cache for OpenCL backend
  return mCLRuntime->onGetCache();
  // return {nullptr, 0};
}

bool CoDLRuntime::onMeasure(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                            const MNN::Op* op, OpInfo& dstInfo) const {
  return mCLRuntime->onMeasure(inputs, outputs, op, dstInfo);
}

void CoDLRuntime::onMaskOpReady(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op) {
  mCLRuntime->onMaskOpReady(inputs, outputs, op);
}


struct CoDLBackendCreator : RuntimeCreator {
  virtual Runtime* onCreate(const Backend::Info& info) const {
    return new CoDLRuntime(info);
  }

};

CoDLBackend::CoDLBackend(const CoDLRuntime *runtime, const BackendConfig *config) : Backend(MNN_FORWARD_USER_2) {
  mCoDLRuntime = runtime;
  mCPUBackend.reset((CPUBackend *) (mCoDLRuntime->mCPURuntime->onCreate(config)));
  mOpenCLBackend.reset((OpenCL::OpenCLBackend *) (mCoDLRuntime->mCLRuntime->onCreate(config))); 
}

CoDLBackend::~CoDLBackend() { }

void CoDLBackend::onExecuteBegin() const {
  mCPUBackend->onExecuteBegin();
  mOpenCLBackend->onExecuteBegin();
}

void CoDLBackend::onExecuteEnd() const {
  mCPUBackend->onExecuteEnd();
  mOpenCLBackend->onExecuteEnd();
}

void CoDLBackend::onResizeBegin() {
  mCPUBackend->onResizeBegin();
  mOpenCLBackend->onResizeBegin();
}

ErrorCode CoDLBackend::onResizeEnd() {
  auto cpuCode = mCPUBackend->onResizeEnd();
  auto oclCode = mOpenCLBackend->onResizeEnd();
  if (cpuCode != NO_ERROR) {
    return cpuCode;
  } else if (oclCode != NO_ERROR) {
    return oclCode;
  } else {
    return NO_ERROR;
  }
}

Backend::MemObj* CoDLBackend::onAcquire(const Tensor* tensor, StorageType storageType) {
  // TODO: 实现一个 CoDLAllocator, 用于管理 CPU 和 GPU 上的内存, 避免内存泄漏

  auto *obj = mCPUBackend->onAcquire(tensor, storageType);
  delete obj;

  // TODO: FIXME: mempack 不会释放
  auto *mempack = new CoDLCPUGPUMemPack(mCPUBackend.get(), mOpenCLBackend.get(), tensor, storageType);
  ((Tensor *)tensor)->buffer().device = (uint64_t) mempack;

  return new Backend::MemObj;
}

bool CoDLBackend::onClearBuffer() {
  return mCPUBackend->onClearBuffer() && mOpenCLBackend->onClearBuffer();
}

void CoDLBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
  // 将三份buffer全部拷贝
  mCPUBackend->onCopyBuffer(srcTensor, dstTensor);

  // deviceId 是 halide_buffer_t::device, 表示设备上的内存 handle
  // 如果 deviceId 为 0 或者 1, 说明内存在 CPU 上
  auto isSrcInDev = srcTensor->deviceId() != 0 && srcTensor->deviceId() != 1;
  auto isDstInDev = dstTensor->deviceId() != 0 && dstTensor->deviceId() != 1;
  // 至少有一个 tensor 不在 CPU 上, 或者两个都不在 CPU 上
  if (isSrcInDev && isDstInDev) {
    CoDLCPUGPUMemPack *srcObj = (CoDLCPUGPUMemPack *) (srcTensor->buffer().device);
    CoDLCPUGPUMemPack *dstObj = (CoDLCPUGPUMemPack *) (dstTensor->buffer().device);

    mCPUBackend->onCopyBuffer(srcObj->getCPUTensor(), dstObj->getCPUTensor());
    mOpenCLBackend->onCopyBuffer(srcObj->getOCLTensor(), dstObj->getOCLTensor());
  } else if (isSrcInDev) {
    // TODO: 这里假设 src 上的 host 内存跟 memobj 内存是同步的
    // 如果 src 在设备上, 跳过, 因为已经在 CPU 上拷贝过了
  } else if (isDstInDev) {
    // 如果 dst 在设备上, 拷贝两次
    CoDLCPUGPUMemPack *dstObj = (CoDLCPUGPUMemPack *) (dstTensor->buffer().device);
    mCPUBackend->onCopyBuffer(srcTensor, dstObj->getCPUTensor());
    mOpenCLBackend->onCopyBuffer(srcTensor, dstObj->getOCLTensor());
  } else {
    MNN_PRINT("onCopyBuffer: both src and dst are on CPU\n");
  }
}

static inline std::map<std::pair<OpType, GpuMemObject>, CoDLBackend::Creator *>* GetCreator() {
  static std::once_flag of;
  static std::map<std::pair<OpType, GpuMemObject>, CoDLBackend::Creator *> *gCreator = nullptr;
  std::call_once(of, [&] () { gCreator = new std::map<std::pair<OpType, GpuMemObject>, CoDLBackend::Creator *>; });
  return gCreator;
}

void CoDLBackend::addCreator(OpType type, GpuMemObject obj, Creator* creator) {
  auto *map = GetCreator();
  auto key = std::make_pair(type, obj);
  map->insert(std::make_pair(key, creator));
}


Execution* CoDLBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) {
  auto *map = GetCreator();
  auto key = std::make_pair(op->type(), mOpenCLBackend->getOpenCLRuntime()->getGpuMemType());
  auto iter = map->find(key);
  if (iter == map->end()) {
    // TODO: 如果是计算密集型 OP，如 conv mm 等，并行执行；如果是内存密集型 OP，如 Raster 等，CPU 执行
    // TODO: 需要两份inputs和outputs，一份给CPU，一份给GPU
    MNN_PRINT("Don't support type [%s]\n", MNN::EnumNameOpType(op->type()));
    return nullptr;
  }
  return iter->second->onCreate(inputs, outputs, op, this);
}

void* CoDLBackend::onMapTensor(Tensor::MapType mtype, Tensor::DimensionType dtype, const Tensor* srcTensor) {
  // TODO: 这里直接返回了 CPU 上的内存, 实际上应该汇总 CPU 和 GPU 上的内存
  return mCPUBackend->onMapTensor(mtype, dtype, srcTensor);
}

bool CoDLBackend::onUnmapTensor(Tensor::MapType mtype, Tensor::DimensionType dtype, const Tensor* dstTensor, void* mapPtr) {
  return mCPUBackend->onUnmapTensor(mtype, dtype, dstTensor, mapPtr);
}

int CoDLBackend::onSync(Tensor::MapType mtype, bool toCpu, const Tensor* dstTensor) {
  // MARK: onSync 在 CPUBackend 上无操作, 在 OpenCLBackend 上执行 CommandQueue::finish()
  return mOpenCLBackend->onSync(mtype, toCpu, dstTensor);
}

void registerCoDLBackendCreator() {
    MNNInsertExtraRuntimeCreator(MNN_FORWARD_USER_2, new CoDLBackendCreator, true);
};

} // namespace MNN