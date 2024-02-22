#ifndef CODLBACKEND_HPP
#define CODLBACKEND_HPP

#include <map>
#include <memory>
#include <utility>
#include "MNN/ErrorCode.hpp"
#include "MNN/MNNForwardType.h"
#include "MNN/Tensor.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/runtime/OpenCLRuntime.hpp"
#include "core/Backend.hpp"
#include "core/Execution.hpp"
#include "core/BufferAllocator.hpp"
#include "MNN_generated.h"


namespace MNN {


struct CoDLNodePartitionParam {

  enum PartDim {
    // 对于矩阵乘法类型的卷积，N表示N，IC表示M，OC表示K
    PART_DIM_N = 0,
    PART_DIM_IC = 1,
    PART_DIM_OC = 2
  };

  PartDim mPartDim;
  float mPartRatio;
};

class CoDLPartitionStrategy {
public:
  CoDLPartitionStrategy(const std::string &jsonFile);

  CoDLNodePartitionParam getPartitionParam(int n, int m, int k);

private:
  std::string mJsonFile;

  std::map<std::tuple<int, int, int>, CoDLNodePartitionParam> mPartitionMap;
};


class CoDLRuntime : public Runtime {
public:
    friend class CoDLBackend;
    CoDLRuntime(const Backend::Info& info);
    virtual ~ CoDLRuntime();
    int onGetRuntimeStatus(RuntimeStatus statusEnum) const override;
    virtual Backend* onCreate(const BackendConfig* config) const override;
    virtual void onGabageCollect(int level) override;
    virtual float onGetMemoryInMB() override;
    virtual CompilerType onGetCompilerType() const override {
        return Compiler_Loop;
    }
    void onConcurrencyBegin() const;
    void onConcurrencyEnd() const;
    virtual bool onCheckInfo(Backend::Info& info) const override;

    virtual bool onSetCache(const void* buffer, size_t size) override;
    virtual std::pair<const void*, size_t> onGetCache() override;

    virtual bool onMeasure(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           const MNN::Op* op, OpInfo& dstInfo) const override;

    virtual void onMaskOpReady(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const MNN::Op* op) override;

private:
    std::shared_ptr<CPURuntime> mCPURuntime;
    std::shared_ptr<OpenCL::CLRuntime> mCLRuntime;
};

class CoDLBackend final : public Backend {
public:
  CoDLBackend(const CoDLRuntime *runtime, const BackendConfig *config);
  
  virtual ~CoDLBackend();
  virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) override;
  virtual void onExecuteBegin() const override;
  virtual void onExecuteEnd() const override;
  virtual void onResizeBegin() override;
  virtual ErrorCode onResizeEnd() override;

  virtual MemObj* onAcquire(const Tensor* tensor, StorageType storageType) override;
  virtual bool onClearBuffer() override;
  virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const override;
  
  virtual void* onMapTensor(Tensor::MapType mtype, Tensor::DimensionType dtype, const Tensor* srcTensor) override;
  virtual bool onUnmapTensor(Tensor::MapType mtype, Tensor::DimensionType dtype, const Tensor* dstTensor, void* mapPtr) override;
  virtual int onSync(Tensor::MapType mtype, bool toCpu, const Tensor* dstTensor) override;

  CPUBackend* getCPUBackend() const {
    return mCPUBackend.get();
  }

  OpenCL::OpenCLBackend* getOpenCLBackend() const {
    return mOpenCLBackend.get();
  }

  CPUBackend* getBackupCPUBackend() const {
    return mBackupCPUBackend.get();
  }

  CoDLNodePartitionParam getPartitionParam(int n, int m, int k) {
    return mPartitionStrategy->getPartitionParam(n, m, k);
  }

  class Creator {
    public:
        /**
         * @brief create execution for given input, op on metal backend.
         * @param inputs    given input tensors.
         * @param op        given op.
         * @param backend   metal backend.
         * @return created execution if supported, NULL otherwise.
         */
        virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op,
                                    Backend *backend) const = 0;
  };
  /**
    * @brief register creator for given op type.
    * @param type      given op type.
    * @param creator   registering creator.
    */
  static void addCreator(OpType type, GpuMemObject obj, Creator *creator);

private:
  std::shared_ptr<CPUBackend> mCPUBackend;
  std::shared_ptr<OpenCL::OpenCLBackend> mOpenCLBackend;
  std::shared_ptr<CPUBackend> mBackupCPUBackend;
  const CoDLRuntime *mCoDLRuntime;
  std::shared_ptr<CoDLPartitionStrategy> mPartitionStrategy;
};

template <class T>
class CoDLCreatorRegister {
public:
    /**
     * @brief initializer. register T creator for given op type.
     * @param type  given op type.
     */
    CoDLCreatorRegister(OpType type, GpuMemObject obj) {
        T *test = new T;
        CoDLBackend::addCreator(type, obj, test);
    }
};


template <typename T>
class TypedCreator : public CoDLBackend::Creator {
public:
    virtual ~TypedCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op,
                                Backend *backend) const override {
        return new T(backend, op, inputs, outputs);
    }
};

}


#endif
