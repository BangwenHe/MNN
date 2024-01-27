#ifndef CODL_COMMON_EXECUTION_HPP
#define CODL_COMMON_EXECUTION_HPP

#include "MNN/ErrorCode.hpp"
#include "backend/codl/CoDLBackend.hpp"
#include "core/Backend.hpp"
#include "core/Execution.hpp"

namespace MNN {

class CoDLCPUOnlyCommonExecution : public Execution {
public:
  CoDLCPUOnlyCommonExecution(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
  virtual ~CoDLCPUOnlyCommonExecution() = default;

  ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
  ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
  const Op *mOp;
  CoDLBackend *mBackend;
  std::shared_ptr<Execution> mCPUExecution;
  std::vector<Tensor *> mCPUInputs, mCPUOutputs;
  std::vector<Tensor *> mOCLInputs, mOCLOutputs;
};

}

#endif
