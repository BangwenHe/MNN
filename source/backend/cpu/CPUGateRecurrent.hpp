#ifndef CPUGateRecurrent_hpp
#define CPUGateRecurrent_hpp

#include "core/Backend.hpp"
#include "core/Execution.hpp"

namespace MNN {

class CPUGateRecurrent : public Execution {
public:
    CPUGateRecurrent(Backend* b) : Execution(b)  {}
    ~CPUGateRecurrent() = default;

    // 若执行onExecute需要使用缓存，在此函数中申请，若无可不声明
    // virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, 
    //                            const std::vector<Tensor *> &outputs) override;
    // 具体的Op执行函数
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, 
                                const std::vector<Tensor *> &outputs) override;
};

}

#endif  // CPUGateRecurrent_hpp