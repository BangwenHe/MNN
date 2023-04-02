#include "backend/cpu/CPUGateRecurrent.hpp"
#include "MNN/ErrorCode.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include <bits/stdint-uintn.h>
#include <cstdio>

namespace MNN {

// ErrorCode CPUGateRecurrent::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
//     // 考察代码运行过程中是否会存在内存不足的情况

//     return NO_ERROR;
// }

ErrorCode CPUGateRecurrent::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto outputTensor = outputs[0];
    MNN_ASSERT(outputTensor->dimensions() == 4);

    auto xTensor = inputs[0];
    auto g1Tensor = inputs[1];
    auto g2Tensor = inputs[2];
    auto g3Tensor = inputs[3];
    
    auto outputPtr = outputTensor->host<float>();
    auto xPtr = xTensor->host<float>();
    auto g1Ptr = g1Tensor->host<float>();
    auto g2Ptr = g2Tensor->host<float>();
    auto g3Ptr = g3Tensor->host<float>();

    int batches = outputTensor->length(0);
    int channels = outputTensor->length(1);
    int height = outputTensor->length(2);
    int width = outputTensor->length(3);

    printf("%dx%dx%dx%d\n", batches, channels, height, width);

    // 使用循环的 naive 实现
    for (int w = 0; w < width; w++) {
        for (int b = 0; b < batches; b++) {
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < height; h++) {
                    int idx = b * channels * height * width + c * height * width + h * width + w;

                    float g1 = *(g1Ptr + idx);
                    float g2 = *(g2Ptr + idx);
                    float g3 = *(g3Ptr + idx);

                    float h1 = 0.0f;
                    if (w > 0 && h > 0) {
                        h1 = *(outputPtr + idx - width - 1);
                    }

                    float h2 = 0.0f;
                    if (w > 0) {
                        h2 = *(outputPtr + idx - 1);
                    }

                    float h3 = 0.0f;
                    if (w > 0 && h < height - 1) {
                        h3 = *(outputPtr + idx + width - 1);
                    }

                    float x = *(xPtr + idx);
                    auto _outputPtr = outputPtr + idx;

                    *_outputPtr = (1 - g1 - g2 - g3) * x + (h1 * g1 + h2 * g2 + h3 * g3);
                }
            }
        }
    }

    return NO_ERROR;
}

class CPUGateRecurrentCreator : public CPUBackend::Creator {
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, 
                                const std::vector<Tensor *> &outputs, 
                                const MNN::Op *op,
                                Backend *backend) const override {
        return new CPUGateRecurrent(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUGateRecurrentCreator, OpType_GateRecurrent);

}