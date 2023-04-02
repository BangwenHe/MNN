
#include "MNN_generated.h"
#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

class GateRecurrentSizeComputer : public SizeComputer {
public:
    /* onComputeSize 用于计算输出张量的维度 */
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        outputs[0]->buffer().type = inputs[0]->buffer().type;
        outputs[0]->buffer().dimensions = inputs[0]->buffer().dimensions;
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;

        int nDims = inputs[0]->dimensions();
        for (int i = 0; i < nDims; i++) {
            outputs[0]->setLength(i, inputs[0]->length(i));
        }

        return true;
    }

    virtual float onComputeFlops(const MNN::Op* op, 
                                const std::vector<Tensor*>& inputs,
                                const std::vector<Tensor*>& outputs) const override {
        float flops = 1.0f;
        int nDims = outputs[0]->dimensions();

        // 4 * N * C * H * W
        for (int i = 0; i < nDims; i++) {
            flops *= outputs[0]->length(i);
        }

        return flops * 4;
    }
};

REGISTER_SHAPE(GateRecurrentSizeComputer, OpType_GateRecurrent);
}