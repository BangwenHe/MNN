#include <stdio.h>
#include <string>
#include "MNN_generated.h"
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(GateRecurrentOnnx);

MNN::OpType GateRecurrentOnnx::opType() {
    return MNN::OpType_GateRecurrent;
}

MNN::OpParameter GateRecurrentOnnx::type() {
    return MNN::OpParameter_NONE;
}

void GateRecurrentOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode, OnnxScope* scope) {
    // 因为没有参数 `__init__` 没有输入, 也不需要获取 param 和 run
    // dstOp->defaultDimentionFormat = MNN::MNN_DATA_FORMAT_NCHW;
    dstOp->name = "GateRecurrent";
}

REGISTER_CONVERTER(GateRecurrentOnnx, GateRecurrent);
