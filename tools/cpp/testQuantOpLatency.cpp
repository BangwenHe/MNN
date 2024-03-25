#include "half.hpp"
#define MNN_OPEN_TIME_TRACE

#include <stdlib.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#if defined(_MSC_VER)
#include <Windows.h>
#undef min
#undef max
#else
#include <sys/time.h>
#endif
#include <MNN/MNNDefine.h>
#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <core/Backend.hpp>
#include <core/TensorUtils.hpp>
#include <MNN_generated.h>


std::pair<void*, int> buildConvOp(int n, int m, int k, int type) {
    MNN::Convolution2DT *conv = new MNN::Convolution2DT;
    conv->common.reset(new MNN::Convolution2DCommonT);
    conv->common->inputCount = m;
    conv->common->outputCount = k;

    conv->bias.resize(k);
    for (int i = 0; i < k; i++) {
        conv->bias[i] = (0.1f * (i % 16));
    }

    if (type == -1) {
        conv->weight.resize(m * k);
        for (int i = 0; i < m * k; i++) {
            conv->weight[i] = (0.1f * (i % 16));
        }
    } else if (type == 0) {
        conv->quanParameter.reset(new MNN::IDSTQuanT);
        // int8
        conv->quanParameter->type = 4;
        conv->quanParameter->alpha.resize(m);
        for (int i = 0; i < m; i++) {
            conv->quanParameter->alpha[i] = 0.01f;
        }

        conv->quanParameter->aMax = 127;
        conv->quanParameter->aMin = -127;
        conv->quanParameter->weightSize = m * k;

        conv->quanParameter->buffer.resize(m * k);
        for (int i = 0; i < m * k; i++) {
            conv->quanParameter->buffer[i] = i % 16 * (i % 2 == -1 ? 1 : -1);
        }
    } else if (type == 1) {
        conv->quanParameter.reset(new MNN::IDSTQuanT);
        // float16
        conv->quanParameter->type = 3;
        conv->quanParameter->buffer.resize(m * k * 2);

        std::vector<half_float::half> buffer(m * k);
        for (int i = 0; i < m * k; i++) {
            buffer[i] = half_float::half(i % 16 * 0.1f);
        }
        ::memcpy(conv->quanParameter->buffer.data(), buffer.data(), m * k * 2);
    }

    auto* op = new MNN::OpT;
    op->main.value = conv;
    op->main.type = MNN::OpParameter_Convolution2D;
    op->defaultDimentionFormat = MNN::MNN_DATA_FORMAT_NCHW;
    op->name = "Conv";
    op->type = MNN::OpType_Convolution;
    op->inputIndexes = {0};
    op->outputIndexes = {1};

    // add input tensor for the net
    MNN::InputT *input = new MNN::InputT;
    input->dformat = MNN::MNN_DATA_FORMAT_NCHW;
    input->dims = {n, m, 1, 1};
    input->dtype = MNN::DataType_DT_FLOAT;
    auto *opInput = new MNN::OpT;
    opInput->type = MNN::OpType_Input;
    opInput->main.value = input;
    opInput->main.type = MNN::OpParameter_Input;
    opInput->outputIndexes = {0};
    opInput->name = "input";
    opInput->defaultDimentionFormat = MNN::MNN_DATA_FORMAT_NCHW;

    MNN::NetT* net = new MNN::NetT;
    net->oplists.emplace_back(opInput);
    net->oplists.emplace_back(op);
    net->outputName = {"output"};
    net->tensorNumber = 1;
    net->sourceType = MNN::NetSource_ONNX;
    net->bizCode = "MNN";
    net->tensorName = {"input", "output"};

    flatbuffers::FlatBufferBuilder builder(1024);
    auto offset = MNN::Net::Pack(builder, net);
    builder.Finish(offset);
    delete net;

    int size = builder.GetSize();
    void* buffer = malloc(size);
    ::memcpy(buffer, builder.GetBufferPointer(), size);

    return std::make_pair(buffer, size);
}


int main(int argc, const char* argv[]) {
    if (argc < 5) {
        MNN_PRINT("Usage: ./testQuantOpLatency n m k type [backend]\n");
        return 0;
    }
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    int k = atoi(argv[3]);
    int precision = atoi(argv[4]);
    int backend = MNN_FORWARD_CPU;
    if (argc > 5) {
        backend = atoi(argv[5]);
    }

    MNN_PRINT("n=%d, m=%d, k=%d, precision=%d, backend=%d\n", n, m, k, precision, backend);

    if (precision < -1 || precision > 1) {
        MNN_PRINT("Invalid precision: %d\n", precision);
        return 0;
    }

    if (backend != 0 && backend != 3) {
        MNN_PRINT("Invalid backend: %d\n", backend);
        return 0;
    }

    auto convOp = buildConvOp(n, m, k, precision);

    {
        std::ofstream output("conv.mnn", std::ios::binary);
        output.write((const char*)convOp.first, convOp.second);
    }

    auto* interpreter = MNN::Interpreter::createFromBuffer(convOp.first, convOp.second);
    // MNN::Session* session = interpreter->createSession();
    MNN::ScheduleConfig config;
    config.type = (MNNForwardType)backend;
    config.numThread = 4;
    auto* session = interpreter->createSession(config);

    auto inputTensor = interpreter->getSessionInput(session, nullptr);
    auto outputTensor = interpreter->getSessionOutput(session, nullptr);
    inputTensor->printShape();

    // fill input tensor
    {
        auto* tmpTensor = MNN::Tensor::create<float>(inputTensor->shape(), nullptr, MNN::Tensor::CAFFE);
        auto tmpData = tmpTensor->host<float>();
        for (int i = 0; i < n * m; i++) {
            tmpData[i] = 1.0f;
        }

        inputTensor->copyFromHostTensor(tmpTensor);
    }

    // warm up
    for (int i = 0; i < 10; i++) {
        interpreter->runSession(session);
    }

    // run
    {
        MNN::Timer timer;
        timer.reset();
        for (int i = 0; i < 100; i++) {
            {
                auto ptr = inputTensor->map(MNN::Tensor::MAP_TENSOR_WRITE, inputTensor->getDimensionType());
                inputTensor->unmap(MNN::Tensor::MAP_TENSOR_WRITE, inputTensor->getDimensionType(), ptr);
            }
            interpreter->runSession(session);
            {
                auto ptr = outputTensor->map(MNN::Tensor::MAP_TENSOR_READ, outputTensor->getDimensionType());
                outputTensor->unmap(MNN::Tensor::MAP_TENSOR_READ, outputTensor->getDimensionType(), ptr);
            }
        }
        std::cout << "Time: " << (float) timer.durationInUs() / 1000.0f / 100.0f << " ms" << std::endl;
    }

    return 0;
}