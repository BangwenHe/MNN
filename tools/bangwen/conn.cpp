
#include "CaffeOp_generated.h"
#include "MNN/AutoTime.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/MNNForwardType.h"
#include "MNN/Tensor.hpp"
#include "MNN_generated.h"
#include "TensorflowOp_generated.h"
#include "Type_generated.h"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/CPUMatMul.hpp"
#include "backend/cpu/CPURandomUniform.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
#include "backend/opencl/core/runtime/OpenCLRuntime.hpp"
#include "backend/opencl/core/runtime/OpenCLWrapper.hpp"
#include "backend/opencl/execution/buffer/ConvBufExecution.hpp"
#include "backend/opencl/execution/buffer/MatmulBufExecution.hpp"
#include "core/Backend.hpp"
#include "core/BufferAllocator.hpp"
#include "core/TensorUtils.hpp"
#include "flatbuffers/flatbuffers.h"
#include <memory>

static void fillFloat(float *dst, int h, int w, float offset = 0.0f) {
  for (int y = 0; y < h; ++y) {
    auto dstY = dst + w * y;
    for (int x = 0; x < w; ++x) {
      int temp = (x + y) % 31;
      dstY[x] = ((float)temp + offset) * 0.01f;
    }
  }
}

int main() {
  MNN::MNNCoreFunctionInit();

  MNN::Backend::Info info;
  info.numThread = 4;
  info.type = MNN_FORWARD_CPU;
  MNN::CPURuntime cpuRuntime(info);
  MNN::Backend *cpuBackend = cpuRuntime.onCreate(nullptr);

  auto *openclOps =
      MNN::OpenCLSymbolsOperator::createOpenCLSymbolsOperatorSingleInstance();
  info.gpuMode =
      MNN_GPU_MEMORY_BUFFER | MNN_GPU_RECORD_BATCH | MNN_GPU_TUNING_NORMAL;
  info.type = MNN_FORWARD_OPENCL;
  MNN::OpenCL::CLRuntime clRuntime(info, 1, 0);
  MNN::Backend *openclBackend = clRuntime.onCreate(nullptr);

  int n = 197, m = 768, k = 768;
  MNN::Tensor *ta =
      MNN::Tensor::createDevice<float>({n, k}, MNN::Tensor::CAFFE);
  cpuBackend->onAcquireBuffer(ta, MNN::Backend::DYNAMIC);
  MNN::Tensor *tb =
      MNN::Tensor::createDevice<float>({k, m}, MNN::Tensor::CAFFE);
  cpuBackend->onAcquireBuffer(tb, MNN::Backend::DYNAMIC);
  MNN::Tensor *tc =
      MNN::Tensor::createDevice<float>({n, m}, MNN::Tensor::CAFFE);
  cpuBackend->onAcquireBuffer(tc, MNN::Backend::DYNAMIC);

  fillFloat(ta->host<float>(), n, k);
  fillFloat(tb->host<float>(), k, m);
  fillFloat(tc->host<float>(), n, m);

  int testRounds = 10;

  {
    MNN::CPUMatMul cpuMatMul(cpuBackend, false, false, false, true);

    for (int i = 0; i < testRounds; i++) {
      {
        MNN::AutoTime timer(__LINE__, __func__);
        cpuMatMul.onResize({ta, tb}, {tc});
      }

      {
        MNN::AutoTime timer(__LINE__, __func__);
        cpuMatMul.onExecute({ta, tb}, {tc});
      }
    }

    auto *line = MNN::Tensor::createDevice<float>({1, m}, MNN::Tensor::CAFFE);
    cpuBackend->onAcquireBuffer(line, MNN::Backend::DYNAMIC);

    {
      float *p = tc->host<float>();
      float *q = line->host<float>();
      std::memcpy(q, p, sizeof(float) * m);
    }

    line->print();
  }

  {
    MNN::Tensor *oclTa =
        MNN::Tensor::createDevice<float>({n, k}, MNN::Tensor::CAFFE);
    MNN::Tensor *oclTb =
        MNN::Tensor::createDevice<float>({k, m}, MNN::Tensor::CAFFE);
    MNN::Tensor *oclTc =
        MNN::Tensor::createDevice<float>({n, m}, MNN::Tensor::CAFFE);
    MNN::Tensor *line =
        MNN::Tensor::createDevice<float>({1, m}, MNN::Tensor::CAFFE);

    MNN::TensorUtils::getDescribe(oclTa)->setBackend(openclBackend);
    MNN::TensorUtils::getDescribe(oclTb)->setBackend(openclBackend);
    MNN::TensorUtils::getDescribe(oclTc)->setBackend(openclBackend);
    MNN::TensorUtils::getDescribe(line)->setBackend(openclBackend);

    openclBackend->onAcquireBuffer(oclTa, MNN::Backend::DYNAMIC);
    openclBackend->onAcquireBuffer(oclTb, MNN::Backend::DYNAMIC);
    openclBackend->onAcquireBuffer(oclTc, MNN::Backend::DYNAMIC);
    openclBackend->onAcquireBuffer(line, MNN::Backend::STATIC);
    oclTa->copyFromHostTensor(ta);
    oclTb->copyFromHostTensor(tb);

    {
      oclTc->copyFromHostTensor(tc);
      MNN::OpenCL::MatMulBufExecution clMatMul{
          {ta, tb}, nullptr, openclBackend, false, false};
      for (int i = 0; i < testRounds; i++) {
        {
          MNN::AutoTime timer(__LINE__, __func__);
          clMatMul.onResize({oclTa, oclTb}, {oclTc});
        }

        {
          MNN::AutoTime timer(__LINE__, __func__);
          clMatMul.onExecute({oclTa, oclTb}, {oclTc});
        }
      }

      {
        float *p = (float *)oclTc->map(MNN::Tensor::MAP_TENSOR_READ,
                                       MNN::Tensor::CAFFE);
        float *q = (float *)line->map(MNN::Tensor::MAP_TENSOR_WRITE,
                                      MNN::Tensor::CAFFE);

        std::memcpy(q, p, sizeof(float) * m);

        line->unmap(MNN::Tensor::MAP_TENSOR_WRITE, MNN::Tensor::CAFFE, q);
        q = (float *)line->map(MNN::Tensor::MAP_TENSOR_READ,
                               MNN::Tensor::CAFFE);

        line->print();

        line->unmap(MNN::Tensor::MAP_TENSOR_READ, MNN::Tensor::CAFFE, q);
        oclTc->unmap(MNN::Tensor::MAP_TENSOR_READ, MNN::Tensor::CAFFE, p);
      }
    }

    {
      oclTc->copyFromHostTensor(tc);

      MNN::OpT *opt = new MNN::OpT;
      opt->name = "conv_test";
      opt->type = MNN::OpType_Convolution;
      opt->main.type = MNN::OpParameter_Convolution2D;
      opt->main.value = new MNN::Convolution2DT;
      opt->main.AsConvolution2D()->common.reset(new MNN::Convolution2DCommonT);
      opt->main.AsConvolution2D()->common->kernelX = k;
      opt->main.AsConvolution2D()->common->kernelY = m;

      flatbuffers::FlatBufferBuilder builder;
      auto opOffset = MNN::Op::Pack(builder, opt);
      builder.Finish(opOffset);
      uint8_t *buf = builder.GetBufferPointer();
      int size = builder.GetSize();
      const MNN::Op *op = flatbuffers::GetRoot<MNN::Op>(buf);

      MNN::OpenCL::ConvBufExecution clConvBuf{
          {oclTa, oclTb}, {oclTc}, op, openclBackend};

      for (int i = 0; i < testRounds; i++) {
        {
          MNN::AutoTime timer(__LINE__, __func__);
          clConvBuf.onResize({oclTa, oclTb}, {oclTc});
        }

        {
          MNN::AutoTime timer(__LINE__, __func__);
          clConvBuf.onExecute({oclTa, oclTb}, {oclTc});
        }
      }

      {
        float *p = (float *)oclTc->map(MNN::Tensor::MAP_TENSOR_READ,
                                       MNN::Tensor::CAFFE);
        float *q = (float *)line->map(MNN::Tensor::MAP_TENSOR_WRITE,
                                      MNN::Tensor::CAFFE);

        std::memcpy(q, p, sizeof(float) * m);

        line->unmap(MNN::Tensor::MAP_TENSOR_WRITE, MNN::Tensor::CAFFE, q);
        q = (float *)line->map(MNN::Tensor::MAP_TENSOR_READ,
                               MNN::Tensor::CAFFE);

        line->print();

        line->unmap(MNN::Tensor::MAP_TENSOR_READ, MNN::Tensor::CAFFE, q);
        oclTc->unmap(MNN::Tensor::MAP_TENSOR_READ, MNN::Tensor::CAFFE, p);
      }
    }
  }
}
