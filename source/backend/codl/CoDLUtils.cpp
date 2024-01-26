#include "CoDLUtils.hpp"
#include "CoDLCPUGPUMemPack.hpp"

namespace MNN {

void CoDLUtils::partConv2dShape(
    const std::vector<int> &originalInputShape,
    const std::vector<int> &originalOutputShape, MNN_DATA_FORMAT inputFormat,
    MNN_DATA_FORMAT outputFormat, std::vector<int> &partCPUInputShape,
    std::vector<int> &partCPUOutputShape, std::vector<int> &partOCLInputShape,
    std::vector<int> &partOCLOutputShape, const CoDLNodePartitionParam &param) {
  int inputN = originalInputShape[0];
  int inputC = originalInputShape[1];
  int inputH = originalInputShape[2];
  int inputW = originalInputShape[3];

  int outputN = originalOutputShape[0];
  int outputC = originalOutputShape[1];
  int outputH = originalOutputShape[2];
  int outputW = originalOutputShape[3];

  int partCPUInputN = inputN;
  int partCPUInputC = inputC;
  int partCPUInputH = inputH;
  int partCPUInputW = inputW;

  int partCPUOutputN = outputN;
  int partCPUOutputC = outputC;
  int partCPUOutputH = outputH;
  int partCPUOutputW = outputW;

  int partOCLInputN = inputN;
  int partOCLInputC = inputC;
  int partOCLInputH = inputH;
  int partOCLInputW = inputW;

  int partOCLOutputN = outputN;
  int partOCLOutputC = outputC;
  int partOCLOutputH = outputH;
  int partOCLOutputW = outputW;

  if (param.mPartDim == CoDLNodePartitionParam::PART_DIM_N) {
    partCPUInputN = inputN * param.mPartRatio;
    partCPUOutputN = outputN * param.mPartRatio;
    partOCLInputN = inputN - partCPUInputN;
    partOCLOutputN = outputN - partCPUOutputN;
  } else if (param.mPartDim == CoDLNodePartitionParam::PART_DIM_IC) {
    partCPUInputC = inputC * param.mPartRatio;
    partOCLInputC = inputC - partCPUInputC;
  } else if (param.mPartDim == CoDLNodePartitionParam::PART_DIM_OC) {
    partCPUOutputC = outputC * param.mPartRatio;
    partOCLOutputC = outputC - partCPUOutputC;
  }

  if (inputFormat == MNN_DATA_FORMAT_NCHW) {
    partCPUInputShape = {partCPUInputN, partCPUInputC, partCPUInputH,
                         partCPUInputW};
    partOCLInputShape = {partOCLInputN, partOCLInputC, partOCLInputH,
                         partOCLInputW};
  } else if (inputFormat == MNN_DATA_FORMAT_NHWC) {
    partCPUInputShape = {partCPUInputN, partCPUInputH, partCPUInputW,
                         partCPUInputC};
    partOCLInputShape = {partOCLInputN, partOCLInputH, partOCLInputW,
                         partOCLInputC};
  } else if (inputFormat == MNN_DATA_FORMAT_NC4HW4) {
    // TODO: 检查当卷积不再是 1x1 时，是否正确
    partCPUInputShape = {partCPUInputN, partCPUInputC, partCPUInputH,
                         partCPUInputW};
    partOCLInputShape = {partOCLInputN, partOCLInputC, partOCLInputH,
                          partOCLInputW}; 
  }

  if (outputFormat == MNN_DATA_FORMAT_NCHW) {
    partCPUOutputShape = {partCPUOutputN, partCPUOutputC, partCPUOutputH,
                          partCPUOutputW};
    partOCLOutputShape = {partOCLOutputN, partOCLOutputC, partOCLOutputH,
                          partOCLOutputW};
  } else if (outputFormat == MNN_DATA_FORMAT_NHWC) {
    partCPUOutputShape = {partCPUOutputN, partCPUOutputH, partCPUOutputW,
                          partCPUOutputC};
    partOCLOutputShape = {partOCLOutputN, partOCLOutputH, partOCLOutputW,
                          partOCLOutputC};
  } else if (outputFormat == MNN_DATA_FORMAT_NC4HW4) {
    partCPUOutputShape = {partCPUOutputN, partCPUOutputC, partCPUOutputH,
                          partCPUOutputW};
    partOCLOutputShape = {partOCLOutputN, partOCLOutputC, partOCLOutputH,
                          partOCLOutputW};
  }
}

void CoDLUtils::printCoDLTensorShape(const Tensor *tensor) {
  std::vector<int> shape = tensor->shape();
  MNN_PRINT("CoDL Tensor Shape: ");
  for (int i = 0; i < shape.size(); i++) {
    MNN_PRINT("%d%c", shape[i], "x,"[i == shape.size() - 1]);
  }
  
  CoDLCPUGPUMemPack *mem = (CoDLCPUGPUMemPack *)tensor->buffer().device;
  MNN_PRINT(" CPU Tensor Shape: ");
  shape = mem->getCPUTensor()->shape();
  for (int i = 0; i < shape.size(); i++) {
    MNN_PRINT("%d%c", shape[i], "x,"[i == shape.size() - 1]);
  }

  MNN_PRINT(" GPU Tensor Shape: ");
  shape = mem->getOCLTensor()->shape();
  for (int i = 0; i < shape.size(); i++) {
    MNN_PRINT("%d%c", shape[i], "x,"[i == shape.size() - 1]);
  }

  MNN_PRINT("\n");
}

} // namespace MNN