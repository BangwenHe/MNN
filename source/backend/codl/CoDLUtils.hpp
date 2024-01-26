#ifndef CODL_UTILS_HPP
#define CODL_UTILS_HPP

#include "CoDLBackend.hpp"

namespace MNN {

struct CoDLUtils {

  static void partConv2dShape(const std::vector<int> &originalInputShape,
                              const std::vector<int> &originalOutputShape,
                              MNN_DATA_FORMAT inputFormat,
                              MNN_DATA_FORMAT outputFormat,
                              std::vector<int> &partCPUInputShape,
                              std::vector<int> &partCPUOutputShape,
                              std::vector<int> &partOCLInputShape,
                              std::vector<int> &partOCLOutputShape,
                              const CoDLNodePartitionParam &param);

  static void printCoDLTensorShape(const Tensor *tensor);
};

} // namespace MNN

#endif
