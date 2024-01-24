#ifndef CODL_RASTER_HPP
#define CODL_RASTER_HPP

#include "MNN/ErrorCode.hpp"
#include "backend/codl/CoDLBackend.hpp"
#include "core/Backend.hpp"
#include "core/Execution.hpp"

namespace MNN {

class CoDLRaster : public Execution {
public:
    CoDLRaster(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);

    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ~CoDLRaster() = default;
    ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    const Op *mOP;
    CoDLBackend *mBackend;
    std::shared_ptr<Execution> mCPURaster;
    std::shared_ptr<Execution> mOCLRaster;
};

}

#endif
