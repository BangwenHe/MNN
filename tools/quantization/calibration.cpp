//
//  calibration.cpp
//  MNN
//
//  Created by MNN on 2019/04/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "calibration.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <set>
#include <algorithm>
#include <MNN/ImageProcess.hpp>
#include "flatbuffers/util.h"
#include "logkit.h"
#include "quantizeWeight.hpp"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "Helper.hpp"
#include "core/TensorUtils.hpp"
#include "cpp/IDSTEncoder.hpp"

#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Module.hpp>
#include "train/source/nn/NN.hpp"
#include "train/source/datasets/ImageNoLabelDataset.hpp"
#include "train/source/datasets/ImageDataset.hpp"
#include "train/source/optimizer/SGD.hpp"
#include "train/source/transformer/Transformer.hpp"
#include "cpp/ConvertToFullQuant.hpp"
#include "core/ConvolutionCommon.hpp"
#include <MNN/expr/Expr.hpp>
#include "half.hpp"
#include "core/MNNMemoryUtils.h"

using namespace MNN::CV;
using namespace MNN::Train;
using namespace MNN::Express;

static inline void *MNNMemoryAllocAlignZeroAlign(size_t size) {
    return MNNMemoryCallocAlign(size, MNN_MEMORY_ALIGN_DEFAULT);
}
static int ReadBlobDim(unsigned char *&myfile, unsigned int* shape, int shapeBufCnt, bool useInt32) {
    int uSize = myfile[0];
    myfile++;
    if (uSize > 4) {
        printf("Read shape error!\n");
        return 0;
    }
    int copyLength = uSize;
    if (copyLength > shapeBufCnt) {
        copyLength = shapeBufCnt;
    }
    if (useInt32) {
        ::memcpy(shape, myfile, sizeof(unsigned int) * copyLength);
        myfile += copyLength * sizeof(unsigned int);
    } else {
        auto myfileint16 = (uint16_t*)myfile;
        for (int i=0; i<copyLength; ++i) {
            shape[i] = myfileint16[i];
        }
        myfile += copyLength * sizeof(unsigned short);
    }
    return copyLength;
}

static double _log2(double x) {
    return log(x) / log(2);
}

static uint32_t atLestBitsCnt(uint32_t n) {
    for (uint32_t i = 0; i < 32; i++) {
        int32_t t = n << i;
        if (t < 0)
            return 32 - i - (((t << 1) == 0) ? 1 : 0);
    }
    return 0;
}

static void SplitBufToArray(uint8_t *buf, size_t bufLen, uint8_t *arr, size_t arrLen, size_t iNeedBits) {
    unsigned char cMask = (1 << (iNeedBits)) - 1;
    unsigned char *tmp  = (unsigned char *)buf;
    int iOffset         = 0;
    for (unsigned int i = 0; i < arrLen; i++) {
        unsigned char idx = 0;
        long uShift       = 8 - iNeedBits - iOffset % 8;
        if (uShift < 0) {
            idx = (tmp[iOffset / 8] << (0 - uShift)) & cMask;
            idx |= (tmp[(iOffset / 8) + 1] >> (8 + uShift)) & cMask;
        } else {
            idx = (tmp[iOffset / 8] >> uShift) & cMask;
        }
        iOffset += iNeedBits;
        if (iOffset % 8 == 0) {
            tmp += iOffset / 8;
            iOffset = 0;
        }
        arr[i] = idx;
    }
}

// fixme!!! not efficiency
typedef struct _SIMPLE_SET {
    int8_t *UniSet;
    uint32_t UniSetSize;
    uint32_t CurUniCnt;
} SIMPLE_SET, *PSIMPLE_SET;

static PSIMPLE_SET CreateSimpleSet(uint32_t maxSize) {
    PSIMPLE_SET set = (PSIMPLE_SET)calloc(1, sizeof(SIMPLE_SET));
    if (set == nullptr)
        return nullptr;
    set->UniSet     = (int8_t *)calloc(maxSize, sizeof(int8_t));
    set->UniSetSize = maxSize;
    set->CurUniCnt  = 0;
    return set;
}

static void SimpleRank(int8_t *data, uint32_t cnt, int up) {
    if (up) {
        for (uint32_t i = 0; i < cnt; i++) {
            for (uint32_t j = i + 1; j < cnt; j++) {
                if (data[i] > data[j]) {
                    int8_t tmp = data[i];
                    data[i]    = data[j];
                    data[j]    = tmp;
                }
            }
        }
    } else {
        for (uint32_t i = 0; i < cnt; i++) {
            for (uint32_t j = i + 1; j < cnt; j++) {
                if (data[i] < data[j]) {
                    int8_t tmp = data[i];
                    data[i]    = data[j];
                    data[j]    = tmp;
                }
            }
        }
    }
}

static void InsertSimpleSet(PSIMPLE_SET set, int8_t value) {
    if (set->CurUniCnt >= set->UniSetSize)
        return;
    for (uint32_t i = 0; i < set->CurUniCnt; i++) {
        if (set->UniSet[i] == value)
            return;
    }
    set->UniSet[set->CurUniCnt++] = value;
    //    SimpleRank(set->UniSet, set->CurUniCnt, 1);
}

void DestorySimpleSet(PSIMPLE_SET set) {
    if (set->UniSet != nullptr)
        free(set->UniSet);
    free(set);
}

typedef struct _SIMPLE_MAP {
    int8_t *CharCharMap;
    uint32_t CharMapSize;
    uint32_t CurMapCnt;
} SIMPLE_MAP, *PSIMPLE_MAP;

static PSIMPLE_MAP CreateSimpleMap(uint32_t MaxCnt) {
    PSIMPLE_MAP map = (PSIMPLE_MAP)calloc(1, sizeof(SIMPLE_MAP));
    if (map == nullptr)
        return nullptr;
    map->CharMapSize = MaxCnt * sizeof(int8_t);
    map->CurMapCnt   = 0;
    map->CharCharMap = (int8_t *)calloc(1, MaxCnt * 2);
    return map;
}

static void DestroySimpleMap(PSIMPLE_MAP map) {
    if (map->CharCharMap)
        free(map->CharCharMap);
    free(map);
}

static void InsertMap(PSIMPLE_MAP map, int8_t k, int8_t v) {
    for (uint32_t i = 0; i < map->CurMapCnt; i++) {
        if (map->CharCharMap[i * 2] == k) {
            map->CharCharMap[i * 2 + 1] = v;
            return;
        }
    }
    if (map->CurMapCnt >= map->CharMapSize)
        return;
    map->CharCharMap[map->CurMapCnt * 2]     = k;
    map->CharCharMap[map->CurMapCnt * 2 + 1] = v;
    map->CurMapCnt++;
}

static int8_t FindInMap(PSIMPLE_MAP map, int8_t k, int *found) {
    for (uint32_t i = 0; i < map->CurMapCnt; i++) {
        if (map->CharCharMap[i * 2] == k) {
            if (found != nullptr)
                *found = 1;
            return map->CharCharMap[i * 2 + 1];
        }
    }
    if (found != nullptr)
        *found = 0;
    return 0;
}

static void StreamSizeRead(void *dst, int unit, size_t count, unsigned char *&file) {
    ::memcpy(dst, file, unit * count);
    file += (unit * count);
}

static bool isLinearSample(const std::vector<int8_t>& sample, int bit) {
    const int offset = 1 << (bit - 1);
    const int size = 1 << bit;
    if (sample.size() != size) {
        return false;
    }
    for (int i = 0; i < sample.size(); i++) {
        if (static_cast<int>(sample[i]) != i - offset) {
            return false;
        }
    }
    return true;
}


static int8_t *ReadQuanData_c(unsigned char *&s, size_t* len, ConvolutionCommon::Int8Common* result, bool shapeInt32) {
    int8_t *blob      = nullptr;
    uint8_t *idxBuf   = nullptr;
    uint8_t *idxBytes = nullptr;
    uint32_t dataCnt  = 1;

    do {
        // blob shape
        unsigned int shape[32] = {0};
        uint32_t shapeDim        = (uint32_t)ReadBlobDim(s, shape, 32, shapeInt32);
        if (shapeDim == 0 || shapeDim > 32)
            break;
        for (uint32_t i = 0; i < shapeDim; i++)
            dataCnt *= shape[i];

        // sample
        uint32_t sampleCnt = 0;
        StreamSizeRead(&sampleCnt, 1, 1, s);
        if (sampleCnt == 0) {
            sampleCnt = 256;
        }
        result->weightMap.resize(sampleCnt);
        auto samples = result->weightMap.data();
        if (samples == nullptr)
            break;
        StreamSizeRead(samples, 1, sampleCnt, s);
        SimpleRank(samples, sampleCnt, 1);
        uint32_t idxBitsCnt = atLestBitsCnt(sampleCnt);
        idxBitsCnt = idxBitsCnt < 1 ? 1 : idxBitsCnt;
        // index
        size_t idxBufSize   = ceil(idxBitsCnt * dataCnt * 0.125);
        idxBuf              = (uint8_t *)MNNMemoryAllocAlignZeroAlign(idxBufSize);
        if (nullptr == idxBuf) {
            MNN_ERROR("Not enought memory\n");
            break;
        }
        StreamSizeRead(idxBuf, 1, idxBufSize, s);
        blob  = (int8_t *)MNNMemoryAllocAlignZeroAlign((size_t)dataCnt);
        if (nullptr == blob) {
            break;
        }

        if (isLinearSample(result->weightMap, idxBitsCnt) && (idxBitsCnt == 4 || idxBitsCnt == 8)) {
            // fast sample for bit = 4 or 8
            if (idxBitsCnt == 4) {
                for (int i = 0; i < idxBufSize; i++) {
                    int val = idxBuf[i];
                    int x1 = val / 16;
                    int x2 = val % 16;
                    blob[2 * i] = x1 - 8;
                    blob[2 * i + 1] = x2 - 8;
                }
            }
            if (idxBitsCnt == 8) {
                for (int i = 0; i < idxBufSize; i++) {
                    int val = idxBuf[i];
                    blob[i] = val - 64;
                }
            }
        } else {
            // split index value into bytes
            idxBytes = (uint8_t *)MNNMemoryAllocAlignZeroAlign(dataCnt * sizeof(uint8_t));
            if (idxBitsCnt == 0 || nullptr == idxBytes) {
                break;
            }
            SplitBufToArray(idxBuf, (uint32_t)idxBufSize, idxBytes, (uint32_t)dataCnt, (uint32_t)idxBitsCnt);
            int i = 0;
            for (; i < dataCnt; i++) {
                if (idxBytes[i] >= sampleCnt) {
                    MNN_PRINT("iNeedBits is %u\nRead quan weights error with idx:%d\n", idxBitsCnt, (int)idxBytes[i]);
                    break;
                }
                blob[i] = samples[idxBytes[i]];
            }

            if (i < dataCnt) {
                MNNMemoryFreeAlign(blob);
                blob = nullptr;
                break;
            }
            MNNMemoryFreeAlign(idxBytes);
            idxBytes = nullptr;
        }
    } while (0);

    if (idxBuf != nullptr)
        MNNMemoryFreeAlign(idxBuf);
    if (idxBytes != nullptr)
        MNNMemoryFreeAlign(idxBytes);
    if (len)
        *len = blob ? dataCnt : 0;
    return blob;
}

static int8_t *ReadSparseQuanData_c(unsigned char *&myfile, size_t* len, const float* alpha_ptr, size_t alpha_size, ConvolutionCommon::Int8Common* result, bool useInt32) {    // MNN_ERROR("sparse:%d\n", 1);
    unsigned int shape[32];
    uint32_t ucMapSize = 0;
    PSIMPLE_SET setWeight = CreateSimpleSet(256);
    if (setWeight == nullptr) {
        return nullptr;
    }
    std::shared_ptr<unsigned int> __autoReleaseSetWeight(nullptr, [setWeight](void *) { DestorySimpleSet(setWeight); });
    unsigned int nnz;
    unsigned char iIdxNeedBits;
    int8_t *blob = nullptr;
    // 1. weights blob shape(unsigned int32)
    int ShapeDim = ReadBlobDim(myfile, shape, 32, useInt32);
    size_t Size     = sizeof(int8_t);
    for (int i = 0; i < ShapeDim; i++)
        Size *= shape[i];
    blob = (int8_t *)MNNMemoryAllocAlignZeroAlign((size_t)Size);
    if (blob == nullptr)
        return nullptr;
    // 2. nnz
    StreamSizeRead(&nnz, 4, 1, myfile);
    // 3. max_step use # bits () (unsigned char)
    StreamSizeRead(&iIdxNeedBits, 1, 1, myfile);
    // read idx array
    // 4. buf for steps ceil(nnz*step need bits/8)
    AutoStorage<unsigned char> arrIdxBuffer(nnz);
    unsigned char *arrIdx = arrIdxBuffer.get();
    if (nullptr == arrIdx) {
        return nullptr;
    }
    {
        size_t bufLen = (size_t)(ceil(0.125 * iIdxNeedBits * nnz));
        char *buf     = (char *)MNNMemoryAllocAlignZeroAlign(bufLen * sizeof(char));
        if (nullptr == buf) {
            return nullptr;
        }
        StreamSizeRead(buf, 1, bufLen, myfile);
        SplitBufToArray((uint8_t *)buf, (uint32_t)bufLen, (uint8_t *)arrIdx, (uint32_t)nnz, (uint32_t)iIdxNeedBits);
        MNNMemoryFreeAlign(buf);
    }
    // 5. Avalable values Count(unsigned char)
    StreamSizeRead(&ucMapSize, 1, 1, myfile);
    if (0 == ucMapSize) {
        ucMapSize = 256;
    }
    result->weightMap.resize(ucMapSize);
    // 6. valueset(signed char * valueset_size)
    for (int i = 0; i < ucMapSize; i++) {
        int8_t tmp;
        StreamSizeRead(&tmp, 1, 1, myfile);
        InsertSimpleSet(setWeight, tmp);
        result->weightMap[i] = tmp;
    }
    SimpleRank(setWeight->UniSet, setWeight->CurUniCnt, 1);
    // map<unsigned char, signed char> mapWeight;
    PSIMPLE_MAP mapWeight = CreateSimpleMap(256);
    if (mapWeight == nullptr) {
        return nullptr;
    }
    std::shared_ptr<unsigned int> __autoReleaseMapWeight(nullptr, [mapWeight](void *) { DestroySimpleMap(mapWeight); });

    for (int i = 0; i < setWeight->CurUniCnt; i++) {
        InsertMap(mapWeight, i, setWeight->UniSet[i]);
    }
    //    unsigned char iIdx = 0;
    // 7. none zero weights indexes(nnz*ceil(log2(Avalable_values_Count))/8)
    AutoStorage<unsigned char> arrWeightIdxBuffer(nnz);
    unsigned char *arrWeightIdx = arrWeightIdxBuffer.get();
    if (nullptr == arrWeightIdx) {
        return nullptr;
    }
    int iDataNeedBits = (int)ceil(_log2(ucMapSize));
    iDataNeedBits = iDataNeedBits < 1 ? 1 : iDataNeedBits;
    {
        size_t bufLen     = (size_t)(ceil(0.125 * iDataNeedBits * nnz));
        char *buf         = (char *)MNNMemoryAllocAlignZeroAlign(bufLen * sizeof(char));
        if (nullptr == buf) {
            return nullptr;
        }
        StreamSizeRead(buf, 1, bufLen, myfile);
        SplitBufToArray((uint8_t *)buf, (uint32_t)bufLen, (uint8_t *)arrWeightIdx, (uint32_t)nnz,
                        (uint32_t)iDataNeedBits);
        MNNMemoryFreeAlign(buf);
    }
    // set blob data with idx and weight idx
    {
        if (alpha_size == 2 * shape[0]) {
            const int min_value = -(1 << (iDataNeedBits - 1));
            auto alphaPtr = alpha_ptr;
            int area = Size / shape[0];
            for (int i = 0; i < shape[0]; i++) {
                float min = alphaPtr[2*i];
                float scale = alphaPtr[2*i+1];
                int zeroQuant = min_value;
                if (scale > 1e-6) {
                    zeroQuant = round((0.0f - min) / scale) + min_value;
                }
                memset(blob+area*i, zeroQuant, area * sizeof(signed char));
            }
        } else {
            memset(blob, 0, Size * sizeof(signed char)); //backward compability with previous symmetric weight quant
        }
        int iPreIdx = 0;
        for (int i = 0; i < nnz; i++) {
            iPreIdx += arrIdx[i];
            int found    = 0;
            int8_t value = FindInMap(mapWeight, arrWeightIdx[i], &found);
            if (!found) {
                MNN_ERROR("Read quan weights error with idx:%d\n", arrWeightIdx[i]);
                MNNMemoryFreeAlign(blob);
                return nullptr;
            }
            blob[iPreIdx] = value;
        }
    }
    *len = Size;
    return blob;
}

#define DUMP_NUM_DATA(type)                          \
    auto data = tensor->host<type>();                \
    for (int z = 0; z < outside; ++z) {              \
        for (int x = 0; x < width; ++x) {            \
            outputOs << data[x + z * width] << "\t"; \
        }                                            \
        outputOs << "\n";                            \
    }

#define DUMP_CHAR_DATA(type)                                           \
    auto data = tensor->host<type>();                                  \
    for (int z = 0; z < outside; ++z) {                                \
        for (int x = 0; x < width; ++x) {                              \
            outputOs << static_cast<int>(data[x + z * width]) << "\t"; \
        }                                                              \
        outputOs << "\n";                                              \
    }


static void dumpTensor2File(const Tensor* tensor, const std::string& name) {
    auto opCopyName = name;
    for (int j = 0; j < opCopyName.size(); ++j) {
        if (opCopyName[j] == '/') {
            opCopyName[j] = '_';
        }
    }

    std::ofstream outputOs(opCopyName);
    auto type = tensor->getType();

    int dimension = tensor->buffer().dimensions;
    int width     = 1;
    if (dimension > 1) {
        width = tensor->length(dimension - 1);
    }

    const int outside = tensor->elementSize() / width;

    const auto dataType  = type.code;
    const auto dataBytes = type.bytes();

    if (dataType == halide_type_float) {
        DUMP_NUM_DATA(float);
    }
    if (dataType == halide_type_int && dataBytes == 4) {
        DUMP_NUM_DATA(int32_t);
    }
    if (dataType == halide_type_uint && dataBytes == 1) {
        DUMP_CHAR_DATA(uint8_t);
    }
    if (dataType == halide_type_int && dataBytes == 1) {
#ifdef MNN_USE_SSE
        auto data = tensor->host<uint8_t>();
        for (int z = 0; z < outside; ++z) {
            for (int x = 0; x < width; ++x) {
                outputOs << (static_cast<int>(data[x + z * width]) - 128) << "\t";
            }
            outputOs << "\n";
        }
#else
        DUMP_CHAR_DATA(int8_t);
#endif
    }
}



Calibration::Calibration(MNN::NetT* model, MNN::NetT* halfModel, const uint8_t* modelBuffer, const int bufferSize, const std::string& configPath, std::string originalModelFile, std::string destModelFile)
    : _originalModel(model), _halfModel(halfModel), _originalModelFile(originalModelFile), _destModelFile(destModelFile) {
    // when the format of input image is RGB/BGR, channels equal to 3, GRAY is 1
    _channels = 3;

    rapidjson::Document document;
    {
        std::ifstream fileNames(configPath.c_str());
        std::ostringstream output;
        output << fileNames.rdbuf();
        auto outputStr = output.str();
        document.Parse(outputStr.c_str());
        if (document.HasParseError()) {
            MNN_ERROR("Invalid json\n");
            mValid = false;
            return;
        }
    }
    auto picObj = document.GetObject();
    _imageProcessConfig.filterType = CV::BILINEAR;
    _imageProcessConfig.destFormat = BGR;
    {
        if (picObj.HasMember("format")) {
            auto format = picObj["format"].GetString();
            static std::map<std::string, ImageFormat> formatMap{{"BGR", BGR}, {"RGB", RGB}, {"GRAY", GRAY}, {"RGBA", RGBA}, {"BGRA", BGRA}};
            if (formatMap.find(format) != formatMap.end()) {
                _imageProcessConfig.destFormat = formatMap.find(format)->second;
            }
        }
    }

    switch (_imageProcessConfig.destFormat) {
        case GRAY:
            _channels = 1;
            break;
        case RGB:
        case BGR:
            _channels = 3;
            break;
        case RGBA:
        case BGRA:
            _channels = 4;
            break;
        default:
            break;
    }

    _imageProcessConfig.sourceFormat = RGBA;
    _calibrationFileNum = 0;
    {
        if (picObj.HasMember("mean")) {
            auto mean = picObj["mean"].GetArray();
            int cur   = 0;
            for (auto iter = mean.begin(); iter != mean.end(); iter++) {
                _imageProcessConfig.mean[cur++] = iter->GetFloat();
            }
        }
        if (picObj.HasMember("normal")) {
            auto normal = picObj["normal"].GetArray();
            int cur     = 0;
            for (auto iter = normal.begin(); iter != normal.end(); iter++) {
                _imageProcessConfig.normal[cur++] = iter->GetFloat();
            }
        }
        if (picObj.HasMember("center_crop_h")) {
            _preprocessConfig.centerCropHeight = picObj["center_crop_h"].GetFloat();
        }
        if (picObj.HasMember("center_crop_w")) {
            _preprocessConfig.centerCropWidth = picObj["center_crop_w"].GetFloat();
        }
        if (picObj.HasMember("width")) {
            _width = picObj["width"].GetInt();
            _preprocessConfig.targetWidth = _width;
        }
        if (picObj.HasMember("height")) {
            _height = picObj["height"].GetInt();
            _preprocessConfig.targetHeight = _height;
        }
        if (picObj.HasMember("batch_size")) {
            _batch = picObj["batch_size"].GetInt();
        }
        if (picObj.HasMember("quant_bits")) {
            _quant_bits = picObj["quant_bits"].GetInt();
        }
        if (!picObj.HasMember("path")) {
            MNN_ERROR("calibration data path not set in .json config file\n");
            return;
        }
        _calibrationFilePath = picObj["path"].GetString();
        if (picObj.HasMember("used_image_num")) {
            _calibrationFileNum = picObj["used_image_num"].GetInt();
        }
        if (picObj.HasMember("used_sample_num")) {
            _calibrationFileNum = picObj["used_sample_num"].GetInt();
        }
        if (picObj.HasMember("feature_quantize_method")) {
            std::string method = picObj["feature_quantize_method"].GetString();
            if (Helper::featureQuantizeMethod.find(method) != Helper::featureQuantizeMethod.end()) {
                _featureQuantizeMethod = method;
            } else {
                MNN_ERROR("not supported feature quantization method: %s\n", method.c_str());
                return;
            }
        }
        if (picObj.HasMember("weight_quantize_method")) {
            std::string method = picObj["weight_quantize_method"].GetString();
            if (Helper::weightQuantizeMethod.find(method) != Helper::weightQuantizeMethod.end()) {
                _weightQuantizeMethod = method;
            } else {
                MNN_ERROR("not supported weight quantization method: %s\n", method.c_str());
                return;
            }
        }
        DLOG(INFO) << "Use feature quantization method: " << _featureQuantizeMethod;
        DLOG(INFO) << "Use weight quantization method: " << _weightQuantizeMethod;
        if (picObj.HasMember("feature_clamp_value")) {
            float value = (int)picObj["feature_clamp_value"].GetFloat();
            if (value < 0.0f || value > 127.0f) {
                MNN_ERROR("feature_clamp_value should be in (0, 127], got: %f\n", value);
                return;
            }
            _featureClampValue = value;
        }
        if (picObj.HasMember("weight_clamp_value")) {
            float value = (int)picObj["weight_clamp_value"].GetFloat();
            if (value < 0.0f || value > 127.0f) {
                MNN_ERROR("weight_clamp_value should be in (0, 127], got: %f\n", value);
                return;
            }
            _weightClampValue = value;
            if (_quant_bits < 8) {
                _weightClampValue = (float)(1 << (_quant_bits - 1)) - 1.0f;
            }
        }
        DLOG(INFO) << "feature_clamp_value: " << _featureClampValue;
        DLOG(INFO) << "weight_clamp_value: " << _weightClampValue;
        if (picObj.HasMember("winogradOpt") && picObj["winogradOpt"].GetBool() == true) {
            if (_featureQuantizeMethod == "EMA") {
                _winogradOpt = true;
            } else {
                DLOG(ERROR) << "winogradOpt only be available under EMA";
            }
        }
        if (picObj.HasMember("skip_quant_op_names")) {
            auto skip_quant_op_names = picObj["skip_quant_op_names"].GetArray();
            for (auto iter = skip_quant_op_names.begin(); iter != skip_quant_op_names.end(); iter++) {
                std::string skip_quant_op_name = iter->GetString();
                _skip_quant_ops.emplace_back(skip_quant_op_name);
                DLOG(INFO) << "skip quant op name: " << skip_quant_op_name;
            }
        }
        if (picObj.HasMember("debug")) {
            _debug = picObj["debug"].GetBool();
        }
        _inputType = Helper::InputType::IMAGE;
        if (picObj.HasMember("input_type")) {
            std::string type = picObj["input_type"].GetString();
            if (type == "sequence") {
                _inputType = Helper::InputType::SEQUENCE;
            }
        }
    }
    std::shared_ptr<ImageProcess> process(ImageProcess::create(_imageProcessConfig), ImageProcess::destroy);
    _process = process;

    // read images file names
    Helper::readClibrationFiles(_calibrationFiles, _calibrationFilePath.c_str(), &_calibrationFileNum);

    for (auto& op : _originalModel->oplists) {
        if (op->type == MNN::OpType_BatchNorm) {
            _featureQuantizeMethod = "EMA";
            DLOG(INFO) << "this model has BatchNorm, use EMA quantize method instead";
            break;
        }
    }
    for (auto& subgraph : _originalModel->subgraphs) {
        for (auto& op : subgraph->nodes) {
            if (op->type == MNN::OpType_BatchNorm) {
                _featureQuantizeMethod = "EMA";
                DLOG(INFO) << "this model has BatchNorm, use EMA quantize method instead";
                break;
            }
        }
    }

    if (_featureQuantizeMethod == "KL" || _featureQuantizeMethod == "ADMM") {
        _initMNNSession(modelBuffer, bufferSize);
        _initMaps();
    }
}

std::vector<int> Calibration::_getInputShape(std::string filename) {
    std::vector<int> inputShape;
    if (_inputType == Helper::InputType::IMAGE) {
        inputShape.resize(4);
        auto inputTensorDataFormat = MNN::TensorUtils::getDescribe(_inputTensor)->dimensionFormat;
        if (inputTensorDataFormat == MNN::MNN_DATA_FORMAT_NHWC) {
            inputShape[0] = 1;
            inputShape[1] = _height;
            inputShape[2] = _width;
            inputShape[3] = _channels;
        } else {
            inputShape[0] = 1;
            inputShape[1] = _channels;
            inputShape[2] = _height;
            inputShape[3] = _width;
        }
    }
    if (_inputType == Helper::InputType::SEQUENCE) {
        if (!Helper::stringEndWith(filename, ".txt")) {
            MNN_ERROR("Error: only '.txt' files are supported for sequence input.\n");
        }

        std::ifstream f(filename);
        if (!f.is_open()) {
            MNN_ERROR("open file %s failed.\n", filename.c_str());
        }

        std::string line;
        _channels = 0;
        while (std::getline(f, line)) {
            std::stringstream ss(line);
            float v;
            int count = 0;
            while (ss >> v) {
                count++;
            }
            if (count > 0) {
                _channels++;
                _height = count;
            }
        }

        if (_channels == 0) {
            MNN_ERROR("Error: no data found in file %s.", filename.c_str());
        }

        inputShape.resize(3);
        auto inputTensorDataFormat = MNN::TensorUtils::getDescribe(_inputTensor)->dimensionFormat;
        if (inputTensorDataFormat == MNN::MNN_DATA_FORMAT_NHWC) {
            inputShape[0] = 1;
            inputShape[1] = _height;
            inputShape[2] = _channels;
        } else {
            inputShape[0] = 1;
            inputShape[1] = _channels;
            inputShape[2] = _height;
        }
    }

    return inputShape;
}

void Calibration::_resizeIfNeeded(std::string filename, bool force) {
    std::vector<int> inputShape = _getInputShape(filename);
    
    if ((inputShape != _inputTensorDims && _featureQuantizeMethod == "KL") || force) {
        _inputTensorDims = inputShape;
        _interpreter->resizeTensor(_inputTensor, _inputTensorDims);
        _interpreter->resizeSession(_session);
        _interpreterOrigin->resizeTensor(_inputTensorOrigin, _inputTensorDims);
        _interpreterOrigin->resizeSession(_sessionOrigin);
        // _interpreterHalf->resizeTensor(_inputTensorHalf, _inputTensorDims);
        // _interpreterHalf->resizeSession(_sessionHalf);
    }
}

void Calibration::_initMNNSession(const uint8_t* modelBuffer, const int bufferSize) {
    _interpreterOrigin.reset(MNN::Interpreter::createFromBuffer(modelBuffer, bufferSize), MNN::Interpreter::destroy);
    MNN::ScheduleConfig config;
    MNN::BackendConfig backendConfig;
    backendConfig.memory = MNN::BackendConfig::Memory_High;
    config.backendConfig = &backendConfig;
    // 如果是以debug模式编译，则线程数为1
#ifdef DEBUG
    config.numThread = 1;
#endif
    _sessionOrigin     = _interpreterOrigin->createSession(config);
    _inputTensorOrigin = _interpreterOrigin->getSessionInput(_sessionOrigin, NULL);

    _fake_quant_weights();

    flatbuffers::FlatBufferBuilder builder(1024);
    auto offset = MNN::Net::Pack(builder, _originalModel);
    builder.Finish(offset);
    int size      = builder.GetSize();
    auto buffer = builder.GetBufferPointer();

    _interpreter.reset(MNN::Interpreter::createFromBuffer(buffer, size),  MNN::Interpreter::destroy);
    _session     = _interpreter->createSession(config);
    _inputTensor = _interpreter->getSessionInput(_session, NULL);

    if (_featureQuantizeMethod == "ADMM") {
        DCHECK((_calibrationFileNum * 4 * _height * _width) < (INT_MAX / 4)) << "Use Little Number of Images When Use ADMM";
        for (auto file : _calibrationFiles) {
            std::vector<int> sampleShape = _getInputShape(file);
            if (_inputTensorDims.empty()) {
                _inputTensorDims = sampleShape;
            }
            if (sampleShape != _inputTensorDims) {
                MNN_ERROR("samples must have the same shape when using ADMM method for sequence inputs.");
            }
        }
        _inputTensorDims[0] = _calibrationFileNum;
        _interpreter->resizeTensor(_inputTensor, _inputTensorDims);
        _interpreter->resizeSession(_session);
        _interpreterOrigin->resizeTensor(_inputTensorOrigin, _inputTensorDims);
        _interpreterOrigin->resizeSession(_sessionOrigin);
    }


    // _interpreterHalf.reset(MNN::Interpreter::createFromBuffer(modelBuffer, bufferSize), MNN::Interpreter::destroy);
    // _sessionHalf     = _interpreterHalf->createSession(config);
    // _inputTensorHalf = _interpreterHalf->getSessionInput(_sessionHalf, NULL);

    _resizeIfNeeded(_calibrationFiles[0]);
}

void Calibration::_initMaps() {
    _featureInfo.clear();
    _featureInfoOrigin.clear();
    _featureInfoHalf.clear();
    _tensorMap.clear();
        // run mnn once, initialize featureMap, opInfo map
    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        std::string opName = info->name();
        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), opName);
        if (iter != _skip_quant_ops.end()) {
            return false;
        }
        for (auto t : nTensors) {
            auto des = TensorUtils::getDescribe(t);
            if (des->index >= 0) {
                _tensorMap[des->index] = t;
            }
        }
        if (Helper::gNotNeedFeatureOp.find(info->type()) == Helper::gNotNeedFeatureOp.end()) {
            int i = 0;
            for (auto t : nTensors) {
                if (_featureInfo.find(t) == _featureInfo.end() && MNN::TensorUtils::getDescribe(t)->memoryType != MNN::Tensor::InsideDescribe::MEMORY_VIRTUAL) {
                    _featureInfo[t] = std::shared_ptr<TensorStatistic>(
                        new TensorStatistic(t, _featureQuantizeMethod, opName + " input_tensor_" + flatbuffers::NumToString(i), _featureClampValue));
                }
                i++;
            }
        }
        return false;
    };
    MNN::TensorCallBackWithInfo after = [this](const std::vector<MNN::Tensor*>& nTensors,
                                               const MNN::OperatorInfo* info) {
        std::string opName = info->name();
        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), opName);
        if (iter != _skip_quant_ops.end()) {
            return true;
        }
        for (auto t : nTensors) {
            auto des = TensorUtils::getDescribe(t);
            if (des->index >= 0) {
                _tensorMap[des->index] = t;
            }
        }
        if (Helper::gNotNeedFeatureOp.find(info->type()) == Helper::gNotNeedFeatureOp.end()) {
            int i = 0;
            for (auto t : nTensors) {
                if (_featureInfo.find(t) == _featureInfo.end()) {
                    _featureInfo[t] =
                        std::shared_ptr<TensorStatistic>(new TensorStatistic(t, _featureQuantizeMethod, opName + " output_tensor_" + flatbuffers::NumToString(i), _featureClampValue));
                }
                i++;
            }
        }
        return true;
    };
    _interpreter->runSessionWithCallBackInfo(_session, before, after);


    MNN::TensorCallBackWithInfo beforeOrigin = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        std::string opName = info->name();
        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), opName);
        if (iter != _skip_quant_ops.end()) {
            return false;
        }
        if (Helper::gNotNeedFeatureOp.find(info->type()) == Helper::gNotNeedFeatureOp.end()) {
            int i = 0;
            for (auto t : nTensors) {
                if (_featureInfoOrigin.find(t) == _featureInfoOrigin.end()) {
                    _featureInfoOrigin[t] = std::shared_ptr<TensorStatistic>(
                        new TensorStatistic(t, _featureQuantizeMethod, opName + " input_tensor_" + flatbuffers::NumToString(i), _featureClampValue));
                }
                i++;
            }
        }
        return false;
    };
    MNN::TensorCallBackWithInfo afterOrigin = [this](const std::vector<MNN::Tensor*>& nTensors,
                                               const MNN::OperatorInfo* info) {
        std::string opName = info->name();
        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), opName);
        if (iter != _skip_quant_ops.end()) {
            return true;
        }
        if (Helper::gNotNeedFeatureOp.find(info->type()) == Helper::gNotNeedFeatureOp.end()) {
            int i = 0;
            for (auto t : nTensors) {
                if (_featureInfoOrigin.find(t) == _featureInfoOrigin.end()) {
                    _featureInfoOrigin[t] =
                        std::shared_ptr<TensorStatistic>(new TensorStatistic(t, _featureQuantizeMethod, opName + " output_tensor_" + flatbuffers::NumToString(i), _featureClampValue));
                }
                i++;
            }
        }
        return true;
    };
    _interpreterOrigin->runSessionWithCallBackInfo(_sessionOrigin, beforeOrigin, afterOrigin);

    if (_featureQuantizeMethod == "KL") {
        // set the tensor-statistic method of input tensor as THRESHOLD_MAX
        auto inputTensorStatistic = _featureInfo.find(_inputTensor);
        if (inputTensorStatistic != _featureInfo.end()) {
            inputTensorStatistic->second->setThresholdMethod(THRESHOLD_MAX);
        }
    }
}

void Calibration::_computeFeatureMapsRange() {
    // feed input data according to input images
    int count = 0;
    for (const auto& file : _calibrationFiles) {
        for (auto& iter : _featureInfo) {
            iter.second->setVisited(false);
        }

        for (auto& iter : _featureInfo) {
            iter.second->resetUpdatedRangeFlags();
        }
        count++;
        _resizeIfNeeded(file);
        Helper::preprocessInput(_process.get(), _preprocessConfig, file, _inputTensor, _inputType);

        MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                 const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    if (_featureInfo[t]->visited() == false) {
                        _featureInfo[t]->updateRange();
                    }
                }
            }
            return true;
        };
        MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    if (_featureInfo[t]->visited() == false) {
                        _featureInfo[t]->updateRange();
                    }
                }
            }
            return true;
        };

        _interpreter->runSessionWithCallBackInfo(_session, before, after);
        MNN_PRINT("\rComputeFeatureRange: %.2lf %%", (float)count * 100.0f / (float)_calibrationFileNum);
        fflush(stdout);
    }
    MNN_PRINT("\n");
}

void Calibration::_collectFeatureMapsDistribution() {
    for (auto& iter : _featureInfo) {
        iter.second->resetDistribution();
    }
    // feed input data according to input images
    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        for (auto t : nTensors) {
            if (_featureInfo.find(t) != _featureInfo.end()) {
                if (_featureInfo[t]->visited() == false) {
                    _featureInfo[t]->updateDistribution();
                }
            }
        }
        return true;
    };
    MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        for (auto t : nTensors) {
            if (_featureInfo.find(t) != _featureInfo.end()) {
                if (_featureInfo[t]->visited() == false) {
                    _featureInfo[t]->updateDistribution();
                }
            }
        }
        return true;
    };
    int count = 0;
    for (const auto& file : _calibrationFiles) {
        count++;

        for (auto& iter : _featureInfo) {
            iter.second->setVisited(false);
        }

        for (auto& iter : _featureInfo) {
            iter.second->resetUpdatedDistributionFlag();
        }
        _resizeIfNeeded(file);
        Helper::preprocessInput(_process.get(), _preprocessConfig, file, _inputTensor, _inputType);
        _interpreter->runSessionWithCallBackInfo(_session, before, after);

        MNN_PRINT("\rCollectFeatureDistribution: %.2lf %%", (float)count * 100.0f / (float)_calibrationFileNum);
        fflush(stdout);
    }
    MNN_PRINT("\n");
}

void Calibration::_computeFeatureScaleKL() {
    _computeFeatureMapsRange();
    _collectFeatureMapsDistribution();

    _scales.clear();
    for (auto& iter : _featureInfo) {
        AUTOTIME;
        _scales[iter.first] = iter.second->finishAndCompute();
    }
    //_featureInfo.clear();//No need now
}

void Calibration::_computeFeatureScaleADMM() {
    // feed input data according to input images
    int count                           = 0;
    std::vector<int> oneImageTensorDims = _inputTensorDims;
    oneImageTensorDims[0]               = 1;
    auto inputTensorDataFormat          = MNN::TensorUtils::getDescribe(_inputTensor)->dimensionFormat;
    auto dimType                        = MNN::Tensor::CAFFE_C4;
    if (inputTensorDataFormat == MNN::MNN_DATA_FORMAT_NHWC) {
        dimType = MNN::Tensor::TENSORFLOW;
    }

    for (const auto& file : _calibrationFiles) {
        auto curPtr = _inputTensor->host<float>() + count * _inputTensor->stride(0);
        std::shared_ptr<MNN::Tensor> tensorWarp(
            MNN::Tensor::create(oneImageTensorDims, _inputTensor->getType(), curPtr, dimType), MNN::Tensor::destroy);
        Helper::preprocessInput(_process.get(), _preprocessConfig, file, tensorWarp.get(), _inputType);

        count++;
        MNN_PRINT("\rProcessCalibrationFiles: %.2lf %%", (float)count * 100.0f / (float)_calibrationFileNum);
        fflush(stdout);
    }
    MNN_PRINT("\n");
    _scales.clear();

    const int totalLayers = static_cast<int32_t>(_featureInfo.size());
    count                 = 0;

    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        if (Helper::gNotNeedFeatureOp.find(info->type()) == Helper::gNotNeedFeatureOp.end()) {
            for (auto t : nTensors) {
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    if (_featureInfo[t]->visited() == false) {
                        _scales[t] = _featureInfo[t]->computeScaleADMM();
                        count++;
                        MNN_PRINT("\rComputeADMM: %.2lf %%", (float)count * 100.0f / (float)totalLayers);
                        fflush(stdout);
                    }
                }
            }
        }
        return true;
    };
    MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        if (Helper::gNotNeedFeatureOp.find(info->type()) == Helper::gNotNeedFeatureOp.end()) {
            for (auto t : nTensors) {
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    if (_featureInfo[t]->visited() == false) {
                        _scales[t] = _featureInfo[t]->computeScaleADMM();
                        count++;
                        MNN_PRINT("\rComputeADMM: %.2lf %%", (float)count * 100.0f / (float)totalLayers);
                        fflush(stdout);
                    }
                }
            }
        }
        return true;
    };

    _interpreter->runSessionWithCallBackInfo(_session, before, after);
    MNN_PRINT("\n");
}

void Calibration::_fake_quant_weights() {
    auto findAbsMax = [&] (const float* weights, const int size) {
        float absMax = 0;
        for (int i = 0; i < size; i++) {
            if (std::fabs(weights[i]) > absMax) {
                absMax = std::fabs(weights[i]);
            }
        }

        return absMax;
    };

    for (const auto& op : _originalModel->oplists) {
        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), op->name);
        if (iter != _skip_quant_ops.end()) {
            continue;
        }

        const auto opType = op->type;
        if (opType != MNN::OpType_Convolution && opType != MNN::OpType_ConvolutionDepthwise) {
            continue;
        }

        auto param = op->main.AsConvolution2D();
        const int kernelNum = param->common->outputCount;
        std::vector<float> weights = param->weight;
        const int weightSize = static_cast<int32_t>(weights.size());
        const int kernelSize = weightSize / kernelNum;

        for (int i = 0; i < kernelNum; i++) {
            const int offset = i * kernelSize;
            float absMax = findAbsMax(weights.data() + offset, kernelSize);
            float scale = absMax / _weightClampValue;
            if (absMax < 1e-6f) {
                scale = absMax;
            }

            for (int j = 0; j < kernelSize; j++) {
                float value = weights[offset + j];
                float quantValue = std::round(value / scale);
                float clampedValue = std::max(std::min(quantValue, _weightClampValue), -_weightClampValue);
                float dequantValue = scale * clampedValue;
                param->weight[offset + j] = dequantValue;
            }
        }
    }
    DLOG(INFO) << "fake quant weights done.";
}


float CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0.0, denom_a = 0.0, denom_b = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        dot += a[i] * b[i];
        denom_a += a[i] * a[i];
        denom_b += b[i] * b[i];
    }
    return dot / (sqrt(denom_a) * sqrt(denom_b));
}

void Calibration::_fake_invert_quant_weights() {
    int n = _originalModel->oplists.size();

    for (int i = 0; i < n; i++) {
        const auto& op = _originalModel->oplists[i];
        auto& opHalf = _halfModel->oplists[i];
        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), opHalf->name);
        if (iter != _skip_quant_ops.end()) {
            continue;
        }

        const auto opType = opHalf->type;
        if (opType != MNN::OpType_Convolution && opType != MNN::OpType_ConvolutionDepthwise) {
            continue;
        }

        auto param = op->main.AsConvolution2D();
        auto& quantizedWeights = param->quanParameter->buffer;
        const int channel = param->common->outputCount;
        const auto& scales = param->quanParameter->alpha;
        const auto type = param->quanParameter->type;
        auto common = std::make_shared<MNN::ConvolutionCommon::Int8Common>();
        auto originBuffer = (unsigned char*) quantizedWeights.data();
        int8_t *buffer = nullptr;
        size_t bufferSzie = 0;
        if (type == 1) {
            buffer = ReadQuanData_c(originBuffer, &bufferSzie, common.get(), param->quanParameter->shapeInt32);
        } else if (type == 2) {
            buffer = ReadSparseQuanData_c(originBuffer, &bufferSzie, scales.data(), scales.size(), common.get(), param->quanParameter->shapeInt32);
        } else if (type == 4) {
            buffer = quantizedWeights.data();
            bufferSzie = quantizedWeights.size();
        }

        auto paramHalf = opHalf->main.AsConvolution2D();
        std::vector<half_float::half> halfWeights(bufferSzie);
        const int featSize = bufferSzie / channel;
        // invert quant weights
        for (int i = 0; i < channel; i++) {
            std::transform(buffer + i * featSize, 
                            buffer + (i + 1) * featSize, 
                            halfWeights.begin() + i * featSize, 
                            [&](int8_t x) { return half_float::half(x * scales[i]); });
        }

        paramHalf->quanParameter.reset(new MNN::IDSTQuanT);
        paramHalf->quanParameter->type = 3;
        int8_t* halfWeightsPtr = reinterpret_cast<int8_t*>(halfWeights.data());
        paramHalf->quanParameter->buffer.assign(halfWeightsPtr, halfWeightsPtr + bufferSzie * sizeof(half_float::half));
        std::vector<float> quantizedWeightsFloat(bufferSzie);
        std::transform(halfWeights.begin(), halfWeights.end(), quantizedWeightsFloat.begin(), 
                       [](half_float::half x) { return static_cast<float>(x); });
        
        float sim = CosineSimilarity(quantizedWeightsFloat, paramHalf->weight);
        auto name = opHalf->name;
        if (quantizedWeightsFloat.size() != paramHalf->weight.size()) {
            MNN_PRINT("op name: %s, weight size not match\n", name.c_str());
        }
        MNN_PRINT("op name: %s, weight cosine similarity: %f\n", name.c_str(), sim);

        paramHalf->weight.clear();
    }
    DLOG(INFO) << "fake invert quant weights done.";
}

void Calibration::_insertScale() {
    for (const auto iter :  _scales) {
        std::unique_ptr<MNN::TensorDescribeT> describe(new MNN::TensorDescribeT);
        auto des = TensorUtils::getDescribe(iter.first);
        if (des->index < 0) {
            continue;
        }
        describe->index = des->index;
        describe->quantInfo.reset(new MNN::TensorQuantInfoT);
        describe->quantInfo->scale = iter.second.first;
        describe->quantInfo->zero = iter.second.second;
        describe->quantInfo->type = MNN::DataType_DT_INT8;
        describe->quantInfo->min = -1 * _featureClampValue;
        describe->quantInfo->max = 1 * _featureClampValue;
        _originalModel->extraTensorDescribe.emplace_back(std::move(describe));
    }
    for (const auto& op : _originalModel->oplists) {
        const auto opType = op->type;

        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), op->name);
        if (iter != _skip_quant_ops.end()) {
            continue;
        }
        
        if (opType != MNN::OpType_Convolution && opType != MNN::OpType_ConvolutionDepthwise && opType != MNN::OpType_Deconvolution) {
            continue;
        }
        if (op->inputIndexes.size() > 1) {
            continue;
        }
        auto inputTensor = _tensorMap[op->inputIndexes[0]];
        auto outputTensor = _tensorMap[op->outputIndexes[0]];
        // below is Conv/DepthwiseConv weight quant
        const float inputScale  = _scales[inputTensor].first;
        const float outputScale = _scales[outputTensor].first;
        const int inputChannel = inputTensor->channel();
        const int outputChannel = outputTensor->channel();
        auto param                = op->main.AsConvolution2D();
        param->common->inputCount = inputChannel;
        const int channles        = param->common->outputCount;
        param->symmetricQuan.reset(new MNN::QuantizedFloatParamT);
        param->symmetricQuan->nbits = _quant_bits;
        const float* originWeight = param->weight.data();
        int originWeightSize   = static_cast<int32_t>(param->weight.size());
        auto conv2d = param;
        std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
        std::unique_ptr<Tensor> externalWeightTensor, externalBiasTensor;
        if (nullptr != conv2d->quanParameter.get()) {
            flatbuffers::FlatBufferBuilder tempBuilder;
            tempBuilder.Finish(IDSTQuan::Pack(tempBuilder, conv2d->quanParameter.get()));
            tempBuilder.Finish(Convolution2D::Pack(tempBuilder, conv2d));
            auto conv2d = flatbuffers::GetRoot<Convolution2D>(tempBuilder.GetBufferPointer());
            bool forceFloat = true;
            quanCommon = ConvolutionCommon::load(conv2d, nullptr, true, true);
            // Back to float
            originWeight     = quanCommon->weightFloat.get();
            originWeightSize = quanCommon->weightFloat.size();
        }
        const int weightSize      = originWeightSize;
        std::vector<int8_t> quantizedWeight(weightSize);
        std::vector<float> quantizedWeightScale(outputChannel);
        if (_weightQuantizeMethod == "MAX_ABS"){
            SymmetricQuantizeWeight(originWeight, weightSize, quantizedWeight.data(), quantizedWeightScale.data(), outputChannel, _weightClampValue);
        } else if (_weightQuantizeMethod == "ADMM") {
            QuantizeWeightADMM(originWeight, weightSize, quantizedWeight.data(), quantizedWeightScale.data(), outputChannel, _weightClampValue);
        }
        param->quanParameter = IDSTEncoder::encode(originWeight, quantizedWeightScale, weightSize/channles, channles, false, quantizedWeight.data(), -_weightClampValue);
        param->quanParameter->scaleIn = inputScale;
        param->quanParameter->scaleOut = outputScale;
        if (param->common->relu6) {
            param->common->relu  = true;
            param->common->relu6 = false;
        }
        param->weight.clear();
    }
}

void Calibration::_computeQuantError() {
    int count = 0;
    std::map<std::string, std::vector<float>> overflowRatiosMap;
    std::map<std::string, std::vector<float>> tensorCosDistanceMap;

    for (const auto& file : _calibrationFiles) {
        count++;
        _resizeIfNeeded(file, true);
        Helper::preprocessInput(_process.get(), _preprocessConfig, file, _inputTensor, _inputType);

        std::map<std::string, std::vector<float>> fakeQuantedFeatures;

        MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                 const MNN::OperatorInfo* info) {
            if (info->type() == "Raster") {
                return true;
            }
            for (auto t : nTensors) {
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    if (_featureInfo[t]->visited() == false) {
                        auto dequantFeatureAndOverflowRatio = _featureInfo[t]->fakeQuantFeature();
                        fakeQuantedFeatures[_featureInfo[t]->name()] = dequantFeatureAndOverflowRatio.first;
                        overflowRatiosMap[_featureInfo[t]->name()].emplace_back(dequantFeatureAndOverflowRatio.second);
                    }
                }
            }
            return true;
        };
        MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    if (_featureInfo[t]->visited() == false) {
                        auto dequantFeatureAndOverflowRatio = _featureInfo[t]->fakeQuantFeature();
                        fakeQuantedFeatures[_featureInfo[t]->name()] = dequantFeatureAndOverflowRatio.first;
                        overflowRatiosMap[_featureInfo[t]->name()].emplace_back(dequantFeatureAndOverflowRatio.second);
                    }
                }
            }
            return true;
        };

        for (auto& iter : _featureInfo) {
            iter.second->setVisited(false);
        }

        _interpreter->runSessionWithCallBackInfo(_session, before, after);

        Helper::preprocessInput(_process.get(), _preprocessConfig, file, _inputTensorOrigin, _inputType);

        MNN::TensorCallBackWithInfo beforeOrigin = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                 const MNN::OperatorInfo* info) {
            if (info->type() == "Raster") {
                return true;
            }
            for (auto t : nTensors) {
                if (_featureInfoOrigin.find(t) != _featureInfoOrigin.end()) {
                    if (_featureInfoOrigin[t]->visited() == false) {
                        auto name = _featureInfoOrigin[t]->name();
                        float cosDis = _featureInfoOrigin[t]->computeDistance(fakeQuantedFeatures[name]);
                        tensorCosDistanceMap[name].emplace_back(cosDis);
                    }
                }
            }
            return true;
        };
        MNN::TensorCallBackWithInfo afterOrigin = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if (_featureInfoOrigin.find(t) != _featureInfoOrigin.end()) {
                    if (_featureInfoOrigin[t]->visited() == false) {
                        auto name = _featureInfoOrigin[t]->name();
                        float cosDis = _featureInfoOrigin[t]->computeDistance(fakeQuantedFeatures[name]);
                        tensorCosDistanceMap[name].emplace_back(cosDis);
                    }
                }
            }
            return true;
        };

        for (auto& iter : _featureInfoOrigin) {
            iter.second->setVisited(false);
        }

        _interpreterOrigin->runSessionWithCallBackInfo(_sessionOrigin, beforeOrigin, afterOrigin);

        MNN_PRINT("\rcomputeDistance: %.2lf %%", (float)count * 100.0f / (float)_calibrationFileNum);
        fflush(stdout);
    }
    MNN_PRINT("\n\nDebug info:\n\n");

    for (auto& iter : tensorCosDistanceMap) {
        auto name = iter.first;
        float sumCos = 0.0f, sumOverflow = 0.0f;
        for (int i = 0; i < iter.second.size(); i++) {
            sumCos += iter.second[i];
            sumOverflow += overflowRatiosMap[name][i];
        }
        float avgCosDistance = sumCos / _calibrationFiles.size();
        float avgOverflowRatio = sumOverflow / _calibrationFiles.size();

        MNN_PRINT("%s:  cos similarity: %f, overflow ratio: %f\n", name.c_str(), avgCosDistance, avgOverflowRatio);
    }
}

void Calibration::_quantizeModelEMA() {
    auto varMap = Variable::loadMap(_originalModelFile.c_str());
    if (varMap.empty()) {
        MNN_ERROR("Can not load model %s\n", _originalModelFile.c_str());
        return;
    }

    auto inputOutputs = Variable::getInputAndOutput(varMap);
    auto inputs       = Variable::mapToSequence(inputOutputs.first);
    auto outputs      = Variable::mapToSequence(inputOutputs.second);
    if (inputs.size() != 1) {
        MNN_ERROR("Only support input size = 1\n");
        return;
    }
    auto originInfo = inputs[0]->getInfo();
    auto originFormat = NC4HW4;
    auto originType = halide_type_of<float>();
    std::vector<int> originDims;
    if (nullptr != originInfo) {
        originFormat = originInfo->order;
        originDims = originInfo->dim;
        originType = originInfo->type;
    }
    std::shared_ptr<Module> model(NN::extract(inputs, outputs, true), Module::destroy);
    NN::turnQuantize(model.get(), _quant_bits, NN::PerTensor, NN::MovingAverage, _winogradOpt);

    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 2);

    std::shared_ptr<SGD> solver(new SGD(model));
    solver->setLearningRate(1e-5);
    solver->setMomentum(0.9f);
    solver->setWeightDecay(0.00004f);

    DLOG(INFO) << "batch size: " << _batch;
    DLOG(INFO) << "quant bits: " << _quant_bits;
    if (_calibrationFileNum < _batch) {
        MNN_ERROR("_calibrationFileNum %d < batch size %d, set batch size as %d\n", _calibrationFileNum, _batch, _calibrationFileNum);
        _batch = _calibrationFileNum;
    }
    DataLoader* trainDataLoader = nullptr;
    std::shared_ptr<MNN::Tensor> tempInputTensor = nullptr;
    if (_inputType == Helper::InputType::IMAGE) {
        auto converImagesToFormat = _imageProcessConfig.destFormat;
        int resizeHeight = _preprocessConfig.targetHeight;
        int resizeWidth = _preprocessConfig.targetWidth;
        std::vector<float> means, scales;
        for (int i = 0; i < 4; i++) {
            means.emplace_back(_imageProcessConfig.mean[i]);
            scales.emplace_back(_imageProcessConfig.normal[i]);
        }
        std::vector<float> cropFraction = {_preprocessConfig.centerCropHeight, _preprocessConfig.centerCropWidth}; // center crop fraction for height and width
        bool centerOrRandomCrop = false; // true for random crop
        std::shared_ptr<ImageDataset::ImageConfig> datasetConfig(ImageDataset::ImageConfig::create(converImagesToFormat, resizeHeight, resizeWidth, scales, means, cropFraction, centerOrRandomCrop));
        auto trainDataset = ImageNoLabelDataset::create(_calibrationFilePath, datasetConfig.get());

        const int trainBatchSize = _batch;
        const int trainNumWorkers = 0;
        trainDataLoader = trainDataset.createLoader(trainBatchSize, true, false, trainNumWorkers);
        trainDataLoader->reset();
    } else {
        flatbuffers::FlatBufferBuilder builder(1024);
        auto offset = MNN::Net::Pack(builder, _originalModel);
        builder.Finish(offset);
        int size      = builder.GetSize();
        auto buffer = builder.GetBufferPointer();
        _interpreter.reset(MNN::Interpreter::createFromBuffer(buffer, size), MNN::Interpreter::destroy);
        MNN::ScheduleConfig config;
        _session     = _interpreter->createSession(config);
        _inputTensor = _interpreter->getSessionInput(_session, NULL);

        _getInputShape(_calibrationFiles[0]);
        std::vector<float> tempData(_batch * _channels * _height, 0.0f);
        tempInputTensor.reset(MNN::Tensor::create({_batch, _channels, _height}, halide_type_of<float>(), tempData.data(), MNN::Tensor::CAFFE), MNN::Tensor::destroy);
    }
    const int trainIterations = _calibrationFileNum / _batch;

    model->clearCache();
    exe->gc(Executor::FULL);

    model->setIsTraining(true);
    for (int i = 0; i < trainIterations; i++) {
        VARP input;
        if (_inputType == Helper::InputType::IMAGE) {
            auto trainData  = trainDataLoader->next();
            auto example    = trainData[0];
            input = example.first[0];
        } else {
            for (auto& file : _calibrationFiles) {
                for (int j = 0; j < _batch; j++) {
                    auto curPtr = tempInputTensor->host<float>() + j * tempInputTensor->stride(0);
                    std::shared_ptr<MNN::Tensor> tensorWarp(MNN::Tensor::create({1, _channels, _height}, _inputTensor->getType(), curPtr, MNN::Tensor::CAFFE), MNN::Tensor::destroy);
                    Helper::preprocessInput(_process.get(), _preprocessConfig, file, tensorWarp.get(), _inputType);
                }
                input = _Input({_batch, _channels, _height}, MNN::Express::Dimensionformat::NCHW, halide_type_of<float>());
                auto inputPtr = input->writeMap<float>();
                auto tempInputPtr = tempInputTensor->host<float>();
                for (int j = 0; j < _batch * _channels * _height; j++) {
                    inputPtr[j] = tempInputPtr[j];
                }
            }
        }
        auto predicts = model->onForward({_Convert(input, originFormat)});
        for (auto& output : predicts) {
            auto ptr = output->readMap<float>();
        }
        MNN_PRINT("\rquantize with EMA: %.2lf %%", (i + 1) * 100.0f / trainIterations);
        fflush(stdout);
        solver->step(_Scalar<float>(0.0f));
    }
    MNN_PRINT("\n");

    model->setIsTraining(false);
    exe->gc(Executor::PART);
    VARP forwardInput = nullptr;
    if (originInfo != nullptr && originDims.size() > 0) {
        forwardInput = _Input(originDims, originFormat, originType);
    } else {
        if (_inputType == Helper::InputType::IMAGE) {
            forwardInput = _Input({1, _channels, _preprocessConfig.targetHeight, _preprocessConfig.targetWidth}, NC4HW4);
        } else {
            forwardInput = _Input({1, _channels, _height}, NC4HW4);
        }
    }
    forwardInput->setName(inputs[0]->name());
    auto predicts = model->onForward({forwardInput});
    Transformer::turnModelToInfer()->onExecute(predicts);
    for (int i = 0; i < predicts.size(); i++) {
        predicts[i]->setName(outputs[i]->name());
    }
    Variable::save(predicts, _destModelFile.c_str());
    ConvertToFullQuant::convert(_destModelFile);
    
    std::unique_ptr<MNN::NetT> netT;
    {
        std::ifstream input(_destModelFile, std::ifstream::in | std::ifstream::binary);
        std::ostringstream outputOs;
        outputOs << input.rdbuf();
        netT = MNN::UnPackNet(outputOs.str().c_str());
    }
    ComputeUnaryBuffer(netT.get());
    {
        flatbuffers::FlatBufferBuilder builderOutput(1024);
        builderOutput.ForceDefaults(true);
        auto len = MNN::Net::Pack(builderOutput, netT.get());
        builderOutput.Finish(len);
        std::ofstream output(_destModelFile, std::ofstream::binary);
        output.write((const char*)builderOutput.GetBufferPointer(), builderOutput.GetSize());
    }
}

void Calibration::_computeInvertQuantError() {
    {
        flatbuffers::FlatBufferBuilder builderOutput(1024);
        builderOutput.ForceDefaults(true);
        auto len = MNN::Net::Pack(builderOutput, _halfModel);
        builderOutput.Finish(len);
        _interpreterHalf.reset(MNN::Interpreter::createFromBuffer(builderOutput.GetBufferPointer(), builderOutput.GetSize()), MNN::Interpreter::destroy);
    }
    MNN::ScheduleConfig config;
    // when using opencl as backend, the execution graph will be different from cpu
    // config.type = MNN_FORWARD_OPENCL;
    config.type = MNN_FORWARD_CPU;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_Low;
    _sessionHalf = _interpreterHalf->createSession(config);
    _inputTensorHalf = _interpreterHalf->getSessionInput(_sessionHalf, NULL);
    _interpreterHalf->resizeTensor(_inputTensorHalf, _inputTensorDims);
    _interpreterHalf->resizeSession(_sessionHalf);

    std::ostringstream opOrderStream;
    std::map<std::string, std::string> nameToOpName;
    // init _featureInfoHalf
    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        auto opName = info->name();
        auto type = info->type();
        // auto iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), opName);
        opOrderStream << opName << "\n";

        // if (iter != _skip_quant_ops.end()) {
        //     return false;
        // }
        if (Helper::gNotNeedFeatureOp.find(info->type()) == Helper::gNotNeedFeatureOp.end()) {
            int i = 0;
            for (auto t : nTensors) {
                if (_featureInfoHalf.find(t) == _featureInfoHalf.end()) {
                    auto name = opName + " input_tensor_" + flatbuffers::NumToString(i);
                    _featureInfoHalf[t] = std::shared_ptr<TensorStatistic>(
                        new TensorStatistic(t, _featureQuantizeMethod, name, _featureClampValue));
                    nameToOpName[name] = type;
                }
                i++;
            }
        }
        return true;
    };
    MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        auto opName = info->name();
        auto type = info->type();
        // auto iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), opName);
        // if (iter != _skip_quant_ops.end()) {
        //     return false;
        // }
        if (Helper::gNotNeedFeatureOp.find(info->type()) == Helper::gNotNeedFeatureOp.end()) {
            int i = 0;
            for (auto t : nTensors) {
                if (_featureInfoHalf.find(t) == _featureInfoHalf.end()) {
                    auto name = opName + " output_tensor_" + flatbuffers::NumToString(i);
                    _featureInfoHalf[t] = std::shared_ptr<TensorStatistic>(
                        new TensorStatistic(t, _featureQuantizeMethod, name, _featureClampValue));
                    nameToOpName[name] = type;
                }
                i++;
            }
        }
        return true;
    };
    _interpreterHalf->runSessionWithCallBackInfo(_sessionHalf, before, after);

    int count = 0;
    std::map<std::string, std::vector<float>> overflowRatiosMap;
    std::map<std::string, std::vector<float>> tensorCosDistanceMap;
    std::map<std::string, std::vector<float>> tensorCosDistanceMapHalf;

    {
        auto opOrder = opOrderStream.str();
        std::ofstream opOrderFile("op_order.txt");
        opOrderFile << opOrder;
    }
    std::string swinTestOpName = "/features/features.1/features.1.0/attn/Add_2_output_0__matmul_converted";
    bool dump = false;

    for (const auto& file : _calibrationFiles) {
        count++;
        _resizeIfNeeded(file, true);
        _interpreterHalf->resizeTensor(_inputTensorHalf, _inputTensorDims);
        _interpreterHalf->resizeSession(_sessionHalf);
        Helper::preprocessInput(_process.get(), _preprocessConfig, file, _inputTensor, _inputType);
        Helper::preprocessInput(_process.get(), _preprocessConfig, file, _inputTensorHalf, _inputType);
        Helper::preprocessInput(_process.get(), _preprocessConfig, file, _inputTensorOrigin, _inputType);

        std::map<std::string, std::vector<float>> fakeQuantedFeatures;
        std::map<std::string, std::vector<float>> fakeHalfFeatures;

        MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                 const MNN::OperatorInfo* info) {
            if (info->type() == "Raster") {
                return true;
            }
            for (auto t : nTensors) {
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    if (_featureInfo[t]->visited() == false) {
                        auto dequantFeatureAndOverflowRatio = _featureInfo[t]->fakeQuantFeature();
                        fakeQuantedFeatures[_featureInfo[t]->name()] = dequantFeatureAndOverflowRatio.first;
                        overflowRatiosMap[_featureInfo[t]->name()].emplace_back(dequantFeatureAndOverflowRatio.second);
                    }
                }

                if (swinTestOpName == info->name() && dump) {
                    std::string filename = _featureInfo[t]->name() + " int8 " + std::to_string(count);
                    dumpTensor2File(t, filename.c_str());
                }
            }
            return true;
        };
        MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    if (_featureInfo[t]->visited() == false) {
                        auto dequantFeatureAndOverflowRatio = _featureInfo[t]->fakeQuantFeature();
                        fakeQuantedFeatures[_featureInfo[t]->name()] = dequantFeatureAndOverflowRatio.first;
                        overflowRatiosMap[_featureInfo[t]->name()].emplace_back(dequantFeatureAndOverflowRatio.second);
                    }
                }

                if (swinTestOpName == info->name() && dump) {
                    std::string filename = _featureInfo[t]->name() + " int8 " + std::to_string(count);
                    dumpTensor2File(t, filename.c_str());
                }
            }
            return true;
        };

        for (auto& iter : _featureInfo) {
            iter.second->setVisited(false);
        }
        _interpreter->runSessionWithCallBackInfo(_session, before, after);

        MNN::TensorCallBackWithInfo beforeHalf =  [&](const std::vector<MNN::Tensor*>& nTensors,
                                                 const MNN::OperatorInfo* info) {
            if (info->type() == "Raster") {
                return true;
            }
            for (auto t : nTensors) {
                if (_featureInfoHalf.find(t) != _featureInfoHalf.end()) {
                    if (_featureInfoHalf[t]->visited() == false) {
                        auto *ptr = t->host<float>();
                        fakeHalfFeatures[_featureInfoHalf[t]->name()].assign(ptr, ptr + t->elementSize());
                        _featureInfoHalf[t]->setVisited(true);
                    }
                }

                if (swinTestOpName == info->name() && dump) {
                    std::string filename = _featureInfoHalf[t]->name() + " half " + std::to_string(count);
                    dumpTensor2File(t, filename.c_str());
                }
            }
            return true;
        };
        MNN::TensorCallBackWithInfo afterHalf = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if (_featureInfoHalf.find(t) != _featureInfoHalf.end()) {
                    if (_featureInfoHalf[t]->visited() == false) {
                        auto *ptr = t->host<float>();
                        fakeHalfFeatures[_featureInfoHalf[t]->name()].assign(ptr, ptr + t->elementSize());
                        _featureInfoHalf[t]->setVisited(true);
                    }
                }

                if (swinTestOpName == info->name() && dump) {
                    std::string filename = _featureInfoHalf[t]->name() + " half " + std::to_string(count);
                    dumpTensor2File(t, filename.c_str());
                }
            }
            return true;
        }; 
        
        for (auto& iter : _featureInfoHalf) {
            iter.second->setVisited(false);
        }
        _interpreterHalf->runSessionWithCallBackInfo(_sessionHalf, beforeHalf, afterHalf);

        MNN::TensorCallBackWithInfo beforeOrigin = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                 const MNN::OperatorInfo* info) {
            if (info->type() == "Raster") {
                return true;
            }
            for (auto t : nTensors) {
                if (_featureInfoOrigin.find(t) != _featureInfoOrigin.end()) {
                    if (_featureInfoOrigin[t]->visited() == false) {
                        auto& info = _featureInfoOrigin[t];
                        auto name = _featureInfoOrigin[t]->name();
                        auto feature = fakeQuantedFeatures[name];
                        float cosDis = _featureInfoOrigin[t]->computeDistance(feature);
                        tensorCosDistanceMap[name].emplace_back(cosDis);

                        auto halfFeature = fakeHalfFeatures[name];
                        float cosDisHalf = _featureInfoOrigin[t]->computeDistance(halfFeature);
                        tensorCosDistanceMapHalf[name].emplace_back(cosDisHalf);
                    }
                }

                if (swinTestOpName == info->name() && dump) {
                    std::string filename = _featureInfoOrigin[t]->name() + " float " + std::to_string(count);
                    dumpTensor2File(t, filename.c_str());
                }
            }
            return true;
        };
        MNN::TensorCallBackWithInfo afterOrigin = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if (_featureInfoOrigin.find(t) != _featureInfoOrigin.end()) {
                    if (_featureInfoOrigin[t]->visited() == false) {
                        auto name = _featureInfoOrigin[t]->name();
                        auto feature = fakeQuantedFeatures[name];
                        float cosDis = _featureInfoOrigin[t]->computeDistance(feature);
                        tensorCosDistanceMap[name].emplace_back(cosDis);

                        auto halfFeature = fakeHalfFeatures[name];
                        float cosDisHalf = _featureInfoOrigin[t]->computeDistance(halfFeature);
                        tensorCosDistanceMapHalf[name].emplace_back(cosDisHalf);
                    }
                }

                if (swinTestOpName == info->name() && dump) {
                    std::string filename = _featureInfoOrigin[t]->name() + " float " + std::to_string(count);
                    dumpTensor2File(t, filename.c_str());
                }
            }
            return true;
        };

        for (auto& iter : _featureInfoOrigin) {
            iter.second->setVisited(false);
        }
        _interpreterOrigin->runSessionWithCallBackInfo(_sessionOrigin, beforeOrigin, afterOrigin);

        MNN_PRINT("\rcomputeDistance: %.2lf %%", (float)count * 100.0f / (float)_calibrationFileNum);
        fflush(stdout);
    }
    MNN_PRINT("\n\nDebug info:\n\n");

    for (auto& iter : tensorCosDistanceMap) {
        auto name = iter.first;
        float sumCos = 0.0f, sumOverflow = 0.0f;
        for (int i = 0; i < iter.second.size(); i++) {
            sumCos += iter.second[i];
            sumOverflow += overflowRatiosMap[name][i];
        }
        float avgCosDistance = sumCos / _calibrationFiles.size();
        float avgOverflowRatio = sumOverflow / _calibrationFiles.size();

        float sumCosHalf = 0.0f;
        for (int i = 0; i < tensorCosDistanceMapHalf[name].size(); i++) {
            sumCosHalf += tensorCosDistanceMapHalf[name][i];
        }
        float avgCosDistanceHalf = sumCosHalf / _calibrationFiles.size();

        MNN_PRINT("int8 %s %s:  cos similarity: %f, overflow ratio: %f; int8->half: cos similarity: %f\n", 
                    nameToOpName[name].c_str(), name.c_str(), avgCosDistance, avgOverflowRatio, avgCosDistanceHalf);
    }

    hybridQuantModel();
}

void Calibration::runQuantizeModel() {
    if (_featureQuantizeMethod == "EMA") {
        _quantizeModelEMA();
        return;
    }

    if (_featureQuantizeMethod == "KL") {
        _computeFeatureScaleKL();
    } else if (_featureQuantizeMethod == "ADMM") {
        _computeFeatureScaleADMM();
    }
    // if (_debug) {
    //     _computeQuantError();
    // }
    _insertScale();

    if (_debug) {
        _fake_invert_quant_weights();
        _computeInvertQuantError();
    }

    ComputeUnaryBuffer(_originalModel);

    {
        flatbuffers::FlatBufferBuilder builderOutput(1024);
        builderOutput.ForceDefaults(true);
        auto len = MNN::Net::Pack(builderOutput, _originalModel);
        builderOutput.Finish(len);
        std::ofstream output(_destModelFile, std::ofstream::binary);
        output.write((const char*)builderOutput.GetBufferPointer(), builderOutput.GetSize());
    }
}

void Calibration::hybridQuantModel() {
    // 必须保存数据，因为tensor会复用内存，导致内存失效
    std::map<std::string, std::vector<float>> tensorDataMap;
    MNN::TensorCallBackWithInfo beforeOrigin = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                const MNN::OperatorInfo* info) {
        if (info->type() == "Raster") {
            return true;
        }
        for (auto t : nTensors) {
            if (_featureInfoOrigin.find(t) != _featureInfoOrigin.end()) {
                if (_featureInfoOrigin[t]->visited() == false) {
                    auto& info = _featureInfoOrigin[t];
                    auto name = _featureInfoOrigin[t]->name();
                    auto* ptr = t->host<float>();
                    std::vector<float> data(t->elementSize());
                    memcpy(data.data(), ptr, t->size());
                    tensorDataMap[name] = data;
                    _featureInfoOrigin[t]->setVisited(true);
                }
            }
        }
        return true;
    };
    MNN::TensorCallBackWithInfo afterOrigin = [&](const std::vector<MNN::Tensor*>& nTensors,
                                            const MNN::OperatorInfo* info) {
        for (auto t : nTensors) {
            if (_featureInfoOrigin.find(t) != _featureInfoOrigin.end()) {
                if (_featureInfoOrigin[t]->visited() == false) {
                    auto name = _featureInfoOrigin[t]->name();
                    auto* ptr = t->host<float>();
                    std::vector<float> data(t->elementSize());
                    memcpy(data.data(), ptr, t->size());
                    tensorDataMap[name] = data;
                    _featureInfoOrigin[t]->setVisited(true);
                }
            }
        }
        return true;
    };

    std::set<std::string> skipOps;
    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        if (info->type() == "Raster") {
            return true;
        }
        auto opType = info->type();
        for (auto t : nTensors) {
            if (_featureInfo.find(t) != _featureInfo.end()) {
                if (_featureInfo[t]->visited() == false) {
                    _featureInfo[t]->setVisited(true);
                    auto name = _featureInfo[t]->name();
                    auto feature = tensorDataMap[name];
                    _featureInfo[t]->fakeQuantFeature();
                    float sim = _featureInfo[t]->computeDistance(feature);

                    if (sim < mSimThreshold) {
                        skipOps.emplace(info->name());
                        auto* ptr = t->host<float>();
                        std::memcpy(ptr, feature.data(), t->size());
                        MNN_PRINT("op %s tensor %s sim %f\n", opType.c_str(), _featureInfo[t]->name().c_str(), sim);
                    }
                }
            }
        }
        return true;
    };
    MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        if (info->type() == "Raster") {
            return true;
        }
        auto opType = info->type();
        for (auto t : nTensors) {
            if (_featureInfo.find(t) != _featureInfo.end()) {
                if (_featureInfo[t]->visited() == false) {
                    _featureInfo[t]->setVisited(true);
                    auto name = _featureInfo[t]->name();
                    auto feature = tensorDataMap[name];
                    _featureInfo[t]->fakeQuantFeature();
                    float sim = _featureInfo[t]->computeDistance(feature);

                    if (sim < mSimThreshold) {
                        skipOps.emplace(info->name());
                        auto* ptr = t->host<float>();
                        std::memcpy(ptr, feature.data(), t->size());
                        MNN_PRINT("op %s tensor %s sim %f\n", opType.c_str(), _featureInfo[t]->name().c_str(), sim);
                    }
                }
            }
        }
        return true;
    };


    MNN_PRINT("\n\nSTART HYBRID QUANTIZATION\n");
    auto file = _calibrationFiles[0];
    _resizeIfNeeded(file, true);
    Helper::preprocessInput(_process.get(), _preprocessConfig, file, _inputTensor, _inputType);
    Helper::preprocessInput(_process.get(), _preprocessConfig, file, _inputTensorOrigin, _inputType);
    for (auto& iter : _featureInfoOrigin) {
        iter.second->setVisited(false);
    }
    _interpreterOrigin->runSessionWithCallBackInfo(_sessionOrigin, beforeOrigin, afterOrigin);

    for (auto& iter : _featureInfo) {
        iter.second->setVisited(false);
    }
    _interpreter->runSessionWithCallBackInfo(_session, before, after);

    MNN_PRINT("\n\nSKIP OPs\n");
    for (auto& op : skipOps) {
        MNN_PRINT("%s\n", op.c_str());
    }
}

void Calibration::dumpTensorScales(const std::string& modelFile) {
    rapidjson::StringBuffer sb;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(sb);

    writer.StartArray();

    for (auto iter = _originalModel->oplists.begin(); iter != _originalModel->oplists.end(); iter++) {
        auto op           = iter->get();
        const auto opType = op->type;
        const auto name   = op->name;
        
        if (opType == MNN::OpType_Raster) {
            continue;
        }

        writer.StartObject();

        writer.Key("name");
        writer.String(rapidjson::StringRef(name.c_str(), name.size()));

        auto& inputIndexes  = op->inputIndexes;
        const int inputSize = static_cast<int32_t>(inputIndexes.size());

        if (inputSize > 0) {
            writer.Key("inputs");
            writer.StartArray();
            for (int i = 0; i < inputSize; ++i) {
                const auto curInputIndex = inputIndexes[i];
                
                auto input        = _tensorMap[curInputIndex];
                auto inputOpScale = _scales[input];
                
                writer.StartObject();
                writer.Key("tensorIndex");
                writer.Int(curInputIndex);

                writer.Key("scales");
                writer.StartArray();
                writer.Double(inputOpScale.first);
                writer.EndArray();

                writer.Key("zeropoint");
                writer.StartArray();
                writer.Double(inputOpScale.second);
                writer.EndArray();

                writer.EndObject();
            }
            writer.EndArray();
        }
 
        auto& outputIndexes  = op->outputIndexes;
        const int outputSize = static_cast<int32_t>(outputIndexes.size());

        if (outputSize > 0) {
            writer.Key("outputs");
            writer.StartArray();
            for (int i = 0; i < outputSize; ++i) {
                const auto curOutputIndex = outputIndexes[i];
                
                auto output        = _tensorMap[curOutputIndex];
                auto outputOpScale = _scales[output];
                
                writer.StartObject();
                writer.Key("tensorIndex");
                writer.Int(curOutputIndex);

                writer.Key("scales");
                writer.StartArray();
                writer.Double(outputOpScale.first);
                writer.EndArray();

                writer.Key("zeropoint");
                writer.StartArray();
                writer.Double(outputOpScale.second);
                writer.EndArray();

                writer.EndObject();
            }
            writer.EndArray();
        }

        writer.EndObject();
    }
    writer.EndArray();

    std::string scaleFile = modelFile + ".json";
    std::ofstream os(scaleFile);
    if (os.is_open()) {
        os << sb.GetString() << std::endl;
        os.close();
    } else {
        std::cerr << "open scale file " << scaleFile << " fail. error code:" << os.failbit << std::endl;
    }
}

typedef VARP (*unaryProc)(VARP input);
static unaryProc selectUnaryProc(int type) {
    switch (type) {
        case UnaryOpOperation_ABS:
            return MNN::Express::_Abs;
        case UnaryOpOperation_SQUARE:
            return MNN::Express::_Square;
        case UnaryOpOperation_NEG:
            return MNN::Express::_Negative;
        case UnaryOpOperation_RSQRT:
            return MNN::Express::_Rsqrt;
        case UnaryOpOperation_EXP:
            return MNN::Express::_Exp;
        case UnaryOpOperation_COS:
            return MNN::Express::_Cos;
        case UnaryOpOperation_SIN:
            return MNN::Express::_Sin;
        case UnaryOpOperation_SIGMOID:
            return MNN::Express::_Sigmoid;
        case UnaryOpOperation_TANH:
            return MNN::Express::_Tanh;
        case UnaryOpOperation_TAN:
            return MNN::Express::_Tan;
        case UnaryOpOperation_ATAN:
            return MNN::Express::_Atan;
        case UnaryOpOperation_SQRT:
            return MNN::Express::_Sqrt;
        case UnaryOpOperation_RECIPROCAL:
            return MNN::Express::_Reciprocal;
        case UnaryOpOperation_LOG1P:
            return MNN::Express::_Log1p;
        case UnaryOpOperation_LOG:
            return MNN::Express::_Log;
        case UnaryOpOperation_ACOSH:
            return MNN::Express::_Acosh;
        case UnaryOpOperation_SINH:
            return MNN::Express::_Sinh;
        case UnaryOpOperation_ASINH:
            return MNN::Express::_Asinh;
        case UnaryOpOperation_ATANH:
            return MNN::Express::_Atanh;
        case UnaryOpOperation_SIGN:
            return MNN::Express::_Sign;
        case UnaryOpOperation_COSH:
            return MNN::Express::_Cosh;
        case UnaryOpOperation_ERF:
            return MNN::Express::_Erf;
        case UnaryOpOperation_ERFC:
            return MNN::Express::_Erfc;
        case UnaryOpOperation_ERFINV:
            return MNN::Express::_Erfinv;
        case UnaryOpOperation_EXPM1:
            return MNN::Express::_Expm1;
        case UnaryOpOperation_ASIN:
            return MNN::Express::_Asin;
        case UnaryOpOperation_ACOS:
            return MNN::Express::_Acos;
        case UnaryOpOperation_HARDSWISH:
            return MNN::Express::_Hardswish;
        case UnaryOpOperation_GELU:
            return MNN::Express::_Gelu;
        default:
            MNN_ASSERT(false);
            break;
    }
    return nullptr;
}
void Calibration::ComputeUnaryBuffer(MNN::NetT* net) {
    for (auto iter = net->oplists.begin(); iter != net->oplists.end(); ++iter) {
        auto op = iter->get();
        const auto opType = op->type;
        std::map<int, TensorDescribeT*> describes;
        for (auto& des : _originalModel->extraTensorDescribe) {
            describes.insert(std::make_pair(des->index, des.get()));
        }
        if (opType == MNN::OpType_UnaryOp) {
            auto type = op->main.AsUnaryOp()->opType;
            if (type == UnaryOpOperation_ABS || type == UnaryOpOperation_NEG || type == UnaryOpOperation_SIGN) {
                continue;
            }
            op->main.AsUnaryOp()->tableInt8.resize(255);
            auto unaryParam = op->main.AsUnaryOp()->tableInt8.data();

            auto outputId = op->outputIndexes[0];
            if (describes.find(outputId) == describes.end()) {
                continue;
            }
            auto unaryDes = describes.find(outputId)->second;
            float outScale = unaryDes->quantInfo->scale;
            float outZero  = unaryDes->quantInfo->zero;
            auto inputId = op->inputIndexes[0];
            if (describes.find(inputId) == describes.end()) {
                MNN_ERROR("Can't find extraTensorDescribe for %s\n", op->name.c_str());
            }
            unaryDes = describes.find(inputId)->second;
            float inpScale = unaryDes->quantInfo->scale;
            float inpZero  = unaryDes->quantInfo->zero;

            // Read input data.
            std::vector<float> dataInput;
            float fx = 0.f;
            auto input = _Input({255}, NCHW, halide_type_of<float>());
            input->setName("input_tensor");
            auto ptr_in = input->template writeMap<float>();
            for (int i = -127; i <= 127; ++i) {
                fx = (i - inpZero) * inpScale;
                dataInput.push_back(fx);
                ptr_in[i + 127] = fx;
            }
            input->unMap();
            // Compute output data.
            VARP output;
            auto func = selectUnaryProc(type);
            if (nullptr == func) {
                MNN_ERROR("Don't support quantizing UnaryOP: %s to Int8\n", op->name.c_str());
            }
            output = func(input);
            auto gotOutput = output->template readMap<float>();
            // Write output data.
            int val;
            for (int i = 0; i < 255; ++i) {
                val = (int)roundf(gotOutput[i] / outScale) + outZero;
                if (val > 127) {
                    val = 127;
                }
                if (val < -127) {
                    val = -127;
                }
                unaryParam[i] = val;
                            }
        }
    }
}

int quant_main(int argc, const char* argv[]) {
    if (argc < 4) {
        DLOG(INFO) << "Usage: ./quantized.out src.mnn dst.mnn preTreatConfig.json\n";
        return 0;
    }
    const char* modelFile      = argv[1];
    const char* preTreatConfig = argv[3];
    const char* dstFile        = argv[2];
    DLOG(INFO) << ">>> modelFile: " << modelFile;
    DLOG(INFO) << ">>> preTreatConfig: " << preTreatConfig;
    DLOG(INFO) << ">>> dstFile: " << dstFile;
    std::unique_ptr<MNN::NetT> netT;
    {
        std::shared_ptr<MNN::Interpreter> interp(MNN::Interpreter::createFromFile(modelFile), MNN::Interpreter::destroy);
        if (nullptr == interp.get()) {
            return 0;
        }
        netT = MNN::UnPackNet(interp->getModelBuffer().first);
    }

    // temp build net for inference
    flatbuffers::FlatBufferBuilder builder(1024);
    auto offset = MNN::Net::Pack(builder, netT.get());
    builder.Finish(offset);
    int size      = builder.GetSize();
    auto ocontent = builder.GetBufferPointer();

    // model buffer for creating mnn Interpreter
    std::unique_ptr<uint8_t> modelForInference(new uint8_t[size]);
    memcpy(modelForInference.get(), ocontent, size);

    std::unique_ptr<uint8_t> modelOriginal(new uint8_t[size]);
    memcpy(modelOriginal.get(), ocontent, size);

    std::unique_ptr<uint8_t> modelHalfBuffer(new uint8_t[size]);
    memcpy(modelHalfBuffer.get(), ocontent, size);

    netT.reset();
    netT = MNN::UnPackNet(modelOriginal.get());

    std::unique_ptr<MNN::NetT> netTHalf = MNN::UnPackNet(modelHalfBuffer.get());

    // quantize model's weight
    DLOG(INFO) << "Calibrate the feature and quantize model...";
    std::shared_ptr<Calibration> calibration(
        new Calibration(netT.get(), netTHalf.get(), modelForInference.get(), size, preTreatConfig, std::string(modelFile), std::string(dstFile)));
    if (!calibration->valid()) {
        return 0;
    }
    calibration->runQuantizeModel();
    calibration->dumpTensorScales(dstFile);
    DLOG(INFO) << "Quantize model done!";

    return 0;
}
