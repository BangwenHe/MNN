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

//#define FEED_INPUT_NAME_VALUE

using namespace MNN;

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

static void dumpTensor2File(const Tensor* tensor, const char* file, std::ofstream& orderFile) {
    orderFile << file << std::endl;
    std::ofstream outputOs(file);
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

static void _loadInputFromFile(Tensor* inputTensor, std::string pwd, std::string name) {
    MNN::Tensor givenTensor(inputTensor, inputTensor->getDimensionType());
    {
        int size_w = inputTensor->width();
        int size_h = inputTensor->height();
        int bpp    = inputTensor->channel();
        int batch  = inputTensor->batch();
        MNN_PRINT("Input size:%d\n", inputTensor->elementSize());
        inputTensor->printShape();

        std::ostringstream fileName;
        fileName << pwd << name;
        std::ifstream input(fileName.str().c_str());
        FUNC_PRINT_ALL(fileName.str().c_str(), s);

        if (givenTensor.getType().code == halide_type_int) {
            auto size           = givenTensor.elementSize();
            const auto bytesLen = givenTensor.getType().bytes();
            if (bytesLen == 4) {
                auto inputData = givenTensor.host<int32_t>();
                double temp;
                for (int i = 0; i < size; ++i) {
                    input >> temp;
                    inputData[i] = temp;
                }
            } else if (bytesLen == 1) {
                auto inputData = givenTensor.host<int8_t>();
                double pixel      = 0;
                for (int i = 0; i < size; ++i) {
                    input >> pixel;
                    inputData[i] = static_cast<int8_t>(pixel);
                }
            }
        } else if (givenTensor.getType().code == halide_type_uint) {
            auto size = givenTensor.elementSize();
            {
                FUNC_PRINT(givenTensor.getType().bytes());
                auto inputData = givenTensor.host<uint8_t>();
                for (int i = 0; i < size; ++i) {
                    double p;
                    input >> p;
                    inputData[i] = (uint8_t)p;
                }
            }
        } else if (givenTensor.getType().code == halide_type_float) {
            auto inputData = givenTensor.host<float>();
            auto size      = givenTensor.elementSize();
            for (int i = 0; i < size; ++i) {
                input >> inputData[i];
                // inputData[i] = 1.0f;
            }
        }
        inputTensor->copyFromHostTensor(&givenTensor);
    }
}

static inline int64_t getTimeInUs() {
    uint64_t time;
#if defined(_MSC_VER)
    LARGE_INTEGER now, freq;
    QueryPerformanceCounter(&now);
    QueryPerformanceFrequency(&freq);
    uint64_t sec  = now.QuadPart / freq.QuadPart;
    uint64_t usec = (now.QuadPart % freq.QuadPart) * 1000000 / freq.QuadPart;
    time          = sec * 1000000 + usec;
#else
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    time = static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
#endif
    return time;
}

static int test_main(int argc, const char* argv[]) {
    if (argc < 2) {
        MNN_PRINT("========================================================================\n");
        MNN_PRINT("Arguments: model.MNN runLoops savePath \n");
        MNN_PRINT("========================================================================\n");
        return -1;
    }

    std::string cmd = argv[0];
    std::string pwd = "./";
    auto rslash     = cmd.rfind("/");
    if (rslash != std::string::npos) {
        pwd = cmd.substr(0, rslash + 1);
    }

    // read args
    const char* fileName = argv[1];

    int runTime = 1;
    if (argc > 2) {
        runTime = ::atoi(argv[2]);
    }

    auto type = MNN_FORWARD_USER_2;
    std::string savePath = "partition.json";
    if (argc > 3) {
        savePath = argv[3];
    }

    // create net
    MNN_PRINT("Open Model %s\n", fileName);
    std::shared_ptr<MNN::Interpreter> net =
        std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(fileName), MNN::Interpreter::destroy);
    if (nullptr == net) {
        return 0;
    }
    net->setCacheFile(".tempcache");
    net->setSessionMode(Interpreter::Session_Debug);

    // create session
    int numThread = 4;
    int precision = BackendConfig::Precision_Low;
    int memory = BackendConfig::Memory_Normal;
    MNN::ScheduleConfig config;
    config.type      = type;
    /*modeNum means gpuMode for GPU usage, Or means numThread for CPU usage.*/
    config.numThread = numThread;
    // If type not fount, let it failed
    config.backupType = type;
    BackendConfig backendConfig;
    // config.path.outputs.push_back("ResizeBilinear_2");
    // backendConfig.power = BackendConfig::Power_High;
    backendConfig.precision = static_cast<MNN::BackendConfig::PrecisionMode>(precision);
    backendConfig.memory = static_cast<MNN::BackendConfig::MemoryMode>(memory);
    config.backendConfig     = &backendConfig;
    MNN::Session* session    = NULL;
    MNN::Tensor* inputTensor = nullptr;
    {
        AUTOTIME;
        session = net->createSession(config);
        if (nullptr == session) {
            return 0;
        }
        inputTensor = net->getSessionInput(session, NULL);
    }
    int resizeStatus = 0;
    net->getSessionInfo(session, MNN::Interpreter::RESIZE_STATUS, &resizeStatus);
    if (resizeStatus != 0) {
        MNN_ERROR("Resize error, can't execute MNN\n");
        return 0;
    }

    float memoryUsage = 0.0f;
    net->getSessionInfo(session, MNN::Interpreter::MEMORY, &memoryUsage);
    float flops = 0.0f;
    net->getSessionInfo(session, MNN::Interpreter::FLOPS, &flops);
    int backendType[2];
    net->getSessionInfo(session, MNN::Interpreter::BACKENDS, backendType);
    MNN_PRINT("Session Info: memory use %f MB, flops is %f M, backendType is %d\n", memoryUsage, flops, backendType[0]);
    // Set Other Inputs to Zero
    auto allInput = net->getSessionInputAll(session);
    for (auto& iter : allInput) {
        auto inputTensor = iter.second;
        auto size = inputTensor->size();
        if (size <= 0) {
            continue;
        }
        MNN::Tensor tempTensor(inputTensor, inputTensor->getDimensionType());
        ::memset(tempTensor.host<void>(), 0, tempTensor.size());
        inputTensor->copyFromHostTensor(&tempTensor);
    }
    MNN_PRINT("===========> Session Resize Done.\n");
    MNN_PRINT("===========> Session Start partition...\n");

    {
        auto start = getTimeInUs();
        net->partitionSession(session, savePath, runTime);
        auto end = getTimeInUs();
        float searchTime = (end - start) / 1000.0f;
        MNN_PRINT("Partition Time: %f ms\n", searchTime);
        MNN_PRINT("===========> Session Partition Done. Save result to %s\n", savePath.c_str());
    }


// #ifndef __ANDROID__
//     {
//         std::string allPath = savePath + "_all.json";

//         auto start = getTimeInUs();
//         net->partitionSessionAll(session, allPath, runTime);
//         auto end = getTimeInUs();
//         float searchTime = (end - start) / 1000.0f;
//         MNN_PRINT("PartitionAll Time: %f ms\n", searchTime);
//         MNN_PRINT("===========> Session Partition Done. Save result to %s\n", allPath.c_str());
//     }
// #endif


    return 0;
}

int main(int argc, const char* argv[]) {
    // For Detect Memory Leak, set circle as true
    bool circle = false;
    do {
        test_main(argc, argv);
    } while (circle);
    return 0;
}