#include <stdlib.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <MNN/MNNDefine.h>
#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <core/Backend.hpp>
#include <core/TensorUtils.hpp>
#include <MNN_generated.h>
#include <numeric>


static inline int64_t getTimeInUs() {
    uint64_t time;
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    time = static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
    return time;
}



int main(int argc, char* argv[]) {
    if (argc < 2) {
        MNN_PRINT("Usage: ./branchDetect.out model.mnn rounds\n");
        return 0;
    }
    
    std::string modelFile = argv[1];
    int rounds = 10;
    if (argc > 2) {
        rounds = atoi(argv[2]);
    }

    MNN_PRINT("modelFile: %s\n", modelFile.c_str());
    std::shared_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(modelFile.c_str()));
    net->setSessionMode(MNN::Interpreter::Session_Debug);
    // std::shared_ptr<MNN::Interpreter> gpuNet(MNN::Interpreter::createFromFile(modelFile.c_str()));
    // gpuNet->setSessionMode(MNN::Interpreter::Session_Debug);
    // gpuNet->setSessionMode(MNN::Interpreter::Session_Debug);

    auto bufferAndLength = net->getModelBuffer();
    const void* buffer = bufferAndLength.first;
    size_t length = bufferAndLength.second;

    auto netT = MNN::UnPackNet(buffer);
    std::vector<MNN::OpT*> inputOps, outputOps;
    for (auto& op : netT->oplists) {
        if (op->inputIndexes.size() == 0) {
            inputOps.push_back(op.get());
        }

        // auto findOutputFunc = [&op] (const std::string& outputName) {
        //     return op->name.find(outputName) != std::string::npos;
        // };
        // if (std::find_if(netT->outputName.begin(), netT->outputName.end(), findOutputFunc) != netT->outputName.end()) {
        if (std::find(netT->outputName.begin(), netT->outputName.end(), op->name) != netT->outputName.end()) {
            outputOps.push_back(op.get());
        }
    }

    std::map<MNN::OpT*, std::vector<MNN::OpT*>> nextOps;
    for (auto& op : netT->oplists) {
        for (auto& index : op->outputIndexes) {
            for (auto& nextOp : netT->oplists) {
                if (std::find(nextOp->inputIndexes.begin(), nextOp->inputIndexes.end(), index) != nextOp->inputIndexes.end()) {
                    nextOps[op.get()].push_back(nextOp.get());
                }
            }
        }
    }

    std::vector<std::vector<MNN::OpT*>> branchNodes;
    std::map<MNN::OpT*, bool> visited;
    for (auto& op : netT->oplists) {
        visited[op.get()] = false;
    }

    // 根据 netT 中的 oplist 每个 OP 的 inputIndexes 和 outputIndexes，通过DFS查找所有的分支节点
    visited[inputOps[0]] = true;
    visited[outputOps[0]] = true;

    for (auto& op : netT->oplists) {
        if (visited[op.get()] == true) {
            continue;
        }

        auto inputIndexes = op->inputIndexes;
        auto outputIndexes = op->outputIndexes;
        auto opname = op->name;
        MNN_PRINT("opname: %s\n", opname.c_str());

        std::vector<MNN::OpT*> branchNode;
        std::vector<MNN::OpT*> stack;
        stack.push_back(op.get());
        while (!stack.empty()) {
            auto node = stack.back();
            stack.pop_back();
            if (visited[node] == true) {
                continue;
            }
            visited[node] = true;
            branchNode.push_back(node);
            
            for (auto& nextOp : nextOps[node]) {
                stack.push_back(nextOp);
            }
        }

        if (branchNode.size() > 1) {
            branchNodes.push_back(branchNode);
        }
    }

    for (int i = 0; i < branchNodes.size(); i++) {
        MNN_PRINT("Branch %d: ", i);
        for (int j = 0; j < branchNodes[i].size(); j++) {
            MNN_PRINT("%s ", branchNodes[i][j]->name.c_str());
        }
        MNN_PRINT("\n");
    }

    
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    int branchIndex = 0;
    auto* session = net->createSession(config);
    auto inputTensor = net->getSessionInput(session, nullptr);
    auto outputTensor = net->getSessionOutput(session, nullptr);
    auto inputHost = inputTensor->host<float>();
    auto outputHost = outputTensor->host<float>();
    for (int i = 0; i < inputTensor->size(); i++) {
        inputHost[i] = 1.0f;
    }

    MNN::ScheduleConfig gpuConfig;
    gpuConfig.type = MNN_FORWARD_OPENCL;
    MNN::BackendConfig backendConfig;
    backendConfig.memory = MNN::BackendConfig::Memory_Low;
    gpuConfig.backendConfig = &backendConfig;
    int gpuBranchIndex = 1;
    if (branchNodes.size() < 2) {
        gpuBranchIndex = 0;
    }

    // auto* gpuSession = gpuNet->createSession(gpuConfig);
    // auto gpuInputTensor = gpuNet->getSessionInput(gpuSession, nullptr);
    // auto gpuOutputTensor = gpuNet->getSessionOutput(gpuSession, nullptr);
    auto* gpuSession = net->createSession(gpuConfig);
    auto gpuInputTensor = net->getSessionInput(gpuSession, nullptr);
    auto gpuOutputTensor = net->getSessionOutput(gpuSession, nullptr);
    gpuInputTensor->copyFromHostTensor(inputTensor);

    auto searchFunc = [] (MNN::OpT* op, const MNN::OperatorInfo* info) {
        return op->name == info->name();
    };

    std::map<MNN::Tensor*, std::pair<bool, std::string>> isCPUBranchTensor;
    std::map<MNN::Tensor*, std::pair<bool, std::string>> isGPUBranchTensor;
    {
        MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
            for (auto& t : tensors) {
                if (std::find_if(branchNodes[branchIndex].begin(), branchNodes[branchIndex].end(), [&info] (MNN::OpT* op) {return op->name == info->name();}) == branchNodes[branchIndex].end()) {
                    return true;
                }
            }
            for (auto& t : tensors) {
                isCPUBranchTensor[t] = std::make_pair(true, info->name());
            }
            return true;
        };

        MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
            for (auto& t : tensors) {
                if (std::find_if(branchNodes[branchIndex].begin(), branchNodes[branchIndex].end(), [&info] (MNN::OpT* op) {return op->name == info->name();}) == branchNodes[branchIndex].end()) {
                    return true;
                }
            }
            for (auto& t : tensors) {
                isCPUBranchTensor[t] = std::make_pair(true, info->name());
            }
            return true;
        };

        MNN::TensorCallBackWithInfo gpuBefore = [&](const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
            for (auto& t : tensors) {
                if (std::find_if(branchNodes[gpuBranchIndex].begin(), branchNodes[gpuBranchIndex].end(), [&info] (MNN::OpT* op) {return op->name == info->name();}) == branchNodes[gpuBranchIndex].end()) {
                    return true;
                }
            }
            for (auto& t : tensors) {
                isGPUBranchTensor[t] = std::make_pair(true, info->name());
            }
            return true;
        };

        MNN::TensorCallBackWithInfo gpuAfter = [&](const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
            for (auto& t : tensors) {
                if (std::find_if(branchNodes[gpuBranchIndex].begin(), branchNodes[gpuBranchIndex].end(), [&info] (MNN::OpT* op) {return op->name == info->name();}) == branchNodes[gpuBranchIndex].end()) {
                    return true;
                }
            }
            for (auto& t : tensors) {
                isGPUBranchTensor[t] = std::make_pair(true, info->name());
            }
            return true;
        };

        net->runSessionWithCallBackInfo(session, before, after);
        // gpuNet->runSessionWithCallBackInfo(gpuSession, gpuBefore, gpuAfter);
        net->runSessionWithCallBackInfo(gpuSession, gpuBefore, gpuAfter);
        MNN_PRINT("CPU-GPU map created\n");

        // print cpu branch tensor
        MNN_PRINT("CPU branch tensor: \n");
        for (auto& t : isCPUBranchTensor) {
            if (t.second.first == true) {
                MNN_PRINT("%s ", t.second.second.c_str());
            }
        }
        MNN_PRINT("\n");

        // print gpu branch tensor
        MNN_PRINT("GPU branch tensor: \n");
        for (auto& t : isGPUBranchTensor) {
            if (t.second.first == true) {
                MNN_PRINT("%s ", t.second.second.c_str());
            }
        }
        MNN_PRINT("\n");
    }

    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
        for (auto& t : tensors) {
            if (isCPUBranchTensor[t].first == false) {
                return false;
            }
        }
        return true;
    };
    MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
        return true;
    };

    MNN::TensorCallBackWithInfo gpuBefore = [&](const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
        for (auto& t : tensors) {
            if (isGPUBranchTensor[t].first == false) {
                return false;
            }
        }
        return true;
    };
    MNN::TensorCallBackWithInfo gpuAfter = [&](const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
        return true;
    };

    auto summarizeTime = [] (const std::vector<uint64_t>& times) {
        MNN_PRINT("time: ");
        for (auto& t : times) {
            MNN_PRINT("%f ", (float) t / 1000.0f);
        }
        auto avgTime = std::accumulate(times.begin(), times.end(), 0) / times.size();
        auto minTime = *std::min_element(times.begin(), times.end());
        MNN_PRINT("\nAvg time: %f ms, Min time: %f ms\n", (float) avgTime / 1000.0f, (float) minTime / 1000.0f);
    };

    net->resizeSession(session);
    // gpuNet->resizeSession(gpuSession);
    net->resizeSession(gpuSession);

    std::vector<uint64_t> times;

    // MNN_PRINT("CPU single\n");
    // for (int i = 0; i < rounds; i++) {
    //     auto start = getTimeInUs();
    //     net->runSessionWithCallBackInfo(session, before, after);
    //     auto end = getTimeInUs();
    //     times.push_back(end - start);
    // }
    // summarizeTime(times);

    MNN_PRINT("GPU single\n");
    times.clear();
    for (int i = 0; i < rounds; i++) {
        auto start = getTimeInUs();
        // gpuNet->runSessionWithCallBackInfo(gpuSession, gpuBefore, gpuAfter, false);
        net->runSessionWithCallBackInfo(gpuSession, gpuBefore, gpuAfter, false);
        gpuOutputTensor->wait(MNN::Tensor::MAP_TENSOR_READ, true);
        auto end = getTimeInUs();
        times.push_back(end - start);
    }
    summarizeTime(times);

    MNN_PRINT("CPU-GPU Parallel\n");
    times.clear();
    for (int i = 0; i < rounds; i++) {
        auto start = getTimeInUs();
        {
            MNN::AutoTime _t(__LINE__, "GPU sche");
            // gpuNet->runSessionWithCallBackInfoAsync(gpuSession, gpuBefore, gpuAfter, false);
            net->runSessionWithCallBackInfoAsync(gpuSession, gpuBefore, gpuAfter, false);
        }
        {
            MNN::AutoTime _t(__LINE__, "CPU comp");
            net->runSessionWithCallBackInfo(session, before, after);
        }
        {
            MNN::AutoTime _t(__LINE__, "GPU wait");
            gpuOutputTensor->wait(MNN::Tensor::MAP_TENSOR_READ, true);
        }
        auto end = getTimeInUs();
        times.push_back(end - start);
        MNN_PRINT("=====================================\n");
    }
    summarizeTime(times);

    // MNN_PRINT("CPU single full\n");
    // times.clear();
    // for (int i = 0; i < rounds; i++) {
    //     auto start = getTimeInUs();
    //     net->runSession(session);
    //     auto end = getTimeInUs();
    //     times.push_back(end - start);
    // }
    // summarizeTime(times);

    MNN_PRINT("GPU single full\n");
    times.clear();
    for (int i = 0; i < rounds; i++) {
        auto start = getTimeInUs();
        // gpuNet->runSession(gpuSession);
        net->runSession(gpuSession);
        gpuOutputTensor->wait(MNN::Tensor::MAP_TENSOR_READ, true);
        auto end = getTimeInUs();
        times.push_back(end - start);
    }
    summarizeTime(times);

    return 0;
}