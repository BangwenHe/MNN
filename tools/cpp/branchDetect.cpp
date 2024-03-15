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

    std::shared_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(modelFile.c_str()));
    auto bufferAndLength = net->getModelBuffer();
    const void* buffer = bufferAndLength.first;
    size_t length = bufferAndLength.second;

    auto netT = MNN::UnPackNet(buffer);
    std::vector<MNN::OpT*> inputOps, outputOps;
    for (auto& op : netT->oplists) {
        if (op->inputIndexes.size() == 0) {
            inputOps.push_back(op.get());
        }

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

    // print nextOps
    for (auto& op : nextOps) {
        MNN_PRINT("op: %s => ", op.first->name.c_str());
        for (auto& nextOp : op.second) {
            MNN_PRINT("nextOp: %s\n", nextOp->name.c_str());
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
    int gpuBranchIndex = 1;
    auto* gpuSession = net->createSession(gpuConfig);
    auto gpuInputTensor = net->getSessionInput(gpuSession, nullptr);
    auto gpuOutputTensor = net->getSessionOutput(gpuSession, nullptr);
    gpuInputTensor->copyFromHostTensor(inputTensor);

    auto searchFunc = [] (MNN::OpT* op, const MNN::OperatorInfo* info) {
        return op->name == info->name();
    };

    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
        for (auto& t : tensors) {
            if (std::find_if(branchNodes[branchIndex].begin(), branchNodes[branchIndex].end(), [&info] (MNN::OpT* op) {return op->name == info->name();}) == branchNodes[branchIndex].end()) {
                return false;
            }
        }
        return true;
    };
    MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
        // for (auto& t : tensors) {
        //     if (std::find_if(branchNodes[branchIndex].begin(), branchNodes[branchIndex].end(), [&info] (MNN::OpT* op) {return op->name == info->name();}) == branchNodes[branchIndex].end()) {
        //         return false;
        //     }
        // }
        return true;
    };

    MNN::TensorCallBackWithInfo gpuBefore = [&](const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
        for (auto& t : tensors) {
            if (std::find_if(branchNodes[gpuBranchIndex].begin(), branchNodes[gpuBranchIndex].end(), [&info] (MNN::OpT* op) {return op->name == info->name();}) == branchNodes[gpuBranchIndex].end()) {
                return false;
            }
        }
        return true;
    };
    MNN::TensorCallBackWithInfo gpuAfter = [&](const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
        // for (auto& t : tensors) {
        //     if (std::find_if(branchNodes[gpuBranchIndex].begin(), branchNodes[gpuBranchIndex].end(), [&info] (MNN::OpT* op) {return op->name == info->name();}) == branchNodes[gpuBranchIndex].end()) {
        //         return false;
        //     }
        // }
        return true;
    };


    for (int i = 0; i < rounds; i++) {
        MNN::AutoTime timer(__LINE__, __func__);
        net->runSessionWithCallBackInfo(gpuSession, gpuBefore, gpuAfter);
        net->runSessionWithCallBackInfo(session, before, after);
        gpuOutputTensor->wait(MNN::Tensor::MAP_TENSOR_READ, true);
    }

    for (int i = 0; i < rounds; i++) {
        MNN::AutoTime timer(__LINE__, __func__);
        net->runSession(session);
    }

    return 0;
}