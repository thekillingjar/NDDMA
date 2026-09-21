/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_TE_FUSION_SOURCE_INC_TE_FUSION_TYPES_H_
#define ATC_OPCOMPILER_TE_FUSION_SOURCE_INC_TE_FUSION_TYPES_H_

#include <string>
#include <memory>
#include <ctime>
#include <nlohmann/json.hpp>
#include "tensor_engine/tbe_op_info.h"
#include "graph/op_kernel_bin.h"
#include "graph/node.h"

namespace te {
namespace fusion {
enum class JIT_MODE {
    JIT_USE_BINARY      = 0,     // use binary first
    JIT_COMPILE_ONLINE  = 1,     // compile online
    JIT_AUTO            = 2      // compile online for static shape and use binary for dynamic shape
};

enum class CompileCacheMode {
    Disable = 0,
    Enable,
    Force
};

enum class OpDebugLevel {
    Disable = 0,  // Disable debug
    Level1,       // Enable TBE pipe_all and generate the operator CCE file and Python-CCE mapping file (.json)
    Level2,       // Include level1 and enable the CCE compiler -O0-g
    Level3,       // Disable debug and keep generating kernel file (.o and .json)
    Level4        // Include level3 and generate cce file (.cce) and the UB fusion computing description file (.json)
};

enum class CompileResultType {
    Online = 0,
    Cache,
    Binary
};

enum class PreCompileResultType {
    Online = 0,
    Cache,
    Config
};

struct PreCompileResult {
    std::string opPattern;        // op pattern
    std::string coreType;         // core type
    std::string prebuiltOptions;  // pre-build options
    PreCompileResult(const std::string &pattern) : opPattern(pattern) {}
};
using PreCompileResultPtr = std::shared_ptr<PreCompileResult>;

using JsonInfoPtr = std::shared_ptr<nlohmann::json>;
struct CompileResult {
    std::string jsonPath;    // json file path
    std::string binPath;     // bin file path
    std::string headerPath;  // header file path for task fusion of nano
    JsonInfoPtr jsonInfo;
    ge::OpKernelBinPtr kernelBin;
    CompileResult() : jsonInfo(nullptr), kernelBin(nullptr) {}
};
using CompileResultPtr = std::shared_ptr<CompileResult>;

enum class OP_TASK_STATUS {
    OP_TASK_FAIL          = -1,
    OP_TASK_PENDING       = 0,
    OP_TASK_SUCC          = 1,
    OP_TASK_DO_NOT_SAVE_TASK = 2
};

struct OpBuildTaskResult {
    int type;
    uint64_t graphId;
    uint64_t taskId;

    int statusCode;
    std::string result;
    std::string infoMsg;

    std::string errArgs;
    std::string pyExceptMsg;

    // built compile info
    std::string compile_info_key;
    // built compile info
    std::string compile_info_str;

    bool backToSingleOpBinReuse;

    // built json file path
    std::string jsonFilePath;

    CompileResultType compileRetType = CompileResultType::Online;
    CompileResultPtr compileRetPtr = nullptr;

    PreCompileResultType preCompileRetType = PreCompileResultType::Online;
    PreCompileResultPtr preCompileRetPtr = nullptr;
};
using OpBuildTaskResultPtr = std::shared_ptr<OpBuildTaskResult>;

struct OpBuildTask {
    uint64_t graphId;
    uint64_t taskId;
    uint64_t sgt_slice_shape_index;
    std::vector<ge::Node *> opNodes;
    ge::OpDescPtr outNode;

    OP_TASK_STATUS status;
    // Except debug mode, kernel is equal to opUniqueKey in other scene.
    // Debug mode, kernel will be updated to the value of kernelName in binary json file.
    std::string kernel;  // kernel name
    std::string opUniqueKey; // op unique key

    TbeOpInfoPtr pTbeOpInfo;
    std::string jsonStr;

    TbeOpInfo *pPrebuildOp;

    OpBuildTaskResultPtr opRes;
    std::string pre_task;
    std::string post_task;
    std::string missSupportInfo;
    BUILD_TYPE buildType;
    bool isHighPerformace;
    bool isHighPerformaceRes;
    bool isBuildBinarySingleOp;
    bool isBuildBinaryFusionOp;
    bool newCompile;
    int64_t maxKernelId;
    std::time_t start_time;
    nlohmann::json supportInfoJson;
    std::string taskFusionUniqueKey;
    std::string attrKernelName;
    nlohmann::json generalizedJson;
    std::string superKernelUniqueKey;

    OpBuildTask() :
        graphId(), taskId(), sgt_slice_shape_index(), opNodes(), outNode(),
        status(), kernel(), opUniqueKey(),
        pTbeOpInfo(), jsonStr(), pPrebuildOp(nullptr),
        opRes(), pre_task(), post_task(), missSupportInfo(), buildType(),
        isHighPerformace(), isHighPerformaceRes(), isBuildBinarySingleOp(),
        isBuildBinaryFusionOp(), newCompile(), maxKernelId(), start_time(),
        supportInfoJson(), taskFusionUniqueKey(), attrKernelName(), generalizedJson(), superKernelUniqueKey() {}
    OpBuildTask(uint64_t graph_id, uint64_t task_id, uint64_t sgt_index, std::vector<ge::Node *> op_nodes,
        ge::OpDescPtr out_node, OP_TASK_STATUS task_status) :
        graphId(graph_id), taskId(task_id), sgt_slice_shape_index(sgt_index), opNodes(std::move(op_nodes)),
        outNode(out_node), status(task_status), kernel(), opUniqueKey(),
        pTbeOpInfo(), jsonStr(), pPrebuildOp(nullptr),
        opRes(), pre_task(), post_task(), missSupportInfo(), buildType(),
        isHighPerformace(), isHighPerformaceRes(), isBuildBinarySingleOp(),
        isBuildBinaryFusionOp(), newCompile(), maxKernelId(), start_time(),
        supportInfoJson(), taskFusionUniqueKey(), attrKernelName(), generalizedJson(), superKernelUniqueKey() {}
};
using OpBuildTaskPtr = std::shared_ptr<OpBuildTask>;

struct CheckNodeInfo {
    int64_t scopeId;
    std::string backend;
    std::string graphName;
    CheckNodeInfo() : scopeId(0) {}
};

enum class NullType {
    NORMAL, // default
    SINGLE_VALUE, // a nan/inf/-inf
    LIST_VALUE // a list of nan/inf/-inf
};

struct NullDesc {
    NullType nullType = NullType::NORMAL;
    std::vector<std::string> nullDesc;
};
}
}
#endif  // ATC_OPCOMPILER_TE_FUSION_SOURCE_INC_TE_FUSION_TYPES_H_