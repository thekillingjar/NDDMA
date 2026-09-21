/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "binary/binary_manager.h"
#include "binary/shape_generalization.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include "graph/anchor.h"
#include "inc/te_fusion_check.h"
#include "python_adapter/python_api_call.h"
#include "common/common_utils.h"
#include "common/fusion_common.h"
#include "common/tbe_op_info_cache.h"
#include "common/te_context_utils.h"
#include "binary/generate_simple_key.h"

namespace te {
namespace fusion {
using namespace ge;
namespace {
    const std::string SINGLE_OP_JIT_COMPILE = "op_jit_compile";
}

bool BinaryManager::SingleOpReuseOmBinary(const OpBuildTaskPtr &opTask, json &binListJson) const
{
    TE_FUSION_TIMECOST_START(SingleOpReuseOmBinary);
    GeneralizedResult generalizedResult;
    if (!ShapeGeneralization::GeneralizeOps(opTask, generalizedResult)) {
        TE_DBGLOG("Generalize ops(%s) did not succeed. Not reuse om binary files", GetTaskNodeName(opTask).c_str());
        return false;
    }

    if (!BinaryMatchWithStaticKeyAndDynInfo(opTask, generalizedResult, binListJson)) {
        TE_DBGLOG("Node(%s) staticKey or dynInfo not match. Need to compile", GetTaskNodeName(opTask).c_str());
        return false;
    }

    std::string jsonFilePath;
    if (!IsBinaryFileValid(true, opTask, binListJson, jsonFilePath)) {
        TE_DBGLOG("Node(%s) binary file invalid. Not reuse om binary files.", GetTaskNodeName(opTask).c_str());
        return false;
    }

    // set build result
    if (!SetOmBinaryReuseResult(opTask, jsonFilePath)) {
        TE_DBGLOG("Node(%s) not set binary reuse result. Not reuse om binary files.",
            GetTaskNodeName(opTask).c_str());
        return false;
    }

    TE_INFOLOG("The node(%s) binary om json file(%s) can be reused. No need to compile",
        GetTaskNodeName(opTask).c_str(), jsonFilePath.c_str());
    TE_FUSION_TIMECOST_END(SingleOpReuseOmBinary, "SingleOpReuseOmBinary");
    return true;
}

bool BinaryManager::MatchSingleOpAttrs(const json &opParamsJson, const json &binJson)
{
    TE_DBGLOG("Match single op attrs");

    const auto opKeyIter = opParamsJson.find(ATTRS);
    const auto binKeyIter = binJson.find(ATTRS);
    if (opKeyIter == opParamsJson.end() && binKeyIter == binJson.end()) {
        TE_DBGLOGF("Both json has no attrs. Match did not succeed");
        return true;
    }
    if (opKeyIter == opParamsJson.end() || binKeyIter == binJson.end()) {
        TE_DBGLOGF("One json has no attrs. Match did not succeed.");
        return false;
    }

    const json &opAttrs = opKeyIter.value();
    const json &binAttrs = binKeyIter.value();
    if (!opAttrs.is_array() || !binAttrs.is_array()) {
        TE_DBGLOGF("Format error! 'attrs' should be array. Match did not succeed.");
        return false;
    }

    if (opAttrs.size() != binAttrs.size()) {
        TE_DBGLOGF("Op attrs size(%zu) is not same as binary size(%zu). Match did not succeed", opAttrs.size(),
            binAttrs.size());
        return false;
    }

    TE_DBGLOG("Op attrs are %s, bin attrs are %s", opAttrs.dump().c_str(), binAttrs.dump().c_str());

    for (auto opIter = opAttrs.begin(), binIter = binAttrs.begin();
        opIter != opAttrs.end() && binIter != binAttrs.end(); ++opIter, ++binIter) {
        const json &opValue = *opIter;
        const json &binValue = *binIter;

        if (!MatchSingleAttr(opValue, binValue)) {
            TE_DBGLOGF("Match single op attr did not succeed");
            return false;
        }
    }

    TE_DBGLOG("Match single op attrs successfully");
    return true;
}

bool BinaryManager::GetOpJitCompileForBinReuse(const OpBuildTaskPtr &opTask, bool &needCheckNext) const
{
    if (opTask == nullptr) {
        return false;
    }
    bool isOpReuseBinary = false;
    for (const auto &node : opTask->opNodes) {
        ConstTbeOpInfoPtr tbeOpInfo = TbeOpInfoCache::Instance().GetTbeOpInfoByNode(node);
        if (tbeOpInfo == nullptr) {
            continue;
        }
        JitCompileType opJitCompile = tbeOpInfo->GetOpJitCompile();
        TE_DBGLOG("Jit compile of op[%s, %s] is [%d], IsUnknownShape is %d", node->GetName().c_str(),
            node->GetType().c_str(), opJitCompile, tbeOpInfo->GetIsUnknownShape());

        if (opJitCompile != JitCompileType::DEFAULT) {
            needCheckNext = false;
        }

        if (opJitCompile == JitCompileType::REUSE_BINARY && tbeOpInfo->GetIsUnknownShape()) {
            // reuse binary when in dynamic shape scenario
            return true;
        } else if (opJitCompile == JitCompileType::ONLINE) {
            isOpReuseBinary = false;
        } else if (opJitCompile == JitCompileType::STATIC_BINARY_DYNAMIC_ONLINE && !tbeOpInfo->GetIsUnknownShape()) {
            // reuse binary when in static shape scenario
            return true;
        } else if (opJitCompile == JitCompileType::STATIC_BINARY_DYNAMIC_BINARY) {
            // reuse binary when in any scenario
            return true;
        }
    }
    return isOpReuseBinary;
}

bool BinaryManager::GetSingleOpSceneFlag(const OpBuildTaskPtr &opTask, bool &isSingleOpScene) const
{
    for (auto &node : opTask->opNodes) {
        ConstTbeOpInfoPtr tbeOpInfo = TbeOpInfoCache::Instance().GetTbeOpInfoByNode(node);
        if (tbeOpInfo == nullptr) {
            TE_WARNLOGF("Node %s GetTbeOpInfoByNode failed", opTask->outNode->GetName().c_str());
            return false;
        }

        tbeOpInfo->GetOriSingleOpSceneFlag(isSingleOpScene);
        if (isSingleOpScene) {
            break;
        }
    }
    return true;
}

bool BinaryManager::CheckJitCompileForBinReuse(const OpBuildTaskPtr &opTask) const
{
    // check if it is in single op scenario(such as ACL)
    bool isSingleOpScene = false;
    if (!GetSingleOpSceneFlag(opTask, isSingleOpScene)) {
        return false;
    }
    TE_DBGLOG("Node(%s), single op scenario is %d", GetTaskNodeName(opTask).c_str(), isSingleOpScene);
    // resue binary for static shape only when it isn't in single op scenario
    if (isSingleOpScene && !IsDynamicNodes(opTask)) {
        TE_DBGLOG("Node(%s), single static op scenario can't reuse binary.",
            GetTaskNodeName(opTask).c_str());
        return false;
    }

    bool opJitCompileFlag = false;
    bool bres = ge::AttrUtils::GetBool(opTask->opNodes.at(0)->GetOpDesc(), SINGLE_OP_JIT_COMPILE, opJitCompileFlag);
    if (bres) {
        if (opJitCompileFlag) {
            TE_DBGLOG("op[%s] set op_jit_compile true. Need to compile.", GetTaskNodeName(opTask).c_str());
            return false;
        }
        TE_DBGLOG("op[%s] set op_jit_compile false", GetTaskNodeName(opTask).c_str());
        return true;
    }

    // check if op.jit_compile is set to false in op info config
    bool needCheckNext = true;
    bool isOpReuseBinary = GetOpJitCompileForBinReuse(opTask, needCheckNext);
    TE_DBGLOG("Op[%s] reuse from op jit compile, isOpReuseBinary is [%d], opJitCompile configured flag[%d]",
              GetTaskNodeName(opTask).c_str(), isOpReuseBinary, !needCheckNext);
    // reuse binary when op.jit_compile is false and it is in dynamic shape scenario
    if (isOpReuseBinary) {
        TE_DBGLOG("Op node(%s) reuse binary when op config - jit compile is set to false",
            GetTaskNodeName(opTask).c_str());
        return true;
    }
    if (!needCheckNext) {
        TE_DBGLOG("Op[%s] get opJitCompile configured flag[%d]. Need to compile",
                  GetTaskNodeName(opTask).c_str(), !needCheckNext);
        return isOpReuseBinary;
    }

    // reuse binary when ge.jit_compile is set to false and it isn't in single op scenario
    JIT_MODE jitMode = TeContextUtils::GetJitCompile();
    TE_DBGLOG("Node(%s), get jit_compile from ge is %d", GetTaskNodeName(opTask).c_str(), jitMode);
    // use binary when jit_compile is set to false or (jit_compile is set to auto and it is dynamic shape)
    if (jitMode == JIT_MODE::JIT_USE_BINARY || (jitMode == JIT_MODE::JIT_AUTO && IsDynamicNodes(opTask))) {
        TE_DBGLOG("Node(%s), use binary when jit_compile is set to false "
            "or (jit_compile is set to auto and it is dynamic shape)",
            GetTaskNodeName(opTask).c_str());
        return true;
    }

    // reuse binary when ge.jit_compile is true and op.jit_compile is true
    // and it is in all range scenario(shape range=[(1,-1),...] or shape=(-2,))
    if (IsSpecShapeAndRange(opTask)) {
        TE_DBGLOG("Node(%s), reuse binary when shape is all shape(-1 or -2, no range).",
            GetTaskNodeName(opTask).c_str());
        return true;
    }

    TE_DBGLOG("Get jit_compile for node(%s) is %d. Need to compile", GetTaskNodeName(opTask).c_str(), jitMode);
    return false;
}

} // namespace fusion
} // namespace te

