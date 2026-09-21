/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TE_FUSION_FUSION_COMMON_H
#define TE_FUSION_FUSION_COMMON_H

#include <string>
#include <vector>
#include "graph/node.h"
#include "inc/te_fusion_log.h"
#include "tensor_engine/tbe_op_info.h"
#include "inc/te_fusion_util_constants.h"
#include "python_adapter/py_wrapper.h"

namespace te {
namespace fusion {
bool CheckMemoryL1MemoryL2(const std::string &opType, const std::vector<TbeOpParam> &inPutsOrOutPuts);

std::string GetInputNodesDesc(const ge::Node &node);

std::string GetNodesName(const std::vector<ge::Node *> &nodes);

PyObject *GenerateDictFromContext();

void GenerateExtraInfoForFusionOpContext(const std::vector<ge::Node *> &oriNodes, PyObject *pyContextDict);

void GenerateExtraInfoForContext(const TbeOpInfo &opInfo, const ge::OpDescPtr &opDescPtr, PyObject *pyContextDict);

bool SetContextItem(PyObject *pyContextDict, const std::string &contextKey,
                    const std::string &switchKey, const std::string &optKey);

bool SetContextItem(PyObject *pyContextDict, const std::string &optKey, const std::string &custom_opp_path);

void SetContextItem(std::map<std::string, std::string> &contextDict, const std::string &contextKey,
                    const std::string &switchKey, const std::string &optKey);

void GenerateContextDict(map<string, string> &contextDict);

void SetLicensePassOptList(std::string &passOptList);

bool GetCheckNodeInfo(const ge::OpDescPtr &opDescPtr, CheckNodeInfo &checkNodeInfo);

bool VerifyNodeInfo(const CheckNodeInfo &currentCheckNodeInfo, const CheckNodeInfo &checkNodeInfo);

bool CheckNodeList(const std::vector<ge::Node *> &teGraphNode);

bool CheckNodeList(const std::vector<ge::Node *> &teGraphNode, CheckNodeInfo &checkNodeInfo,
                   std::vector<std::string> &verifyOpType);

void GetVariableAttrValue(const TbeOpInfo &opInfo, std::vector<std::string> &variableAttrs);

bool JudgeAttrIsVariableAttr(const TbeAttrValue &attrValue, const std::vector<std::string> &variableAttrs);

std::string GetSessionGraphId(const ge::Node *node);

bool GetOpInputKeyName(const ge::Node *node, std::vector<std::string> &key_name);

bool TbeAttrDtypeToString(const ATTR_DTYPE &dtype, std::string &res);

bool HasCustomOp(const std::vector<ge::Node *> &nodes);

std::string GetTaskNodeName(const OpBuildTaskPtr &opTask);

void UpdateStaticOpModulePath(const std::string &opsPathNamePrefix, std::string &opModule, size_t pos, bool flag);

void UpdateOpModuleFromStaicToDynamicAndAddPrefix(const TbeOpInfo &tbeOpInfo, std::string &opModule);

bool IsOpParameterValid(const std::string &opModule, const std::string &opFuncName);

void OriginalOpNamesSplicing(const std::vector<std::string> &originalOpNames, std::string &opNameStr);

void GetNodeListStr(const std::vector<ge::Node *> &teGraphNode, std::string &opNames);

ge::Node *GetPreviousNode(const ge::Node *node, const uint32_t index);

bool GetSubOpLoc(int64_t skCount, int64_t skSubId, std::string &locStr);
}  // namespace fusion
}  // namespace te
#endif  // TE_FUSION_FUSION_COMMON_H
