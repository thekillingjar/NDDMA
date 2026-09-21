/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <Python.h>
#include <set>
#include <map>
#include <functional>
#include <utility>
#include <sstream>
#include <climits>
#include <tuple>
#include <iostream>
#include <string>
#include <cstdlib>
#include <chrono>
#include <random>
#include <ctime>
#include <iomanip>
#include <functional>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <sys/stat.h>

#include "compile/fusion_manager.h"
#include "inc/te_fusion_check.h"
#include "python_adapter/py_decouple.h"
#include "python_adapter/python_api_call.h"
#include "python_adapter/py_wrapper.h"
#include "tensor_engine/fusion_api.h"
#include "common/common_utils.h"
#include "common/fusion_common.h"
#include "common/tbe_op_info_cache.h"
#include "common/te_config_info.h"
#include "common/te_context_utils.h"
#include "common/te_file_utils.h"
#include "assemble_json/te_json_assemble.h"
#include "compile/te_compile_task_cache.h"
#include "python_adapter/python_adapter_manager.h"
#include "graph/anchor.h"
#include "graph/ge_context.h"
#include "graph/ge_local_context.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "ge_common/ge_api_types.h"

namespace te {
namespace fusion {
bool TeFusionManager::MakeBinaryRootDir(std::string &rootPath)
{
    std::string lockRealPath = RealPath(rootPath);
    if (lockRealPath.empty()) {
        TE_DBGLOG("Compile path[%s] does not exist, try to create one", rootPath.c_str());
        bool ret = mkdir(rootPath.c_str(), S_IRWXU | S_IRGRP | S_IXGRP);
        if (ret != 0) {
            TE_ERRLOG("Create directory [%s] failed, error number: %d.", rootPath.c_str(), errno);
            return false;
        }
    }
    return true;
}

/*
 * if error occurs, false is returned;
 * if build result exists, true is returned and isFound = true;
 * if build result does not exist and the same task is running, true is returned, isBuilding = true, isFound = false
 * if build result does not exist and no task is running, true is returned, isBuilding = false, isFound = false
 */
bool TeFusionManager::CheckResultInBinaryBuildDir(const OpBuildTaskPtr &opTask, bool &isBuilding, bool &isFound) const
{
    // obj & json file is building
    if (FindLockFpHandle(opTask->kernel) != nullptr) {
        TE_INFOLOG("Lock file[%s] handle is not null", opTask->kernel.c_str());
        isBuilding = true;
        return true;
    }

    std::string kernelName = opTask->kernel;
    std::size_t pos = opTask->kernel.find("__kernel");
    if (pos != std::string::npos) {
        kernelName = opTask->kernel.substr(0, pos);
    }
    std::string objFilePath = TeConfigInfo::Instance().GetKernelMetaParentDir() + "/kernel_meta/" + kernelName + ".o";
    std::string jsonFilePath = TeConfigInfo::Instance().GetKernelMetaParentDir() +
            "/kernel_meta/" + kernelName + ".json";
    FILE* objFile = TeFileUtils::LockFile(objFilePath, false, F_RDLCK);
    FILE* jsonFile = TeFileUtils::LockFile(jsonFilePath, false, F_RDLCK);

    // obj & json file is ready
    if (objFile != nullptr && jsonFile != nullptr) {
        TE_INFOLOG("%s and %s exist.", objFilePath.c_str(), jsonFilePath.c_str());
        TeFileUtils::ReleaseLockFile(objFile, objFilePath);
        TeFileUtils::ReleaseLockFile(jsonFile, jsonFilePath);
        opTask->opRes->result = "success";
        opTask->opRes->statusCode = 0;
        opTask->opRes->jsonFilePath = jsonFilePath;
        opTask->opRes->compileRetType = CompileResultType::Online;
        isFound = true;
        return true;
    }
    if (objFile != nullptr) {
        TeFileUtils::ReleaseLockFile(objFile, objFilePath);
    }
    if (jsonFile != nullptr) {
        TeFileUtils::ReleaseLockFile(jsonFile, jsonFilePath);
    }

    // will build obj & json file, lock
    std::string lockPath = "";
    const std::string& kernelMetaParentDir = TeConfigInfo::Instance().GetKernelMetaParentDir();
    int size = static_cast<int>(kernelMetaParentDir.size());
    for (int i = 0; i < size; ++i) {
        lockPath += kernelMetaParentDir[i];
        if (kernelMetaParentDir[i] == '\\' ||
            kernelMetaParentDir[i] == '/' || i == size - 1) {
            if (!MakeBinaryRootDir(lockPath)) {
                return false;
            }
        }
    }

    std::string lockKernelName = RealPath(lockPath) + "/" + opTask->kernel + ".lock";
    FILE *fp = fopen(lockKernelName.c_str(), "a+");
    if (fp == nullptr) {
        TE_ERRLOG("Failed to open file [%s].", lockKernelName.c_str());
        return false;
    }

    int res = chmod(lockKernelName.c_str(), FILE_AUTHORITY);
    TE_DBGLOG("Update file[%s] authority result: %d.", lockKernelName.c_str(), res);

    if (!TeFileUtils::FcntlLockFileSet(fileno(fp), F_WRLCK, 0)) {
        TE_INFOLOG("Can not Lock file[%s].", lockKernelName.c_str());
        fclose(fp);
        isBuilding = true;
        return true;
    }

    SaveLockFpHandle(opTask->kernel, fp);
    TE_DBGLOG("Lock file[%s] success", lockKernelName.c_str());

    // unlock and close file after compile complete or process fail
    return true;
}

/*
 * check if can reuse binary build cache
 * if can reuse disk cache or task cache, true is returned;
 * if can not reuse disk cache or task cache, false is returned
 * if error occurs, false is returned and hasError is set to true
 */
bool TeFusionManager::CanReuseBuildBinaryCache(OpBuildTaskPtr &opTask, bool &hasError)
{
    // chech if build result exists already
    bool isFound = false;
    bool isBuilding = false;
    if (CheckResultInBinaryBuildDir(opTask, isBuilding, isFound)) {
        if (isFound) {
            TE_INFOLOG("Build result is ready, no need to create task.");
            opTask->status = OP_TASK_STATUS::OP_TASK_SUCC;
            return true;
        }
        if (isBuilding) {
            // 正在编译，尝试复用task缓存，如果复用失败，继续走在线编译
            TE_INFOLOG("The same kernel[%s] is building, insert task[%lu:%lu] to waiting list",
                       opTask->kernel.c_str(), opTask->graphId, opTask->taskId);
            OpBuildTaskResultPtr opRes;
            bool bres = false;
            if (opTask->isBuildBinaryFusionOp) {
                bres = CanReuseTaskCache(opTask->kernel, fusionOpsKernel_, opTask, opRes);
            } else {
                bres = CanReuseTaskCache(opTask->kernel, opKernelMap_, opTask, opRes);
            }
            TE_INFOLOG("try to reuse same kernel[%s] for task [%lu:%lu] result is [%u].",
                       opTask->kernel.c_str(), opTask->graphId, opTask->taskId, bres);
            return bres;
        }
        return false;
    }
    opTask->status = OP_TASK_STATUS::OP_TASK_FAIL;
    TE_ERRLOG("Error occurred, build quit");
    hasError = true;
    return false;
}

bool TeFusionManager::CheckBuildBinaryOpTaskResult(OpBuildTaskPtr &opTask)
{
    TE_INFOLOG("Build Binary task dispatched, taskID[%lu:%lu], kernel=[%s]",
               opTask->graphId, opTask->taskId,
               opTask->kernel.c_str());

    opTask->status = OP_TASK_STATUS::OP_TASK_PENDING;
    auto kernelMap = &opKernelMap_;
    if (opTask->isBuildBinaryFusionOp) {
        kernelMap = &fusionOpsKernel_;
    }
    auto opIter = kernelMap->find(opTask->kernel);
    if (opIter == kernelMap->end()) {
        (void)kernelMap->emplace(opTask->kernel, std::make_pair(nullptr, std::vector<OpBuildTaskPtr>({opTask})));
        TE_INFOLOG("Insert task[%lu:%lu] to op kernel map, kernel name is %s",
                   opTask->graphId, opTask->taskId, opTask->kernel.c_str());
        return true;
    }
    TE_ERRLOG("Insert task[%lu:%lu] into op kernel map, duplicated kernel name is: %s",
              opTask->graphId, opTask->taskId,
              opTask->kernel.c_str());
    opTask->status = OP_TASK_STATUS::OP_TASK_FAIL;
    return false;
}

bool TeFusionManager::BuildBinarySingleOp(OpBuildTaskPtr &opTask)
{
    std::vector<ge::Node *> &teGraphNode = opTask->opNodes;
    ge::Node *pNode = teGraphNode[0];
    TE_FUSION_CHECK(pNode == nullptr, {
        TE_ERRLOG("Failed to get teGraphNode[0].");
        return false;
    });

    TE_DBGLOG("Start to build single op [%s], taskID[%lu:%lu], json: %s.", pNode->GetName().c_str(),
              opTask->graphId, opTask->taskId, opTask->generalizedJson["supportInfo"].dump().c_str());
    std::string keyName;
    bool res = TbeOpInfoCache::Instance().GetOpKeyNameByNode(pNode, keyName);
    TE_FUSION_CHECK(!res, {
        TE_ERRLOG("Failed to obtain op key with name=[%s].", pNode->GetName().c_str());
        return false;
    });

    TbeOpInfoPtr pOpInfo;
    res = TbeOpInfoCache::Instance().GetTbeOpInfoByName(keyName, pOpInfo);
    TE_FUSION_CHECK(!res, {
        TE_ERRLOGF("Fail to get tbe op info of node %s.", pNode->GetName().c_str());
        return false;
    });
    TbeOpInfo &opInfo = *pOpInfo;

    std::string kernelName;
    if (opTask->generalizedJson.find("kernelName") != opTask->generalizedJson.end()) {
        kernelName = opTask->generalizedJson["kernelName"];
        opTask->kernel = kernelName;
    } else {
        TE_ERRLOG("Failed to get kernel name.");
        return false;
    }
    (void)opInfo.SetKernelName(kernelName);
    std::string opsPathNamePrefix;
    (void)ge::AttrUtils::GetStr(opTask->opNodes[0]->GetOpDesc(), OPS_PATH_NAME_PREFIX, opsPathNamePrefix);
    opInfo.SetOpsPathNamePrefix(opsPathNamePrefix);
    opTask->pTbeOpInfo = pOpInfo;

    res = PythonApiCall::Instance().SetL1SpaceSize(opInfo.GetL1Space());
    TE_FUSION_CHECK_WITH_DUMP_PYERR(!res, {
        TE_ERRLOG("Failed to set node[%s] L1 info.", pNode->GetName().c_str());
        return false;
    });

    std::string opModule;
    (void)opInfo.GetModuleName(opModule);
    TE_FUSION_CHECK(!PythonApiCall::Instance().UpdateSingleOpModule(opInfo, opModule), {
        TE_ERRLOG("Failed to update and import single op module.");
        return false;
    });

    std::string opImplMode = opInfo.GetOpImplMode();
    std::string opType = pNode->GetType();

    std::string opPattern = TeCompileTaskCache::Instance().GetOpPattern(keyName);

    std::string opFuncName;
    (void)opInfo.GetFuncName(opFuncName);

    std::string extraParams;
    (void)opInfo.GetExtraParams(extraParams);

    std::string opImplSwitch;
    opInfo.GetOpImplSwitch(opImplSwitch);

    opTask->isHighPerformace = (TeContextUtils::GetPerformanceMode() == "high") ? true : false;

    std::vector<ConstTbeOpInfoPtr> tbeOpInfoVec = { pOpInfo };
    std::map<std::string, std::string> optionsMap;
    res = TeJsonAssemble::GenerateOptionsMap(tbeOpInfoVec, optionsMap);
    TE_FUSION_CHECK(!res, {
        TE_ERRLOG("Failed to generate options map.");
        return false;
    });

    std::map<std::string, std::string> contextDict;
    GenerateContextDict(contextDict);

    std::string passOptList;
    SetLicensePassOptList(passOptList);

    std::string relationStr;
    GenerateSingleOpRelationParam(opTask->opNodes.at(0), relationStr);

    bool hasError = false;
    if (CanReuseBuildBinaryCache(opTask, hasError)) {
        TE_INFOLOG("Can reuse build binary cache");
        return true;
    }
    if (hasError) {
        TE_ERRLOG("Failed to reuse build binary cache; an error occurred.");
        return false;
    }

    nlohmann::json taskDesc;
    taskDesc["graph_id"] = opTask->graphId;
    taskDesc["task_id"] = opTask->taskId;
    taskDesc["l1_size"] = PythonApiCall::Instance().GetL1SpaceSize();
    taskDesc["op_module"] = opModule;
    taskDesc["op_name"] = pNode->GetName();
    taskDesc["op_type"] = opType;
    taskDesc["op_func"] = opFuncName;
    // remove "__kernel0" in kernel name, since "__kernel0" will be added when building
    std::size_t pos = kernelName.find("__kernel");
    if (pos != std::string::npos) {
        taskDesc["kernel_name"] = kernelName.substr(0, pos);
    } else {
        taskDesc["kernel_name"] = kernelName;
    }
    std::string optionalInputMode;
    std::string dynamicParamMode;
    if (opTask->generalizedJson.find("supportInfo") != opTask->generalizedJson.end()) {
        taskDesc["inputs"] = opTask->generalizedJson["supportInfo"]["inputs"];
        taskDesc["outputs"] = opTask->generalizedJson["supportInfo"]["outputs"];
        taskDesc["attrs"] = opTask->generalizedJson["supportInfo"]["attrs"];
        taskDesc["int64_mode"] = opTask->generalizedJson["supportInfo"]["int64Mode"];
        if (opTask->generalizedJson["supportInfo"].find("dynamicParamMode") !=
            opTask->generalizedJson["supportInfo"].end()) {
            dynamicParamMode = opTask->generalizedJson["supportInfo"]["dynamicParamMode"];
        }
        if (opTask->generalizedJson["supportInfo"].find("optionalInputMode") !=
            opTask->generalizedJson["supportInfo"].end()) {
            optionalInputMode = opTask->generalizedJson["supportInfo"]["optionalInputMode"];
        }
    }
    taskDesc["unknown_shape"] = true;
    taskDesc["options"] = optionsMap;
    if (!opTask->pre_task.empty()) {
        taskDesc["pre_task"] = opTask->pre_task;
    }
    if (!opTask->post_task.empty()) {
        taskDesc["post_task"] = opTask->post_task;
    }
    taskDesc["is_dynamic_impl"] = true;
    taskDesc["op_pattern"] = opPattern;
    taskDesc["context_param"] = contextDict;
    taskDesc["pass_opt_list"] = passOptList;
    taskDesc["extra_params"] = extraParams;
    taskDesc["relation_param"] = relationStr;
    taskDesc["op_impl_switch"] = opImplSwitch;
    taskDesc["op_impl_mode"] = opImplMode;
    taskDesc["optional_input_mode"] = optionalInputMode;
    taskDesc["dynamic_param_mode"] = dynamicParamMode;
    TE_INFOLOGF("Python call dispatch_binary_single_op_compile_task, json is %s.", taskDesc.dump().c_str());
    if (!PythonAdapterManager::Instance().DispatchBinarySingleOpCompileTask(taskDesc.dump())) {
        TE_ERRLOG("Failed to call the Python function dispatch_binary_single_op_compile_task.");
        opTask->status = OP_TASK_STATUS::OP_TASK_FAIL;
        return false;
    }
    return CheckBuildBinaryOpTaskResult(opTask);
}

bool TeFusionManager::BuildBinaryFusionOp(OpBuildTaskPtr &opTask)
{
    std::vector<ConstTbeOpInfoPtr> tbeOpInfoVec;
    for (ge::Node *node : opTask->opNodes) {
        TE_FUSION_NOTNULL(node);
        ConstTbeOpInfoPtr opInfo = TbeOpInfoCache::Instance().GetTbeOpInfoByNode(node);
        if (opInfo == nullptr) {
            TE_ERRLOG("Node[%s] get tbe op information by node failed.", node->GetName().c_str());
            return false;
        }
        tbeOpInfoVec.emplace_back(opInfo);
    }

    // use kernel_name in json
    if (opTask->generalizedJson.find("kernelName") != opTask->generalizedJson.end()) {
        opTask->kernel = opTask->generalizedJson["kernelName"];
    } else {
        TE_ERRLOG("Failed to get kernel name.");
        return false;
    }

    std::map<std::string, std::string> optionsMap;
    bool res = TeJsonAssemble::GenerateOptionsMap(tbeOpInfoVec, optionsMap);
    TE_FUSION_CHECK(!res, {
        TE_ERRLOG("Failed to generate options map.");
        return false;
    });

    std::map<std::string, std::string> contextDict;
    GenerateContextDict(contextDict);

    std::string passOptList;
    SetLicensePassOptList(passOptList);

    std::string relationStr;
    GenerateFusionOpRelationParam(opTask->opNodes, relationStr);

    ge::Node *firstNode = opTask->opNodes[0];
    std::string fixpipeFusionFlag;
    (void)ge::AttrUtils::GetStr(firstNode->GetOpDesc(), FUSION_OP_BUILD_OPTIONS, fixpipeFusionFlag);

    std::string optionalInputMode;
    (void)ge::AttrUtils::GetStr(firstNode->GetOpDesc(), OPTIONAL_INPUT_MODE, optionalInputMode);

    bool hasError = false;
    if (CanReuseBuildBinaryCache(opTask, hasError)) {
        TE_INFOLOG("Can reuse build binary cache");
        return true;
    }
    if (hasError) {
        TE_ERRLOG("Failed to reuse build binary cache; an error occurred.");
        return false;
    }

    nlohmann::json taskDesc;
    taskDesc["graph_id"] = opTask->graphId;
    taskDesc["task_id"] = opTask->taskId;
    taskDesc["l1_size"] = PythonApiCall::Instance().GetL1SpaceSize();
    taskDesc["graph_desc"] = opTask->jsonStr;
    taskDesc["generalized_desc"] = opTask->generalizedJson["supportInfo"].dump().c_str();
    taskDesc["op_name"] = firstNode->GetName();
    taskDesc["kernel_name"] = opTask->kernel;
    taskDesc["options"] = optionsMap;
    if (!opTask->pre_task.empty()) {
        taskDesc["pre_task"] = opTask->pre_task;
    }
    if (!opTask->post_task.empty()) {
        taskDesc["post_task"] = opTask->post_task;
    }
    taskDesc["context_param"] = contextDict;
    taskDesc["pass_opt_list"] = passOptList;
    taskDesc["reused_relation"] = relationStr;
    taskDesc["fixpipe_ub_cfg"] = fixpipeFusionFlag;
    taskDesc["optional_input_mode"] = optionalInputMode;
    TE_INFOLOGF("Python call dispatch_binary_fusion_op_compile_task, json is %s.", taskDesc.dump().c_str());
    if (!PythonAdapterManager::Instance().DispatchBinaryFusionOpCompileTask(taskDesc.dump())) {
        TE_ERRLOG("Failed to call the Python function dispatch_binary_fusion_op_compile_task.");
        opTask->status = OP_TASK_STATUS::OP_TASK_FAIL;
        return false;
    }
    return CheckBuildBinaryOpTaskResult(opTask);
}
}
}
