/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "compile/prebuild_manager.h"
#include "binary/binary_manager.h"
#include "graph/node.h"
#include "inc/te_fusion_util_constants.h"
#include "inc/te_fusion_log.h"
#include "inc/te_fusion_check.h"
#include "common/common_utils.h"
#include "common/fusion_common.h"
#include "common/te_config_info.h"
#include "common/tbe_op_info_cache.h"
#include "compile/te_compile_task_cache.h"
#include "assemble_json/te_json_assemble.h"
#include "python_adapter/py_wrapper.h"
#include "python_adapter/py_decouple.h"
#include "python_adapter/py_object_utils.h"
#include "python_adapter/python_api_call.h"
#include "python_adapter/python_adapter_manager.h"
#include "python_adapter/pyobj_assemble_utils.h"
#include "compile/fusion_manager.h"

namespace te {
namespace fusion {
using namespace ge;
/*
 * @brief: get PreBuildManager instance
 * @return PreBuildManager*: PreBuildManager instance
 */
PreBuildManager &PreBuildManager::GetInstance()
{
    static PreBuildManager preBuildManager;
    return preBuildManager;
}

/*
 * @brief: pre build single op, save op parameters and record op pattern
 * @param [in] opinfo: op total parameter set
 * @return bool: pre build op ok or not
 */
bool PreBuildManager::PreBuildTbeOpInner(TbeOpInfo &opinfo, uint64_t taskId, uint64_t graphId)
{
    std::vector<ge::Node *> teGraphNode;
    ge::NodePtr nodePtr = nullptr;
    ge::OpDescPtr outNodeDesc = nullptr;
    (void)opinfo.GetNode(nodePtr);
    if (nodePtr != nullptr) {
        teGraphNode.push_back(nodePtr.get());
        outNodeDesc = nodePtr->GetOpDesc();
    }
    auto task = std::make_shared<OpBuildTask>(graphId, taskId, 0, teGraphNode,
                                              outNodeDesc, OP_TASK_STATUS::OP_TASK_FAIL);
    TE_FUSION_CHECK((task == nullptr), {
        REPORT_TE_INNER_ERROR("Task is null, taskID[%lu:%lu].", graphId, taskId);
        return false;
    });
    task->pPrebuildOp = &opinfo;
    task->buildType = opinfo.GetBuildType();
    bool res = DoPrebuildOp(task);
    if (!res) {
        REPORT_TE_INNER_ERROR("Do prebuild failed, taskID[%lu:%lu].", graphId, taskId);
        return false;
    }

    TeFusionManager *teFusionM = TeFusionManager::GetInstance();
    TE_FUSION_CHECK((teFusionM == nullptr), {
        TE_ERRLOG("Failed to get tbe fusion manager instance.");
        return false;
    });
    const std::lock_guard<std::mutex> fusionMgrLock(TeFusionManager::mtx_);
    TE_DBGLOG("Get TeFusionManager lock.");
    bool saveRes = teFusionM->SaveBuildTask(task);
    if (!saveRes) {
        TE_ERRLOG("Save prebuild task failed, taskID[%lu:%lu]", graphId, taskId);
    }
    return saveRes;
}

bool PreBuildManager::DoPrebuildOp(const OpBuildTaskPtr &opTask)
{
    TbeOpInfo &opinfo = *opTask->pPrebuildOp;
    bool res = false;
    TE_DBGLOG("Start to prebuild single op.");

    if (!TeFusionManager::RefreshCacheAndSinalManager()) {
        return false;
    }

    // save op moudle/name/args info
    TbeOpInfoPtr pTbeOp = opinfo.shared_from_this();
    TE_FUSION_CHECK((pTbeOp == nullptr), {
        TE_ERRLOG("Record prebuild op info failed, it is NULL!");
        return false;
    });
    TeConfigInfo::Instance().SetOpDebugConfig(opinfo.GetOpDebugConfig());
    std::string opRealName = opinfo.GetRealName();
    std::string opName;
    if (opTask->opNodes.size() > 0) {
        bool bres = TbeOpInfoCache::Instance().GetOpKeyNameByNode(opTask->opNodes[0], opName);
        TE_FUSION_CHECK(!bres, {
            TE_ERRLOG("Failed to get node key name.");
            return false;
        });
    } else {
        opName = opinfo.GetName();
    }

    CheckOpImplMode(pTbeOp);

    TbeOpInfoCache::Instance().UpdateTbeOpInfo(opName, pTbeOp);
    TbeOpInfoPtr selectedOpInfo = pTbeOp;

    const std::lock_guard<std::mutex> fusionMgrLock(TeFusionManager::mtx_);

    PythonApiCall::Instance().ResetL1SpaceSize();
    BinaryManager::FuzzyBuildSecondaryGeneralization(opTask, opName, selectedOpInfo);

    PyLockGIL gil;
    opTask->pTbeOpInfo = selectedOpInfo;
    std::string kernelName;
    PyObject *pyInputs = nullptr;
    PyObject *pyOutputs = nullptr;
    PyObject *pyAttrs = nullptr;
    res = AssembleOpArgs(*selectedOpInfo, kernelName, pyInputs, pyOutputs, pyAttrs, true);
    TE_FUSION_CHECK(!res, {
        REPORT_TE_INNER_ERROR("Failed to assemble op args with op[%s].", opName.c_str());
        return false;
    });
    AUTO_PY_DECREF(pyInputs);
    AUTO_PY_DECREF(pyOutputs);
    AUTO_PY_DECREF(pyAttrs);

    std::vector<ConstTbeOpInfoPtr> tbeOpInfoVec = {opTask->pTbeOpInfo};
    PyObject *optionValues = PyObjectUtils::GenPyOptionsInfo(tbeOpInfoVec);
    TE_FUSION_CHECK(optionValues == nullptr, {
        TE_ERRLOG("Failed to generate options info.");
        return false;
    });
    AUTO_PY_DECREF(optionValues);
    // build op common cfg
    res = PythonApiCall::Instance().SetOpParams(*selectedOpInfo, pyOutputs);
    TE_FUSION_CHECK(!res, {
        TE_ERRLOG("Failed to set op parameters.");
        return false;
    });

    if (!TeJsonAssemble::GeneratePrebuildJsonAndKernelName(opTask->opNodes, opTask->jsonStr, opTask->kernel)) {
        TE_ERRLOG("Failed to generate json and kernel name.");
        return false;
    }
    TE_INFOLOG("Core type of node[%s] is [%s].", opName.c_str(), opTask->pTbeOpInfo->GetOpCoreType().c_str());
    if (NeedSkipPrebuild(opTask, opinfo)) {
        TE_INFOLOG("Node [%s] needs to skip prebuild.", opName.c_str());
        return true;
    }

    // search for equal Op
    OpBuildTaskResultPtr opRes;
    res = CanReusePreBuildCache(selectedOpInfo, opTask, opRes);
    TE_FUSION_CHECK(res, {
        TE_INFOLOG("Prebuild success, pattern=[%s]. cache prebuild op found, taskID[%lu:%lu]",
                   opTask->kernel.c_str(), opTask->graphId, opTask->taskId);
        if (opRes != nullptr && opRes->preCompileRetPtr != nullptr) {
            opinfo.SetPattern(opRes->preCompileRetPtr->opPattern);
            TE_DBGLOG("Set pattern %s core type %s for node %s", opRes->preCompileRetPtr->opPattern.c_str(),
                       opRes->preCompileRetPtr->coreType.c_str(), opName.c_str());
            (void)opinfo.SetOpCoreType(opRes->preCompileRetPtr->coreType);
            (void)pTbeOp->SetOpCoreType(opRes->preCompileRetPtr->coreType);
        }
        return true;
    });

    bool ifUnknownShape = selectedOpInfo->GetIsUnknownShape();
    PyObject *pyTrue = HandleManager::Instance().get_py_true();
    PyObject *pyFalse = HandleManager::Instance().get_py_false();
    if (pyTrue == nullptr || pyFalse == nullptr) {
        TE_ERRLOG("Failed to get pyTrue or pyFalse from class HandleManager.");
        return false;
    }

    PyObject *pyUnknownShape = ifUnknownShape ? pyTrue : pyFalse;
    PyObject *pyInt64Mode = selectedOpInfo->GetFlagUseInt64() ? pyTrue : pyFalse;
    PyObject *pyIsDynamicImpl = selectedOpInfo->IsDynamicImpl() ? pyTrue : pyFalse;
    PyObject *contextDict = GenerateDictFromContext();
    TE_FUSION_CHECK(contextDict == nullptr, {
        TE_ERRLOG("Failed to generate context dict.");
        return false;
    });

    res = SetPrebuildTargetsToCtx(opinfo, contextDict);
    TE_FUSION_CHECK(!res, {
        TE_ERRLOG("Failed to SetPrebuildTargetsToCtx.");
        return false;
    });
    AUTO_PY_DECREF(contextDict);

    std::string passOptList;
    SetLicensePassOptList(passOptList);

    std::string opModule;
    std::string opFuncName;
    (void)selectedOpInfo->GetModuleName(opModule);
    selectedOpInfo->GetFuncName(opFuncName);

    res = PythonApiCall::Instance().UpdateSingleOpModule(*selectedOpInfo, opModule);

    TE_FUSION_CHECK(!res, {
        TE_ERRLOG("Failed to update and import single op module for %s.", opName.c_str());
        return false;
    });

    const std::string &opType = selectedOpInfo->GetOpType();
    TE_INFOLOG("Prebuilding op: node[%s], op type[%s], module name[%s], opFuncName[%s], op inputs: %s, outputs: %s, attrs: %s.", opName.c_str(), opType.c_str(),
               opModule.c_str(), opFuncName.c_str(), PyObjectToStr(pyInputs).c_str(),
               PyObjectToStr(pyOutputs).c_str(), PyObjectToStr(pyAttrs).c_str());

    std::string extraParams;
    std::string opImplSwitch;
    (void)selectedOpInfo->GetExtraParams(extraParams);
    selectedOpInfo->GetOpImplSwitch(opImplSwitch);
    std::string opImplMode = selectedOpInfo->GetOpImplMode();
    res = PythonAdapterManager::Instance().BuildOpAsync(opTask, "dispatch_prebuild_task", "ssssO(OOOO)OOOssss",
                                  opModule.c_str(), opRealName.c_str(), opType.c_str(), opFuncName.c_str(),
                                  pyUnknownShape, pyInputs, pyOutputs, pyAttrs, optionValues, pyInt64Mode,
                                  pyIsDynamicImpl, contextDict, passOptList.c_str(), extraParams.c_str(),
                                  opImplSwitch.c_str(), opImplMode.c_str());
    TE_FUSION_CHECK(res, {
        TeCompileTaskCache::Instance().SetOpPreBuildTask(selectedOpInfo, opTask);
        TE_INFOLOG("Prebuild task dispatched, taskID[%lu:%lu], module[%s], func[%s]",
                   opTask->graphId, opTask->taskId, opModule.c_str(), opFuncName.c_str());
        return true;
    });

    TE_ERRLOG("Failed to call 'dispatch_prebuild_task', taskID[%lu:%lu], module[%s], func[%s]",
              opTask->graphId, opTask->taskId, opModule.c_str(), opFuncName.c_str());
    return false;
}

bool PreBuildManager::CanReusePreBuildCache(const TbeOpInfoPtr &tbeOpInfoPtr, const OpBuildTaskPtr &opTask,
                                            OpBuildTaskResultPtr &opRes)
{
    auto match = TeCompileTaskCache::Instance().GetOpBuildTaskPair(tbeOpInfoPtr);
    if (match == nullptr) {
        return false;
    }

    // fuzz build, need build missing shape range
    // fuzzy cache build, in this case(not single process) first is nullptr not mean another equal task is running
    if (!opTask->missSupportInfo.empty()) {
        TE_INFOLOG("incremental building kernel[%s].", opTask->kernel.c_str());
        return false;
    }

    if (match->first == nullptr) {
        // kernel not generated yet, there's another equal task running background.
        // save current task to the queue, when previous equal task finished, all tasks in this
        // queue became finished
        if (match->second.size() <= 0) {
            TE_WARNLOG("No matched task. Size = %zu.", match->second.size());
            return false;
        }
        TE_DBGLOG("Op[%lu] background build task running[%lu],[%s] enqueue taskID[%lu:%lu] [%s]",
                  opTask->opNodes.size(), match->second.size(), match->second[0]->kernel.c_str(),
                  opTask->graphId, opTask->taskId, opTask->kernel.c_str());

        // set current op's kernel name
        opTask->kernel = match->second[0]->kernel;
        opTask->status = OP_TASK_STATUS::OP_TASK_PENDING;
        match->second.push_back(opTask);
        return true;
    }

    opRes = match->first;
    TE_DBGLOG("Get same input/output/attr op, taskID[%lu:%lu], opRes->result is %s",
              opTask->graphId, opTask->taskId, opRes->result.c_str());

    opTask->opRes = opRes;
    if (opRes->statusCode != 0) {
        opTask->status = OP_TASK_STATUS::OP_TASK_FAIL;
    } else {
        opTask->status = OP_TASK_STATUS::OP_TASK_SUCC;
    }

    return true;
}

// report warning if op in op_impl_mode list do not support op_impl_mode
void PreBuildManager::CheckOpImplMode(const TbeOpInfoPtr &tbeOpInfo)
{
    const std::string &opType = tbeOpInfo->GetOpType();
    const std::string &opImplMode = tbeOpInfo->GetOpImplMode();
    if (opImplMode.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lockGuard(opImplModeSupportedMapMutex_);
    auto iter = opImplModeSupportedMap_.find(opType);
    if (iter != opImplModeSupportedMap_.end()) {
        // only once for each optype
        return;
    }

    bool isSupported = false;
    if (!PythonApiCall::Instance().IsOpImplModeSupported(tbeOpInfo, isSupported)) {
        TE_WARNLOG("Failed to call check_op_impl_mode, opType: %s", opType.c_str());
        return;
    }
    opImplModeSupportedMap_.emplace(opType, isSupported);

    if (!isSupported) {
        TE_DBGLOG("Op impl mode is not supported by op type[%s], the op impl mode[%s] will be ignore.",
                  opType.c_str(), opImplMode.c_str());
    } else {
        TE_DBGLOG("Op impl mode is supported by op type[%s], the op impl mode is [%s]",
                  opType.c_str(), opImplMode.c_str());
    }
}

bool PreBuildManager::NeedSkipPrebuild(const OpBuildTaskPtr &opTask, TbeOpInfo &tbeOpInfo)
{
    bool isCoreTypeFixed = !tbeOpInfo.IsNeedPreCompile() || TeConfigInfo::Instance().IsDisableOpCompile();
    TE_DBGLOG("Op[%s] core type fixed: %d", tbeOpInfo.GetRealName().c_str(), isCoreTypeFixed);

    // if coretype is not needed or coretype check is disabled
    if (isCoreTypeFixed) {
        OpBuildTaskResultPtr opResPtr = nullptr;
        TE_FUSION_MAKE_SHARED(opResPtr = std::make_shared<OpBuildTaskResult>(), return false);
        TE_FUSION_MAKE_SHARED(opResPtr->preCompileRetPtr = std::make_shared<PreCompileResult>("Opaque"), return false);
        opResPtr->preCompileRetPtr->coreType = "Default";

        const std::string &prebuildPattern = tbeOpInfo.GetPrebuildPattern();
        if (!prebuildPattern.empty() && prebuildPattern != STR_UNDEFINDED) {
            tbeOpInfo.SetPattern(prebuildPattern);
            opResPtr->result = "success";
            opResPtr->preCompileRetType = PreCompileResultType::Config;
            opResPtr->preCompileRetPtr->opPattern = prebuildPattern;
            UpdateOpTaskForPreBuildCache(opTask, opResPtr);
            TE_DBGLOG("Op[%s] prebuildPattern jump prebuild, prebuildPattern is %s.", tbeOpInfo.GetRealName().c_str(),
                      prebuildPattern.c_str());
            return true;
        }

        // when ub_fuison is closed, fe will set pattern to opaque
        if (TeConfigInfo::Instance().IsDisableUbFusion()) {
            opResPtr->result = "success";
            opResPtr->preCompileRetType = PreCompileResultType::Config;
            if (tbeOpInfo.IsNeedPreCompile()) {
                opResPtr->preCompileRetPtr->coreType = "AiCore";
            }
            UpdateOpTaskForPreBuildCache(opTask, opResPtr);
            TE_DBGLOG("Op[%s] UbFusion disabled, pattern is %s.",
                      tbeOpInfo.GetRealName().c_str(), tbeOpInfo.GetPattern().c_str());
            return true;
        }

        if (tbeOpInfo.IsSingleOpScene()) {
            tbeOpInfo.SetPattern("Opaque");
            opResPtr->result = "success";
            opResPtr->preCompileRetType = PreCompileResultType::Config;
            opResPtr->preCompileRetPtr->opPattern = "Opaque";
            UpdateOpTaskForPreBuildCache(opTask, opResPtr);
            TE_DBGLOG("Op[%s] SingleOpScene jump prebuild.", tbeOpInfo.GetRealName().c_str());
            return true;
        }
    }

    if (CanReusePreBuildDiskCache(opTask)) {
        TE_DBGLOG("Op[%s] reuse prebuild disk cache.", tbeOpInfo.GetRealName().c_str());
        return true;
    }

    return false;
}

bool PreBuildManager::SetPrebuildTargetsToCtx(const TbeOpInfo &tbeOpInfo, PyObject *contextDict)
{
    /*
     * The FE judge the coreType, Use need_precompile display the judgment result.
     * if need_precompile = false, coreType fixed.
     */
    bool coreTypeFixed = !tbeOpInfo.IsNeedPreCompile() || TeConfigInfo::Instance().IsDisableOpCompile();
    bool patternFixed = (!tbeOpInfo.GetPrebuildPattern().empty() && tbeOpInfo.GetPrebuildPattern() != STR_UNDEFINDED)
                         || TeConfigInfo::Instance().IsDisableUbFusion();
    TE_DBGLOG("Op[%s] coreTypeFixed [%s], prebuildPattern [%s], patternFixed [%s].", tbeOpInfo.GetName().c_str(),
        coreTypeFixed ? "True" : "False", tbeOpInfo.GetPrebuildPattern().c_str(), patternFixed ? "True" : "False");

    PyObject *pyCoreType = HandleManager::Instance()._Py_BuildValue("s", "coreType");
    TE_FUSION_NOTNULL(pyCoreType);
    PyObject *pyPattern = HandleManager::Instance()._Py_BuildValue("s", "pattern");
    TE_FUSION_NOTNULL(pyPattern);

    PyObject *pyTargetsList = nullptr;
    if (!coreTypeFixed && !patternFixed) {
        pyTargetsList = HandleManager::Instance().TE_PyList_New(2);  // 2 elements in the list
        TE_FUSION_CHECK((pyTargetsList == nullptr ||
            HandleManager::Instance().TE_PyList_SetItem(pyTargetsList, 0, pyCoreType) != 0 ||
            HandleManager::Instance().TE_PyList_SetItem(pyTargetsList, 1, pyPattern) != 0), {
            TE_PY_DECREF(pyCoreType);
            TE_PY_DECREF(pyPattern);
            TE_ERRLOG("Op[%s] TE_PyList_SetItem failed.", tbeOpInfo.GetName().c_str());
            return false;
        });
    } else if (!coreTypeFixed) {
        pyTargetsList = HandleManager::Instance().TE_PyList_New(1);
        TE_FUSION_CHECK((pyTargetsList == nullptr ||
            HandleManager::Instance().TE_PyList_SetItem(pyTargetsList, 0, pyCoreType) != 0), {
            TE_PY_DECREF(pyCoreType);
            TE_PY_DECREF(pyPattern);
            TE_ERRLOG("Op[%s] TE_PyList_SetItem pyCoreType failed.", tbeOpInfo.GetName().c_str());
            return false;
        });
        TE_PY_DECREF(pyPattern);
    } else if (!patternFixed) {
        pyTargetsList = HandleManager::Instance().TE_PyList_New(1);
        TE_FUSION_CHECK((pyTargetsList == nullptr ||
            HandleManager::Instance().TE_PyList_SetItem(pyTargetsList, 0, pyPattern) != 0), {
            TE_PY_DECREF(pyCoreType);
            TE_PY_DECREF(pyPattern);
            TE_ERRLOG("Op[%s] TE_PyList_SetItem pyPattern failed.", tbeOpInfo.GetName().c_str());
            return false;
        });
        TE_PY_DECREF(pyCoreType);
    } else {
        TE_PY_DECREF(pyCoreType);
        TE_PY_DECREF(pyPattern);
        return true;
    }
    int ires = HandleManager::Instance().TE_PyDict_SetItemString(contextDict, "prebuild_targets", pyTargetsList);
    TE_PY_DECREF(pyTargetsList);
    TE_FUSION_CHECK_WITH_DUMP_PYERR((ires != 0), {
        TE_PY_DECREF(pyCoreType);
        TE_PY_DECREF(pyPattern);
        TE_ERRLOG("Op[%s] set prebuild_targets failed", tbeOpInfo.GetName().c_str());
        return false;
    });
    return true;
}

void PreBuildManager::UpdateOpTaskForPreBuildCache(const OpBuildTaskPtr &opTask, const OpBuildTaskResultPtr &opResPtr)
{
    opTask->status = OP_TASK_STATUS::OP_TASK_SUCC;
    opTask->opRes = opResPtr;
    opTask->opRes->statusCode = 0;

    if (HasCustomOp(opTask->opNodes)) {
        TE_DBGLOGF("DiskCache reuse of the custom node[%s] succeeded.", opTask->opNodes[0]->GetName().c_str());
    }

    TE_DBGLOG("Node[%s] no need to precompile, kernelName[%s].",
              opTask->opNodes[0]->GetName().c_str(), opTask->kernel.c_str());
}

bool PreBuildManager::CanReusePreBuildDiskCache(const OpBuildTaskPtr &opTask)
{
    if (opTask->opNodes.size() == 0) {
        TE_WARNLOGF("There is no node in opTask.");
        return false;
    }

    if (!TeCacheManager::Instance().IsCacheEnable()) {
        return false;
    }

    TE_DBGLOG("Pre CompileCache kernel_name is [%s]", opTask->kernel.c_str());
    PreCompileResultPtr preCompileRetPtr = TeCacheManager::Instance().MatchPreCompileCache(opTask->kernel);
    if (preCompileRetPtr != nullptr) {
        OpBuildTaskResultPtr opResPtr = nullptr;
        TE_FUSION_MAKE_SHARED(opResPtr = std::make_shared<OpBuildTaskResult>(), return false);
        opResPtr->result = "success";
        opResPtr->preCompileRetType = PreCompileResultType::Cache;
        opResPtr->preCompileRetPtr = preCompileRetPtr;
        TE_DBGLOG("Kernel [%s] cache JSON file path exists.", opTask->kernel.c_str());
        UpdateOpTaskForPreBuildCache(opTask, opResPtr);
        return true;
    }

    return false;
}
} // namespace fusion
} // namespace te