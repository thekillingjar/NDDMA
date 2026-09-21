/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tensor_engine/fusion_api.h"
#include <vector>
#include <Python.h>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <string>
#include <dirent.h>
#include <unistd.h>
#include <sstream>
#include <nlohmann/json.hpp>
#include "inc/te_fusion_check.h"
#include "inc/te_fusion_error_code.h"
#include "binary/binary_manager.h"
#include "common/common_utils.h"
#include "common/te_file_utils.h"
#include "common/te_config_info.h"
#include "common/fusion_common.h"
#include "common/signal_manager.h"
#include "assemble_json/te_json_assemble.h"
#include "compile/fusion_manager.h"
#include "compile/prebuild_manager.h"
#include "compile/superkernel_task.h"
#include "file_handle/te_file_handle.h"
#include "dfxinfo_manager/trace_utils.h"
#include "dfxinfo_manager/dfxinfo_manager.h"
#include "python_adapter/python_api_call.h"
#include "python_adapter/python_adapter_manager.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/ge_context.h"
#include "register/op_check.h"
#include "registry/op_impl_space_registry_v2.h"
#include "exe_graph/lowering/exe_res_generation_ctx_builder.h"
#include "common/tbe_op_info_cache.h"

using namespace ge;
namespace te {
using namespace fusion;
namespace {
bool ParseCheckOpSupportedInfo(std::string &jsonStr, CheckSupportedInfo &checkSupportedInfo)
{
    try {
        nlohmann::json jsonDesc = nlohmann::json::parse(jsonStr);
        std::string isSupported = jsonDesc["isSupported"];
        TE_DBGLOG("The isSupported value is %s", isSupported.c_str());
        if (isSupported == "True") {
            checkSupportedInfo.isSupported = FULLY_SUPPORTED;
        } else if (isSupported == "False") {
            checkSupportedInfo.isSupported = NOT_SUPPORTED;
        } else if (isSupported == "Unknown") {
            checkSupportedInfo.isSupported = PARTIALLY_SUPPORTED;
        } else {
            TE_WARNLOG("Invalid isSupported value[%s]", isSupported.c_str());
            return false;
        }
        if (jsonDesc.contains("dynamicCompileStatic")) {
            std::string dynamicCompileStatic = jsonDesc["dynamicCompileStatic"];
            TE_DBGLOG("The dynamicCompileStatic value is %s", dynamicCompileStatic.c_str());
            if (dynamicCompileStatic == "True") {
                checkSupportedInfo.dynamicCompileStatic = true;
            } else if (dynamicCompileStatic == "False") {
                checkSupportedInfo.dynamicCompileStatic = false;
            } else {
                TE_WARNLOG("Invalid dynamicCompileStatic value[%s]", dynamicCompileStatic.c_str());
                return false;
            }
        }
        if (jsonDesc.contains("reason")) {
            std::string reason = jsonDesc["reason"];
            checkSupportedInfo.reason = reason;
            TE_DBGLOG("Reason value is %s", reason.c_str());
        }
        checkSupportedInfo.allImplChecked = true;
    } catch (std::exception &e) {
        REPORT_TE_INNER_ERROR("Failed to parser jsonStr: %s, reason is %s", jsonStr.c_str(), e.what());
        return false;
    }

    return true;
}

bool UpdateFinComTaskListFromPy(TeFusionManager *&pInstance, const bool &needRecordLog, const uint64_t &graphId,
                                vector<FinComTask> &tasks)
{
    if (!pInstance->GetFinishedCompilationTask(graphId)) {
        TE_ERRLOG("GetFinishedCompilationTask false");
        return false;
    }

    (void)std::copy_if(pInstance->finComTaskList.begin(), pInstance->finComTaskList.end(),
                       std::back_inserter(tasks), [&graphId](FinComTask &i){return i.graphId == graphId;});

    pInstance->finComTaskList.erase(std::remove_if(pInstance->finComTaskList.begin(),
                                                   pInstance->finComTaskList.end(),
                                                   [&graphId](FinComTask t){return t.graphId == graphId;}),
                                    pInstance->finComTaskList.end());
    bool taskFinished = (tasks.size() > 0 || needRecordLog);
    if (taskFinished) {
        TE_DBGLOG("Reported tasks size: [%zu], graphId[%lu].", tasks.size(), graphId);
    }

    return true;
}
}

/**
 * @brief: Tbe Initialize proccess
 * @param [in] options            ddk version
 * @return [out] bool             succ or fail for Tbe Initialize
 */
extern "C" bool TbeInitialize(const std::map<std::string, std::string>& options, bool* isSupportParallelCompilation)
{
    TE_FUSION_TIMECOST_START(InitTeFusion);
    TE_INFOLOG("Begin to do TbeInitialize");
    const std::lock_guard<std::mutex> fusionMgrLock(TeFusionManager::mtx_);
    TE_INFOLOG("Get TeFusionManager lock.");
    TraceUtils::SubmitGlobalTrace("Begin to initialize TeFusion.");
    int setEnvRes = setenv("CONTEXT_MODELCOMPILING", "TRUE", 1);
    TE_FUSION_CHECK(setEnvRes == -1, {
        TeInnerErrMessageReport(EM_INNER_ERROR, "failed to set environment variable.");
        return false;
    });

    if (!TeConfigInfo::Instance().Initialize(options)) {
        TeInnerErrMessageReport(EM_INNER_ERROR, "Failed to initialize TeConfigInfo.");
        return false;
    }
    TraceUtils::SubmitGlobalTrace("Te config info has been initialized.");

    SignalManager::Instance().Initialize();
    TraceUtils::SubmitGlobalTrace("Signal manager has been initialized.");

    DfxInfoManager::Instance().Initialize();
    TraceUtils::SubmitGlobalTrace("Dfx manager has been initialized.");

    TE_FUSION_TIMECOST_START(InitPyAdapter);
    if (!PythonAdapterManager::Instance().Initialize()) {
        TeInnerErrMessageReport(EM_INNER_ERROR, "Failed to initialize TeConfigInfo.");
        return false;
    }
    TE_FUSION_TIMECOST_END(InitPyAdapter, "PythonAdapterManager::Initialize");
    TraceUtils::SubmitGlobalTrace("Python adapter manager has been initialized.");

    if (isSupportParallelCompilation != nullptr) {
        *isSupportParallelCompilation = PythonAdapterManager::Instance().IsInitParallelCompilation();
    }
    if (PythonAdapterManager::Instance().IsInitParallelCompilation()) {
        CreateKernelMetaTempDir();
        TraceUtils::SubmitGlobalTrace("Kernel meta tempDir has been created.");
    }

    // init binary manager
    BinaryManager::Instance().Initialize(options);

    auto now = std::chrono::high_resolution_clock::now();
    uint64_t ts = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    srand(static_cast<uint32_t>(ts));

    TeFusionManager *managerInstance = TeFusionManager::GetInstance();
    TE_FUSION_CHECK(managerInstance == nullptr, {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "failed to initialize tbe.");
        return false;
    });
    // record pid and kernelMetaPath to buildPidTimeIDInfo.json
    RecordPidTimeIdInfo();
    TE_FUSION_TIMECOST_END(InitTeFusion, "TeFusion::Initialize");
    return true;
}

/**
 * @brief: tbe finalize proccess
 * @return [out] bool             succ or fail for Tbe Finalize
 */
extern "C" bool TbeFinalize()
{
    TE_INFOLOG("Begin to Finalize tbe");
    const std::lock_guard<std::mutex> fusionMgrLock(TeFusionManager::mtx_);
    TE_DBGLOG("Get TeFusionManager::mtx_");

    TraceUtils::SubmitGlobalTrace("Begin to finalize TeFusion.");
    TeFusionManager::GetInstance()->ReportPendingTaskInfo();
    TE_INFOLOG("Reuse binary op count is [%d].", TeFusionManager::GetInstance()->reuseCount);

    string npuCollectPath = GetNpuCollectPath();
    string kernelMetaPath = TeConfigInfo::Instance().GetKernelMetaParentDir() + "/kernel_meta";
    bool saveKernelMeta = (TeConfigInfo::Instance().GetEnvSaveKernelMeta() == "1");
    TE_INFOLOG("npuCollectPath = %s, kernelMetaPath = %s, saveKernelMetaFlag[%d].",
               npuCollectPath.c_str(), kernelMetaPath.c_str(), saveKernelMeta);

    if (!npuCollectPath.empty()) {
        if (TeFileUtils::CreateMultiLevelDir(npuCollectPath)) {
            /* cp all files from kernel_meta to npuCollectPath/kernel_meta/pid_device_id/ */
            TeFileUtils::CopyDirFileToNewDir(kernelMetaPath, npuCollectPath);
            TraceUtils::SubmitGlobalTrace("Files under kernel meta dir has been copy to npu collect path.");
        }
    }

    if (npuCollectPath != kernelMetaPath && !saveKernelMeta) {
        ClearKernelMetaProc(kernelMetaPath);
    }

    // remove pid and kernelMetaPath record and remove remain files
    RemovePidTimeIdInfo();
    RemoveAllWorkFiles();
    TraceUtils::SubmitGlobalTrace("Files under kernel meta dir and kernel meta temp dir has been removed.");
    TeCacheManager::Instance().Finalize();
    BinaryManager::Instance().Finalize();
    SignalManager::Instance().Finalize();
    if (!PythonAdapterManager::Instance().Finalize()) {
        return false;
    }

    DfxInfoManager::Instance().Finalize();
    TeFusionManager::GetInstance()->Finalize();
    SuperKernelTaskManager::GetInstance().Finalize();
    TE_INFOLOG("Tbe finalized. Dynamic handle closed.");
    return true;
}

extern "C" bool TbeFinalizeSessionInfo(const std::string &session_graph_id)
{
    TbeOpInfoCache::Instance().FinalizeSessionInfo(session_graph_id);
    TeCacheManager::Instance().ClearOpArgsCache(session_graph_id);
    (void)PythonAdapterManager::Instance().CallGcCollect();
    return true;
}

/*
 * @brief: check tbe op capability
 * @param [in] opinfo: op total parameter set
 *
 * @param [out] IsSupport: result to support or not
 *              reason: When check supported returns true, reason is empty. Otherwise, reason
 *              is set as the unsupported reason which is generated by operator implementation.
 * @return bool: check process ok or not
 */
extern "C" bool CheckOpSupported(TbeOpInfo& opinfo, CheckSupportedInfo &checkSupportedInfo)
{
    TE_DBGLOG("Start to do Opchecksupport");
    bool res = true;
    std::string opName;
    std::string opModule;
    std::string opType;
    std::string opFuncName = FUNC_CHECK_SUPPORTED;

    const std::lock_guard<std::mutex> pythonApiCallLock(PythonApiCall::mtx_);
    TE_DBGLOG("get PythonApiCall lock");

    (void)opinfo.GetName(opName);
    (void)opinfo.GetModuleName(opModule);
    (void)opinfo.GetOpType(opType);

    TE_DBGLOG("Name=[%s], Module=[%s], FuncName=[%s], OpType=[%s].", opName.c_str(), opModule.c_str(),
              opFuncName.c_str(), opType.c_str());

    res = IsOpParameterValid(opModule, opFuncName);
    TE_FUSION_CHECK(!res, {
        TE_ERRLOG("IsOpParameterValid return false. Name=[%s], Module=[%s], FuncName=[%s]",
                  opName.c_str(), opModule.c_str(), opFuncName.c_str());
        return false;
    });

    ge::NodePtr nodePtr = nullptr;
    ge::AscendString result;
    (void)opinfo.GetNode(nodePtr);
    
    if (nodePtr != nullptr && nodePtr->GetOpDesc() != nullptr && gert::DefaultOpImplSpaceRegistryV2::GetInstance()
        .GetSpaceRegistry(static_cast<gert::OppImplVersionTag>(nodePtr->GetOpDesc()->GetOppImplVersion())) != nullptr) {
        const auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry(
            static_cast<gert::OppImplVersionTag>(nodePtr->GetOpDesc()->GetOppImplVersion()));
        auto funcsPtr = spaceRegistry->GetOpImpl(nodePtr->GetType().c_str());
        if (funcsPtr != nullptr && funcsPtr->check_support != nullptr) {
            gert::ExeResGenerationCtxBuilder exeCtxBuilder;
            auto resPtrHolder = exeCtxBuilder.CreateOpCheckContext(*nodePtr.get());
            if (resPtrHolder != nullptr) {
                TE_DBGLOG("CheckOpSupported start to call registered check_support function with OpCheckContext, Name=[%s], Module=[%s], FuncName=[%s].",
                           opName.c_str(), opModule.c_str(), opFuncName.c_str());
                auto opCheckContent = reinterpret_cast<gert::OpCheckContext *>(resPtrHolder->context_);
                if (funcsPtr->check_support(opCheckContent, result) == ge::GRAPH_SUCCESS) {
                    std::string jsonStr = result.GetString();
                    return ParseCheckOpSupportedInfo(jsonStr, checkSupportedInfo);
                }
                TE_WARNLOG("CheckOpSupported failed to call registered check_support function with OpCheckContext, Name=[%s], Module=[%s], FuncName=[%s].",
                            opName.c_str(), opModule.c_str(), opFuncName.c_str());
                return false;
            }
        }
    }
    // call cpp function if exists
    ge::AscendString opTypeString(opType.c_str());
    optiling::OP_CHECK_FUNC cppFunc =
                                optiling::OpCheckFuncRegistry::GetOpCapability(FUNC_CHECK_SUPPORTED, opTypeString);
    if (cppFunc != nullptr && nodePtr != nullptr) {
        ge::Operator op = OpDescUtils::CreateOperatorFromNode(nodePtr);
        ge::graphStatus ret = (*cppFunc)(op, result);
        if (ret == ge::SUCCESS) {
            std::string jsonStr = result.GetString();
            return ParseCheckOpSupportedInfo(jsonStr, checkSupportedInfo);
        }
        TE_WARNLOG("Failed to call Cpp function, Name=[%s], Module=[%s], FuncName=[%s]",
                   opName.c_str(), opModule.c_str(), opFuncName.c_str());
        return false;
    }
    TE_DBGLOG("Cpp function doesn't exist, continue to call python. Name=[%s], Module=[%s], FuncName=[%s]",
              opName.c_str(), opModule.c_str(), opFuncName.c_str());

    CheckSupportedResult isSupport;
    std::string reason;
    res = PythonApiCall::Instance().CheckSupported(opinfo, isSupport, reason);
    if (!res) {
        TE_WARNLOG("Failed to call python function. Name=[%s], Module=[%s], FuncName=[%s].",
                   opName.c_str(), opModule.c_str(), opFuncName.c_str());
        const std::lock_guard<std::mutex> fusionMgrLock(TeFusionManager::mtx_);
        TE_DBGLOG("Get TeFusionManager lock.");
        UpdateOpModuleFromStaicToDynamicAndAddPrefix(opinfo, opModule);
        TeFusionManager *teFusionM = TeFusionManager::GetInstance();
        TE_FUSION_CHECK((teFusionM == nullptr), {
            TE_ERRLOG("Failed to get tbe fusion manager instance.");
            return false;
        });
        auto emres = teFusionM->reportedErr_.emplace(std::make_pair(EM_INNER_WARNING + opName, 1));
        auto errIter = emres.first;
        if (!emres.second) {
            TE_WARNLOG("CheckSupported warning has already reported %ld times", errIter->second);
            errIter->second += 1;
        } else {
            REPORT_TE_INNER_WARN("Op[type:%s, name:%s] checkSupported by tbe[%s] failed.",
                                 opType.c_str(), opName.c_str(), opModule.c_str());
        }
        return false;
    }
    TE_INFOLOG("Operation [type: %s, name: %s] is %ssupported by module %s.", opType.c_str(), opName.c_str(),
        isSupport ? "" : "not ", opModule.c_str());
    checkSupportedInfo.isSupported = isSupport;
    checkSupportedInfo.reason = reason;
    return true;
}

/**
 * @brief pre build tbe op
 * @return [out] bool
 */
extern "C" bool PreBuildTbeOp(TbeOpInfo& opinfo, uint64_t taskId, uint64_t graphId)
{
    std::string opName;
    std::string opRealName;
    std::string opModule;
    std::string opFuncName;
    std::string opType;

    (void)opinfo.GetName(opName);
    (void)opinfo.GetRealName(opRealName);
    (void)opinfo.GetModuleName(opModule);
    (void)opinfo.GetFuncName(opFuncName);
    opinfo.GetOpType(opType);

    // func begin log
    TE_DBGLOG("Prebuild taskID:[%lu:%lu] Name=[%s], Module=[%s], FuncName=[%s].",
              graphId, taskId, opRealName.c_str(), opModule.c_str(), opFuncName.c_str());

    bool res = IsOpParameterValid(opModule, opFuncName);
    TE_FUSION_CHECK(!res, {
        TE_ERRLOG("IsOpParameterValid failed, Name=[%s], Module=[%s], FuncName=[%s].",
                  opRealName.c_str(), opModule.c_str(), opFuncName.c_str());
        return false;
    });

    res = PreBuildManager::GetInstance().PreBuildTbeOpInner(opinfo, taskId, graphId);

    TE_FUSION_CHECK(!res, {
        TE_ERRLOG("Op[type:%s, name:%s] prebuild with tbe[%s] failed.",
                  opType.c_str(), opRealName.c_str(), opModule.c_str());
        return false;
    });

    return true;
}

/**
 * @brief: build fusion op
 */
extern "C" OpBuildResCode TeFusion(std::vector<ge::Node *> teGraphNode, ge::OpDescPtr opDesc,
                                   const std::vector<ge::NodePtr> &toBeDel, uint64_t taskId, uint64_t graphId,
                                   const std::string &strategy)
{
    return TeFusionV(teGraphNode, opDesc, toBeDel, taskId, graphId, INVALID_SGT_INDEX, strategy);
}

/**
 * @brief: build sgt fusion op
 */
extern "C" OpBuildResCode TeFusionV(std::vector<ge::Node *> teGraphNode, ge::OpDescPtr opDesc,
                                    const std::vector<ge::NodePtr> &toBeDel, uint64_t taskId, uint64_t graphId,
                                    uint64_t sgtThreadIndex, const std::string &strategy)
{
    TE_DBGLOG("Start TeFusionV.");
    (void)toBeDel;
    // check input
    TE_FUSION_CHECK(!CheckNodeList(teGraphNode), {
        REPORT_TE_INNER_ERROR("Failed to check tbe nodes with taskID[%lu:%lu].", graphId, taskId);
        return OP_BUILD_FAIL;
    });

    const std::lock_guard<std::mutex> fusionMgrLock(TeFusionManager::mtx_);
    TE_DBGLOG("Get TeFusionManager lock.");
    TraceUtils::SubmitCompileDetailTrace(graphId, teGraphNode.at(0)->GetOpDesc()->GetId(),
                                         teGraphNode.at(0)->GetOpDesc()->GetType(), "get mutex lock");

    TeFusionManager *teFusionM = TeFusionManager::GetInstance();
    TE_FUSION_CHECK((teFusionM == nullptr), {
        TE_ERRLOG("Failed to get tbe fusion manager instance.");
        return OP_BUILD_FAIL;
    });

    auto task = std::make_shared<OpBuildTask>(graphId, taskId, sgtThreadIndex,
                                                          teGraphNode, opDesc, OP_TASK_STATUS::OP_TASK_PENDING);
    TE_FUSION_CHECK((task == nullptr), {
        TE_ERRLOG("Task is null, taskID[%lu:%lu].", graphId, taskId);
        return OP_BUILD_FAIL;
    });
    task->buildType = ACCURATELY_BUILD;

    return teFusionM->BuildTbeOp(task, strategy);
}

/**
 * @brief: get finished compilation task list
 * @return [out] list             finished compilation task list
 */
extern "C" bool WaitAllFinished(uint64_t graphId, vector<FinComTask> &tasks)
{
    const std::lock_guard<std::mutex> fusionMgrLock(TeFusionManager::mtx_);
    TeFusionManager *pInstance = TeFusionManager::GetInstance();
    if (pInstance == nullptr) {
        TE_WARNLOG("Wrong WaitAllFinished call. graphId[%lu]", graphId);
        return true;
    }
    bool needRecordLog = pInstance->UpdateInhibitionInfoForLog();
    if (needRecordLog) {
        TE_DBGLOG("GraphId[%ld] get TeFusionManager lock", graphId);
    }

    pInstance->PrintProgressHint();
    bool res = true;
    // get from finishedTask_
    auto cashedCompilTaskItr = pInstance->finishedTask_.find(graphId);
    if (cashedCompilTaskItr != pInstance->finishedTask_.end()) {
        for (auto &taskInfo : cashedCompilTaskItr->second) {
            auto &tmpBuildTaskPtr = taskInfo.second;
            TE_FUSION_CHECK(tmpBuildTaskPtr == nullptr,
                TE_WARNLOG("tmpBuildTaskPtr is null. graphId[%lu]", graphId);
                continue);
            FinComTask finComTaskItem;
            finComTaskItem.graphId = tmpBuildTaskPtr->graphId;
            finComTaskItem.taskId = tmpBuildTaskPtr->taskId;
            finComTaskItem.status = (tmpBuildTaskPtr->status == OP_TASK_STATUS::OP_TASK_SUCC) ? 0 : 1;

            if (tmpBuildTaskPtr->pPrebuildOp == nullptr) {
                finComTaskItem.teNodeOpDesc = tmpBuildTaskPtr->outNode;
                TE_FUSION_CHECK(!(pInstance->SetBuildResult(tmpBuildTaskPtr, tmpBuildTaskPtr->opRes)), {
                    TE_ERRLOG("Task[%lu:%lu] set build result failed.",
                              finComTaskItem.graphId, finComTaskItem.taskId);
                    return false;
                });
            } else {
                TE_FUSION_CHECK(!(pInstance->SetPreBuildResult(tmpBuildTaskPtr, tmpBuildTaskPtr->opRes)), {
                    TE_ERRLOG("Task[%lu:%lu] set prebuild result failed.",
                              finComTaskItem.graphId, finComTaskItem.taskId);
                    return false;
                });
            }
            tasks.push_back(finComTaskItem);
            TE_DBGLOG("Get finished task[%lu:%lu] status[%s] from FinishedTask", finComTaskItem.graphId,
                      finComTaskItem.taskId, (finComTaskItem.status == 0) ? "success" : "unsuccess");
            if (!tmpBuildTaskPtr->superKernelUniqueKey.empty()) {
                TE_INFOLOG("Superkernel compile task finished, kernel_name: %s.",
                            tmpBuildTaskPtr->superKernelUniqueKey.c_str());
            }
        }

        pInstance->finishedTask_.erase(cashedCompilTaskItr);
    } else {
        res = UpdateFinComTaskListFromPy(pInstance, needRecordLog, graphId, tasks);
    }

    return res;
}


/**
 * @brief select tbe op format
 * @param [in] tbeOpInfo            op info
 * @param [out] opDtypeformat       op date format
 * @return [out] bool               succ or fail
 */
extern "C" bool SelectTbeOpFormat(const TbeOpInfo &tbeOpInfo, std::string &opDtypeFormat)
{
    TE_DBGLOG("Start to select tbe op format");
    bool res = true;
    std::string opName;
    std::string opModule;
    std::string opType;
    std::string opFuncName = FUNC_OP_SELECT_FORMAT;

    const std::lock_guard<std::mutex> pythonApiCallLock(PythonApiCall::mtx_);
    TE_DBGLOG("Get PythonApiCall lock");

    TbeOpInfo &opinfo = const_cast<TbeOpInfo&>(tbeOpInfo);
    (void)opinfo.GetName(opName);
    (void)opinfo.GetModuleName(opModule);
    (void)opinfo.GetOpType(opType);
    TE_DBGLOG("Name=[%s], Module=[%s], FuncName=[%s], OpType=[%s].",
              opName.c_str(), opModule.c_str(), opFuncName.c_str(), opType.c_str());

    res = IsOpParameterValid(opModule, opFuncName);
    TE_FUSION_CHECK(!res, {
        TE_WARNLOG("IsOpParameterValid return false.");
        return false;
    });

    ge::NodePtr nodePtr = nullptr;
    ge::AscendString result;
    (void)opinfo.GetNode(nodePtr);

    if (nodePtr != nullptr && nodePtr->GetOpDesc() != nullptr && gert::DefaultOpImplSpaceRegistryV2::GetInstance()
        .GetSpaceRegistry(static_cast<gert::OppImplVersionTag>(nodePtr->GetOpDesc()->GetOppImplVersion())) != nullptr) {
        const auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry(
            static_cast<gert::OppImplVersionTag>(nodePtr->GetOpDesc()->GetOppImplVersion()));
        auto funcsPtr = spaceRegistry->GetOpImpl(nodePtr->GetType().c_str());
        if (funcsPtr != nullptr && funcsPtr->op_select_format != nullptr) {
            gert::ExeResGenerationCtxBuilder exeCtxBuilder;
            auto resPtrHolder = exeCtxBuilder.CreateOpCheckContext(*nodePtr.get());
            if (resPtrHolder != nullptr) {
                TE_DBGLOG("SelectTbeOpFormat start to call registered select_op_format function with OpCheckContext, Name=[%s], Module=[%s], FuncName=[%s].",
                           opName.c_str(), opModule.c_str(), opFuncName.c_str());
                auto opCheckContent = reinterpret_cast<gert::OpCheckContext *>(resPtrHolder->context_);
                if (funcsPtr->op_select_format(opCheckContent, result) == ge::GRAPH_SUCCESS) {
                    opDtypeFormat = result.GetString();
                    TE_DBGLOG("SelectTbeOpFormat succeeded to call registered select_op_format function with OpCheckContext, opDtypeFormat is %s.",
                               opDtypeFormat.c_str());
                    return true;
                }
                TE_WARNLOG("SelectTbeOpFormat failed to call registered select_op_format function with OpCheckContext, Name=[%s], Module=[%s], FuncName=[%s].",
                            opName.c_str(), opModule.c_str(), opFuncName.c_str());
                return false;
            }
        }
    }

    // call cpp function if exists
    ge::AscendString opTypeString(opType.c_str());
    optiling::OP_CHECK_FUNC cppFunc =
                                optiling::OpCheckFuncRegistry::GetOpCapability(FUNC_OP_SELECT_FORMAT, opTypeString);
    if (cppFunc != nullptr && nodePtr != nullptr) {
        ge::Operator op = OpDescUtils::CreateOperatorFromNode(nodePtr);
        ge::graphStatus ret = (*cppFunc)(op, result);
        if (ret == ge::SUCCESS) {
            opDtypeFormat = result.GetString();
            TE_INFOLOGF("Call Cpp function success, opDtypeFormat is %s.", opDtypeFormat.c_str());
            return true;
        }
        TE_WARNLOG("Failed to call Cpp function, Name=[%s], Module=[%s], FuncName=[%s]",
                   opName.c_str(), opModule.c_str(), opFuncName.c_str());
        return false;
    }
    TE_DBGLOG("Cpp function doesn't exist, continue to call python. Name=[%s], Module=[%s], FuncName=[%s].",
              opName.c_str(), opModule.c_str(), opFuncName.c_str());

    res = PythonApiCall::Instance().SelectTbeOpFormat(opinfo, opDtypeFormat);
    TE_FUSION_CHECK(!res, {
        TE_WARNLOG("Failed to call python function. Name=[%s], Module=[%s]", opName.c_str(), opModule.c_str());
        return false;
    });

    TE_INFOLOGF("Call Python success, opDtypeFormat is %s.", opDtypeFormat.c_str());
    return true;
}

extern "C" LX_QUERY_STATUS GetOpInfo(const TbeOpInfo &tbeOpInfo, std::string &result)
{
    TE_DBGLOG("Start to get op info");
    bool res = true;

    std::string opName;
    std::string opModule;
    std::string opFuncName = FUNC_GET_OP_SUPPORT_INFO;
    const std::lock_guard<std::mutex> pythonApiCallLock(PythonApiCall::mtx_);
    TE_DBGLOG("Get PythonApiCall lock.");

    TbeOpInfo &opinfo = const_cast<TbeOpInfo&>(tbeOpInfo);

    (void)opinfo.GetName(opName);
    (void)opinfo.GetModuleName(opModule);

    TE_DBGLOG("Query LxFusion info begin. Name=[%s], Module=[%s].", opName.c_str(), opModule.c_str());

    res = IsOpParameterValid(opModule, opFuncName);
    TE_FUSION_CHECK(!res, {
        TE_ERRLOG("Check op parameter failed.");
        return LX_QUERY_FAIL;
    });

    ge::NodePtr nodePtr = nullptr;
    ge::AscendString ascendResult;
    (void)opinfo.GetNode(nodePtr);
    
    if (nodePtr != nullptr && nodePtr->GetOpDesc() != nullptr && gert::DefaultOpImplSpaceRegistryV2::GetInstance()
        .GetSpaceRegistry(static_cast<gert::OppImplVersionTag>(nodePtr->GetOpDesc()->GetOppImplVersion())) != nullptr) {
        const auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry(
            static_cast<gert::OppImplVersionTag>(nodePtr->GetOpDesc()->GetOppImplVersion()));
        auto funcsPtr = spaceRegistry->GetOpImpl(nodePtr->GetType().c_str());
        if (funcsPtr != nullptr && funcsPtr->get_op_support_info != nullptr) {
            gert::ExeResGenerationCtxBuilder exeCtxBuilder;
            auto resPtrHolder = exeCtxBuilder.CreateOpCheckContext(*nodePtr.get());
            if (resPtrHolder != nullptr) {
                TE_DBGLOG("GetOpInfo start to call registered get_op_info function with OpCheckContext, Name=[%s], Module=[%s], FuncName=[%s].",
                           opName.c_str(), opModule.c_str(), opFuncName.c_str());
                auto opCheckContent = reinterpret_cast<gert::OpCheckContext *>(resPtrHolder->context_);
                if (funcsPtr->get_op_support_info(opCheckContent, ascendResult) == ge::GRAPH_SUCCESS) {
                    result = ascendResult.GetString();
                    TE_DBGLOG("GetOpInfo succeeded to call registered get_op_info function with OpCheckContext, opSupportedInfo is %s",
                               result.c_str());
                    return LX_QUERY_SUCC;
                }
                TE_WARNLOG("GetOpInfo failed to call registered get_op_info function with OpCheckContext, Name=[%s], Module=[%s], FuncName=[%s].",
                            opName.c_str(), opModule.c_str(), opFuncName.c_str());
                return LX_QUERY_FAIL;
            }
        }
    }
    return PythonApiCall::Instance().GetOpInfo(opinfo, result);
}

extern "C" OpBuildResCode FuzzBuildTbeOp(uint64_t taskId, uint64_t graphId, ge::Node &node)
{
    TE_DBGLOG("start FuzzBuildTbeOp.");
    const std::lock_guard<std::mutex> fusionMgrLock(TeFusionManager::mtx_);
    TE_DBGLOG("get TeFusionManager lock.");

    std::vector<ge::Node *> teGraphNode;
    teGraphNode.push_back(&node);
    auto opDesc = node.GetOpDesc();
    TE_INFOLOG("FuzzBuildTbeOp begin node[%s], taskID[%lu:%lu].", opDesc->GetName().c_str(), graphId, taskId);
    auto task = std::make_shared<OpBuildTask>(graphId, taskId, INVALID_SGT_INDEX, teGraphNode, opDesc,
        OP_TASK_STATUS::OP_TASK_PENDING);
    TE_FUSION_CHECK((task == nullptr), {
        TE_ERRLOG("Task is null, taskID[%lu:%lu].", graphId, taskId);
        return OP_BUILD_FAIL;
    });
    task->buildType = FUZZILY_BUILD;
    std::string strategy;

    TeFusionManager *teFusionM = TeFusionManager::GetInstance();
    TE_FUSION_CHECK((teFusionM == nullptr), {
        TE_ERRLOG("Failed to get tbe fusion manager instance.");
        return OP_BUILD_FAIL;
    });

    return teFusionM->BuildTbeOp(task, strategy);
}

extern "C" bool QueryOpPattern(const std::vector<std::pair<std::string, std::string>>& opPatternVec)
{
    (void)opPatternVec;
    const std::lock_guard<std::mutex> pythonApiCallLock(PythonApiCall::mtx_);
    TE_DBGLOG("Get PythonApiCall lock");
    return true;
}

extern "C" bool CheckIsTbeGeneralizeFuncRegistered(const TbeOpInfo &tbeOpInfo, bool &hasRegisteredFunc)
{
    const std::lock_guard<std::mutex> pythonApiCallLock(PythonApiCall::mtx_);
    TE_DBGLOG("Get PythonApiCall lock");
    return PythonApiCall::Instance().CheckIsTbeGeneralizeFuncRegistered(tbeOpInfo, hasRegisteredFunc);
}

extern "C" bool TeGeneralize(const TbeOpInfo &tbeOpInfo, const TE_GENERALIZE_TYPE &generalizeType,
                             const ge::NodePtr &nodePtr)
{
    TE_DBGLOG("Start TeGeneralize");
    return BinaryManager::Instance().TeGeneralize(tbeOpInfo, generalizeType, nodePtr);
}

extern "C" bool GetOpSpecificInfo(const TbeOpInfo &tbeOpInfo, std::string &opSpecificInfo)
{
    TE_DBGLOG("Start");
    bool res = true;
    std::string opName;
    std::string opModule;
    std::string opType;
    std::string opFuncName = FUNC_GET_SPECIFIC_INFO;

    const std::lock_guard<std::mutex> pythonApiCallLock(PythonApiCall::mtx_);
    TE_DBGLOG("Get PythonApiCall lock");

    (void)tbeOpInfo.GetName(opName);
    (void)tbeOpInfo.GetModuleName(opModule);
    (void)tbeOpInfo.GetOpType(opType);
    TE_DBGLOG("Name=[%s], Module=[%s], FuncName=[%s], OpType=[%s].",
              opName.c_str(), opModule.c_str(), opFuncName.c_str(), opType.c_str());

    ge::NodePtr nodePtr = nullptr;
    ge::AscendString result;
    (void)tbeOpInfo.GetNode(nodePtr);

    if (nodePtr != nullptr && nodePtr->GetOpDesc() != nullptr && gert::DefaultOpImplSpaceRegistryV2::GetInstance()
        .GetSpaceRegistry(static_cast<gert::OppImplVersionTag>(nodePtr->GetOpDesc()->GetOppImplVersion())) != nullptr) {
        const auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry(
            static_cast<gert::OppImplVersionTag>(nodePtr->GetOpDesc()->GetOppImplVersion()));
        auto funcsPtr = spaceRegistry->GetOpImpl(nodePtr->GetType().c_str());
        if (funcsPtr != nullptr && funcsPtr->get_op_specific_info != nullptr) {
            gert::ExeResGenerationCtxBuilder exeCtxBuilder;
            auto resPtrHolder = exeCtxBuilder.CreateOpCheckContext(*nodePtr.get());
            if (resPtrHolder != nullptr) {
                TE_DBGLOG("GetOpSpecificInfo start to call registered get_op_special_info function with OpCheckContext, Name=[%s], Module=[%s], FuncName=[%s].",
                           opName.c_str(), opModule.c_str(), opFuncName.c_str());
                auto opCheckContent = reinterpret_cast<gert::OpCheckContext *>(resPtrHolder->context_);
                if (funcsPtr->get_op_specific_info(opCheckContent, result) == ge::GRAPH_SUCCESS) {
                    opSpecificInfo = result.GetString();
                    TE_DBGLOG("GetOpSpecificInfo succeeded to call registered get_op_special_info function with OpCheckContext, opSpecificInfo is %s.",
                               opSpecificInfo.c_str());
                    return true;
                }
                TE_WARNLOG("GetOpSpecificInfo failed to call registered get_op_special_info function with OpCheckContext, Name=[%s], Module=[%s], FuncName=[%s].",
                            opName.c_str(), opModule.c_str(), opFuncName.c_str());
                return false;
            }
        }
    }

    // call cpp function if exists
    ge::AscendString opTypeString(opType.c_str());
    optiling::OP_CHECK_FUNC cppFunc =
                                optiling::OpCheckFuncRegistry::GetOpCapability(FUNC_GET_SPECIFIC_INFO, opTypeString);
    if (cppFunc != nullptr && nodePtr != nullptr) {
        ge::Operator op = OpDescUtils::CreateOperatorFromNode(nodePtr);
        ge::graphStatus ret = (*cppFunc)(op, result);
        if (ret == ge::SUCCESS) {
            opSpecificInfo = result.GetString();
            TE_INFOLOG("Call Cpp function success, opSpecificInfo is %s.", opSpecificInfo.c_str());
            return true;
        }
        TE_WARNLOG("Failed to call Cpp function, Name=[%s], Module=[%s], FuncName=[%s]",
                   opName.c_str(), opModule.c_str(), opFuncName.c_str());
        return false;
    }
    TE_DBGLOG("Cpp function doesn't exist, continue to call python. Name=[%s], Module=[%s], FuncName=[%s].",
              opName.c_str(), opModule.c_str(), opFuncName.c_str());

    res = PythonApiCall::Instance().GetOpSpecificInfo(tbeOpInfo, opSpecificInfo);
    TE_FUSION_CHECK(!res, {
        TE_WARNLOG("Failed to call python function. Name=[%s], Module=[%s].", opName.c_str(), opModule.c_str());
        return false;
    });

    TE_INFOLOG("Call Python success, opSpecificInfo is %s.", opSpecificInfo.c_str());
    return true;
}

extern "C" bool DynamicShapeRangeCheck(const TbeOpInfo &tbeOpInfo, bool &isSupported,
                                       std::vector<size_t> &upperLimitedInputIndexs,
                                       std::vector<size_t> &lowerLimitedInputIndexs)
{
    const std::lock_guard<std::mutex> pythonApiCallLock(PythonApiCall::mtx_);
    TE_DBGLOG("Get PythonApiCall lock");
    TeFusionManager *teFusionM = TeFusionManager::GetInstance();
    TE_FUSION_CHECK((teFusionM == nullptr), {
        TE_WARNLOG("Failed to get tbe fusion manager instance.");
        return false;
    });

    return PythonApiCall::Instance().DynamicShapeRangeCheck(tbeOpInfo, isSupported,
                                                            upperLimitedInputIndexs, lowerLimitedInputIndexs);
}

extern "C" bool GetOpUniqueKeys(const TbeOpInfo &tbeOpInfo, std::vector<std::string> &opUniqueKeyList)
{
    TE_DBGLOG("Start GetOpUniqueKeys.");
    std::string opName;
    std::string opModule;

    (void)tbeOpInfo.GetName(opName);
    (void)tbeOpInfo.GetModuleName(opModule);

    const std::lock_guard<std::mutex> pythonApiCallLock(PythonApiCall::mtx_);
    TE_DBGLOG("Get PythonApiCall lock");

    ge::NodePtr nodePtr = nullptr;
    (void)tbeOpInfo.GetNode(nodePtr);
    if (nodePtr == nullptr) {
        TE_WARNLOG("Node of op[%s, %s] is null", opName.c_str(), opModule.c_str());
        return false;
    }
    std::string jsonStr;
    bool ret = TeJsonAssemble::GenerateOpJson(nodePtr, tbeOpInfo, jsonStr);
    TE_FUSION_CHECK(!ret, {
        TE_WARNLOG("Failed to get generate json. Name=[%s], Module=[%s]", opName.c_str(), opModule.c_str());
        return false;
    });

    bool res = PythonApiCall::Instance().GetOpUniqueKeys(tbeOpInfo, jsonStr, opUniqueKeyList);
    TE_FUSION_CHECK(!res, {
        TE_WARNLOG("Failed to get op unique keys. Name=[%s], Module=[%s]", opName.c_str(), opModule.c_str());
        return false;
    });

    TE_DBGLOGF("Node of op[%s,%s] get op unique key success, jsonStr is [%s], Op keylist size is [%zu].",
        opName.c_str(), opModule.c_str(), jsonStr.c_str(), opUniqueKeyList.size());
    return true;
}

extern "C" OpBuildResCode TaskFusion(const std::vector<ge::Node *> &graphNodes, ge::OpDescPtr opDesc,
                                     const uint64_t taskId, const uint64_t graphId)
{
    TE_DBGLOG("start TaskFusion, taskId is [%lu:%lu].", graphId, taskId);
    auto task = std::make_shared<OpBuildTask>(graphId, taskId, INVALID_SGT_INDEX,
                                                          graphNodes, opDesc, OP_TASK_STATUS::OP_TASK_PENDING);

    const std::lock_guard<std::mutex> fusionMgrLock(TeFusionManager::mtx_);
    TE_DBGLOG("Get TeFusionManager lock.");
    TeFusionManager *teFusionM = TeFusionManager::GetInstance();
    TE_FUSION_CHECK((teFusionM == nullptr), {
        TE_ERRLOG("Failed to get tbe fusion manager instance.");
        return OP_BUILD_FAIL;
    });

    return teFusionM->BuildTaskFusion(task);
}

extern "C" OpBuildResCode BuildSuperKernel(const std::vector<ge::Node *> &graphNodes, ge::OpDescPtr opDesc,
                                     const uint64_t taskId, const uint64_t graphId)
{
    TE_DBGLOG("start BuildSuperKernel, taskId is [%lu: %lu].", graphId, taskId);
    auto task = std::make_shared<OpBuildTask>(graphId, taskId, INVALID_SGT_INDEX,
                                              graphNodes, opDesc, OP_TASK_STATUS::OP_TASK_PENDING);
    return SuperKernelTaskManager::GetInstance().BuildSuperKernel(task);
}

extern "C" bool IsOppKernelInstalled(bool isOm, int64_t implType)
{
    return BinaryManager::Instance().JudgeBinKernelInstalled(isOm, implType);
}

extern "C" void GetAllCompileStatistics(std::vector<std::string> &compileStatistics)
{
    DfxInfoManager::Instance().GetStatisticsMsgs(compileStatistics);
}

extern "C" std::string GetKernelMetaDir()
{
    return TeConfigInfo::Instance().GetKernelMetaParentDir() + "/kernel_meta";
}
} // namespace te
