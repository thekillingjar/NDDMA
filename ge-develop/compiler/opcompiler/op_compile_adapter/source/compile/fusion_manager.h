/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TE_FUSION_OP_H
#define TE_FUSION_OP_H

#include <Python.h>
#include <vector>
#include <queue>
#include <map>
#include <set>
#include <mutex>
#include <functional>
#include <unordered_map>
#include <chrono>
#include <nlohmann/json.hpp>

#include "graph/node.h"
#include "inc/te_fusion_util_constants.h"
#include "inc/te_fusion_log.h"
#include "inc/te_fusion_check.h"
#include "inc/te_fusion_types.h"
#include "cache/te_cache_manager.h"
#include "python_adapter/python_api_call.h"
#include "binary/fusion_binary_info.h"

namespace te {
namespace fusion {
using std::function;
using nlohmann::json;
// grashid, taskid
using OpTaskKey = std::pair<uint64_t, uint64_t>;

/**
 * @brief: the Fusion Manager Class
 */
class TeFusionManager {
public:
    static TeFusionManager *GetInstance();
    /*
     * @brief: clear instance source when session end
     */
    void Finalize();
    /*
     * @brief: get same op parameter kernel name
     * @param [in] pOr: op info pointer
     * @param [out] kernalName: reuse same input parameter kernel name
     * @param [in] opTask: op build task
     * @return bool: check result
     */
    template<typename Key, typename Map>
    bool CanReuseTaskCache(const Key &opKey, Map &opMap, const OpBuildTaskPtr &opTask,
                           OpBuildTaskResultPtr &opRes) const
    {
        auto opIter = opMap.find(opKey);
        if (opIter == opMap.end()) {
            return false;
        }

        // fuzz build, need build missing shape range
        // fuzzy cache build, in this case(not single process) first is nullptr not mean another equal task is running
        if (!opTask->missSupportInfo.empty()) {
            TE_INFOLOG("incremental building kernel[%s].", opTask->kernel.c_str());
            return false;
        }

        // have found the op of equal parameters
        auto &match = opIter->second;
        if (match.first == nullptr) {
            // kernel not generated yet, there's another equal task running background.
            // save current task to the queue, when previous equal task finished, all tasks in this
            // queue became finished
            if (match.second.size() <= 0) {
                TE_WARNLOG("No matched task. Size = %zu.", match.second.size());
                return false;
            }
            TE_DBGLOG("Op[%lu] background build task running[%lu],[%s] enqueue taskID[%lu:%lu] [%s]",
                      opTask->opNodes.size(), match.second.size(), match.second[0]->kernel.c_str(),
                      opTask->graphId, opTask->taskId, opTask->kernel.c_str());

            // set current op's kernel name
            opTask->kernel = match.second[0]->kernel;
            opTask->status = OP_TASK_STATUS::OP_TASK_PENDING;
            match.second.push_back(opTask);
            return true;
        }

        opRes = match.first;
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

    /**
     * @brief build fusion op
     * @param [in]  opTask      op build task
     * @return  bool            true: success, false: fail
     */
    bool BuildFusionOp(OpBuildTaskPtr &opTask, const std::string &opCompileStrategyStr);

    /**
     * @brief get finished compilation task from python
     * @param [in] NA
     * @return [out] NA
     */
    bool GetFinishedCompilationTask(uint64_t graphId);

    bool SaveBuildTask(const OpBuildTaskPtr &task);

    void ReportPendingTaskInfo() const;

    bool SaveFinishedTask(const OpBuildTaskPtr &task);

    OpBuildResCode BuildTbeOp(OpBuildTaskPtr &opTask, const std::string &opCompileStrategyStr);

    static bool SetBuildResult(OpBuildTaskPtr &relBuildTaskPtr, OpBuildTaskResultPtr &opBuildResult);

    static void SetCompileResultAttr(const ge::OpDescPtr &opDesc, const CompileResultPtr &compileRetPtr);

    static bool SetPreBuildResult(const OpBuildTaskPtr &relBuildTaskPtr, const OpBuildTaskResultPtr &opBuildResult);

    bool UpdateInhibitionInfoForLog();

    bool IsTimeToPrintProgressHint();

    void PrintProgressHint();

    static bool IsFusionCheckTask(const OpBuildTaskPtr &opTask);

    OpBuildResCode BuildTaskFusion(OpBuildTaskPtr &opTask);

    bool TaskFusion(OpBuildTaskPtr &opTask);

    bool TaskFusionProcess(const OpBuildTaskPtr &opTask);

    static bool RefreshCacheAndSinalManager();

    bool GetCompileFileFromCache(const std::string &kernelName, std::string &jsonPath, std::string &binPath);

    // finished compilation task list
    std::vector<FinComTask> finComTaskList;
    // instantly finished op build task, no need to dispatch to background worker
    std::map<uint64_t, std::map<uint64_t, OpBuildTaskPtr>> finishedTask_;

    std::map<std::string, long> reportedErr_;
    std::time_t taskStatisticsTime_ = 0;
    std::unordered_map<std::string, FILE *> lockFpHandle_; // lock build dir handle: <kernelName, handle>

    // lock for singleton mode
    static std::mutex mtx_;

    int reuseCount = 0;

    TeFusionManager()
    {
    }

    ~TeFusionManager()
    {
    }

private:
    /*
     * @brief: build single op
     * @param [in] opTask:      opBuildTask
     * @return bool:            build op ok or not
     */
    bool BuildSingleOp(OpBuildTaskPtr &opTask);

    bool BuildBinarySingleOp(OpBuildTaskPtr &opTask);

    bool BuildBinaryFusionOp(OpBuildTaskPtr &opTask);

    static bool SetJsonDescAndKernelName(OpBuildTaskPtr &opTask, const bool isUnique);

    bool CheckResultInBinaryBuildDir(const OpBuildTaskPtr &opTask, bool &isBuilding, bool &isFound) const;

    static bool MakeBinaryRootDir(std::string &rootPath);

    bool CanReuseBuildBinaryCache(OpBuildTaskPtr &opTask, bool &hasError);

    bool CheckBuildBinaryOpTaskResult(OpBuildTaskPtr &opTask);

    /*
     * @brief: check op build is from PreBuildTbeOpInner or BuildSingleOp
     * @param [in] opName: op name
     * @param [out] isPreBuild: true is for PreBuildTbeOpInner, false is for BuildSingleOp
     * @return bool: check result
     */
    static bool CheckPreBuildOp(const std::string &opName, bool &isPreBuild);

    /*
     * @brief: config op L1 info to python
     * @param [in] pNode: op node
     * @return bool: save L1 info parameter to python ok or not
     */
    static bool SetOpParamsL1Info(ge::Node *pNode);

    static void GetOpModuleName(const OpBuildTaskPtr &relBuildTaskPtr, string &opModuleNames);

    static void ReportBuildErrMessage(const OpBuildTaskPtr &relBuildTaskPtr, const string &opModuleNames,
                                      const OpBuildTaskResultPtr &taskRes, const string &errorMsg);

    bool CompilePendingTask(std::vector<OpBuildTaskPtr> &vecOpBuildTask,
        const OpBuildTaskPtr &finBuildTaskPtr, const OpBuildTaskResultPtr &opBuildResult);

    bool FinishPendingTask(const OpBuildTaskPtr &relBuildTaskPtr,
                           const OpBuildTaskResultPtr &opBuildResult);

    static bool SyncOpTuneParams();

    static bool SetOpArgsToNode(const OpBuildTaskPtr &opTask);

    static void GenerateSingleOpRelationParam(const ge::Node *curNode, std::string &relationJson);

    static void GenerateFusionOpRelationParam(const std::vector<ge::Node *> &teGraphNode, std::string &relationJson);

    static void GetFusionOpOuterInputCnt(std::unordered_set<ge::Node *> &allNodes,
                                         const ge::Node *currentNode, int32_t &outerInputCnt);

    static void FusionOpRelationGetInplacelist(const ge::OutDataAnchorPtr &anchor, const ge::Node *currentNode,
                                               std::vector<int32_t> &inplaceOutIdxList, int32_t &realIdx,
                                               std::unordered_set<ge::Node *> &allNodes);

    static bool DfsFindOuterInput(const ge::Node *curNode, size_t outputIdx,
                                  const std::unordered_set<ge::Node *> &allNodes, uint32_t &dfsMaxCnt);

    static bool DfsFindOuterInputSingleAnchor(const ge::InDataAnchorPtr &anchor,
                                              const std::unordered_set<ge::Node *> &allNodes,
                                              const ge::Node *curNode, size_t outputIdx, uint32_t &dfsMaxCnt);

    static bool UpdateOpTaskForCompileCache(const OpBuildTaskPtr &opTask, const CompileResultPtr &compileResultPtr);

    bool ParallelCompilationProcess(const OpBuildTaskPtr &opTask);

    static void DumpFusionOpInfoToJsonFile(const OpBuildTaskPtr &opTask);

    void DelDispatchedTaskByKey(uint64_t graphId, uint64_t taskId);

    void DelDispatchedTask(std::vector<OpBuildTaskPtr> &vecOpBuildTask, int index);

    std::pair<OpBuildTaskResultPtr, std::vector<OpBuildTaskPtr>>* FindBuildTaskList(
        const OpBuildTaskPtr &relBuildTaskPtr);

    FILE* FindLockFpHandle(const std::string &kernelName) const;

    void SaveLockFpHandle(const std::string &kernelName, FILE *fp) const;

    static bool CanReuseBuildDiskCache(const OpBuildTaskPtr &opTask);

    static void UpdatePreops(const OpBuildTaskPtr &opTask);

    static bool IsOpdebugCompile(const std::vector<ge::Node *> &nodes);

    TeFusionManager& operator=(const TeFusionManager& op)
    {
        if (&op == this) {
            return *this;
        }
        return *this;
    }

    // Op->Kernel map
    std::unordered_map<std::string,   // kernelName
                       std::pair<OpBuildTaskResultPtr, std::vector<OpBuildTaskPtr>>> opKernelMap_;

    // cached json->kernel_name map
    std::unordered_map<std::string,           // json description of fusion op
                       std::pair<OpBuildTaskResultPtr, std::vector<OpBuildTaskPtr>>> fusionOpsKernel_;

    std::unordered_map<std::string,           // json description of fusion op
                       std::pair<OpBuildTaskResultPtr, std::vector<OpBuildTaskPtr>>> taskFusionMap_;

    // all op build request by FE
    std::map<OpTaskKey, OpBuildTaskPtr> dispatchedTask_;

    // compile progress promot, record last print time
    std::time_t lastPrintTime_ = 0;
};

/*
 * @brief: initial python env and create TeFusionManager instance
 * @param [in] options: soc version
 * @return TeFusionManager*: TeFusionManager instance
 */
bool SetNodeCompileInfoAttr(const ge::OpDescPtr &opDesc, const OpBuildTaskResultPtr &opRes);
} // namespace fusion
} // namespace te
#endif
