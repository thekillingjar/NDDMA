/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TE_FUSION_API_H
#define TE_FUSION_API_H

#include <string>
#include <map>

#include "graph/node.h"
#include "tensor_engine/tbe_op_info.h"

/*lint -e148*/
namespace te {
/**
 * @brief: Tbe Initialize proccess
 * @param [in] options            ddk version
 * @return [out] bool             succ or fail for Tbe Initialize
 */
extern "C" bool TbeInitialize(const std::map<std::string, std::string> &options, bool *isSupportParallelCompilation);

/**
 * @brief: tbe finalize proccess
 * @return [out] bool             succ or fail for Tbe Finalize
 */
extern "C" bool TbeFinalize();

extern "C" bool TbeFinalizeSessionInfo(const std::string &session_graph_id);

/**
 * @brief: get finished compilation task list
 * @return [out] list             finished compilation task list
 */
extern "C" bool WaitAllFinished(uint64_t graphId, vector<FinComTask> &tasks);

/*
 * @brief: check tbe op capability
 * @param [in] opinfo: op total parameter set
 *
 * @param [out] result: contains the results returned by op
 * @return bool: check process ok or not
 */
extern "C" bool CheckOpSupported(TbeOpInfo& opinfo, CheckSupportedInfo &checkSupportedInfo);

/**
 * @brief pre build tbe op
 * @return [out] bool                 succ or fail of prebuild
 */
extern "C" bool PreBuildTbeOp(TbeOpInfo& opinfo, uint64_t taskId, uint64_t graphId);

/**
 * @brief build fusion op
 * @param [in] teGraphNode            graph node ptr vector
 * @param [in] strategy               op optimization strategy, empty or json str with key
 *                                    module_name, object_name and object_value
 * @param [in] sgtThreadIndex         This index shows the index of sgt slice shape,
 * when compiling ffts nodes, we need to get the real sgt slice
 * shape from nodes. Usually, one node has two different sgt
 * slice shapes and we call them head slice and tail slice.
 * The index of head slice is 0 and the index of tail slice is 1.
 * @return [in] outputNode            recode json file path in it
 * @return [out] bool                 succ or fail of building fusion op
 */
extern "C" OpBuildResCode TeFusion(std::vector<ge::Node *> teGraphNode, ge::OpDescPtr opDesc,
                                   const std::vector<ge::NodePtr> &toBeDel,
                                   uint64_t taskId, uint64_t graphId, const std::string &strategy);

extern "C" OpBuildResCode TeFusionV(std::vector<ge::Node *> teGraphNode, ge::OpDescPtr opDesc,
                                    const std::vector<ge::NodePtr> &toBeDel, uint64_t taskId, uint64_t graphId,
                                    uint64_t sgtThreadIndex, const std::string &strategy);

/**
 * @brief select tbe op format
 * @param [in] tbeOpInfo            op info
 * @param [out] opDtypeformat       op date format
 * @return [out] bool               succ or fail
 */
extern "C" bool SelectTbeOpFormat(const TbeOpInfo &tbeOpInfo, std::string &opDtypeFormat);

/**
 * @brief clear instance
 * @return [out] bool                 succ or fail of TeFusionEnd
 */
extern "C" bool TeFusionEnd();

/**
 * @brief get op info
 * @param [in] tbeOpInfo            op info
 * @param [out] result              op segmentation information
 * @return [out] LX_QUERY_STATUS    LX_QUERY_FAIL, LX_QUERY_NOT_FOUND or LX_QUERY_SUCC
 */
extern "C" LX_QUERY_STATUS GetOpInfo(const TbeOpInfo &tbeOpInfo, std::string &result);

/**
 * @brief fuzz build tbe op
 * @param [in] tbeOpInfo            op info
 * @param [in] taskId               fuzz build task id
 * @param [in] graphId              fuzz build graph id
 * @param [in] node                 tbe op to compile
 * @return [out] OpBuildResCode                 succ or fail of building fusion op
 */
extern "C" OpBuildResCode FuzzBuildTbeOp(uint64_t taskId, uint64_t graphId, ge::Node &node);

/**
* @brief: query op registered patterns
* @param [out] opPatternVec op_pattern list, key is op_type, value is pattern(Elewise/func/...).
* @return [out] bool        succ or fail
*/
extern "C" bool QueryOpPattern(const std::vector<std::pair<std::string, std::string>>& opPatternVec);

/**
 * @brief check is this op has registered generalize func
 * @param [in] tbeOpInfo          op info
 * @param [in] hasRegisteredFunc  true or false
 * @return [out] bool             succ or fail of check
 */
extern "C" bool CheckIsTbeGeneralizeFuncRegistered(const TbeOpInfo &tbeOpInfo, bool &hasRegisteredFunc);

/**
 * @brief shape or value generalize with corresponding generalize type
 * @param [in] tbeOpInfo          op info
 * @param [in] generalizeType    generalize type(TE_GENERALIZE_TYPE)
 * @param [in] nodePtr            node
 * @return [out] bool             succ or fail of check
 */
extern "C" bool TeGeneralize(const TbeOpInfo &tbeOpInfo, const TE_GENERALIZE_TYPE &generalizeType,
                             const ge::NodePtr &nodePtr);

/**
 * @brief get op specific info from tbe
 * @param [in] tbeOpInfo          op info
 * @param [out] opSpecificInfo    op specific info
 * @return [out] bool             succ or fail of get
 */
extern "C" bool GetOpSpecificInfo(const TbeOpInfo &tbeOpInfo, std::string &opSpecificInfo);

/**
 * @brief Check the validity of shape range by tbe
 * @param [in] tbeOpInfo                  op info
 * @param [out] isSupported               shape range is valid or not
 * @param [out] upperLimitedInputIndexs   shape range upper unSupported input indexs
 * @param [out] lowerLimitedInputIndexs   shape range lower unSupported input indexs
 * @return [out] bool                     succ or fail of check
 */
extern "C" bool DynamicShapeRangeCheck(const TbeOpInfo &tbeOpInfo, bool &isSupported,
                                       std::vector<size_t> &upperLimitedInputIndexs,
                                       std::vector<size_t> &lowerLimitedInputIndexs);

/**
 * @brief get op specific info from tbe
 * @param [in] tbeOpInfo          op info
 * @param [out] opUniqueKeyList     op UniqueKey list
 * @return [out] bool             succ or fail of get
 */
extern "C" bool GetOpUniqueKeys(const TbeOpInfo &tbeOpInfo,  std::vector<std::string> &opUniqueKeyList);

/**
 * @brief support task fusion
 * @param [in] graphNodes     graph node ptr vector
 * @param [in] taskId         task fusion task id
 * @param [in] graphId        task fusion graph id
 * @return [out] OpBuildResCode  succ or fail of task fusion
 */
extern "C" OpBuildResCode TaskFusion(const std::vector<ge::Node *> &graphNodes, ge::OpDescPtr opDesc,
                                     const uint64_t taskId, const uint64_t graphId);

extern "C" OpBuildResCode BuildSuperKernel(const std::vector<ge::Node *> &graphNodes, ge::OpDescPtr opDesc,
                                           const uint64_t taskId, const uint64_t graphId);

extern "C" bool IsOppKernelInstalled(bool isOm, int64_t implType);

/**
 * @brief get all compile staticistics messages, include disk cache, binary, task reuse and online compile statistics
 * @return [out] compileStatistics compile staticistics messages
 */
extern "C" void GetAllCompileStatistics(std::vector<std::string> &compileStatistics);
} // namespace te
/*lint +e148*/
#endif
