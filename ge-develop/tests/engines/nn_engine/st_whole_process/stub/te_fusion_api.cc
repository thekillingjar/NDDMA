/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "te_fusion_api.h"
#include "fe_llt_utils.h"
namespace te {
/**
 * @brief: Tbe Initialize proccess
 * @param [in] options            ddk version
 * @return [out] bool             succ or fail for Tbe Initialize
 */
bool TbeInitialize(const std::map<std::string, std::string> &options, bool *isSupportParallelCompilation) {
  return te::TeStub::GetInstance().GetImpl()->TbeInitialize(options, isSupportParallelCompilation);
}

/**
 * @brief: tbe finalize proccess
 * @return [out] bool             succ or fail for Tbe Finalize
 */
bool TbeFinalize() {
  return te::TeStub::GetInstance().GetImpl()->TbeFinalize();
}

/**
 * @brief: get finished compilation task list
 * @return [out] list             finished compilation task list
 */
bool WaitAllFinished(uint64_t graphId, vector<FinComTask> &tasks) {
  return te::TeStub::GetInstance().GetImpl()->WaitAllFinished(graphId, tasks);
}

/*
 * @brief: check tbe op capability
 * @param [in] opinfo: op total parameter set
 * @param [out] IsSupport: result to support or not
 * @return bool: check process ok or not
 */
bool CheckTbeSupported(TbeOpInfo& opinfo, CheckSupportedResult &isSupport, std::string &reason) {
  return te::TeStub::GetInstance().GetImpl()->CheckTbeSupported(opinfo, isSupport, reason);
}

bool CheckOpSupported(TbeOpInfo& opinfo, CheckSupportedInfo &result) {
  bool res = te::TeStub::GetInstance().GetImpl()->CheckTbeSupported(opinfo, result.isSupported, result.reason);
  return res;
}

bool GetOpUniqueKeys(const TbeOpInfo &tbeOpInfo, std::vector<std::string> &opUniqueKeys) {
  return te::TeStub::GetInstance().GetImpl()->GetOpUniqueKeys(tbeOpInfo, opUniqueKeys);
}

/**
 * @brief pre build tbe op
 * @return [out] bool                 succ or fail of prebuild
 */
bool PreBuildTbeOp(TbeOpInfo& opinfo, uint64_t taskId, uint64_t graphId) {
  return te::TeStub::GetInstance().GetImpl()->PreBuildTbeOp(opinfo, taskId, graphId);
}

/**
 * @brief build fusion op
 * @param [in] teGraphNode            graph node ptr vector
 * @param [in] strategy               op optimization strategy, empty or json str with key
 *                                    module_name, object_name and object_value
 * @return [in] outputNode            recode json file path in it
 * @return [out] bool                 succ or fail of building fusion op
 */
OpBuildResCode TeFusion(std::vector<ge::Node *> teGraphNode, ge::OpDescPtr opDesc,
                        const std::vector<ge::NodePtr> &toBeDel,
                        uint64_t taskId, uint64_t graphId, const std::string &strategy) {
  return te::TeStub::GetInstance().GetImpl()->TeFusion(teGraphNode, opDesc, toBeDel, taskId,
                                                       graphId, strategy);
}

OpBuildResCode TeFusionV(std::vector<ge::Node *> teGraphNode, ge::OpDescPtr opDesc,
                         const std::vector<ge::NodePtr> &toBeDel, uint64_t taskId, uint64_t graphId,
                         uint64_t sgtThreadIndex, const std::string &strategy) {
  return te::TeStub::GetInstance().GetImpl()->TeFusionV(teGraphNode, opDesc, toBeDel, taskId,
                                                        graphId, sgtThreadIndex, strategy);
}

/**
 * @brief select tbe op format
 * @param [in] tbeOpInfo            op info
 * @param [out] opDtypeformat       op date format
 * @return [out] bool               succ or fail
 */
bool SelectTbeOpFormat(const TbeOpInfo &tbeOpInfo, std::string &opDtypeFormat) {
  return te::TeStub::GetInstance().GetImpl()->SelectTbeOpFormat(tbeOpInfo, opDtypeFormat);
}

/**
 * @brief clear instance
 * @return [out] bool                 succ or fail of TeFusionEnd
 */
bool TeFusionEnd() {
  return te::TeStub::GetInstance().GetImpl()->TeFusionEnd();
}

/**
 * @brief get op info
 * @param [in] tbeOpInfo            op info
 * @param [out] result              op segmentation information
 * @return [out] LX_QUERY_STATUS    LX_QUERY_FAIL, LX_QUERY_NOT_FOUND or LX_QUERY_SUCC
 */
LX_QUERY_STATUS GetOpInfo(const TbeOpInfo &tbeOpInfo, std::string &result) {
  return te::TeStub::GetInstance().GetImpl()->GetOpInfo(tbeOpInfo, result);
}

/**
 * @brief fuzz build tbe op
 * @param [in] tbeOpInfo            op info
 * @param [in] taskId               fuzz build task id
 * @param [in] graphId              fuzz build graph id
 * @param [in] node                 tbe op to compile
 * @return [out] OpBuildResCode                 succ or fail of building fusion op
 */
OpBuildResCode FuzzBuildTbeOp(uint64_t taskId, uint64_t graphId, ge::Node &node) {
  return te::TeStub::GetInstance().GetImpl()->FuzzBuildTbeOp(taskId, graphId, node);
}

/**
 * @brief set fuzz build support info to node
 * @param [in] nodePtr              original node
 * @param [in] supportInfoStr       generate support info
 * @param [in] indirectIndexs       indexs of current node indirect link to original node
 * @param [in] directIndexMap       index map of current node input index direct link to original node
 * @return [out] bool               succ or fail of set fuzz build support info to node
 */
bool SetFuzzyBuildSupportInfoToNode(ge::NodePtr &nodePtr, const std::string &supportInfoStr,
                                    const std::vector<size_t> &indirectIndexs,
                                    const std::vector<std::pair<size_t, size_t>> &directIndexMap) {
  return te::TeStub::GetInstance().GetImpl()->SetFuzzyBuildSupportInfoToNode(nodePtr, supportInfoStr,
                                                                     indirectIndexs, directIndexMap);
}

/**
 * @brief check is this op has registered tbegeneralize func
 * @param [in] tbeOpInfo            op info
 * @param [in] hasRegisteredFunc    true or false
 * @return [out] bool               succ or fail of check
 */
bool CheckIsTbeGeneralizeFuncRegistered(const TbeOpInfo &tbeOpInfo, bool &hasRegisteredFunc) {
  return te::TeStub::GetInstance().GetImpl()->CheckIsTbeGeneralizeFuncRegistered(tbeOpInfo, hasRegisteredFunc);
}

/**
 * @brief shape or value generalize with generalize type
 * @param [in] tbeOpInfo            op info
 * @param [in] generalize_type      generalize type
 * @param [in] nodePtr              node
 * @return [out] bool               succ or fail of check
 */
bool TeGeneralize(const TbeOpInfo &tbeOpInfo,
                  const TE_GENERALIZE_TYPE &generalize_type,
                  const ge::NodePtr &nodePtr) {
  return te::TeStub::GetInstance().GetImpl()->TeGeneralize(tbeOpInfo, generalize_type, nodePtr);
}

/**
 * @brief get op specific info from tbe
 * @param [in] tbeOpInfo         op info
 * @param [in] opSpecificInfo    op specific info
 * @param [out] bool             succ or fail
 */
bool GetOpSpecificInfo(const TbeOpInfo &tbeOpInfo, std::string &opSpecificInfo) {
  return te::TeStub::GetInstance().GetImpl()->GetOpSpecificInfo(tbeOpInfo, opSpecificInfo);
}

/**
 * @brief Check the validity of shape range by tbe
 * @param [in] tbeOpInfo                  op info
 * @param [out] isSupported               shape range is valid or not
 * @param [out] unSupportedInputIndexs    shape range unSupported input indexs
 * @param [out] bool                      succ or fail
 */
bool DynamicShapeRangeCheck(const TbeOpInfo &tbeOpInfo, bool &isSupported,
                            std::vector<size_t> &upperLimitedInputIndexs,
                            std::vector<size_t> &lowerLimitedInputIndexs) {
  return te::TeStub::GetInstance().GetImpl()->DynamicShapeRangeCheck(tbeOpInfo, isSupported,
                                                             upperLimitedInputIndexs,
                                                             lowerLimitedInputIndexs);
}

bool TeStubApi::TbeInitialize(const std::map<std::string, std::string> &options,
                              bool *isSupportParallelCompilation) {
  if (te::TeStub::GetInstance().tbe_initialize_ == nullptr) {
    return true;
  }
  auto initialize_func =
          (bool(*)(const std::map<std::string, std::string> &, bool*))(te::TeStub::GetInstance().tbe_initialize_);

  return initialize_func(options, isSupportParallelCompilation);
}

bool TeStubApi::TbeFinalize() {
  return true;
}

bool TeStubApi::CheckTbeSupported(TbeOpInfo& opinfo,
                                  CheckSupportedResult &isSupport,
                                  std::string &reason) {
  isSupport = te::FULLY_SUPPORTED;
  return true;
}

bool TeStubApi::CheckOpSupported(TbeOpInfo& opinfo, CheckSupportedInfo &result) {
  result.isSupported = te::FULLY_SUPPORTED;
  return true;
}

bool TeStubApi::PreBuildTbeOp(TbeOpInfo& opinfo, uint64_t taskId, uint64_t graphId) {
  //之后TEFUSION会设置一个串行预编译的接口，当前采用打桩方式
  string pattern = "Opaque";
  string core_type = "AiCore";
  string op_type;
  opinfo.GetOpType(op_type);
  if (op_type == "Aipp") {
    pattern = "aipp";
  } else if (op_type == "StridedWrite") {
    pattern = "strided_write";
  } else if (op_type == "StridedRead") {
    pattern = "strided_read";
  } else if (op_type == "Pooling") {
    pattern = "Pool2d";
    std::vector<TbeAttrValue> attr_values;
    opinfo.GetAttrValues(attr_values);
    for (const auto &attr : attr_values) {
      if (attr.GetName() == "mode") {
        int64_t value;
        attr.GetValue(value);
        if (value == 1) {
          pattern = "Convolution";
        }
      }
    }
  } else if (op_type == "LeakyRelu" || op_type == "Add" || op_type == "Mul" || op_type == "Relu" ||
             op_type == "Gelu" || op_type == "GeluGrad") {
    pattern = "ElemWise";
  } else if (op_type == "FullyConnection") {
    pattern = "Matmul";
  } else if (op_type == "Conv2D") {
    pattern = "Convolution";
  } else if (op_type == "AscendRequant") {
    pattern = "requant";
  } else if (op_type == "AscendQuant") {
    pattern = "quant";
  } else if (op_type == "AscendDequant") {
    pattern = "dequant";
  } else if (op_type == "AscendAntiQuant") {
    pattern = "anti_quant";
  } else if (op_type == "MatMul") {
    pattern = "Matmul";
  }

  FinComTask new_task;
  new_task.graphId = graphId;
  new_task.taskId = taskId;
  new_task.status = 0;
  opinfo.SetPattern(pattern);
  opinfo.SetOpCoreType(core_type);
  auto iter = tasks_.find(graphId);

  if (iter == tasks_.end()) {
    vector<FinComTask> tasks = {new_task};
    tasks_.emplace(std::make_pair(graphId, tasks));
  } else {
    iter->second.emplace_back(new_task);
  }
  return true;
}

OpBuildResCode TeStubApi::TeFusion(std::vector<ge::Node *> teGraphNode, ge::OpDescPtr opDesc,
                                   const std::vector<ge::NodePtr> &toBeDel,
                                   uint64_t taskId, uint64_t graphId,
                                   const std::string &strategy) {
  return TeFusionV(teGraphNode, opDesc, toBeDel, taskId, graphId, 0xffffffff, strategy);
}

OpBuildResCode TeStubApi::TeFusionV(std::vector<ge::Node *> teGraphNode, ge::OpDescPtr opDesc,
                                    const std::vector<ge::NodePtr> &toBeDel, uint64_t taskId,
                                    uint64_t graphId, uint64_t sgtThreadIndex,
                                    const std::string &strategy) {
  /* 编译流程耗时较长，目前没有绝对的必要采取真实实现。统一采用打桩方式。 */
  string json_file_path =
      fe::GetCodeDir() + "/tests/engines/nn_engine/depends/te_fusion/compilation_result/te_op_312acb789acb79acb789cb7a89cef8978_1.json";
  ge::AttrUtils::SetStr(opDesc, "json_file_path", json_file_path);
  FinComTask new_task;
  new_task.graphId = graphId;
  new_task.taskId = taskId;
  new_task.status = 0;
  new_task.teNodeOpDesc = opDesc;
  auto iter = tasks_.find(graphId);
  if (iter == tasks_.end()) {
    vector<FinComTask> tasks = {new_task};
    tasks_.emplace(std::make_pair(graphId, tasks));
  } else {
    iter->second.emplace_back(new_task);
  }
  return OP_BUILD_SUCC;
}

bool TeStubApi::WaitAllFinished(uint64_t graphId, vector<FinComTask> &tasks) {
  auto iter = tasks_.find(graphId);
  if (iter != tasks_.end()) {
    tasks = std::move(iter->second);
    tasks_.erase(graphId);
  } else {
    if (te::TeStub::GetInstance().wait_all_finished_ != nullptr) {
      auto wait_all_finished_func =
              (bool(*)(uint64_t, vector<FinComTask> &))(te::TeStub::GetInstance().wait_all_finished_);
      return wait_all_finished_func(graphId, tasks);
    }
  }
  return true;
}

bool TeStubApi::SelectTbeOpFormat(const TbeOpInfo &tbeOpInfo, std::string &opDtypeFormat) {
  return true;
}

bool TeStubApi::GetOpUniqueKeys(const TbeOpInfo &tbeOpInfo, std::vector<std::string> &opUniqueKeys) {
  opUniqueKeys.push_back("11223344");
  return true;
}

bool TeStubApi::TeFusionEnd() {
  return true;
}

LX_QUERY_STATUS TeStubApi::GetOpInfo(const TbeOpInfo &tbeOpInfo, std::string &result) {
  return LX_QUERY_SUCC;
}

OpBuildResCode TeStubApi::FuzzBuildTbeOp(uint64_t taskId, uint64_t graphId, ge::Node &node) {
  return OP_BUILD_SUCC;
}

bool TeStubApi::SetFuzzyBuildSupportInfoToNode(ge::NodePtr &nodePtr, const std::string &supportInfoStr,
                                                       const std::vector<size_t> &indirectIndexs,
                                                       const std::vector<std::pair<size_t, size_t>> &directIndexMap) {
  return true;
}

bool TeStubApi::CheckIsTbeGeneralizeFuncRegistered(const TbeOpInfo &tbeOpInfo,
                                                           bool &hasRegisteredFunc) {
  return true;
}

bool TeStubApi::TeGeneralize(const TbeOpInfo &tbeOpInfo,
                                     const TE_GENERALIZE_TYPE &generalize_type,
                                     const ge::NodePtr &nodePtr) {
  return true;
}

bool TeStubApi::GetOpSpecificInfo(const TbeOpInfo &tbeOpInfo,
                                          std::string &opSpecificInfo) {
  return true;
}

bool TeStubApi::DynamicShapeRangeCheck(const TbeOpInfo &tbeOpInfo, bool &isSupported,
                                               std::vector<size_t> &upperLimitedInputIndexs,
                                               std::vector<size_t> &lowerLimitedInputIndexs) {
  return true;
}

/**
* @brief: query op registered patterns
* @param [out] opPatternVec op_pattern list, key is op_type, value is pattern(Elewise/func/...).
* @return [out] bool        succ or fail
*/
bool TeStubApi::QueryOpPattern(const std::vector<std::pair<std::string, std::string>>& opPatternVec) {
  return true;
}

OpBuildResCode TeStubApi::BuildSuperKernel(const std::vector<ge::Node *> &graphNodes, ge::OpDescPtr opDesc,
    const uint64_t taskId, const uint64_t graphId) {
  string json_file_path =
          fe::GetCodeDir() + "/tests/engines/nn_engine/depends/te_fusion/compilation_result/te_op_312acb789acb79acb789cb7a89cef8978_1.json";
  ge::AttrUtils::SetStr(opDesc, "json_file_path", json_file_path);
  FinComTask new_task;
  new_task.graphId = graphId;
  new_task.taskId = taskId;
  new_task.status = 0;
  new_task.teNodeOpDesc = opDesc;
  auto iter = tasks_.find(graphId);
  if (iter == tasks_.end()) {
    vector<FinComTask> tasks = {new_task};
    tasks_.emplace(std::make_pair(graphId, tasks));
  } else {
    iter->second.emplace_back(new_task);
  }
  return OP_BUILD_SUCC;
}

std::string TeStubApi::GetKernelMetaDir() {
  return "./kernel_meta";
}
} // namespace te

