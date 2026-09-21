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
#include "tbe_op_func_manager.h"

namespace te {
namespace {
std::string GetCodeDir() {
  static std::string gCachedCodeDir;
  if (gCachedCodeDir.empty()) {
    const char *code_path_ptr = std::getenv("AIR_CODE_DIR");
    if (code_path_ptr != nullptr) {
      gCachedCodeDir = string(code_path_ptr);
    }
  }
  return gCachedCodeDir;
}
const unordered_map<std::string, std::string> op_type_pattern_map {
    {"Aipp", "aipp"},
    {"StridedWrite", "strided_write"},
    {"StridedRead", "strided_read"},
    {"Pooling", "Pool2d"},
    {"FullyConnection", "Matmul"},
    {"Conv2D", "Convolution"},
    {"AscendRequant", "requant"},
    {"AscendQuant", "quant"},
    {"AscendDequant", "dequant"},
    {"AscendAntiQuant", "anti_quant"},
    {"MatMul", "Matmul"},
    {"LeakyRelu", "ElemWise"},
    {"Add", "ElemWise"},
    {"Mul", "ElemWise"},
    {"RealDiv", "ElemWise"},
    {"Square", "ElemWise"},
    {"Sqrt", "ElemWise"},
    {"Relu", "ElemWise"},
    {"Gelu", "ElemWise"},
    {"GeluGrad", "ElemWise"}
};
}
class TeFusionStub {
 public:
  static TeFusionStub &Instance() {
    static TeFusionStub instance;
    return instance;
  }
  void Reset() {
    tasks_.clear();
    tbe_op_infos_.clear();
  }
  void GetOpPattern(const TbeOpInfo& opinfo, std::string &op_pattern) {
    op_pattern = "Opaque";
    string op_type;
    opinfo.GetOpType(op_type);
    auto iter = op_type_pattern_map.find(op_type);
    if (iter != op_type_pattern_map.end()) {
      op_pattern = iter->second;
    }
    if (op_type == "Pooling" && op_pattern != "Convolution") {
      std::vector<TbeAttrValue> attr_values;
      opinfo.GetAttrValues(attr_values);
      for (const auto &attr : attr_values) {
        if (attr.GetName() == "mode") {
          int64_t value;
          attr.GetValue(value);
          if (value == 1) {
            op_pattern = "Convolution";
          }
        }
      }
    }
  }
  bool PreBuildTbeOp(TbeOpInfo& opinfo, uint64_t taskId, uint64_t graphId) {
    string op_name;
    opinfo.GetRealName(op_name);
    te::TbeOpInfoPtr op_info_ptr = opinfo.shared_from_this();
    tbe_op_infos_.emplace(op_name, op_info_ptr);

    string op_pattern;
    GetOpPattern(opinfo, op_pattern);

    FinComTask new_task;
    new_task.graphId = graphId;
    new_task.taskId = taskId;
    new_task.status = 0;
    opinfo.SetPattern(op_pattern);
    string core_type = opinfo.GetOpType() == "Relu" ? "Default" : "AiCore";
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

  OpBuildResCode TeFusionV(std::vector<ge::Node *> teGraphNode, ge::OpDescPtr opDesc,
                           const std::vector<ge::NodePtr> &toBeDel, uint64_t taskId, uint64_t graphId,
                           uint64_t sgtThreadIndex, const std::string &strategy) {
    string json_file_path =
            GetCodeDir() + "/tests/engines/nn_engine/depends/te_fusion/compilation_result/te_op_312acb789acb79acb789cb7a89cef8978_1.json";
    ge::AttrUtils::SetStr(opDesc, "json_file_path", json_file_path);
    FinComTask new_task;
    new_task.graphId = graphId;
    new_task.taskId = taskId;
    new_task.status = 0;
    for (ge::Node *node : teGraphNode) {
      if (node == nullptr) {
        continue;
      }
      ge::OpDescPtr op_desc = node->GetOpDesc();
      // l1_fusion failed
      bool l1_compile_fail = false;
      if (ge::AttrUtils::HasAttr(op_desc, "_l1_fusion_scope") &&
          ge::AttrUtils::GetBool(op_desc, "_l1_compile_fail", l1_compile_fail) && l1_compile_fail) {
        new_task.status = 1;
      }
      // ub fusion failed
      int64_t scope_id = 0;
      bool ub_compile_fail = false;
      if (ge::AttrUtils::GetInt(op_desc, "fusion_scope", scope_id) && scope_id >= 0 &&
          ge::AttrUtils::GetBool(op_desc, "_ub_compile_fail", ub_compile_fail) && ub_compile_fail) {
        new_task.status = 1;
      }
      // compile failed
      bool compile_fail = false;
      if (ge::AttrUtils::GetBool(op_desc, "_compile_fail", compile_fail) && compile_fail) {
        new_task.status = 1;
      }
      // fusion check
      bool is_fusion_check = false;
      bool fusion_check_fail = false;
      ge::AttrUtils::GetBool(op_desc, "_auto_fusion_pattern", is_fusion_check);
      ge::AttrUtils::GetBool(op_desc, "_fusion_check_fail", fusion_check_fail);
      if (is_fusion_check && fusion_check_fail) {
        new_task.status = 1;
      }
    }

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
  OpBuildResCode TaskFusion(const std::vector<ge::Node *> &graphNodes, ge::OpDescPtr opDesc,
                            const uint64_t taskId, const uint64_t graphId) {
    string json_file_path =
            GetCodeDir() + "/tests/engines/nn_engine/depends/te_fusion/compilation_result/te_op_312acb789acb79acb789cb7a89cef8978_1.json";
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
  bool WaitAllFinished(uint64_t graphId, vector<FinComTask> &tasks) {
    auto iter = tasks_.find(graphId);
    if (iter != tasks_.end()) {
      tasks = std::move(iter->second);
      tasks_.erase(graphId);
    }
    return true;
  }

  bool CheckTbeSupported(TbeOpInfo& opinfo, CheckSupportedResult &isSupport, std::string &reason) {
    string op_type;
    opinfo.GetOpType(op_type);
    CheckSupportFunc func;
    if (TbeOpFuncManager::Instance().GetCheckSupportFunction(op_type, func)) {
      return func(opinfo, isSupport, reason);
    }
    isSupport = te::FULLY_SUPPORTED;
    return true;
  }
  bool SelectTbeOpFormat(const TbeOpInfo &tbeOpInfo, std::string &opDtypeFormat) {
    string op_type;
    tbeOpInfo.GetOpType(op_type);
    SelectFormatFunc func;
    if (TbeOpFuncManager::Instance().GetSelectFormatFunction(op_type, func)) {
      return func(tbeOpInfo, opDtypeFormat);
    }
    return true;
  }

  OpBuildResCode BuildSuperKernel(const std::vector<ge::Node *> &graphNodes, ge::OpDescPtr opDesc,
      const uint64_t taskId, const uint64_t graphId) {
    string json_file_path =
            GetCodeDir() + "/tests/engines/nn_engine/depends/te_fusion/compilation_result/te_op_312acb789acb79acb789cb7a89cef8978_1.json";
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

  std::string GetKernelMetaDir() {
    return "./kernel_meta";
  }

 private:
  TeFusionStub() {}
  ~TeFusionStub() {}
  std::map<uint64_t, vector<FinComTask>> tasks_;
  std::map<string, te::TbeOpInfoPtr> tbe_op_infos_;
};
/**
 * @brief: Tbe Initialize proccess
 * @param [in] options            ddk version
 * @return [out] bool             succ or fail for Tbe Initialize
 */
bool TbeInitialize(const std::map<std::string, std::string> &options, bool *isSupportParallelCompilation) {
  TeFusionStub::Instance().Reset();
  return true;
}

/**
 * @brief: tbe finalize proccess
 * @return [out] bool             succ or fail for Tbe Finalize
 */
bool TbeFinalize() {
  TeFusionStub::Instance().Reset();
  return true;
}

bool TbeFinalizeSessionInfo(const std::string &session_graph_id) {
  return true;
}

bool CheckTbeSupported(TbeOpInfo& opinfo, CheckSupportedResult &isSupport, std::string &reason) {
  return TeFusionStub::Instance().CheckTbeSupported(opinfo, isSupport, reason);
}

bool CheckOpSupported(TbeOpInfo& opinfo, CheckSupportedInfo &result) {
  bool res = TeFusionStub::Instance().CheckTbeSupported(opinfo, result.isSupported, result.reason);
  return res;
}

bool SelectTbeOpFormat(const TbeOpInfo &tbeOpInfo, std::string &opDtypeFormat) {
  return TeFusionStub::Instance().SelectTbeOpFormat(tbeOpInfo, opDtypeFormat);
}

bool GetOpUniqueKeys(const TbeOpInfo &tbeOpInfo, std::vector<std::string> &opUniqueKeys) {
  opUniqueKeys.push_back("11223344");
  return true;
}

/**
 * @brief pre build tbe op
 * @return [out] bool                 succ or fail of prebuild
 */
bool PreBuildTbeOp(TbeOpInfo& opinfo, uint64_t taskId, uint64_t graphId) {
  return TeFusionStub::Instance().PreBuildTbeOp(opinfo, taskId, graphId);
}

OpBuildResCode TeFusion(std::vector<ge::Node *> teGraphNode, ge::OpDescPtr opDesc,
                        const std::vector<ge::NodePtr> &toBeDel,
                        uint64_t taskId, uint64_t graphId, const std::string &strategy) {
  uint64_t sgtThreadIndex = 0xffffffff;
  return TeFusionStub::Instance().TeFusionV(teGraphNode, opDesc, toBeDel, taskId, graphId, sgtThreadIndex, strategy);
}

OpBuildResCode TeFusionV(std::vector<ge::Node *> teGraphNode, ge::OpDescPtr opDesc,
                         const std::vector<ge::NodePtr> &toBeDel, uint64_t taskId, uint64_t graphId,
                         uint64_t sgtThreadIndex, const std::string &strategy) {
  return TeFusionStub::Instance().TeFusionV(teGraphNode, opDesc, toBeDel, taskId, graphId, sgtThreadIndex, strategy);
}

OpBuildResCode TaskFusion(const std::vector<ge::Node *> &graphNodes, ge::OpDescPtr opDesc,
                          const uint64_t taskId, const uint64_t graphId) {
  return TeFusionStub::Instance().TaskFusion(graphNodes, opDesc, taskId, graphId);
}

/**
 * @brief: get finished compilation task list
 * @return [out] list             finished compilation task list
 */
bool WaitAllFinished(uint64_t graphId, vector<FinComTask> &tasks) {
  return TeFusionStub::Instance().WaitAllFinished(graphId, tasks);
}

/**
 * @brief get op info
 * @param [in] tbeOpInfo            op info
 * @param [out] result              op segmentation information
 * @return [out] LX_QUERY_STATUS    LX_QUERY_FAIL, LX_QUERY_NOT_FOUND or LX_QUERY_SUCC
 */
LX_QUERY_STATUS GetOpInfo(const TbeOpInfo &tbeOpInfo, std::string &result) {
  return LX_QUERY_SUCC;
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
  return OP_BUILD_SUCC;
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
  return true;
}

/**
 * @brief check is this op has registered tbegeneralize func
 * @param [in] tbeOpInfo            op info
 * @param [in] hasRegisteredFunc    true or false
 * @return [out] bool               succ or fail of check
 */
bool CheckIsTbeGeneralizeFuncRegistered(const TbeOpInfo &tbeOpInfo, bool &hasRegisteredFunc) {
  return true;
}

/**
 * @brief shape or value generalize with generalize type
 * @param [in] tbeOpInfo            op info
 * @param [in] generalize_type      generalize type
 * @param [in] nodePtr              node
 * @return [out] bool               succ or fail of check
 */
bool TeGeneralize(const TbeOpInfo &tbeOpInfo, const TE_GENERALIZE_TYPE &generalize_type, const ge::NodePtr &nodePtr) {
  return true;
}

/**
 * @brief get op specific info from tbe
 * @param [in] tbeOpInfo         op info
 * @param [in] opSpecificInfo    op specific info
 * @param [out] bool             succ or fail
 */
bool GetOpSpecificInfo(const TbeOpInfo &tbeOpInfo, std::string &opSpecificInfo) {
  return true;
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
  return true;
}

/**
* @brief: query op registered patterns
* @param [out] opPatternVec op_pattern list, key is op_type, value is pattern(Elewise/func/...).
* @return [out] bool        succ or fail
*/
bool QueryOpPattern(const std::vector<std::pair<std::string, std::string>>& opPatternVec) {
  return true;
}

void GetAllCompileStatistics(std::vector<std::string> &statistics_msg) {
  statistics_msg.push_back("Disk cache statistics:match times:132|reuse times:45");
  statistics_msg.push_back("Binary statistics:match times:132|reuse times:86");
}

bool IsOppKernelInstalled(bool isOm, int64_t implType) {
  return true;
}

extern "C" OpBuildResCode BuildSuperKernel(const std::vector<ge::Node *> &graphNodes, ge::OpDescPtr opDesc,
    const uint64_t taskId, const uint64_t graphId) {
  return TeFusionStub::Instance().BuildSuperKernel(graphNodes, opDesc, taskId, graphId);
}

extern "C" std::string GetKernelMetaDir()
{
  return TeFusionStub::Instance().GetKernelMetaDir();
}
} // namespace te

