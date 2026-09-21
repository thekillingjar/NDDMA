/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_JSON_PARSER_TBE_JSON_PARSE_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_JSON_PARSER_TBE_JSON_PARSE_H_

#include <climits>
#include <string>
#include <vector>
#include "common/fe_inner_attr_define.h"
#include "common/fe_inner_error_codes.h"
#include "common/aicore_util_constants.h"
#include "common/aicore_util_types.h"
#include "common/fe_op_info_common.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"
#include "graph_optimizer/json_parser/tbe_json_parse_impl.h"

namespace fe {
using NodeBaseInfo = struct tagNodeBaseInfo {
  size_t workspace_num;
  size_t input_num;
  size_t output_num;
  size_t offset_index;
};

class TbeJsonFileParse {
 public:
  explicit TbeJsonFileParse(ge::Node& node);

  ~TbeJsonFileParse() {};

  Status PackageTvmJsonInfo(const CompileResultInfo &compile_result);

 private:
  ge::Node& node_;
  ge::OpDescPtr op_desc_;
  int32_t cube_ratio_{0};
  int32_t vector_ratio_{0};
  std::shared_ptr<std::vector<ge::NodePtr>> ffts_related_thread_nodes_;
  TbeJsonFileParseImplPtr json_parser_impl_;
  std::string attr_prefix_;
  using ParseFunc = Status(TbeJsonFileParse::*)();
  std::vector<std::pair<std::string, ParseFunc>> parse_func_map_ = {
      {kKeyTaskRation, &TbeJsonFileParse::ParseTvmTaskRatio},
      {kKeyOldCoreType, &TbeJsonFileParse::ParseTvmOldCoreType},
      {kKeyCoreType, &TbeJsonFileParse::ParseTvmCoreType},
      {kKeyMagic, &TbeJsonFileParse::ParseTvmMagic},
      {kKeyBlockDim, &TbeJsonFileParse::ParseTvmBlockDim},
      {kKeyScheduleMode, &TbeJsonFileParse::ParseScheduleMode},
      {kKeyModeInArgsFirstField, &TbeJsonFileParse::ParseTvmModeInArgsFirstField},
      {kKeyIntercoreSync, &TbeJsonFileParse::ParseTvmInterCoreSync},
      {kKeyBatchBindOnly, &TbeJsonFileParse::ParseBatchBindOnly},
      {kKeyWorkSpace, &TbeJsonFileParse::ParseTvmWorkSpace},
      {kKeyParameters, &TbeJsonFileParse::ParseTvmParameters},
      {kKeyWspMode, &TbeJsonFileParse::ParseTvmWspMode},
      {kMetaData, &TbeJsonFileParse::ParseTvmMetaData},
      {kKernelBinId, &TbeJsonFileParse::ParseTvmKernelBinId},
      {kKeyKernelName, &TbeJsonFileParse::ParseTvmKernelName},
      {kKeyCompressParameters, &TbeJsonFileParse::ParseConvCompressParameters},
      {kKeyWeightRepeat, &TbeJsonFileParse::ParseWeightRepeat},
      {kKeySupportInfo, &TbeJsonFileParse::ParseSupportInfo},
      {kKeyOpParaSize, &TbeJsonFileParse::ParseOpParaSize},
      {kBinFile, &TbeJsonFileParse::PackageTvmBinFile},
      {kHeadFilePath, &TbeJsonFileParse::PackageHeadFilePath},
      {kKeyKernelList, &TbeJsonFileParse::ParseTvmKernelList},
      {kKeyGlobalWorkspace, &TbeJsonFileParse::ParseGlobleWorkspaceStatus},
      {kKeyOptionalInputMode, &TbeJsonFileParse::ParseOptionalInputMode},
      {kKeyOptionalOutputMode, &TbeJsonFileParse::ParseOptionalOutputMode},
      {kKeyDynamicParamMode, &TbeJsonFileParse::ParseDynamicParamMode},
      {kKeyKBHit, &TbeJsonFileParse::ParseOpKBHitrate},
      {kKeyCompileResult, &TbeJsonFileParse::ParseCompileResult},
      {kKeyDfxOption, &TbeJsonFileParse::ParseOpDfxOptions},
      {kKeyLocalMemSize, &TbeJsonFileParse::ParseLocalMemSize},
      {kRunInfo, &TbeJsonFileParse::ParseAndSetTilingInfo},
      {kSupportSuperKernel, &TbeJsonFileParse::ParseSuperKernelSupportFlag},
      {kFatbinInfo, &TbeJsonFileParse::ParseFatbinInfo}
  };
  size_t total_param_size_{0};
  TbeJsonFileParse& operator=(const TbeJsonFileParse& op) {
    if (&op == this) {
      return *this;
    }
    return *this;
  }

  Status PackageTvmBinFile();
  Status PackageHeadFilePath();
  Status ParseTvmMagic();
  Status ParseTvmBlockDim();
  Status ParseScheduleMode();
  Status ParseBatchBindOnly();
  Status ParseTvmWorkSpace();
  Status ParseTvmParameters();
  Status ParseTvmWspMode();
  Status ParseTvmMetaData();
  Status ParseTvmKernelBinId();
  Status ParseTvmKernelName();
  Status ParseConvCompressParameters();
  Status ParseSupportInfo();
  Status ParseOpParaSize();
  Status ParseWeightRepeat();
  Status ParseTvmOldCoreType();
  Status ParseTvmCoreType();
  Status ParseTvmTaskRatio();
  Status ParseTvmStrTaskRatio();
  Status ParseTvmModeInArgsFirstField();
  Status ParseTvmInterCoreSync();
  Status ParseTvmKernelList();
  Status ParseGlobleWorkspaceStatus();
  Status ParseOpKBHitrate();
  Status ParseCompileResult();
  Status ParseOpDfxOptions();
  Status ParseLocalMemSize();
  void ProcMixCoreType();
  Status ParseAndSetTilingInfo();
  bool ParseAndSetTilingData(const std::string &tiling_data_str, RunInfoPtr &tiling_info);
  void HexDecode(const std::string &hex_str, std::vector<unsigned char> &bytes) const;
  void GetWorkspaceAtomicFlagAndOutputIndexFlag(const std::vector<int64_t> &parameters_index,
                                                const NodeBaseInfo &node_info,
                                                std::vector<int64_t> &output_index,
                                                std::vector<int64_t> &workspace_index,
                                                bool &workspace_atomic_flag, bool &output_index_flag) const;

  Status SetAtomicInfo(std::vector<int64_t> &parameters_index, AtomicInitInfo &atomic_init_info);

  void SetAtomicInitInfo(const bool &output_index_flag, bool &workspace_atomic_flag,
                         std::vector<int64_t> &output_index, std::vector<int64_t> &workspace_index,
                         AtomicInitInfo &atomic_init_info);

  Status ParseOptionalInputMode();

  Status ParseOptionalOutputMode();

  Status ParseDynamicParamMode();

  bool IsModelBinaryReuse(const CompileResultInfo &compile_result);

  bool SetRelatedNodesListInt(const std::string &attr_name, const std::vector<int64_t> &value);
  bool SetRelatedNodesListInt(const std::string &attr_name, const std::vector<int32_t> &value);
  bool SetRelatedNodesListStr(const std::string &attr_name, const std::vector<string> &value);
  bool SetRelatedNodesInt(const std::string &attr_name, const int64_t &value);
  bool SetRelatedNodesBool(const std::string &attr_name, const bool &value);
  bool SetRelatedNodesStr(const string &attr_name, const string &value);
  bool SetRelatedNodesStrPrefixWithOpName(const string &prefix, const string &attr_name, const string &value);
  bool SetRelatedNodesBytes(const string &attr_name, const ge::Buffer &value);
  bool SetRelatedNodesListDataType(const string &attr_name, const std::vector<ge::DataType> &value);
  bool SetRelatedNodesListFloat(const string &attr_name, const std::vector<float> &value);
  void SetRelatedNodesWorkspace(const std::vector<int64_t> &tvm_workspace_sizes);
  bool ClearRelatedNodesStr(const string &attr_name);
  template<class T>
  bool SetRelatedNodesExtAttr(const std::string &name, const T &value) {
    bool ret = op_desc_->SetExtAttr(name, value);
    if (ffts_related_thread_nodes_ != nullptr) {
      for (auto &ele : *ffts_related_thread_nodes_) {
        ret &= ele->GetOpDesc()->SetExtAttr(name, value);
      }
    }
    return ret;
  }
  bool IsEnhancedMixAiCore();
  bool IsDynamicTaskRatio() const;
  bool IsMixDynamicTaskRatio() const;

  const std::string& GetAttrPrefix() const;
  Status ParseSuperKernelSupportFlag();
  Status ParseFatbinInfo();
};
using TbeJsonFileParsePtr = std::shared_ptr<TbeJsonFileParse>;
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_JSON_PARSER_TBE_JSON_PARSE_H_
