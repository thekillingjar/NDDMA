/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_JSON_PARSER_TBE_JSON_PARSE_IMPL_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_JSON_PARSER_TBE_JSON_PARSE_IMPL_H_

#include <climits>
#include <string>
#include <vector>
#include <memory>
#include <nlohmann/json.hpp>
#include "common/fe_inner_error_codes.h"
#include "common/fe_log.h"
#include "common/util/op_info_util.h"
#include "graph/compute_graph.h"
#include "common/tile_fwk_op_info.h"

namespace fe {
using JsonExpcetion = nlohmann::json::exception;

const std::vector<std::string> kBinaryMagicTypesVec = {
    "RT_DEV_BINARY_MAGIC_ELF_AICPU",
    "RT_DEV_BINARY_MAGIC_ELF",
    "RT_DEV_BINARY_MAGIC_ELF_AIVEC",
    "RT_DEV_BINARY_MAGIC_ELF_AICUBE",
    "FFTS_BINARY_MAGIC_ELF_MIX_AIC",
    "FFTS_BINARY_MAGIC_ELF_MIX_AIV"
};

using KeyJsonList = struct SJsonList {
  std::string coreType;
  std::string jsonFileName;
};

using KeyWorkSpace = struct SWorkSpace {
  int32_t num;
  std::vector<int64_t> size;
  std::vector<int32_t> type;
};

using KeyGlobalWorkspaceSpecWorkspace = struct SGlobalWorkspaceSpecWorkspace {
  int64_t size;
  int64_t type;
};

struct AtomicInitInfo {
  std::vector<int32_t> dtype_list;
  std::vector<int64_t> init_value_int64_list;
  std::vector<float> init_value_float_list;
};

struct TilingWithRatio {
  std::vector<std::string> tiling_key_vec;
  std::vector<int64_t> c_ratio_vec;
  std::vector<int64_t> v_ratio_vec;
};

struct OpTilingInfo {
  int tiling_key = 0;
  int tiling_cond = 0;
  int schedule_mode = 0;
  int local_memory_size = 0;
  int block_dim = 0;
  int aicpu_block_dim = 0;
  bool clear_atomic = false;
  std::string tiling_data_str;
  std::vector<int64_t> tvm_workspace_sizes;
};

// attr
constexpr char kKeyBlockDim[] = "blockDim";
constexpr char kKeyBatchBindOnly[] = "batchBindOnly";
constexpr char kKeyMagic[] = "magic";
constexpr char kKeyOldCoreType[] = "core_type";
constexpr char kKeyCoreType[] = "coreType";
constexpr char kKeyTaskRation[] = "taskRation";
constexpr char kKeyModeInArgsFirstField[] = "modeInArgsFirstField";
constexpr char kKeyIntercoreSync[] = "intercoreSync";
constexpr char kKeyParameters[] = "parameters";
constexpr char kKeyWspMode[] = "wspMode";
constexpr char kKeyCompressParameters[] = "compress_parameters";
constexpr char kKeyWeightRepeat[] = "weight_repeat";
constexpr char kKeySupportInfo[] = "supportInfo";
constexpr char kKeyOpParaSize[] = "opParaSize";
constexpr char kKeyOriOpParaSize[] = "oriOpParaSize";
constexpr char kKeyKernelList[] = "kernelList";
constexpr char kKeyKernelName[] = "kernelName";
constexpr char kKeyGlobalWorkspace[] = "globalworkspace_spec_workspace";
constexpr char kKeyWorkSpace[] = "workspace";
constexpr char kKeyKBHit[] = "KBHit";   // knowledge bank
constexpr char kKeyBinFileSuffix[] = "binFileSuffix";
constexpr char kKeyBinFileName[] = "binFileName";
constexpr char kKeyHeadFileSuffix[] = "headerFileSuffix";
constexpr char kKeySha256[] = "sha256";
constexpr char kKeyJsonList[] = "jsonList";
constexpr char kKeyOptionalInputMode[] = "optionalInputMode";
constexpr char kKeyOptionalOutputMode[] = "optionalOutputMode";
constexpr char kKeyDynamicParamMode[] = "dynamicParamMode";
constexpr char kKeyCompileResult[] = "compileResult";
constexpr char kKeyScheduleMode[] = "schedule_mode";
constexpr char kKeyDebugOptions[] = "debugOptions";
constexpr char kKeyDebugBufSize[] = "debugBufSize";
constexpr char kKeyLocalMemSize[] = "localMemorySize";
constexpr char kMetaData[] = "metaData";
constexpr char kKernelBinId[] = "kernelBinId";
constexpr char kBinFile[] = "BinFile";
constexpr char kHeadFilePath[] = "HeadFilePath";
constexpr char kRunInfo[] = "runInfo";
constexpr char kSupportSuperKernel[] = "supportSuperKernel";
constexpr char kFatbinInfo[] = "fatbinInfo";
constexpr int32_t kMaxBlockDim = 65535;
constexpr size_t kNumTwo = 2;

constexpr char kKeyDfxOption[] = "dfxOption";

// nlohmann::json need this function to parse json
void from_json(const nlohmann::json &json_value, KeyJsonList &json_list);

class TbeJsonFileParseImpl {
 public:
  TbeJsonFileParseImpl() {};

  ~TbeJsonFileParseImpl() {};

  /*
  *  @ingroup fe
  *  @brief   package the json info together
  *  @param   [in]  info
  *  @param   [out] tvm_file_path_
  *  @return  SUCCESS or FAILED
  */
  Status Initialize(const std::string &json_file_path);

  Status Initialize(const std::string &json_file_path, const TbeJsonPtr &json_ptr, const ge::OpKernelBinPtr &bin_ptr);

  TbeJsonFileParseImpl &operator=(const TbeJsonFileParseImpl &op) {
    if (&op == this) {
      return *this;
    }
    return *this;
  }

  template <typename T>
  Status ParseJsonAttr(const nlohmann::json &json_value, const bool is_force_need,
                       const std::string &key, const T &default_value, T &value) const {
  if (json_value.find(key) == json_value.end()) {
    if (is_force_need) {
       FE_LOGE("attr [%s] is necessary, but not found", key.c_str());
       return FAILED;
     } else {
       value = default_value;
       return SUCCESS;
     }
    }

   try {
      value = json_value.at(key).get<T>();
    } catch (JsonExpcetion &e) {
      FE_LOGE("get attr[%s] failed, exception[%s]", key.c_str(), e.what());
      value = default_value;
      return FAILED;
    }
    return SUCCESS;
  }

  template <typename T>
  Status ParseJsonAttr(const bool is_force_need, const std::string &key, const T &default_value, T &value) const {
    return ParseJsonAttr(json_handle_, is_force_need, key, default_value, value);
  }

  /*
  *  @ingroup fe
  *  @brief   reading binary files
  *  @param   [in]  file_name(or path)
  *  @param   [out] buffer
  *  @return  SUCCESS or FAILED
  */
  Status ReadBytesFromBinaryFile(const std::string& file_name, std::vector<char>& buffer) const;

  /*
  *  @ingroup fe
  *  @brief  joint the path of bin file, if success, renew the op_desc.name
  *  @param   [in] handle
  *  @param   [out] op_desc_->name
  *  @return  SUCCESS or FAILED
  */
  Status PackageTvmBinFile(std::vector<char> &buffer) const;

  /*
   *  @ingroup fe
   *  @brief  joint the path of bin file, if success, renew the op_desc.name
   *  @param   [in] handle
   *  @param   [out] op_desc_->name
   *  @return  SUCCESS or FAILED
   */
  Status ParseHeadFilePath(std::string &head_file_path) const;

  /*
  *  @ingroup fe
  *  @brief  parse the workspace info in handle
  *  @param   [in] handle
  *  @param   [out] op_desc_, tvm_workspace_sizes_
  *  set workspace according to block_dim info in handle
  *  @return SUCCESS or FAILED
  */
  Status ParseTvmWorkSpace(std::vector<int64_t> &tvm_workspace_sizes, std::vector<int64_t> &tvm_workspace_types) const;

  /*
  *  @ingroup fe
  *  @brief  parse the parameters info in handle
  *  @param   [in] handle
  *  @param   [out] op_desc_, set output according to output info in handle
  *  @param   [out] op_desc_, set dtype_list and init_values_list according to parameters info in handle
  *  @return SUCCESS or FAILED
  */
  Status ParseTvmParameters(std::vector<int64_t> &parameters_index, AtomicInitInfo &atomic_init_info);

  /*
  *  @ingroup fe
  *  @brief  parse the meta_data info in handle
  *  @param   [in] handle
  *  @param   [out] op_desc_, set meta_data according to meta_data info in handle
  *  @return SUCCESS or FAILED
  */
  Status ParseTvmMetaData(std::string &meta_data) const;

  Status ParseOpDfxOptions(std::vector<std::string> &opt_list, int64_t &buffer_size) const;

  /*
   *  @ingroup fe
   *  @brief  parse the kernel bin id info in handle
   *  @param   [in] handle
   *  @param   [out] op_desc_, set kernel bin id according to kernel bin id info in handle
   *  @return SUCCESS or FAILED
   */
  Status ParseTvmKernelBinId(std::string &kernel_bin_id) const;

  Status ParseGlobleWorkspaceStatus(KeyGlobalWorkspaceSpecWorkspace &global_work_space) const;

  Status ParseTvmKernelList(std::string &kernel_list_first) const;

  Status ParseListTilingRatio(TilingWithRatio &tiling_ratio) const;

  Status ParseTvmTaskRatio(int32_t &cube_ratio, int32_t &vector_ratio, bool &dyn_ratio) const;

  Status ParseTvmJsonList(std::vector<KeyJsonList> &json_list) const;

  void SetTvmDirPath(const std::string &json_file_path);

  ge::OpKernelBinPtr GetOpKernelBinPtr() const;

  const std::string& GetTvmDirPath() const;

  Status ParseRunInfo(OpTilingInfo &run_info, bool &has_run_info) const;

  Status ParseFatbin(const ge::OpKernelBinPtr &fatbin, FatbinKernelInfoMap &fatbin_kernel_info_map) const;

  Status ParseFatbinJson(FatbinKernelInfoMap &fatbin_kernel_info_map) const;

 private:
  nlohmann::json json_handle_;
  ge::OpKernelBinPtr op_kernel_bin_;
  std::string tvm_dir_path_;
};
using TbeJsonFileParseImplPtr = std::unique_ptr<TbeJsonFileParseImpl>;
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_JSON_PARSER_TBE_JSON_PARSE_IMPL_H_
