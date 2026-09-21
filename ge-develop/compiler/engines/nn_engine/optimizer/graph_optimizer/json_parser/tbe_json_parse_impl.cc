/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/json_parser/tbe_json_parse_impl.h"
#include <fstream>
#include <thread>
#include <ext/stdio_filebuf.h>
#include <sys/file.h>
#include "common/fe_inner_attr_define.h"
#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "common/string_utils.h"
#include "common/util/json_util.h"
#include "tensor_engine/fusion_types.h"
#include "graph/utils/op_type_utils.h"

namespace fe {
namespace {
constexpr size_t kMixChildJsonNum = 2;
constexpr size_t kRatioVecSize = 2;
constexpr int32_t kTmpMixRation = 1;

int32_t TransGeWorkspaceType(int32_t type) {
  int32_t res_type = 0;
  switch (type) {
    case te::TBE_MEMORY_DDR:
      res_type = RT_MEMORY_HBM;
      break;

    case te::TBE_MEMORY_L1:
      res_type = RT_MEMORY_L1;
      break;

    case te::TBE_MEMORY_L2:
      res_type = RT_MEMORY_L2;
      break;

    default:
      res_type = RT_MEMORY_HBM;
      break;
  }
  return res_type;
}
} // namespace

void from_json(const nlohmann::json &json_value, KeyJsonList &json_list) {
  json_value.at("coreType").get_to(json_list.coreType);
  json_value.at("jsonFileName").get_to(json_list.jsonFileName);
}

Status TbeJsonFileParseImpl::ParseListTilingRatio(TilingWithRatio &tiling_ratio) const {
  std::unordered_set<std::string> tiling_key_set;
  int32_t cube_ratio;
  int32_t vector_ratio;
  auto kernel_value = json_handle_.find(kKeyKernelList).value();
  for (auto it = kernel_value.begin(); it != kernel_value.end(); ++it) {
    auto tiling_key = it->at("tilingKey");
    std::string tiling_key_val = std::to_string(tiling_key.get<uint64_t>());
    if (tiling_key_set.count(tiling_key_val) == 1) {
      FE_LOGE("There is a repeated tiling key [%s] in the kernel list.", tiling_key_val.c_str());
      return FAILED;
    }

    auto task_ratio = it->at("taskRation");
    std::string task_ratio_str = task_ratio.get<std::string>();
    std::vector<std::string> ratio_vec = StringUtils::Split(task_ratio_str, ':');
    if (ratio_vec.size() != kRatioVecSize) {
      FE_LOGE("Task ratio[%s] format is invalid.", task_ratio_str.c_str());
      return FAILED;
    }
    try {
      cube_ratio = std::stoi(ratio_vec[0].c_str());
      vector_ratio = std::stoi(ratio_vec[1].c_str());
    } catch (std::invalid_argument &) {
      FE_LOGE("Task ratio[%s] format is invalid.", task_ratio_str.c_str());
      return FAILED;
    }
    tiling_ratio.tiling_key_vec.emplace_back(tiling_key_val);
    tiling_ratio.c_ratio_vec.emplace_back(cube_ratio);
    tiling_ratio.v_ratio_vec.emplace_back(vector_ratio);
    FE_LOGD("Parsing tilingKey[%s] with c/v ratio of [%d/%d].", tiling_key_val.c_str(), cube_ratio, vector_ratio);
    tiling_key_set.insert(tiling_key_val);
  }
  return SUCCESS;
}

Status TbeJsonFileParseImpl::ParseTvmKernelList(std::string &kernel_list_first) const {
  auto kernel_list = json_handle_.find(kKeyKernelList);
  if (kernel_list == json_handle_.end()) {
    return SUCCESS;
  }

  try {
    if (kernel_list.value().empty()) {
      FE_LOGD("kernel list is empty.");
      return SUCCESS;
    }
    nlohmann::json first_name = kernel_list.value().begin()->at("kernelName");
    kernel_list_first = first_name.get<std::string>();
    FE_LOGI("The name of the first node in the kernel list is %s.", kernel_list_first.c_str());
  } catch (const JsonExpcetion &e) {
    FE_LOGE("ParseTvmKernelList encountered an exception for kernel name: %s", e.what());
  }
  return SUCCESS;
}

/* OpKernelBinPtr
 *  @ingroup fe
 *  @brief  parse the workspace info in handle
 *  @param   [in] handle
 *  @param   [out] op_desc_, tvm_workspace_sizes_  set workspace according to
 * block_dim info in handle
 *  @return SUCCESS or FAILED
 */
Status TbeJsonFileParseImpl::ParseTvmWorkSpace(vector<int64_t> &tvm_workspace_sizes,
                                               vector<int64_t> &tvm_workspace_types) const {
  auto j_workspace = json_handle_.find(kKeyWorkSpace);
  if (j_workspace == json_handle_.end()) {
    return SUCCESS;
  }
  try {
    KeyWorkSpace workspace;
    if (ParseJsonAttr(j_workspace.value(), true, "num", workspace.num, workspace.num) != SUCCESS) {
      return FAILED;
    }
    if (ParseJsonAttr(j_workspace.value(), true, "size", workspace.size, workspace.size) != SUCCESS) {
      return FAILED;
    }

    if (workspace.num != static_cast<int32_t>(workspace.size.size())) {
      FE_LOGE("workspace num:%d is not equal to workspace sizes size:%zu", workspace.num, workspace.size.size());
      return FAILED;
    }

    for (int32_t index = 0; index < workspace.num; ++index) {
      tvm_workspace_sizes.emplace_back(workspace.size[index] == UNKNOWN_WORK_SPACE_SIZE ?
                                       workspace.size[index] : (workspace.size[index] + DATA_MEMORY_ALIGN_SIZE));
    }

    if (ParseJsonAttr(j_workspace.value(), false, "type", workspace.type, workspace.type) == SUCCESS) {
      if (workspace.type.empty()) {
        return SUCCESS;
      }
      if (!workspace.type.empty() && workspace.num != static_cast<int32_t>(workspace.type.size())) {
        FE_LOGE("workspace num: %d does not match workspace types size: %zu", workspace.num, workspace.type.size());
        return FAILED;
      }

      for (int32_t index = 0; index < workspace.num; ++index) {
        tvm_workspace_types.emplace_back(TransGeWorkspaceType(workspace.type[index]));
      }
    }
  } catch (const JsonExpcetion &e) {
    FE_LOGE("Failed to get the workspace, exception: %s", e.what());
    return FAILED;
  }

  return SUCCESS;
}
Status TbeJsonFileParseImpl::ParseTvmParameters(std::vector<int64_t> &parameters_index,
                                                AtomicInitInfo &atomic_init_info)
{
  if (json_handle_.find(kKeyParameters) == json_handle_.end()) {
    return SUCCESS;
  }

  try {
    nlohmann::json &j_parameters = json_handle_.at(kKeyParameters);
    if (!j_parameters.is_array()) {
      FE_LOGE("[SubGraphOpt][ParseJson][ParseTvmParameters] json info is not an array");
      return FAILED;
    }
    for (auto it = j_parameters.begin(); it != j_parameters.end(); ++it) {
      nlohmann::json &parameter_object = *it;
      if (parameter_object.is_null()) {
        parameters_index.emplace_back(0);
        continue;
      }

      if (!parameter_object.is_object()) {
        parameters_index = j_parameters.get<std::vector<int64_t>>();
        return SUCCESS;
      }
      parameters_index.emplace_back(1);
      const std::string dtype_str = "dtype";
      const std::string init_value_str = "init_value";
      const std::string int_dtype_str = "int";
      const std::string float_dtype_str = "float";
      const std::string &dtype = parameter_object.at(dtype_str).get<std::string>();
      ge::DataType ge_dtype;
      FE_CHECK(TransStringToDtype(dtype, ge_dtype) != SUCCESS, FE_LOGE("dtype %s is invalid", dtype.c_str()),
               return FAILED);
      atomic_init_info.dtype_list.emplace_back(static_cast<int32_t>(ge_dtype));
      if (dtype.find(int_dtype_str) != std::string::npos) {
        atomic_init_info.init_value_int64_list.emplace_back(parameter_object.at(init_value_str).get<int64_t>());
      } else if (dtype.find(float_dtype_str) != std::string::npos) {
        atomic_init_info.init_value_float_list.emplace_back(parameter_object.at(init_value_str).get<float>());
      } else {
        FE_LOGE("[SubGraphOpt][ParseJson][ParseTvmParameters] The data type [%s] is not supported.", dtype.c_str());
        return FAILED;
      }
    }
  } catch (const JsonExpcetion &e) {
    FE_LOGE("Failed to get parameters, exception: %s", e.what());
    return FAILED;
  }

  return SUCCESS;
}

/*
*  @ingroup fe
*  @brief  parse the meta_data info in handle
*  @param   [in] handle
*  @param   [out] op_desc_, set meta_data according to meta_data info in handle
*  @return SUCCESS or FAILED
*/
Status TbeJsonFileParseImpl::ParseTvmMetaData(std::string &meta_data) const {
  // 1. get bin_file_suffix
  std::string binfile_suffix;
  if (ParseJsonAttr(false, kKeyBinFileSuffix, binfile_suffix, binfile_suffix) == FAILED) {
    return FAILED;
  }
  // do not register MetaData when bin_file_suffix is not .so
  if (binfile_suffix != ".so") {
    FE_LOGD("'BinFileSuffix' is not '.so', skip.");
    return SUCCESS;
  }

  // 2. get bin_file_name
  std::string binfile_name;
  if (ParseJsonAttr(false, kKeyBinFileName, binfile_name, binfile_name) == FAILED) {
    return FAILED;
  }
  // check kernel name
  if (binfile_name.empty()) {
    FE_LOGE("BinFileName is empty. Please check the JSON file.");
    return FAILED;
  }

  // sha256, set sha256 to be null string when it is not specifed in json file
  std::string sha256;
  if (ParseJsonAttr(false, kKeySha256, sha256, sha256) == FAILED) {
    return FAILED;
  }
  if (sha256.empty()) {
    FE_LOGE("BinFileSuffix is so, but the JSON information is insufficient. Please check the JSON file.");
    return FAILED;
  }

  // 3. concat meta data
  meta_data.append(binfile_name);
  meta_data.append(binfile_suffix);
  meta_data.append(", version, ");
  meta_data.append(sha256);
  meta_data.append(", shared");
  FE_LOGD("Add meta data: %s.", meta_data.c_str());
  return SUCCESS;
}

Status TbeJsonFileParseImpl::ParseTvmKernelBinId(std::string &kernel_bin_id) const {
  std::string binfile_name;
  std::string sha256;
  (void)ParseJsonAttr(false, kKeyBinFileName, binfile_name, binfile_name);
  (void)ParseJsonAttr(false, kKeySha256, sha256, sha256);
  kernel_bin_id.append(binfile_name);
  kernel_bin_id.append(sha256);
  FE_LOGD("Kernel bin ID is [%s].", kernel_bin_id.c_str());
  return SUCCESS;
}

Status TbeJsonFileParseImpl::ParseGlobleWorkspaceStatus(KeyGlobalWorkspaceSpecWorkspace &global_work_space) const {
  auto j_global_work = json_handle_.find(kKeyGlobalWorkspace);
  if (j_global_work == json_handle_.end()) {
    FE_LOGD("[SubGraphOpt][Compile][ParseGlobalWorkspaceStatus] cannot get GlobalWorkspaceSpecWorkspace");
    return NOT_CHANGED;
  }
  try {
    if (ParseJsonAttr(j_global_work.value(), true, "size", global_work_space.size, global_work_space.size) != SUCCESS) {
      return FAILED;
    }
    if (ParseJsonAttr(j_global_work.value(), false, "type", global_work_space.type,
                      global_work_space.type) != SUCCESS) {
      FE_LOGD("[SubGraphOpt][Compile][ParseGlobalWorkspaceStatus] Get 'type' from json data.");
      return FAILED;
    }
  } catch (const JsonExpcetion &e) {
    FE_LOGE("[SubGraphOpt][Compile][ParseGlobalWorkspaceStatus] get the %s failed, exception:%s",
            kKeyGlobalWorkspace, e.what());
    return FAILED;
  }
  return SUCCESS;
}

Status TbeJsonFileParseImpl::PackageTvmBinFile(vector<char> &buffer) const {
  std::string binfile_suffix = "";
  std::string binfile_name = "";
  (void)ParseJsonAttr(false, kKeyBinFileSuffix, binfile_suffix, binfile_suffix);
  (void)ParseJsonAttr(false, kKeyBinFileName, binfile_name, binfile_name);
  if (binfile_suffix.empty() || binfile_name.empty()) {
    FE_LOGE("Bin file name or binfile suffix is empty. Please check the JSON file.");
    return FAILED;
  }

  std::string bin_file_path = tvm_dir_path_;
  bin_file_path.append("/");
  bin_file_path.append(binfile_name);
  bin_file_path.append(binfile_suffix);
  if (binfile_suffix != ".so") {
    bool ret = ReadBytesFromBinaryFile(bin_file_path, buffer);
    if (ret != SUCCESS) {
      FE_LOGE("Failed to read bin file, file path is %s.", bin_file_path.c_str());
      return PARAM_INVALID;
    }
  } else {
    FE_LOGW("Auto-fused node %s, do not set bin file.", bin_file_path.c_str());
  }

  return SUCCESS;
}

Status TbeJsonFileParseImpl::ParseHeadFilePath(std::string &head_file_path) const {
  std::string bin_file_name;
  std::string header_file_suffix;
  (void)ParseJsonAttr(false, kKeyBinFileName, bin_file_name, bin_file_name);
  (void)ParseJsonAttr(false, kKeyHeadFileSuffix, header_file_suffix, header_file_suffix);
  if (bin_file_name.empty() || header_file_suffix.empty()) {
    return SUCCESS;
  }
  head_file_path = tvm_dir_path_;
  head_file_path.append("/");
  head_file_path.append(bin_file_name);
  head_file_path.append(header_file_suffix);
  return SUCCESS;
}

Status TbeJsonFileParseImpl::ParseTvmJsonList(std::vector<KeyJsonList> &json_list) const {
  try {
    json_handle_.at(kKeyJsonList).get_to(json_list);
  } catch (const JsonExpcetion &e) {
    FE_LOGE("[SubGraphOpt][ParseJson][ParseTvmJsonList] get jsonList failed, exception:%s", e.what());
    return FAILED;
  }
  return SUCCESS;
}

/*
*  @ingroup fe
*  @brief   package the json info together
*  @param   [in]  info
*  @param   [out] tvm_file_path_
*  @return  SUCCESS or FAILED
*/
Status TbeJsonFileParseImpl::Initialize(const std::string &json_file_path) {
  FE_LOGD("Start to parse op_json_file %s.", json_file_path.c_str());
  if (ReadJsonFile(json_file_path, json_handle_) != SUCCESS) {
    FE_LOGE("ReadJsonFile failed.");
    return FAILED;
  }
  SetTvmDirPath(json_file_path);
  return SUCCESS;
}

Status TbeJsonFileParseImpl::Initialize(const std::string &json_file_path,
                                        const TbeJsonPtr &json_ptr, const ge::OpKernelBinPtr &bin_ptr) {
  FE_LOGD("Begin initializing JSON file parsing implementation, JSON file path is [%s].", json_file_path.c_str());
  if (json_file_path.empty()) {
    FE_LOGE("JSON file path is empty.");
    return FAILED;
  }
  SetTvmDirPath(json_file_path);

  // try to parse from json and bin object first
  if (json_ptr != nullptr && bin_ptr != nullptr) {
    FE_LOGD("Parsed JSON info from JSON and binary object [%s].", bin_ptr->GetName().c_str());
    json_handle_ = *json_ptr;
    op_kernel_bin_ = bin_ptr;
  } else {
    FE_LOGD("Trying to parse JSON info from the JSON file [%s].", json_file_path.c_str());
    if (ReadJsonFile(json_file_path, json_handle_) != SUCCESS) {
      FE_LOGE("Failed to read JSON from the file [%s].", json_file_path.c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

void TbeJsonFileParseImpl::SetTvmDirPath(const std::string &json_file_path) {
  size_t find_pos = json_file_path.find_last_of("\\/");
  if (find_pos != std::string::npos) {
    tvm_dir_path_ = json_file_path.substr(0, find_pos);
  } else {
    tvm_dir_path_ = ".";
  }
}

Status TbeJsonFileParseImpl::ParseTvmTaskRatio(int32_t &cube_ratio, int32_t &vector_ratio, bool &dyn_ratio) const {
  std::string task_ratio;
  dyn_ratio = false;
  if (ParseJsonAttr(false, kKeyTaskRation, task_ratio, task_ratio) == FAILED) {
    return FAILED;
  }
  if (task_ratio.empty()) {
    return SUCCESS;
  }
  FE_LOGD("Parse TvmTaskRatio:[%s].", task_ratio.c_str());
  if (task_ratio == kDynRatioTilingKey) {
    dyn_ratio = true;
    cube_ratio = kTmpMixRation;
    vector_ratio = kTmpMixRation;
    return SUCCESS;
  }
  std::vector<std::string> cube_vector_ratio;
  cube_vector_ratio = StringUtils::Split(task_ratio, ':');
  if (cube_vector_ratio.size() != kMixChildJsonNum) {
    return FAILED;
  }
  try {
    cube_ratio = std::stoi(cube_vector_ratio[0]);
    vector_ratio = std::stoi(cube_vector_ratio[1]);
  } catch (...) {
    FE_LOGE("[SubGraphOpt][Compile][ParseTvmTaskRatio] Ratio value [%s] is invalid.", task_ratio.c_str());
    return FAILED;
  }

  if (cube_ratio == 0 && vector_ratio == 0) {
    return FAILED;
  }
  return SUCCESS;
}

Status TbeJsonFileParseImpl::ParseOpDfxOptions(std::vector<std::string> &opt_list, int64_t &buffer_size) const {
  std::string debug_options;
  if (ParseJsonAttr(false, kKeyDebugOptions, debug_options, debug_options) == FAILED) {
    return FAILED;
  }
  if (debug_options.empty()) {
    return SUCCESS;
  }
  if (ParseJsonAttr(true, kKeyDebugBufSize, buffer_size, buffer_size) == FAILED || buffer_size <= 0) {
    return FAILED;
  }
  FE_LOGD("Parse debug options[%s], buffer_size[%ld].", debug_options.c_str(), buffer_size);
  opt_list = StringUtils::Split(debug_options, ',');
  return SUCCESS;
}

/*
*  @ingroup fe
*  @brief   reading binary files
*  @param   [in]  file_name(or path), buffer, length
*  @param   [out] length
*  @return  SUCCESS or FAILED
*/
Status TbeJsonFileParseImpl::ReadBytesFromBinaryFile(const string& file_name, std::vector<char>& buffer) const {
  if ((file_name.empty())) {
    FE_LOGE("incorrect parameter: file path is null");
    return FAILED;
  }

  string real_path = RealPath(file_name);
  if (real_path.empty()) {
    FE_LOGE("File path '%s' not valid.", file_name.c_str());
    return FAILED;
  }

  std::ifstream if_stream(real_path.c_str(), std::ios::binary | std::ios::ate);
  if (!if_stream.is_open()) {
    FE_LOGE("Failed to read file %s.", file_name.c_str());
    return FAILED;
  }
  try {
    std::streamsize size = if_stream.tellg();
    if (size <= 0) {
      if_stream.close();
      FE_LOGE("File length is less than or equal to 0, which is not valid.");
      return FAILED;
    }

    if (size > INT_MAX) {
      if_stream.close();
      FE_LOGE("File size %ld exceeds the limit of %d.", size, INT_MAX);
      return FAILED;
    }
    if_stream.seekg(0, std::ios::beg);

    buffer.resize(size);
    if_stream.read(&buffer[0], size);
    FE_LOGD("Release lock file(%s).", real_path.c_str());
    if_stream.close();
    FE_LOGD("Read size: %ld bytes.", size);
  } catch (const ifstream::failure& e) {
    FE_LOGE("Failed to read file %s. Exception: %s", file_name.c_str(), e.what());
    if_stream.close();
    return FAILED;
  }
  return SUCCESS;
}

ge::OpKernelBinPtr TbeJsonFileParseImpl::GetOpKernelBinPtr() const {
  return op_kernel_bin_;
}

const std::string& TbeJsonFileParseImpl::GetTvmDirPath() const {
  return tvm_dir_path_;
}

Status TbeJsonFileParseImpl::ParseRunInfo(OpTilingInfo &run_info, bool &has_run_info) const {
  auto j_runinfo = json_handle_.find(kRunInfo);
  if (j_runinfo == json_handle_.end()) {
    has_run_info = false;
    return SUCCESS;
  }
  has_run_info = true;
  // mc2 static compile json has attr tilingData
  run_info.aicpu_block_dim = 0;
  try {
    if (ParseJsonAttr(j_runinfo.value(), true, "block_dim", run_info.block_dim, run_info.block_dim) != SUCCESS) {
      FE_LOGE("parse block_dim failed.");
      return FAILED;
    }
    if (ParseJsonAttr(j_runinfo.value(), false, "aicpu_block_dim", run_info.aicpu_block_dim, run_info.aicpu_block_dim) != SUCCESS) {
      FE_LOGE("parse aicpu_block_dim failed.");
      return FAILED;
    }
    std::vector<int64_t> tvm_workspace_types;
    if (ParseTvmWorkSpace(run_info.tvm_workspace_sizes, tvm_workspace_types) == FAILED) {
      REPORT_FE_ERROR("Failed to parse workspaces.");
      return FAILED;
    }
    if (ParseJsonAttr(j_runinfo.value(), true, "clear_atomic", run_info.clear_atomic, run_info.clear_atomic) != SUCCESS) {
      return FAILED;
    }
    if (ParseJsonAttr(j_runinfo.value(), true, "local_memory_size", run_info.local_memory_size, run_info.local_memory_size) != SUCCESS) {
      FE_LOGE("Failed to parse local_memory_size.");
      return FAILED;
    }
    if (ParseJsonAttr(j_runinfo.value(), true, "schedule_mode", run_info.schedule_mode, run_info.schedule_mode) != SUCCESS) {
      FE_LOGE("Failed to parse schedule_mode.");
      return FAILED;
    }
    if (ParseJsonAttr(j_runinfo.value(), true, "tiling_cond", run_info.tiling_cond, run_info.tiling_cond) != SUCCESS) {
      FE_LOGE("parse tiling_cond failed.");
      return FAILED;
    }
    if (ParseJsonAttr(j_runinfo.value(), true, "tiling_key", run_info.tiling_key, run_info.tiling_key) != SUCCESS) {
      FE_LOGE("parse tiling_key failed.");
      return FAILED;
    }
    if (ParseJsonAttr(j_runinfo.value(), true, "tiling_data", run_info.tiling_data_str, run_info.tiling_data_str) != SUCCESS) {
      FE_LOGE("parse tiling_data failed.");
      return FAILED;
    }
    if (run_info.tiling_data_str.empty()) {
      FE_LOGE("Tiling data string is empty.");
      return FAILED;
    }
  } catch (const JsonExpcetion &e) {
    FE_LOGE("Failed to get the workspace, exception: %s", e.what());
    return FAILED;
  }
  return SUCCESS;
}

Status TbeJsonFileParseImpl::ParseFatbin(const ge::OpKernelBinPtr &fatbin,
                                         FatbinKernelInfoMap &fatbin_kernel_info_map) const {
  const uint8_t* fatbin_ptr = fatbin->GetBinData();
  FE_LOGD("Fatbin data size is %zu.", fatbin->GetBinDataSize());
  // get fatbin header info
  uint64_t tiling_key_num = 0;
  if (memcpy_s(&tiling_key_num, sizeof(uint64_t), fatbin_ptr, sizeof(uint64_t)) != EOK) {
    FE_LOGE("Failed to get tiling key num.");
    return FAILED;
  }
  FE_LOGD("Tiling key num is %lu.", tiling_key_num);
  FatbinHeaderInfo fatbin_header_info(tiling_key_num);
  if (memcpy_s(fatbin_header_info.tilingKeyList.data(), sizeof(uint64_t) * tiling_key_num,
      fatbin_ptr + sizeof(uint64_t), sizeof(uint64_t) * tiling_key_num) != EOK) {
    FE_LOGE("Failed to get tiling key list.");
    return FAILED;
  }
  if (memcpy_s(fatbin_header_info.binOffsets.data(), sizeof(size_t) * tiling_key_num,
      fatbin_ptr + sizeof(uint64_t) + sizeof(uint64_t) * tiling_key_num,
      sizeof(size_t) * tiling_key_num) != EOK) {
    FE_LOGE("Failed to get bin offset list.");
    return FAILED;
  }
  if (fatbin_header_info.tilingKeyList.size() != tiling_key_num ||
      fatbin_header_info.binOffsets.size() != tiling_key_num) {
    FE_LOGE("Tiling key list size %zu or bin offset list size %zu is not equal to tiling key num.",
            fatbin_header_info.tilingKeyList.size(), fatbin_header_info.binOffsets.size(), tiling_key_num);
    return FAILED;
  }

  for (size_t i = 0; i < tiling_key_num; ++i) {
    size_t subkernel_bin_size = 0;
    if (i == (tiling_key_num - 1)) {
      subkernel_bin_size = fatbin->GetBinDataSize() - fatbin_header_info.binOffsets[i];
    } else {
      subkernel_bin_size = fatbin_header_info.binOffsets[i + 1] - fatbin_header_info.binOffsets[i];
    }
    std::vector<char> subkernel(fatbin_ptr + fatbin_header_info.binOffsets[i],
                                fatbin_ptr + fatbin_header_info.binOffsets[i] + subkernel_bin_size);
    SubkernelInfo subkernel_info;
    FE_MAKE_SHARED(subkernel_info.subkernel_ptr =
        std::make_shared<ge::OpKernelBin>("tmp", std::move(subkernel)), return FAILED);
    fatbin_kernel_info_map.emplace(fatbin_header_info.tilingKeyList[i], subkernel_info);
  }
  if (fatbin_kernel_info_map.empty()) {
    FE_LOGE("Failed to get subkernel info.");
    return FAILED;
  }
  return SUCCESS;
}

Status TbeJsonFileParseImpl::ParseFatbinJson(FatbinKernelInfoMap &fatbin_kernel_info_map) const {
  auto kernel_list = json_handle_.find(kKeyKernelList);
  if (kernel_list == json_handle_.end()) {
    FE_LOGE("Failed to get kernelList key from json.");
    return FAILED;
  }
  try {
    for (auto kernel_iter = kernel_list.value().begin(); kernel_iter != kernel_list.value().end(); ++kernel_iter) {
      uint64_t config_key = kernel_iter->at("configKey").get<uint64_t>();
      auto iter = fatbin_kernel_info_map.find(config_key);
      if (iter == fatbin_kernel_info_map.end()) {
        FE_LOGE("Failed to match config key %lu from fatbin_kernel_info_map", config_key);
        return FAILED;
      }
      iter->second.block_dim = kernel_iter->at("blockDim").get<int32_t>();
      iter->second.workspace_size = kernel_iter->at("workspaceSize").get<int64_t>();
      iter->second.kernel_name = kernel_iter->at("kernelName").get<std::string>();
    }
  } catch (const JsonExpcetion &e) {
    FE_LOGE("Failed to parse kernelList, exception: %s.", e.what());
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace fe
