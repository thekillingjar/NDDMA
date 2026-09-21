/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cpu_optimizer.h"

#include <dirent.h>
#include <sys/types.h>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <utility>

#include "config/config_file.h"
#include "error_code/error_code.h"
#include "ge/ge_api_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/op_kernel_bin.h"
#include "graph/utils/attr_utils.h"
#include "common/util/trace_manager/trace_manager.h"
#include "util/log.h"
#include "util/util.h"
#include "cpu_engine_util.h"
#include "cpu_engine/aicpu_engine/engine/aicpu_engine.h"
#include "common/aicpu_ops_kernel_builder/kernel_builder.h"
#include "common/aicpu_ops_kernel_info_store/kernel_info.h"

namespace {
constexpr const char *kPlaceholderOpType = "PlaceHolder";
constexpr const char *kFrameworkOp = "FrameworkOp";
constexpr const char *kEndOpType = "End";
const std::string kConfigFile = "/config.ini";
const std::string kOpsCpuPathPrefix = "opp/built-in/";
const std::string kOpsCpuPathPrefixOld = "/ops/built-in/";
const std::string kCustAicpuPathPrefix = "vendors/config.ini";
const std::string kCustAicpuPathPrefixOld = "ops/vendors/";
constexpr const char *kCustKernelSoPathBasedOnEnv =
    "op_impl/custom/cpu/aicpu_kernel/custom_impl/";
constexpr const char *kLibCpuKernelsSoPathBasedOnEnv =
    "opp/op_impl/built-in/aicpu/aicpu_kernel/lib/";
constexpr const char *kCustKernelSoPathNew =
    "/op_impl/cpu/aicpu_kernel/impl/";
constexpr const char *kLibCpuKernelsSoPathBasedOnEnvNew =
    "opp/built-in/op_impl/aicpu/aicpu_kernel/lib/";
constexpr const char *kLibCpuKernelsSoName = "libcpu_kernels.so";
constexpr const char *kLibAicpuKernelsSoName = "libaicpu_kernels.so";
constexpr const char *kLibPtKernelsSoName = "libpt_kernels.so";
constexpr const char *kCustomKernelSoRelativePath =
    "/ops/op_impl/custom/cpu/aicpu_kernel/custom_impl/";
constexpr const char *kCpuKernelsSoRelativePath =
    "/ops/op_impl/built-in/aicpu/aicpu_kernel/lib/";
constexpr const char *kCpuKernelsSoRelativePathNew =
    "op_impl/aicpu/aicpu_kernel/lib/";
constexpr const char *kAscendAicpuPathEnvName = "ASCEND_AICPU_PATH";
// load libcpu_kernels.so in model
constexpr uint64_t kLoadTypeForCpuKernelsInModel = 1;
// load libcpu_kernels.so in installation package
constexpr uint64_t kLoadTypeForCpuKernelsInPkg = 0;
// regex pattern for libcpu_kernels.so
constexpr const char *kPatternForCpuKernelsSo = "libcpu_kernels_v.+\\.so";
// regex pattern for libpt_kernels.soF
constexpr const char *kPatternForPtKernelsSo = "libpt_kernels_v.+\\.so";
constexpr int DOT_THIRD_CHAR_NUM = 2;

// global variable: so name with version map, key: so name, value: so name with
// version
std::map<std::string, std::string> g_so_with_version_map;
// global variable: mutex to prevent g_so_with_version_map
std::mutex g_so_with_version_map_mutex;
// global variable: <graph_id, soName>
std::set<std::pair<uint32_t, std::string>> g_so_in_graph_set;
// global variable: mutex for g_soInGraph set
std::mutex g_so_in_graph_set_mutex;
// soc version
std::string g_soc_version = "Ascend310";

std::mutex g_cust_mutex;
}  // namespace

namespace aicpu {

using AttrValueMap = google::protobuf::Map<string, aicpuops::AttrValue>;

ge::Status CpuOptimizer::Initialize() {
  InitLoadCpuKernelsType();
  Optimizer::InitOpCheckMode();

  std::string soc_version;
  if (ge::GetContext().GetOption(ge::SOC_VERSION, soc_version) ==
      ge::GRAPH_SUCCESS) {
    AICPUE_LOGI("Get soc version [%s] success.", soc_version.c_str());
  } else {
    AICPUE_LOG_RUN_INFO("Get soc version abnormal. Please check!");
    return ge::SUCCESS;
  }
  CheckAndSetSocVersion(soc_version);
  return ge::SUCCESS;
}

void CpuOptimizer::CheckAndSetSocVersion(
    const std::string &soc_version_from_ge) const {
  if (soc_version_from_ge.find("Ascend310P") == 0) {
    g_soc_version = "Ascend310P";
  } else if (soc_version_from_ge.find("Ascend310") == 0) {
    g_soc_version = "Ascend310";
  } else if (soc_version_from_ge.find("Ascend910") == 0) {
    g_soc_version = "Ascend910";
  } else if (soc_version_from_ge.find("Ascend710") == 0) {
    g_soc_version = "Ascend710";
  } else {
    AICPUE_LOG_RUN_INFO("Check soc version [%s] abnormal, Please check!",
                        soc_version_from_ge.c_str());
  }
}

void CpuOptimizer::SetCustUserInfos(map<std::string, std::string> info)
{
  cust_user_infos_ = info;
}

void CpuOptimizer::GetCustUserInfos(map<std::string, std::string> &info) const
{
  const std::lock_guard<std::mutex> lock(g_cust_mutex);
  info = cust_user_infos_;
}

ge::Status CpuOptimizer::OptimizeOriginalGraph(ge::ComputeGraph &graph,
                                               const std::map<std::string, OpFullInfo> &all_op_info) const {
  for (const ge::NodePtr &curr_node : graph.GetDirectNode()) {
    AICPU_CHECK_NOTNULL(curr_node)
    ge::OpDescPtr curr_op_desc_ptr = curr_node->GetOpDesc();
    AICPU_CHECK_NOTNULL(curr_op_desc_ptr)
    std::string op_type;
    AICPU_CHECK_RES(GetOriginalType(curr_op_desc_ptr, op_type));

    std::string kernel_lib_name = GetKernelLibNameByOpType(op_type, all_op_info);
    if (kernel_lib_name == kHostCpuKernelInfoChoice) {
      AICPUE_LOGI("[%s] don't need to clear 64-byte alignment attribute", kHostCpuKernelInfoChoice.c_str());
      return ge::SUCCESS;
    }

    auto const iter = all_op_info.find(op_type);
    if (iter == all_op_info.end()) {
      AICPUE_LOGD("don't support op type[%s], op name[%s]", curr_op_desc_ptr->GetType().c_str(),
                  curr_op_desc_ptr->GetName().c_str());
      continue;
    }

    FACTORY_ENGINE::FactoryType engine_ptr = FACTORY_ENGINE::Produce(kAicpuEngine);
    AICPU_CHECK_NOTNULL(engine_ptr)
    AicpuOpsKernelInfoStorePtr aicpu_ops_kernel_info_store_ptr = engine_ptr->GetAicpuOpsKernelInfoStore();
    AICPU_CHECK_NOTNULL(aicpu_ops_kernel_info_store_ptr);
    string un_supported_reason;
    if (aicpu_ops_kernel_info_store_ptr->CheckSupported(curr_op_desc_ptr, un_supported_reason)) {
      if (curr_op_desc_ptr->HasAttr(kAttrName64BytesFlag)) {
        AICPUE_LOGD("op support, op type is %s, op name is %s, begin clean attribute %s",
                    curr_op_desc_ptr->GetType().c_str(), curr_op_desc_ptr->GetName().c_str(),
                    kAttrName64BytesFlag.c_str());
        if (curr_op_desc_ptr->DelAttr(kAttrName64BytesFlag) != ge::SUCCESS) {
          AICPUE_LOGW("DelAttr %s failed.", kAttrName64BytesFlag.c_str());
        }
      }
    }

    if (curr_op_desc_ptr->GetType() == "GridSampler3D") {
      ge::Status status = InsertTransposeNode(curr_op_desc_ptr, graph, curr_node);
      if (status != ge::SUCCESS) {
        AICPUE_LOGE("InsertTransposeNode failed.");
        return status;
      }
    }
  }
  return ge::SUCCESS;
}

ge::Status CpuOptimizer::OptimizeFusedGraph(
    ge::ComputeGraph &graph,
    const std::map<string, OpFullInfo> &all_op_info) const {
  ge::TraceManager::GetInstance().SetTraceOwner(kModuleName, kTraceCpuFusedOptimizer, graph.GetName());
  uint32_t graph_id = 0;
  bool exist_graph_id = true;
  if (!ge::AttrUtils::GetInt(graph, kAttrNameRootGraphId, graph_id)) {
    exist_graph_id = false;
    AICPUE_LOGW("Failed to get attr _root_graph_id from subgraph.");
  }
  AICPUE_LOGI("Success to get attr[%s] from subgraph[%u].",
              kAttrNameRootGraphId.c_str(), graph_id);

  for (ge::NodePtr &node : graph.GetDirectNode()) {
    AICPU_CHECK_NOTNULL(node)
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    AICPU_CHECK_NOTNULL(op_desc_ptr)
    std::string op_type = op_desc_ptr->GetType();
    AICPU_IF_BOOL_EXEC(
        ((op_type == kPlaceholderOpType) || (op_type == kEndOpType)),
        AICPUE_LOGD("Current op type is [%s]. Don't need to fuse.",
                    op_type.c_str());
        continue)
    // if op type is framework_op, get original op
    AICPU_IF_BOOL_EXEC(
        (op_type == kFrameworkOp),
        AICPU_CHECK_RES(GetFrameworkOpType(op_desc_ptr, op_type)))

    // now only fuse aicpu op
    std::string kernel_lib_name =
        GetKernelLibNameByOpType(op_type, all_op_info);
    if ((kernel_lib_name != kAicpuKernelInfoChoice) &&
        (kernel_lib_name != kCustAicpuKernelInfoChoice) &&
        (kernel_lib_name != kHostCpuKernelInfoChoice)) {
      continue;
    }
    auto iter = all_op_info.find(op_type);
    AICPU_IF_BOOL_EXEC(iter == all_op_info.end(),
                       AICPU_REPORT_INNER_ERR_MSG(
                           "Can't find op type[%s] in op info store, op[%s].",
                           op_type.c_str(), node->GetName().c_str());
                       return ErrorCode::CREATE_NODEDEF_FAILED)
    OpFullInfo op_full_info = iter->second;
    std::string kernel_so = op_full_info.kernelSo;
    std::string func_name = op_full_info.functionName;
    bool async_flag = op_full_info.flagAsync;
    int workspace_size = op_full_info.workspaceSize;
    FWKAdapter::FWKExtTopicType topic_type = op_full_info.topicType;
    std::string resource = op_full_info.resource;
    bool support_block_flag = op_full_info.flagSupportBlockDim;
    int block_dim_index = op_full_info.blockDimByIndex;

    bool user_defined = op_full_info.userDefined;
    if (kernel_lib_name != kHostCpuKernelInfoChoice) {
      (void)ge::AttrUtils::SetStr(op_desc_ptr, "kernelSo", kernel_so);
      (void)ge::AttrUtils::SetStr(op_desc_ptr, "funcName", func_name);
    }
    (void)ge::AttrUtils::SetBool(op_desc_ptr, kAsyncFlag, async_flag);
    // now this is just for cust op
    (void)ge::AttrUtils::SetInt(op_desc_ptr, "workspaceSize", workspace_size);
    (void)ge::AttrUtils::SetStr(op_desc_ptr, "opKernelLib", kernel_lib_name);
    (void)ge::AttrUtils::SetInt(op_desc_ptr, kTopicType, topic_type);
    (void)ge::AttrUtils::SetBool(op_desc_ptr, kCustAicpuFlag, false);
    (void)ge::AttrUtils::SetStr(op_desc_ptr, kResource, resource);
    (void)ge::AttrUtils::SetBool(op_desc_ptr, kSupportBlockDim, support_block_flag);
    (void)ge::AttrUtils::SetInt(op_desc_ptr, kBlockDimByIndex, block_dim_index);

    if ((op_check_mode_) && (!user_defined) &&
        (kernel_so == kLibAicpuKernelsSoName)) {
      (void)ge::AttrUtils::SetStr(op_desc_ptr, "needCheckCpu", op_type);
    }
    AICPU_CHECK_RES_WITH_LOG(
        CheckAndSetUnknowType(op_desc_ptr, all_op_info),
        "Call CheckAndSetUnknowType function failed. op[%s].",
        node->GetName().c_str())
    if (kernel_lib_name != kHostCpuKernelInfoChoice) {
      AICPU_CHECK_RES(SetCustKernelBinFile(op_desc_ptr, all_op_info, graph_id, exist_graph_id))
    }
    FftsPlusInfo ffts_info;
    if (GetAicpuFftsPlusInfo(op_desc_ptr, ffts_info) == ge::SUCCESS) {
      ge::Status state = BuildFftsPlusAicpuNodeDef(op_desc_ptr, ffts_info);
      if (state != ge::SUCCESS) {
        AICPU_REPORT_INNER_ERR_MSG("[%s]BuildFftsInfoAicpuNodeDef fail state[%u]", node->GetName().c_str(), state);
        return state;
      }
      continue;
    }
    aicpuops::NodeDef node_def;
    BuildAicpuNodeDef(op_desc_ptr, node_def);
    auto ret = InsertAicpuNodeDefAttrToOp(op_desc_ptr, node_def, kCustomizedOpDef);
    if (ret != ge::SUCCESS) {
      return ret;
    }
  }

  return ge::SUCCESS;
}

ge::Status CpuOptimizer::SetCustKernelBinFile(
    ge::OpDescPtr op_desc_ptr,
    const std::map<std::string, OpFullInfo> &all_op_info, uint32_t graph_id,
    bool exist_graph_id) const {
  const std::string op_type = op_desc_ptr->GetType();
  OpFullInfo op_full_info;
  auto iter = all_op_info.find(op_type);
  AICPU_IF_BOOL_EXEC(
      iter == all_op_info.end(),
      AICPU_REPORT_INNER_ERR_MSG("can not find op type[%s] for op[%s].",
                               op_type.c_str(), op_desc_ptr->GetName().c_str());
      return ErrorCode::NONE_KERNELINFOSTORE)
  op_full_info = iter->second;

  std::string cust_kernel_so_real_path;
  if (!CheckSoNeedLoadInModel(op_full_info, cust_kernel_so_real_path, op_type)) {
    return ge::SUCCESS;
  }
  AICPUE_LOGI("cust so real path is %s.", cust_kernel_so_real_path.c_str());
  AICPU_IF_BOOL_EXEC(
      cust_kernel_so_real_path.empty(),
      AICPU_REPORT_INNER_ERR_MSG("Get cust kernel so real path failed.");
      return ErrorCode::STR_IS_EMPTY);
  AICPU_IF_BOOL_EXEC(
      (!PackageBinFile(op_desc_ptr, cust_kernel_so_real_path, op_full_info,
                       graph_id, exist_graph_id)),
      AICPU_REPORT_INNER_ERR_MSG(
          "Call PackageBinFile function failed, op[%s].",
          op_desc_ptr->GetName().c_str());
      return ErrorCode::PACKAGE_BIN_FILE);
  return ge::SUCCESS;
}

bool CpuOptimizer::isVendorCustPathEmpty(const std::string &path) const {
  DIR *dir = opendir(path.c_str());
  if (!dir) {
    return true;
  }

  struct dirent *entry = nullptr;
  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_name[0] == '.' &&
       (entry->d_name[1] == '\0' ||
       (entry->d_name[1] == '.' && entry->d_name[DOT_THIRD_CHAR_NUM] == '\0'))) {
      continue;
    }
    (void)closedir(dir);
    return false;
  }

  (void)closedir(dir);
  return true;
}

const string CpuOptimizer::GetCustKernelSoPath(const std::string op_type) const {
  std::string cust_user_name;
  map<std::string, std::string> cust_user_info;
  GetCustUserInfos(cust_user_info);
  AICPUE_LOGI("cust user info size is %zu", cust_user_info.size());
  auto itor = cust_user_info.find(op_type);
  if (itor != cust_user_info.end()) {
    cust_user_name = itor->second;
  }
  AICPUE_LOGD("cust_user_name = %s.", cust_user_name.c_str());

  std::string cust_kernel_so_real_path;
  const char *path_env = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ASCEND_OPP_PATH, path_env);
  AICPUE_LOGI("cust user name is %s, path env is %s.", cust_user_name.c_str(), path_env);
  if (path_env != nullptr) {
    std::string path = path_env;
    if (path[path.size() - 1] != '/') {
      path.append("/");
    }
    cust_kernel_so_real_path = path + kCustKernelSoPathBasedOnEnv;
    if (cust_user_name != "") {
      cust_kernel_so_real_path = cust_user_name + kCustKernelSoPathNew;
      if (isVendorCustPathEmpty(cust_kernel_so_real_path)) {
        AICPUE_LOGI("vendor path is empty so we use built in cust path.");
        cust_kernel_so_real_path = path_env + kAicpuBuiltInCustKernelFile;
      }
    }
  } else {
    std::string real_file_path = GetSoPath(reinterpret_cast<void *>(&CpuOptimizer::Initialize));
    AICPU_IF_BOOL_EXEC(real_file_path.empty(),
                       AICPU_REPORT_INNER_ERR_MSG("Call GetSoPath function failed, so path is empty.");
                       return cust_kernel_so_real_path);
    std::string path = Stringcat(real_file_path, "../..", kCustAicpuPathPrefixOld);
    cust_kernel_so_real_path = Stringcat(real_file_path, "../..", kCustomKernelSoRelativePath);

    std::vector<string> custom_user;
    if (ReadConfigFile(path + kConfigFile, custom_user)) {
      if (custom_user.size() > 0) {
        cust_kernel_so_real_path = path + cust_user_name + kCustKernelSoPathNew;
      }
    }
  }
  
  AICPUE_LOGI("after all path check cust so real path is %s.", cust_kernel_so_real_path.c_str());
  return cust_kernel_so_real_path;
}

const std::string CpuOptimizer::GetCpuKernelSoPath() const {
  std::string cpu_kernel_so_real_path;
  const char* path_env = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ASCEND_AICPU_PATH, path_env);
  if (path_env != nullptr) {
    std::string path = std::string(path_env);
    std::string cpu_path;
    if (path[path.size() - 1] != '/') {
      path.append("/");
    }
    cpu_path = path + kOpsCpuPathPrefix;

    cpu_kernel_so_real_path = path + kLibCpuKernelsSoPathBasedOnEnvNew + g_soc_version;
    // 如果存在新的算子包路径，则不再兼容读取老路径。
    if (!IsPathExist(cpu_kernel_so_real_path)) {
      cpu_kernel_so_real_path = path + kLibCpuKernelsSoPathBasedOnEnv + g_soc_version;
    }
  } else {
    std::string real_file_path = GetSoPath(reinterpret_cast<void *>(&CpuOptimizer::Initialize));
    AICPU_IF_BOOL_EXEC(real_file_path.empty(),
                       AICPU_REPORT_INNER_ERR_MSG("Call GetSoPath function failed, so path is empty.");
                       return cpu_kernel_so_real_path);
    std::string path = Stringcat(real_file_path, "../..", kOpsCpuPathPrefixOld);
    cpu_kernel_so_real_path = path + kCpuKernelsSoRelativePathNew + g_soc_version + "/";
    if (!IsPathExist(cpu_kernel_so_real_path)) {
      cpu_kernel_so_real_path = Stringcat(real_file_path, "../..", kCpuKernelsSoRelativePath, g_soc_version, "/");
    }
  }

  AICPUE_LOGI("cpu_kernel_so_real_path = %s.", cpu_kernel_so_real_path.c_str());
  return cpu_kernel_so_real_path;
}

bool CpuOptimizer::PackageBinFile(ge::OpDescPtr op_desc_ptr,
                                  const std::string &bin_folder_path,
                                  const OpFullInfo &op_full_info,
                                  uint32_t graph_id,
                                  bool exist_graph_id) const {
  std::string bin_file_name;
  AICPUE_LOGI("Save so name [%s] and graph id [%u] success.", bin_file_name.c_str(), graph_id);
  if (GetBinFileName(op_full_info, bin_folder_path, bin_file_name) !=
      ge::SUCCESS) {
    AICPU_REPORT_INNER_ERR_MSG("Call GetBinFileName failed.");
    return false;
  }

  if (exist_graph_id) {
    std::lock_guard<std::mutex> so_in_graph_set_mutex(g_so_in_graph_set_mutex);
    std::pair<uint32_t, std::string> pair =
        std::make_pair(graph_id, bin_file_name);
    auto iter = g_so_in_graph_set.find(pair);
    if (iter != g_so_in_graph_set.end()) {
      AICPUE_LOGI("So [%s] has exist in current sub graph [%u].",
                  bin_file_name.c_str(), graph_id);
      // kernel_so need refresh to real name for libcpu_kernels_${version}.so
      (void)ge::AttrUtils::SetStr(op_desc_ptr, "kernelSo", bin_file_name);
      (void)ge::AttrUtils::SetBool(op_desc_ptr, kCustAicpuFlag, true);
      return true;
    }
    (void)g_so_in_graph_set.insert(pair);
    AICPUE_LOGI("Save so name [%s] and graph id [%u] success.", bin_file_name.c_str(), graph_id);
  }

  std::string bin_file_path = bin_folder_path;
  AICPU_IF_BOOL_EXEC(
      (bin_file_name.empty() || bin_file_path.empty()),
      AICPU_REPORT_INNER_ERR_MSG("bin file name[%s] or bin file path[%s] is "
                               "empty. please check json file.",
                               bin_file_name.c_str(), bin_file_path.c_str());
      return false);

  if (bin_file_path[bin_file_path.size() - 1] != '/') {
    bin_file_path.append("/");
  }
  bin_file_path.append(bin_file_name);

  AICPUE_LOGI("Bin file path is [%s]", bin_file_path.c_str());
  std::vector<char> buffer;
  bool ret = ReadBytesFromBinaryFile(bin_file_path, buffer);
  if (!ret) {
    AICPU_REPORT_INNER_ERR_MSG(
        "Call ReadBytesFromBinaryFile failed"
        " to read bin file[%s].",
        bin_file_path.c_str());
    return false;
  }

  std::string key = op_desc_ptr->GetName();

  ge::OpKernelBinPtr cust_aicpu_kernel_ptr =
      std::make_shared<ge::OpKernelBin>(key, move(buffer));
  AICPU_IF_BOOL_EXEC(
      cust_aicpu_kernel_ptr == nullptr,
      AICPU_REPORT_INNER_ERR_MSG("Create OpKernelBin object failed.");
      return false);
  op_desc_ptr->SetExtAttr(ge::OP_EXTATTR_CUSTAICPU_KERNEL,
                          cust_aicpu_kernel_ptr);
  // kernel_so need refresh to real name for libcpu_kernels_${version}.so
  (void)ge::AttrUtils::SetStr(op_desc_ptr, "kernelSo", bin_file_name);
  (void)ge::AttrUtils::SetBool(op_desc_ptr, kCustAicpuFlag, true);

  AICPUE_LOGI("Set extend attr cust aicpu kernel success!");
  return true;
}

void CpuOptimizer::InitLoadCpuKernelsType() {
  // get load_type_for_cpu_kernels from config file.
  std::string load_type_for_cpu_kernels;
  if (ConfigFile::GetInstance().GetValue(kLoadCpuKernelsInModel,
                                         load_type_for_cpu_kernels)) {
    uint64_t result = kDefaultLoadTypeForCpuKernels;
    if (StringToNum(load_type_for_cpu_kernels, result).state != ge::SUCCESS) {
      AICPUE_LOGW(
          "Tran LoadCpuKernelsInModel [%s] to integer failed. default value is "
          "0.",
          load_type_for_cpu_kernels.c_str());
      return;
    }
    // if load_type_for_cpu_kernels from config file is not 0 or 1, print
    // warning log.
    if ((result != kLoadTypeForCpuKernelsInModel) &&
        (result != kLoadTypeForCpuKernelsInPkg)) {
      AICPUE_LOGW("load_type_for_cpu_kernels is [%s], default value is [%lu].",
                  load_type_for_cpu_kernels.c_str(),
                  kDefaultLoadTypeForCpuKernels);
      return;
    }
    load_type_for_cpu_kernels_ = result;
    return;
  }
  AICPUE_LOGW(
      "Get OpFusionMinNum from config file failed. default value is 2.");
}

bool CpuOptimizer::CheckSoNeedLoadInModel(const OpFullInfo &op_full_info,
                                          std::string &file_path, const std::string op_type) const {
  // only user defined kernels and libcpu_kernels.so and libpt_kernels.so
  // perhaps load in model
  if ((!op_full_info.userDefined) &&
      (op_full_info.kernelSo != kLibCpuKernelsSoName) &&
      (op_full_info.kernelSo != kLibPtKernelsSoName)) {
    return false;
  }
  // user defined kernels need load in model
  if (op_full_info.userDefined) {
    file_path = GetCustKernelSoPath(op_type);
    return true;
  }

  // libcpu_kernels.so and libpt_kernels.so load in model only when
  // load_type_for_cpu_kernels_ == 1
  if (load_type_for_cpu_kernels_ != kLoadTypeForCpuKernelsInModel) {
    return false;
  }
  // libpt_kernels.so must load in model
  file_path = GetCpuKernelSoPath();
  return true;
}

ge::Status CpuOptimizer::GetBinFileName(const OpFullInfo &op_full_info,
                                        const std::string &bin_folder_path,
                                        std::string &bin_file_name) const {
  // get so name for user defined kernels from json config file
  std::string kernel_so = op_full_info.kernelSo;
  if (op_full_info.userDefined) {
    bin_file_name = kernel_so;
    return ge::SUCCESS;
  }

  std::lock_guard<std::mutex> so_with_version_map_lock(
      g_so_with_version_map_mutex);
  auto iter = g_so_with_version_map.find(kernel_so);
  if (iter != g_so_with_version_map.end()) {
    bin_file_name = iter->second;
    AICPUE_LOGI("Kernel so name [%s] has been searched, real bin file name is [%s]",
                kernel_so.c_str(), bin_file_name.c_str());
    return ge::SUCCESS;
  }

  // realpath
  std::string folder_path = RealPath(bin_folder_path);
  AICPU_IF_BOOL_EXEC(
      folder_path.empty(),
      AICPU_REPORT_INNER_ERR_MSG("Get realpath[%s] failed, [%s]",
                               bin_folder_path.data(), strerror(errno));
      return GET_BIN_FILE_NAME_FAILED)

  // open dir
  DIR *dir;
  dir = opendir(folder_path.c_str());
  AICPU_IF_BOOL_EXEC(
      dir == nullptr,
      AICPU_REPORT_INNER_ERR_MSG("Open dir[%s] failed.", folder_path.c_str());
      return GET_BIN_FILE_NAME_FAILED)
  uint32_t so_count = 0;
  dirent *direntp = nullptr;
  // max search 1024 files to prevent dead circulation
  for (int index = 0; index < 1024; index++) {
    direntp = readdir(dir);
    if (direntp == nullptr) {
      break;
    }
    // DT_REG: a regular file
    if (direntp->d_type != DT_REG) {
      continue;
    }
    // check libcpu_kernels.so
    std::string file_name(direntp->d_name);
    AICPUE_LOGI(
        "file_name is [%s], file_name length is [%zu], d_name is [%s], d_reclen is "
        "[%u].",
        file_name.c_str(), file_name.length(), direntp->d_name,
        direntp->d_reclen);
    std::string pattern = (kernel_so == kLibPtKernelsSoName)
                              ? kPatternForPtKernelsSo
                              : kPatternForCpuKernelsSo;
    if (!ValidateStr(file_name, pattern)) {
      continue;
    }

    bin_file_name = file_name;
    so_count++;
    // check so count: only one is allowed.
    AICPU_IF_BOOL_EXEC(
        so_count > 1, (void)closedir(dir); AICPU_REPORT_INNER_ERR_MSG(
            "multi cpu kernels so exist in dir path[%s].", folder_path.c_str());
        return MULTI_CPU_KERNELS_SO_EXIST)
  }

  // check libcpu_kernels.so exist
  AICPU_IF_BOOL_EXEC(
      so_count == 0, (void)closedir(dir); AICPU_REPORT_INNER_ERR_MSG(
          "no cpu kernels so exist in dir path[%s].", folder_path.c_str());
      return NO_CPU_KERNELS_SO_EXIST)

  // record bin_file_name
  g_so_with_version_map[kernel_so] = bin_file_name;
  AICPUE_LOGI("Kernel so is [%s], bin file name is [%s].", kernel_so.c_str(),
              bin_file_name.c_str());
  ;

  // close dir: 0 success
  if (closedir(dir) != 0) {
    AICPUE_LOGW("close dir failed, errno: [%s], dir path: [%s].", strerror(errno),
                folder_path.c_str());
  }
  return ge::SUCCESS;
}
}  // namespace aicpu
