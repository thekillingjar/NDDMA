/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include<mutex>
#include "aicpu_ops_kernel_info_store.h"

#include "base/err_msg.h"
#include "config/config_file.h"
#include "error_code/error_code.h"
#include "kernel_info.h"
#include "util/constant.h"
#include "util/log.h"
#include "util/util.h"
#include "graph/utils/type_utils.h"

using namespace std;
using namespace ge;
namespace {
std::mutex g_cust_mutex;
constexpr uint32_t kStridedSliceGradInputIndex1 = 1U;
constexpr uint32_t kStridedSliceGradInputIndex2 = 2U;
constexpr uint32_t kStridedSliceGradInputIndex3 = 3U;
}

namespace aicpu {

Status AicpuOpsKernelInfoStore::Initialize(const map<string, string> &options) {
  // check options
  auto iter = options.find(ge::SOC_VERSION);
  AICPU_IF_BOOL_EXEC(iter == options.end(),
      AICPU_REPORT_INNER_ERR_MSG(
          "cannot find [%s] in param of aicpu op store initialize function.",
          ge::SOC_VERSION.c_str());
      return INPUT_PARAM_VALID);

  // lhisi not load TFKernel
  set<string> blacklist;
  if ((iter->second.find("Hi") != string::npos) ||
      (iter->second.find("SD") != string::npos)) {
    blacklist.insert("TFKernel");
  }

  AICPU_CHECK_RES(Finalize());

  string kernel_libs;
  string ops_kernel_config = Stringcat(engine_name_, "OpsKernel");
  if (!ConfigFile::GetInstance().GetValue(ops_kernel_config, kernel_libs)) {
    AICPU_REPORT_INNER_ERR_MSG("[%s] not exist.", ops_kernel_config.c_str());
    return LOAD_PRIORITY_ITEM_FAILED;
  }

  // Parse string kernel_libs contains separator
  ConfigFile::GetInstance().SplitValue(kernel_libs, kernel_lib_names_, blacklist);
  if (kernel_lib_names_.empty()) {
    AICPUE_LOGW("kernelLibNames is empty.");
    return SUCCESS;
  }

  AICPUE_LOGI("First load [%s] Info Store", kernel_lib_names_[0].c_str());
  // Load all kernel libraries
  AICPU_CHECK_RES(LoadKernelLibs(options));
  // Get all op infos
  FillKernelInfos();
  AICPUE_LOGI("Aicpu kernel info store initialize success.");
  return SUCCESS;
}

Status AicpuOpsKernelInfoStore::Finalize() {
  kernel_libs_.clear();
  kernel_lib_names_.clear();
  op_infos_.clear();
  op_full_infos_.clear();
  cust_users_.clear();
  return SUCCESS;
}

Status AicpuOpsKernelInfoStore::LoadKernelLibs(
    const map<string, string> &options) {
  for (const string &name : kernel_lib_names_) {
    FACTORY_KERNELINFO::FactoryType kernel_info_ptr =
        FACTORY_KERNELINFO::Produce(name);
    if (kernel_info_ptr == nullptr) {
      AICPU_REPORT_INNER_ERR_MSG("create kernel in for store[%s] failed",
          name.c_str());
      return KERNELINFOSTORE_INSTANCE_FAILED;
    }
    Status flag = kernel_info_ptr->Initialize(options);
    if (flag != SUCCESS) {
      AICPU_REPORT_INNER_ERR_MSG("kernel in for store[%s] initialize failed",
          name.c_str());
      return KERNELINFOSTORE_INITIALIZE_FAILED;
    }
    kernel_libs_[name] = kernel_info_ptr;
  }
  return SUCCESS;
}

void AicpuOpsKernelInfoStore::FillKernelInfos() {
  const std::lock_guard<std::mutex> lock(g_cust_mutex);
  for (auto lib_iter = kernel_lib_names_.crbegin();
       lib_iter != kernel_lib_names_.crend(); ++lib_iter) {
    string kernel_name = *lib_iter;
    const KernelInfoPtr &kernel_lib_ptr = kernel_libs_[kernel_name];
    // how can be null...
    AICPU_IF_BOOL_EXEC(
        kernel_lib_ptr == nullptr,
        AICPU_REPORT_INNER_ERR_MSG("kernel lib is nullptr, kernel name[%s].",
            kernel_name.c_str());
        return);
    map<string, OpFullInfo> op_full_infos;
    kernel_lib_ptr->GetOpInfos(op_full_infos);
    kernel_lib_ptr->GetCustUserInfo(cust_users_);
    for (auto opIter = op_full_infos.cbegin(); opIter != op_full_infos.cend();
         ++opIter) {
      const string &op_type = opIter->first;
      const OpFullInfo &op_full_info = opIter->second;
      OpInfo op_info;
      op_info.engine = engine_name_;
      op_info.computeCost = op_full_info.computeCost;
      op_info.flagAsync = op_full_info.flagAsync;
      op_info.flagPartial = op_full_info.flagPartial;
      op_info.opKernelLib = op_full_info.opKernelLib;
      op_info.isAtomic = false;
      op_infos_[op_type] = op_info;
      op_full_infos_[op_type] = op_full_info;
    }
  }
}

void AicpuOpsKernelInfoStore::GetAllOpsKernelInfo(
    map<string, OpInfo> &infos) const {
  infos = op_infos_;
}

void AicpuOpsKernelInfoStore::GetCustUserNameInfo(map<string, string> &infos) const {
  infos = cust_users_;
}

bool AicpuOpsKernelInfoStore::CheckSupported(const OpDescPtr &op_desc_ptr,
                                             string &unsupported_reason) const {
  AICPU_CHECK_NOTNULL_ERRCODE(op_desc_ptr, false);

  string op_type = op_desc_ptr->GetType();
  if (op_type.empty()) {
    AICPU_REPORT_INNER_ERR_MSG("op type is empty, op[%s]",
        op_desc_ptr->GetName().c_str());
    return false;
  }

  // check whether the op is in aicpu ops kernel info store
  auto iter = op_full_infos_.find(op_type);
  if (iter == op_full_infos_.end()) {
    AICPUE_LOGI("Internal kernel info store not include this op[%s].",
                op_type.c_str());
    unsupported_reason = "Aicpu kernel info store not include this op ";
    unsupported_reason.append(op_type);
    return false;
  }

  auto op_full_info = iter->second;
  const map<string, string> data_types = op_full_info.inOutDataType;
  const map<string, string> in_out_real_name = op_full_info.inOutRealName;
  const string transdata_op_type = "TransData";
  const string stridedslicegrad_op_type = "StridedSliceGrad";
  if (op_type == transdata_op_type) {
    AICPU_IF_BOOL_EXEC(!(CheckTransDataSupported(op_desc_ptr, data_types, unsupported_reason)),
                       AICPUE_LOGW("Check not supported, op[%s].", op_type.c_str());
                       return false)
  } else if ((op_type == stridedslicegrad_op_type) && (engine_name_ == kTfEngine)) {
    AICPU_IF_BOOL_EXEC(!(CheckStridedSliceGradSupported(op_desc_ptr, unsupported_reason)),
                       AICPUE_LOGW("Check not supported, op[%s].", op_type.c_str());
                       return false)
  }
  if (op_type == kFrameworkOp) {
    (void)AttrUtils::SetBool(op_desc_ptr, kAttrNameTfDebug, true);
  }
  if (!op_desc_ptr->GetAllInputsDesc().empty()) {
    bool ret = CheckInputSupported(op_desc_ptr, data_types, in_out_real_name,
                                   unsupported_reason);
    AICPU_IF_BOOL_EXEC(
        !(ret), AICPUE_LOGI("Check input not supported, op[%s].", op_type.c_str());
        return false)
  }

  if (!op_desc_ptr->GetAllOutputsDesc().empty()) {
    bool ret = CheckOutputSupported(op_desc_ptr, data_types, in_out_real_name,
                                    unsupported_reason);
    AICPU_IF_BOOL_EXEC(
        !(ret), AICPUE_LOGI("Check output not supported, op[%s].", op_type.c_str());
        return false)
  }
  return true;
}

bool AicpuOpsKernelInfoStore::CheckStridedSliceGradSupported(const OpDescPtr &op_desc_ptr,
                                                             string &unsupported_reason) const {
  AICPU_CHECK_NOTNULL_ERRCODE(op_desc_ptr, false);

  auto begin_data_type = op_desc_ptr->GetInputDescPtr(kStridedSliceGradInputIndex1)->GetDataType();
  auto end_data_type = op_desc_ptr->GetInputDescPtr(kStridedSliceGradInputIndex2)->GetDataType();
  auto strides_data_type = op_desc_ptr->GetInputDescPtr(kStridedSliceGradInputIndex3)->GetDataType();
  if ((begin_data_type == end_data_type) && (begin_data_type == strides_data_type) &&
      (end_data_type == strides_data_type)) {
    return true;
  }

  unsupported_reason = Stringcat("StridedSliceGrad op, data type is not the same. "
      "begin ", begin_data_type, " end ", end_data_type, " strides ", strides_data_type);
  AICPUE_LOGW("The %s.", unsupported_reason.c_str());
  return false;
}

bool AicpuOpsKernelInfoStore::CheckTransDataSupported(const OpDescPtr &op_desc_ptr,
                                                      const map<string, string> data_types,
                                                      string &unsupported_reason) const {
  AICPU_CHECK_NOTNULL_ERRCODE(op_desc_ptr, false);

  string op_type = op_desc_ptr->GetType();
  if (op_type.empty()) {
    AICPU_REPORT_INNER_ERR_MSG("op type is empty, op[%s]",
        op_desc_ptr->GetName().c_str());
    return false;
  }

  int groups  = 1;
  if (!AttrUtils::GetInt(op_desc_ptr, "groups", groups)) {
    // groups是可选属性，没有默认为1
    AICPUE_LOGW("Transdata Op does not have attr groups, using default value 1");
  }

  auto input_desc = op_desc_ptr->GetInputDesc(0);
  auto input_format = static_cast<Format>(GetPrimaryFormat(static_cast<int32_t>(input_desc.GetFormat())));
  auto output_desc = op_desc_ptr->GetOutputDesc(0);
  auto output_format = static_cast<Format>(GetPrimaryFormat(static_cast<int32_t>(output_desc.GetFormat())));

  // If groups equals 1 with aicpu, there is a risk of performance deterioration, in addition to FZC04 to HWCN.
  const bool format_is_normal =
      ((groups <= 1) && ((input_format != FORMAT_FRACTAL_Z_C04) && (output_format != FORMAT_HWCN)));
  if (format_is_normal) {
    unsupported_reason = Stringcat("Transdata op, groups should be greater than 1, but now is ", groups);
    AICPUE_LOGW("The %s.", unsupported_reason.c_str());
    return false;
  }

  bool supported = false;
  // check format
  const bool output_format_fz =
      ((input_format == FORMAT_NCHW) || (input_format == FORMAT_HWCN)) && output_format == FORMAT_FRACTAL_Z;
  const bool input_format_fz =
      ((output_format == FORMAT_NCHW) || (output_format == FORMAT_HWCN)) && input_format == FORMAT_FRACTAL_Z;
  const bool output_format_fz3d =
      ((input_format == FORMAT_NCDHW) || (input_format == FORMAT_DHWCN) || (input_format == FORMAT_NDHWC)) &&
      output_format == FORMAT_FRACTAL_Z_3D;
  const bool input_format_fz3d =
      ((output_format == FORMAT_NCDHW) || (output_format == FORMAT_DHWCN) || (output_format == FORMAT_NDHWC)) &&
      input_format == FORMAT_FRACTAL_Z_3D;
  const bool fzc04_to_hwcn = ((input_format == FORMAT_FRACTAL_Z_C04) && (output_format == FORMAT_HWCN));
  const bool nd_to_nz_c0_16 = ((input_format == FORMAT_ND) && (output_format == FORMAT_FRACTAL_NZ_C0_16) && 
      (engine_name_ == kHostCpuEngine));
  const bool nd_to_nz_c0_32 = ((input_format == FORMAT_ND) && (output_format == FORMAT_FRACTAL_NZ_C0_32) && 
      (engine_name_ == kHostCpuEngine));
  if (output_format_fz) {
    supported = true;
  }

  if (input_format_fz) {
    supported = true;
  }

  if (output_format_fz3d) {
    supported = true;
  }

  if (input_format_fz3d) {
    supported = true;
  }

  if (fzc04_to_hwcn) {
    supported = true;
  }
  
  if (nd_to_nz_c0_16 || nd_to_nz_c0_32) {
    supported = true;
  }

  auto input_format_str = TypeUtils::FormatToSerialString(input_format);
  auto output_format_str = TypeUtils::FormatToSerialString(output_format);

  if (!supported) {
    unsupported_reason = Stringcat("Transdata op, input format is ", input_format_str, " , output format is ",
                                   output_format_str, " is unsupported by current kernel info store.");
    AICPUE_LOGW("The %s.", unsupported_reason.c_str());
  }

  if (supported) {
    // check input data type
    auto input_data_type = op_desc_ptr->GetInputDescPtr(0)->GetDataType();
    set<DataType> dst_data_type;
    GetDataType(data_types, "input0", dst_data_type);
    if (dst_data_type.find(input_data_type) == dst_data_type.end()) {
      string type_str;
      (void)ConvertDataType2String(type_str, input_data_type);
      string err_msg = Stringcat("data_type ", type_str,
                                 " of input src is unsupported by current kernel info store. op type[Transdata]");
      unsupported_reason = err_msg;
      AICPUE_LOGW("The %s.", err_msg.c_str());
      supported = false;
    }
  }

  if (supported) {
    auto output_data_type = op_desc_ptr->GetOutputDescPtr(0)->GetDataType();
    set<DataType> dst_data_type;
    GetDataType(data_types, "output0", dst_data_type);
    if (dst_data_type.find(output_data_type) == dst_data_type.end()) {
      string type_str;
      (void)ConvertDataType2String(type_str, output_data_type);
      string err_msg = Stringcat("data_type ", type_str,
                                 " of output dst is unsupported by current kernel info store. op type[Transdata]");
      unsupported_reason = err_msg;
      AICPUE_LOGW("The %s.", err_msg.c_str());
      supported = false;
    }
  }
  return supported;
}

bool AicpuOpsKernelInfoStore::CheckInputSupported(
    const OpDescPtr &op_desc_ptr, const map<string, string> data_types,
    const map<string, string> in_out_real_name, string &unsupported_reason) const {
  uint32_t input_index = 0;
  for (const ge::GeTensorDescPtr& input_desc_ptr : op_desc_ptr->GetAllInputsDescPtr()) {
    const string input_name = op_desc_ptr->GetInputNameByIndex(input_index);
    for (auto it = in_out_real_name.begin(); it != in_out_real_name.end(); it++) {
      const string input_real_name = it->first;
      if (input_name.compare(input_real_name) == 0) {
        const string data_type_name = it->second;
        set<DataType> dst_data_type;
        GetDataType(data_types, data_type_name, dst_data_type);
        const DataType type = input_desc_ptr->GetDataType();
        if (dst_data_type.find(type) == dst_data_type.end()) {
          string type_str;
          (void)ConvertDataType2String(type_str, type);
          string err_msg =
              Stringcat("data_type ", type_str, " of input[", std::to_string(input_index), ", ", input_real_name,
                        "] is unsupported, op type[", op_desc_ptr->GetType(), "].");
          unsupported_reason = err_msg;
          AICPUE_LOGI("The %s.", err_msg.c_str());
          return false;
        }
      }
    }
    input_index++;
  }

  return true;
}

bool AicpuOpsKernelInfoStore::CheckOutputSupported(
    const OpDescPtr &op_desc_ptr, const map<string, string> data_types,
    const map<string, string> in_out_real_name, string &unsupported_reason) const {
  uint32_t output_index = 0;
  for (const ge::GeTensorDescPtr& output_desc_ptr : op_desc_ptr->GetAllOutputsDescPtr()) {
    const string output_name = op_desc_ptr->GetOutputNameByIndex(output_index);
    for (auto it = in_out_real_name.begin(); it != in_out_real_name.end(); it++) {
      const string output_real_name = it->first;
      if (output_name.compare(output_real_name) == 0) {
        const string data_type_name = it->second;
        set<DataType> dst_data_type;
        GetDataType(data_types, data_type_name, dst_data_type);
        const DataType type = output_desc_ptr->GetDataType();
        if (dst_data_type.find(type) == dst_data_type.end()) {
          string type_str;
          (void)ConvertDataType2String(type_str, type);
          string err_msg =
              Stringcat("dataType ", type_str, " of output[", std::to_string(output_index), ", ", output_real_name,
                        " is unsupported, op type[", op_desc_ptr->GetType(), "].");
          unsupported_reason = err_msg;
          AICPUE_LOGI("The %s.", err_msg.c_str());
          return false;
        }
      }
    }
    output_index++;
  }
  return true;
}

void AicpuOpsKernelInfoStore::opsFlagCheck(const Node &node, string &ops_flag) {
  OpDescPtr op_desc_ptr = node.GetOpDesc();
  AICPU_IF_BOOL_EXEC(op_desc_ptr == nullptr,
      AICPU_REPORT_INNER_ERR_MSG(
            "op desc is nullptr, op[%s].", node.GetName().c_str());
      return);
  string op_type = op_desc_ptr->GetType();
  auto iter = op_full_infos_.find(op_type);
  if (iter != op_full_infos_.end()) {
    ops_flag = (iter->second).opsFlag;
  } else {
    AICPUE_LOGW("Find [ops_flag] failed in this op type [%s]", op_type.c_str());
  }
}

void AicpuOpsKernelInfoStore::GetAllOpsFullKernelInfo(
    map<string, OpFullInfo> &infos) const {
  infos = op_full_infos_;
}

/**
 * For ops in AIcore, Call CompileOp before Generate task in AICPU
 * @param node_vec Node information
 * @return status whether operation successful
 */
ge::Status AicpuOpsKernelInfoStore::CompileOp(vector<ge::NodePtr> &node_vec) {
  if (node_vec.empty()) {
      AICPUE_LOGI("AicpuOpsKernelInfoStore's node_vec is empty in CompileOp.");
      return ge::SUCCESS;
  }

  AICPUE_LOGI("AicpuOpsKernelInfoStore's start CompileOp.");
  map<string, OpFullInfo> all_op_info;
  GetAllOpsFullKernelInfo(all_op_info);
  for (ge::NodePtr &node : node_vec) {
    AICPU_CHECK_NOTNULL(node)
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    AICPU_CHECK_NOTNULL(op_desc_ptr)
    std::string op_type = op_desc_ptr->GetType();
    // check function op and framework op
    if ((op_type == kFunctionOp) || (op_type == kFrameworkOp)) {
      std::string err_msg = Stringcat("Can not create node def for function op and framework op[",
          node->GetName(), "] in CompileOp, op type[", op_type, "].");
      AICPU_REPORT_INNER_ERR_MSG("%s.", err_msg.c_str());
      return ErrorCode::NODE_DEF_NOT_EXIST;
    }

    std::string kernel_name = GetKernelLibNameByOpType(op_type, all_op_info);
    auto iter = kernel_libs_.find(kernel_name);
    if (iter == kernel_libs_.end()) {
        AICPU_REPORT_INNER_ERR_MSG("kernel lib[%s] is not exist.", kernel_name.c_str());
        return KERNEL_TYPE_INVALID;
    }
    const KernelInfoPtr &kernel_lib_ptr = iter->second;
    AICPU_CHECK_RES(kernel_lib_ptr->CompileOp(node));
  }

  AICPUE_LOGI("AicpuOpsKernelInfoStore's last Op run CompileOp Success.");
  return ge::SUCCESS;
}
}  // namespace aicpu
