/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "kernel_register_info_builder.h"
#include <unordered_map>
#include "common/checker.h"
#include "graph/debug/ge_attr_define.h"
#include "common/ge_common/ge_types.h"
#include "common/tbe_handle_store/kernel_store.h"
#include "framework/common/taskdown_common.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "graph/load/model_manager/model_manager.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
namespace {
constexpr const ge::char_t *kAttrMemsetKernelBinId = "_memset_kernel_bin_id";

Status GetBinaryMagic(const OpDescPtr &op_desc, bool is_atomic_node, int32_t &binary_magic) {
  static const std::unordered_map<std::string, uint32_t> binary_magics = {
      {"RT_DEV_BINARY_MAGIC_ELF", RT_DEV_BINARY_MAGIC_ELF},
      {"RT_DEV_BINARY_MAGIC_ELF_AIVEC", RT_DEV_BINARY_MAGIC_ELF_AIVEC},
      {"RT_DEV_BINARY_MAGIC_ELF_AICUBE", RT_DEV_BINARY_MAGIC_ELF_AICUBE}};
  std::string tvm_magic_attr;
  if (is_atomic_node) {
    tvm_magic_attr = kAtomicPrefix + TVM_ATTR_NAME_MAGIC;
  } else {
    tvm_magic_attr = TVM_ATTR_NAME_MAGIC;
  }
  std::string magic_value;
  (void)AttrUtils::GetStr(op_desc, tvm_magic_attr, magic_value);
  auto iter = binary_magics.find(magic_value);
  GE_ASSERT_TRUE(iter != binary_magics.end(),
      "[%s][%s] magic value: %s get from attr %s is invalid.",
      op_desc->GetNamePtr(), op_desc->GetTypePtr(), magic_value.c_str(), tvm_magic_attr.c_str());
  binary_magic = iter->second;
  return SUCCESS;
}

TBEKernelPtr GetTbeKernelBin(const OpDescPtr &op_desc, bool is_atomic_node) {
  std::string kernel_bin_attr = OP_EXTATTR_NAME_TBE_KERNEL;
  if (is_atomic_node) {
    kernel_bin_attr = kAtomicPrefix + kernel_bin_attr;
  }
  auto tbe_kernel_ptr = op_desc->TryGetExtAttr(kernel_bin_attr, TBEKernelPtr());
  // 当前ffts+场景可能会获取到空，不影响功能
  if (tbe_kernel_ptr == nullptr) {
    GELOGW("[%s][%s] Attr %s not exists.",
           op_desc->GetNamePtr(), op_desc->GetTypePtr(), kernel_bin_attr.c_str());
  }
  return tbe_kernel_ptr;
}

Status GetTbeKernelId(const OpDescPtr &op_desc, bool is_atomic_node, uint32_t model_id, std::string &kernel_bin_id) {
  std::string kernel_id_attr;
  if (is_atomic_node) {
    kernel_id_attr = kAttrMemsetKernelBinId;
  } else {
    kernel_id_attr = ATTR_NAME_KERNEL_BIN_ID;
  }
  (void)AttrUtils::GetStr(op_desc, kernel_id_attr, kernel_bin_id);
  if (kernel_bin_id.empty()) {
    (void)AttrUtils::GetStr(op_desc, ATTR_NAME_SESSION_GRAPH_ID, kernel_bin_id);
    kernel_bin_id += std::string("_" + std::to_string(model_id) + op_desc->GetName());
  }
  return SUCCESS;
}
}

Status KernelRegisterInfoBuilder::ConstructAicoreRegisterInfo(const OpDescPtr &op_desc,
    bool is_atomic_node,  uint32_t model_id, KernelRegisterInfo &register_info) {
  AicoreRegisterInfo aicore_register_info;
  int32_t magic = 0;
  GE_ASSERT_SUCCESS(GetBinaryMagic(op_desc, is_atomic_node, magic));
  aicore_register_info.magic = magic;
  aicore_register_info.kernel_bin = GetTbeKernelBin(op_desc, is_atomic_node);
  GE_ASSERT_SUCCESS(GetTbeKernelId(op_desc, is_atomic_node, model_id, aicore_register_info.kernel_bin_name));
  register_info = aicore_register_info;
  return SUCCESS;
}

Status KernelRegisterInfoBuilder::ConstructAicpuRegisterInfo(const std::string &op_type,
    const std::string &so_name, const std::string kernel_name, const std::string &op_kernel_lib,
    KernelRegisterInfo &register_info) {
  AicpuRegisterInfo aicpu_register_info;
  aicpu_register_info.op_type = op_type;
  aicpu_register_info.so_name = so_name;
  aicpu_register_info.kernel_name = kernel_name;
  aicpu_register_info.op_kernel_lib = op_kernel_lib;
  register_info = aicpu_register_info;
  return SUCCESS;
}

Status KernelRegisterInfoBuilder::ConstructCustAicpuRegisterInfo(const OpDescPtr &op_desc,
    KernelRegisterInfo &register_info) {
  CustAicpuRegisterInfo cust_aicpu_register_info;
  const auto cust_aicpu_bin_ptr = op_desc->TryGetExtAttr(OP_EXTATTR_CUSTAICPU_KERNEL, CustAICPUKernelPtr());
  GE_ASSERT_NOTNULL(cust_aicpu_bin_ptr, "[%s][%s] Get cust aicpu kernel bin from attr %s failed.",
      op_desc->GetNamePtr(), op_desc->GetTypePtr(), OP_EXTATTR_CUSTAICPU_KERNEL);
  cust_aicpu_register_info.cust_aicpu_kernel_bin = cust_aicpu_bin_ptr;
  register_info = cust_aicpu_register_info;
  return SUCCESS;
}

Status KernelRegisterInfoBuilder::ConstructTilingDeviceRegisterInfo(const std::string &so_path, uint32_t model_id,
    KernelRegisterInfo &register_info) {
  CustAicpuRegisterInfo tiling_device_register_info;
  auto unique_so_name = ModelManager::GetInstance().GetCustTilingDeviceUniqueSoName(model_id, so_path);
  const auto tiling_device_kernel = ModelManager::GetInstance().GetCustTilingDeviceSoBin(unique_so_name);
  GE_ASSERT_NOTNULL(tiling_device_kernel);
  tiling_device_register_info.cust_aicpu_kernel_bin = tiling_device_kernel;
  register_info = tiling_device_register_info;
  return SUCCESS;
}
}
