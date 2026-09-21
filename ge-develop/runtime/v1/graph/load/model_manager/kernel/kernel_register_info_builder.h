/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef EXECUTOR_GRAPH_LOAD_MODEL_KERNEL_KERNEL_REGISTER_INFO_BUILDER_H
#define EXECUTOR_GRAPH_LOAD_MODEL_KERNEL_KERNEL_REGISTER_INFO_BUILDER_H

#include "common/kernel_handles_manager/kernel_handles_manager.h"
#include "graph/op_desc.h"

namespace ge {
class KernelRegisterInfoBuilder {
 public:
  static Status ConstructAicoreRegisterInfo(const OpDescPtr &op_desc,
      bool is_atomic_node, uint32_t model_id, KernelRegisterInfo &register_info);
  static Status ConstructAicpuRegisterInfo(const std::string &op_type,
      const std::string &so_name, const std::string kernel_name, const std::string &op_kernel_lib,
      KernelRegisterInfo &register_info);
  static Status ConstructCustAicpuRegisterInfo(const OpDescPtr &op_desc, KernelRegisterInfo &register_info);
  static Status ConstructTilingDeviceRegisterInfo(const std::string &so_path, uint32_t model_id,
      KernelRegisterInfo &register_info);
};
}

#endif // EXECUTOR_GRAPH_LOAD_MODEL_KERNEL_KERNEL_REGISTER_INFO_BUILDER_H