/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_MODEL_UTILS_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_MODEL_UTILS_H_

#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/op_desc.h"
#include "graph/utils/tensor_adapter.h"
#include "common/model/ge_root_model.h"
#include "framework/common/types.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace ge {
enum IowMemoryType : uint64_t {
  kFmMemType = 0x1000000000UL,
  kFixMemType,
  kWeightMemType,
  kVarMemType,
  kVarAutoMemType,
  kConstantMemType,
  kAicpuMemMallMemType,
  kAbsoluteMemType
};

class ModelUtils {
 public:
  struct NodeMemInfo {
    NodeMemInfo(const uint64_t mem_type, const ConstOpDescPtr &op_desc,
                const size_t index, const std::string &io_type,
                const int64_t size, const int64_t logical_offset)
        : mem_type_(mem_type),
          op_desc_(op_desc),
          index_(index),
          io_type_(io_type),
          size_(size),
          logical_offset_(logical_offset) {}
    std::string ToString() const {
      std::stringstream ss;
      ss << "type[";
      switch (mem_type_) {
        case RT_MEMORY_HOST:
          ss << "H] ";
          break;
        case RT_MEMORY_HOST_SVM:
          ss << "S] ";
          break;
        case RT_MEMORY_P2P_DDR:
          ss << "P] ";
          break;
        case RT_MEMORY_L1:
        case kRtMemoryUB:
        case RT_MEMORY_TS:
        case RT_MEMORY_HBM:
        default:
          ss << "F] ";
          break;
      }
      ss << "name[" << op_desc_->GetName() << "] ";
      ss << io_type_ << "[" << index_ << "] ";
      ss << "offset[" << logical_offset_ << "] ";
      ss << "size[" << size_ << "]";
      return ss.str();
    }
    uint64_t mem_type_;
    ConstOpDescPtr op_desc_;
    size_t index_;
    std::string io_type_;
    const int64_t size_;
    const int64_t logical_offset_;
  };
  ModelUtils() = default;
  ~ModelUtils() = default;

  /// @ingroup ge
  /// @brief Get input size.
  /// @return std::vector<uint32_t>
  static std::vector<int64_t> GetInputSize(const ConstOpDescPtr &op_desc);

  /// @ingroup ge
  /// @brief Get output size.
  /// @return std::vector<uint32_t>
  static std::vector<int64_t> GetOutputSize(const ConstOpDescPtr &op_desc);

  /// @ingroup ge
  /// @brief Get workspace size.
  /// @return std::vector<uint32_t>
  static std::vector<int64_t> GetWorkspaceSize(const ConstOpDescPtr &op_desc);

  /// @ingroup ge
  /// @brief Get weight size.
  /// @return std::vector<uint32_t>
  static std::vector<int64_t> GetWeightSize(const ConstOpDescPtr &op_desc);

  /// @ingroup ge
  /// @brief Get weights.
  /// @return std::vector<ConstGeTensorPtr>
  static std::vector<ConstGeTensorPtr> GetWeights(const ConstOpDescPtr &op_desc);

  /// @ingroup ge
  /// @brief Get AiCpuOp Input descriptor.
  /// @return std::vector<ccAICPUTensor>
  static std::vector<ccAICPUTensor> GetInputDescs(const ConstOpDescPtr &op_desc);

  /// @ingroup ge
  /// @brief Get AiCpuOp Output descriptor.
  /// @return std::vector<ccAICPUTensor>
  static std::vector<ccAICPUTensor> GetOutputDescs(const ConstOpDescPtr &op_desc);

  /// @ingroup ge
  /// @brief Get input address.
  /// @return std::vector<void*>
  static std::vector<void *> GetInputAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc);
  static std::vector<void *> GetInputAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                           std::vector<uint64_t> &mem_type);

  /// @ingroup ge
  /// @brief Get input address value.
  /// @return std::vector<uint64_t>
  static std::vector<uint64_t> GetInputAddrsValue(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc);
  static std::vector<uint64_t> GetInputAddrsValue(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                                  std::vector<uint64_t> &mem_type);

  /// @ingroup ge
  /// @brief Get input data address.
  /// @return std::vector<void*>
  static std::vector<void *> GetInputDataAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc);
  static std::vector<void *> GetInputDataAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                               std::vector<uint64_t> &mem_type);

  /// @ingroup ge
  /// @brief Get input data address value.
  /// @return std::vector<uint64_t>
  static std::vector<uint64_t> GetInputDataAddrsValue(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc);
  static std::vector<uint64_t> GetInputDataAddrsValue(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                                      std::vector<uint64_t> &mem_type);

  /// @ingroup ge
  /// @brief Get output address.
  /// @return std::vector<void*>
  static std::vector<void *> GetOutputAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc);
  static std::vector<void *> GetOutputAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                            std::vector<uint64_t> &mem_type,
                                            const bool has_optional_addr = false);

  /// @ingroup ge
  /// @brief Get output address value.
  /// @return std::vector<uint64_t>
  static std::vector<uint64_t> GetOutputAddrsValue(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc);
  static std::vector<uint64_t> GetOutputAddrsValue(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                                   std::vector<uint64_t> &mem_type,
                                                   const bool has_optional_addr = false);

  /// @ingroup ge
  /// @brief Get output data address.
  /// @return std::vector<void*>
  static std::vector<void *> GetOutputDataAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc);
  static std::vector<void *> GetOutputDataAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                                std::vector<uint64_t> &mem_type,
                                                const bool has_optional_addr = false);

  /// @ingroup ge
  /// @brief Get output data address value.
  /// @return std::vector<uint64_t>
  static std::vector<uint64_t> GetOutputDataAddrsValue(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc);
  static std::vector<uint64_t> GetOutputDataAddrsValue(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                                       std::vector<uint64_t> &mem_type);

  /// @ingroup ge
  /// @brief Get output tensor description address, and copy output data address to output tensor description address.
  /// @return Status
  static Status GetInputOutputDescAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                        const OpDesc::Vistor<GeTensorDescPtr> &tensor_desc_visitor,
                                        std::vector<void *> &v_addrs,
                                        const bool has_optional_addr = false);

  /// @ingroup ge
  /// @brief Get workspace data address.
  /// @return std::vector<void*>
  static std::vector<void *> GetWorkspaceDataAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc);
  static std::vector<void *> GetWorkspaceDataAddrs(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc,
                                                   std::vector<uint64_t> &mem_type);

  /// @ingroup ge
  /// @brief Get workspace data address value.
  /// @return std::vector<uint64_t>
  static std::vector<uint64_t> GetWorkspaceDataAddrsValue(const RuntimeParam &model_param,
                                                          const ConstOpDescPtr &op_desc);
  static std::vector<uint64_t> GetWorkspaceDataAddrsValue(const RuntimeParam &model_param,
                                                          const ConstOpDescPtr &op_desc,
                                                          std::vector<uint64_t> &mem_type);

  /// @ingroup ge
  /// @brief Calculate aicpu blocing event num
  /// @return Status
  static Status CalculateAicpuBlockingEventNum(const GeModelPtr &ge_model, uint32_t &aicpu_blocking_event_num);

  /// @ingroup ge
  /// @brief Calculate hccl follw stream
  /// @return Status
  static Status CalculateFollowStream(const GeModelPtr &ge_model, uint64_t &hccl_fellow_stream_num);

  /// @ingroup ge
  /// @brief Calculate hccl group ordered event num
  /// @return Status
  static Status CalculateHcclGroupOrderedEventNum(const GeModelPtr &ge_model, uint32_t &hccl_group_ordered_event_num);

  /// @ingroup ge
  /// @brief Calculate the sum of follow stream
  /// @return int64_t
  static uint64_t CalFollowStreamSum(const std::multimap<int64_t, uint64_t> &hccl_stream_map);

  /// @ingroup ge
  /// @brief Get memory runtime base.
  /// @return Status
  static Status GetRtAddress(const RuntimeParam &param, const uintptr_t logic_addr, uint8_t *&mem_addr);
  static Status GetRtAddress(const RuntimeParam &param, const uintptr_t logic_addr, uint8_t *&mem_addr,
                             uint64_t &mem_type);

  /// @ingroup ge
  /// @brief Set device.
  /// @return Status
  static Status SetDevice(const uint32_t device_id);

  /// @ingroup ge
  /// @brief Reset device.
  /// @return Status
  static Status ResetDevice(const uint32_t device_id);

  static bool IsReuseZeroCopyMemory();
  static bool IsGeUseExtendSizeMemory(bool dynamic_graph = false);
  static vector_bit_t GetInputTensorNeedRawData(const OpDescPtr &op_desc);
  static Status InitRuntimeParams(const GeModelPtr &ge_model, RuntimeParam &runtime_param, const uint32_t device_id);
  static std::vector<MemInfo> GetAllMemoryTypeSize(const GeModelPtr &ge_model);
  static Status GetHbmFeatureMapMemInfo(const GeModelPtr &ge_model, std::vector<MemInfo> &all_mem_info,
                                        bool get_zero_copy = false);
  static Status MallocExMem(const uint32_t device_id, RuntimeParam &runtime_param);
  static void FreeExMem(const uint32_t device_id, RuntimeParam &runtime_param,
                        const uint64_t session_id = 0UL, const bool is_online = true);
  static bool IsSuppoprtAddrRefreshable(const uint64_t mem_types);
  static void GetAddrRefreshableFlagsByMemTypes(const std::vector<uint64_t> &mem_types, std::vector<uint8_t> &flags);
  static bool IsFeatureMapOrModelIoType(const uint64_t mem_type);
  static Status GetSpaceRegistries(const ge::GeRootModelPtr &root_model,
                                   std::shared_ptr<gert::OpImplSpaceRegistryV2Array> &space_registries);
  static bool IsAICoreKernel(const ge::ccKernelType kernel_type);
  static Status GetOpMasterDevice(const uint32_t &model_id, const ge::GeRootModelPtr &root_model,
                                  std::unordered_map<std::string, OpSoBinPtr> &built_in_so_bins,
                                  std::unordered_map<std::string, OpSoBinPtr> &cust_so_bins);
  static string GetOpMasterDeviceKey(const uint32_t &model_id, const std::string &so_path);

 private:
  static std::string GetOpMasterDeviceCustKey(const uint32_t &model_id, const std::string &so_path);

 private:
  static Status CreateOmOppDir(std::string &opp_dir);
  static Status RmOmOppDir(const std::string &opp_dir);
  static Status SaveToFile(const std::shared_ptr<ge::OpSoBin> &so_bin, const std::string &opp_path);

  static bool ValidateMemRange(const ConstOpDescPtr &op_desc, const uint64_t total_size, const int64_t offset,
                               const int64_t size);

  /// @ingroup ge
  /// @brief Get variable address.
  /// @return Status
  static Status GetVarAddr(const RuntimeParam &model_param, const int64_t offset, void *&var_addr);
  static Status RefreshAddressByMemType(const RuntimeParam &model_param, const NodeMemInfo &node_mem_info,
                                        void *&mem_addr);
  static Status GetOpMasterDeviceFromOppPackage(const uint32_t &model_id,
                                                std::unordered_map<std::string, OpSoBinPtr> &built_in_so_bins,
                                                std::unordered_map<std::string, OpSoBinPtr> &cust_so_bins);
};
}  // namespace ge

#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_MODEL_UTILS_H_
