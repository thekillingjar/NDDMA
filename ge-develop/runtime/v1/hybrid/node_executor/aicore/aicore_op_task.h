/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_KERNEL_AICORE_OP_TASK_H_
#define GE_HYBRID_KERNEL_AICORE_OP_TASK_H_

#include <memory>
#include <vector>
#include "common/dump/exception_dumper.h"
#include "common/tbe_handle_store/bin_register_utils.h"
#include "platform/platform_info.h"
#include "framework/common/ge_inner_error_codes.h"
#include "hybrid/common/tensor_value.h"
#include "hybrid/node_executor/task_context.h"
#include "proto/task.pb.h"
#include "register/op_tiling.h"
#include "graph/bin_cache/node_compile_cache_module.h"

namespace ge {
namespace hybrid {
class AiCoreOpTask {
 public:
  AiCoreOpTask() = default;
  virtual ~AiCoreOpTask() = default;

  virtual Status Init(const NodePtr &node, const domi::TaskDef &task_def);

  bool IsDynamicShapeSupported() const;

  // do preparation with shape(without actual io memory)
  Status PrepareWithShape(const TaskContext &context);

  virtual Status UpdateArgs(TaskContext &task_context);

  Status LaunchKernel(const rtStream_t stream);

  const std::string& GetName() const;

  const std::string& GetLogName() const {return log_name_;}

  bool GetClearAtomic() const {return clear_atomic_;}

  uint32_t GetBlockDim() const {return block_dim_;}

  void SetSingleOp(const bool is_single_op) {is_single_op_ = is_single_op;}

  void SetOverflowAddr(void *const overflow_addr) {overflow_addr_  = overflow_addr;};

  void SetNeedHostMemOpt(const bool need_host_mem_opt) { need_host_mem_opt_ = need_host_mem_opt; }
  bool IsNeedHostMemOpt() const { return need_host_mem_opt_; }
  bool IsArgsExtendedForHostMemInput() const { return host_mem_input_data_offset_ != 0U; }
  virtual const std::string& GetOpType() const;

  UnknowShapeOpType GetUnknownShapeOpType() const {
    return unknown_shape_op_type_;
  }

  Status UpdateOutputsShape(const TaskContext &context) const;

  void FillExtraOpInfo(ExtraOpInfo &extra_op_info) const {
    extra_op_info.node_info = node_info_;
    extra_op_info.tiling_data = tiling_data_;
    extra_op_info.tiling_key = tiling_key_;
    extra_op_info.args = PtrToValue(args_.get()) + offset_;
  }
  virtual Status UpdateBinHandle(const TaskContext &task_context, const NodeCompileCacheItem &cci);

  void SetModelId(const uint32_t model_id) {
    model_id_ = model_id;
  }

  Status SetPlatformInfo(const NodePtr &node, const fe::PlatFormInfos &platform_info);

 protected:
  Status UpdateTilingInfo(const TaskContext &context);
  virtual std::string GetKeyForOpParamSize() const;
  virtual std::string GetKeyForTbeKernel() const;
  virtual std::string GetKeyForTvmMagic() const;
  virtual std::string GetKeyForTvmMetaData() const;
  virtual std::string GetKeyForKernelName(const OpDesc &op_desc) const;
  virtual Status CalcTilingInfo(const NodePtr &node, const Operator &op) const;
  virtual AttrNameOfBinOnOp GetAttrNamesForBin() const;
  virtual Status AddCompileCacheItem(const KernelLaunchBinType bin_type, const NodePtr &node) const;
  Status DoInit(const NodePtr &node, const domi::TaskDef &task_def);

 private:
  static Status ValidateTaskDef(const domi::TaskDef &task_def);
  Status InitWithTaskDef(const NodePtr &node, const domi::TaskDef &task_def);
  Status InitTilingInfo(const OpDesc &op_desc, bool is_updating = false);
  Status InitWithKernelDef(const OpDesc &op_desc, const domi::TaskDef &task_def);
  Status InitWithKernelDefWithHandle(const OpDesc &op_desc, const domi::TaskDef &task_def);
  Status UpdateShapeToOutputDesc(const TaskContext &context, const GeShape &shape, const int32_t output_index) const;
  Status UpdateArgBase(const TaskContext &task_context, uint32_t &index, const size_t overflow_arg_index);
  Status UpdateHostMemInputArgs(const TaskContext &task_context);
  Status ExpandArgsByLength(const size_t length, const std::string &name);
  void SetTaskTag() const;

  std::string stub_name_;
  void *stub_func_ = nullptr;
  std::unique_ptr<uint8_t[]> args_ = nullptr;
  uint32_t block_dim_ = 1U;
  bool clear_atomic_ = true;
  bool is_single_op_ = false;
  std::vector<int32_t> output_indices_to_skip_;
  std::string node_info_;
  uint64_t tiling_key_ = 0U;
  void *handle_ = nullptr;
  bool is_dynamic_ = false;
  uint64_t log_id_ = 0U;
  std::string log_name_;
  std::string op_type_;
  UnknowShapeOpType unknown_shape_op_type_ = DEPEND_IN_SHAPE;
  std::unique_ptr<TensorBuffer> shape_buffer_ = nullptr;
  std::unique_ptr<uint8_t[]> host_shape_buffer_ = nullptr;
  std::string op_name_;
  std::string tiling_data_;
  uint32_t max_tiling_size_ = 0U;
  bool need_tiling_ = false;
  bool need_overflow_addr_ = false;
  fe::PlatFormInfos platform_infos_;

  uintptr_t *arg_base_ = nullptr;
  void *overflow_addr_ = nullptr;
  uint32_t args_size_ = 0U;
  uint32_t args_size_without_tiling_ = 0U;
  uint32_t max_arg_count_ = 0U;
  std::unique_ptr<rtHostInputInfo_t[]> host_inputs_info_;
  rtArgsEx_t args_ex_ = {};
  uint32_t offset_ = 0U;
  std::unique_ptr<optiling::utils::OpRunInfo> tiling_info_;
  uint32_t tiling_data_idx_ = 0U;
  bool need_host_mem_opt_ = false;
  size_t host_mem_input_data_offset_ = 0U;
  uint32_t model_id_ = 0U;
  uint32_t local_memory_size_ = 0U;
  friend class AtomicAddrCleanOpTask;
};

class AtomicAddrCleanOpTask : public AiCoreOpTask {
 public:
  Status Init(const NodePtr &node, const domi::TaskDef &task_def) override;
  Status UpdateArgs(TaskContext &task_context) override;
  const std::string& GetOpType() const override;

 protected:
  std::string GetKeyForOpParamSize() const override;
  std::string GetKeyForTbeKernel() const override;
  std::string GetKeyForTvmMagic() const override;
  std::string GetKeyForTvmMetaData() const override;
  std::string GetKeyForKernelName(const OpDesc &op_desc) const override;
  Status CalcTilingInfo(const NodePtr &node, const Operator &op) const override;
  AttrNameOfBinOnOp GetAttrNamesForBin() const override;
  Status UpdateBinHandle(const TaskContext &task_context, const NodeCompileCacheItem &cci) override {
    (void)task_context;
    is_dynamic_ = cci.IsSupportDynamic();
    return SUCCESS;
  };

  Status AddCompileCacheItem(const KernelLaunchBinType bin_type, const NodePtr &node) const override {
    (void)bin_type;
    (void)node;
    return SUCCESS;
  };

 private:
  Status InitAtomicAddrCleanIndices(const OpDesc &op_desc);
  std::vector<int32_t> atomic_output_indices_;
  std::vector<int32_t> atomic_workspace_indices_;
  fe::PlatFormInfos platform_infos_;
};
}  // namespace hybrid
}  // namespace ge
#endif  // GE_HYBRID_KERNEL_AICORE_OP_TASK_H_
