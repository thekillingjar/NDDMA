/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_SINGLE_OP_TASK_TBE_TASK_BUILDER_H_
#define GE_SINGLE_OP_TASK_TBE_TASK_BUILDER_H_

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>

#include "graph/op_desc.h"
#include "single_op/single_op.h"
#include "single_op/single_op_model.h"

namespace ge {
class TbeTaskBuilder {
 public:
  TbeTaskBuilder(const std::string &model_name, const NodePtr &node, const domi::TaskDef &task_def);
  virtual ~TbeTaskBuilder() = default;

  Status BuildTask(TbeOpTask &task, const SingleOpModelParam &param);

 protected:
  virtual std::string GetKeyForOpParamSize() const;
  virtual std::string GetKeyForTvmMetaData() const;
  virtual TBEKernelPtr GetTbeKernel(const OpDescPtr &op_desc) const;
  virtual void GetKernelName(const OpDescPtr &op_desc, std::string &kernel_name) const;
  virtual Status InitKernelArgs(void *const args_addr, const size_t arg_size, const SingleOpModelParam &param);

 private:
  friend class MixL2TaskBuilder;
  Status InitTilingInfo(TbeOpTask &task);
  Status SetKernelArgs(TbeOpTask &task, const SingleOpModelParam &param, const OpDescPtr &op_desc);
  Status UpdateTilingArgs(TbeOpTask &task, const size_t index, const size_t tiling_arg_index) const;
  Status RegisterKernel(TbeOpTask &task, const SingleOpModelParam &param);
  Status RegisterKernelWithHandle(const SingleOpModelParam &param);

  Status DoRegisterKernel(const ge::OpKernelBin &tbe_kernel, const char_t *const bin_file_key, void **const bin_handle,
                          const SingleOpModelParam &param);

  Status DoRegisterBinary(const OpKernelBin &kernel_bin, void **const bin_handle,
                          const SingleOpModelParam &param) const;
  Status DoRegisterMeta(void *const bin_handle) const;
  Status GetMagic(uint32_t &magic) const;

  static Status DoRegisterFunction(void *const bin_handle, const char_t *const stub_name,
                                   const char_t *const kernel_name);

  const NodePtr node_;
  const OpDescPtr op_desc_;
  const domi::TaskDef &task_def_;
  const domi::KernelDef &kernel_def_;
  const domi::KernelDefWithHandle &kernel_def_with_handle_;
  const std::string model_name_;
  std::string stub_name_;
  void *handle_ = nullptr;
};

class AtomicAddrCleanTaskBuilder : public TbeTaskBuilder {
 public:
  AtomicAddrCleanTaskBuilder(const std::string &model_name, const NodePtr &node,
                             const domi::TaskDef &task_def)
      : TbeTaskBuilder(model_name, node, task_def) {}
  ~AtomicAddrCleanTaskBuilder() override = default;

 protected:
  std::string GetKeyForOpParamSize() const override;
  std::string GetKeyForTvmMetaData() const override;
  TBEKernelPtr GetTbeKernel(const OpDescPtr &op_desc) const override;
  void GetKernelName(const OpDescPtr &op_desc, std::string &kernel_name) const override;
  Status InitKernelArgs(void *const args_addr, const size_t arg_size, const SingleOpModelParam &param) override;
};

class MixL2TaskBuilder : public TbeTaskBuilder {
 public:
  MixL2TaskBuilder(const std::string &model_name, const NodePtr &node, const domi::TaskDef &task_def)
      : TbeTaskBuilder(model_name, node, task_def), ffts_plus_task_def_(task_def.ffts_plus_task()) {}
  Status BuildMixL2Task(MixL2OpTask &task, SingleOpModelParam &param);
  ~MixL2TaskBuilder() override = default;

 private:
  Status InitMixKernelArgs(MixL2OpTask &task, const size_t addr_base_offset, SingleOpModelParam &param);
  Status HandleSoftSyncOp(MixL2OpTask &task, SingleOpModelParam &param);
  Status InitTilingDataAddrToArgs(MixL2OpTask &task) const;
  const domi::FftsPlusTaskDef &ffts_plus_task_def_;
  uint64_t tiling_data_size_ = 0U;
  uint8_t *tiling_data_addr_ = nullptr;
};
}  // namespace ge

#endif  // GE_SINGLE_OP_TASK_TBE_TASK_BUILDER_H_
