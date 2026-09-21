/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "faker/aicore_taskdef_faker.h"
#include "faker/aicpu_taskdef_faker.h"
#include "framework/common/debug/ge_log.h"
namespace gert {

AiCoreTaskDefFaker::AiCoreTaskDefFaker(std::string stub_name) : with_handle_(false), need_atomic_(false),
                                                                inited_(false), stub_name_(stub_name), with_aicpu_(false) {
}

AiCoreTaskDefFaker::AiCoreTaskDefFaker(bool with_handle, bool need_atomic, std::string stub_name, bool with_aicpu)
        : with_handle_(with_handle), need_atomic_(need_atomic), inited_(false), stub_name_(stub_name), with_aicpu_(with_aicpu) {
}

vector<domi::TaskDef> AiCoreTaskDefFaker::CreateTaskDef(uint64_t op_index) {
  Init();
  auto task_def = TaskDefFaker::CreateTaskDef(op_index);
  if (!with_handle_) {
    if (need_atomic_) {
      task_def[0].mutable_kernel()->set_stub_func(atomic_stub_name_);
      task_def[1].mutable_kernel()->set_stub_func(stub_name_);
      if (!args_format_.empty()) {
        task_def[1].mutable_kernel()->set_stub_func(stub_name_);
      }
    } else {
      task_def[0].mutable_kernel()->set_stub_func(stub_name_);
    }
  } else {
    if (need_atomic_) {
      task_def[0].mutable_kernel()->set_stub_func(atomic_stub_name_);
    }
  }
  if (!args_format_.empty()) {
    task_def[0].mutable_kernel()->mutable_context()->set_args_format(args_format_);
  }
  GELOGD("CreateTaskDef with_aicpu_:%d, size:%zu.", with_aicpu_, task_def.size());
  if (with_aicpu_) {
    domi::KernelExDef *kernel_ex_def = task_def[1].mutable_kernel_ex();
    if (!args_format_.empty()) {
      task_def[1].mutable_kernel()->mutable_context()->set_args_format(args_format_);
    }
    SetTfKernelDef(kernel_ex_def);
  }
  return task_def;
}

std::unique_ptr<TaskDefFaker> AiCoreTaskDefFaker::Clone()  const {
  return std::unique_ptr<AiCoreTaskDefFaker>(new AiCoreTaskDefFaker(*this));
}

AiCoreTaskDefFaker& AiCoreTaskDefFaker::AtomicStubNum(const std::string &name){
  need_atomic_ = true;
  atomic_stub_name_ = name;
  return *this;
}

AiCoreTaskDefFaker& AiCoreTaskDefFaker::WithHandle(){
  with_handle_ = true;
  return *this;
}

AiCoreTaskDefFaker& AiCoreTaskDefFaker::BinData(uint64_t data){
  bin_data = data;
  return *this;
}

AiCoreTaskDefFaker& AiCoreTaskDefFaker::ArgsFormat(const std::string &args_format){
  args_format_ = args_format;
  return *this;
}

void AiCoreTaskDefFaker::Init(){
  if (inited_) {
    return;
  }
  if (need_atomic_) {
    AddTask({kWithoutHandle, kTE_AiCore, bin_data});
  }
  if (with_handle_) {
    AddTask({kWithHandle, kTE_AiCore, bin_data});
  } else {
    AddTask({kWithoutHandle, kTE_AiCore, bin_data});
  }
  if (with_aicpu_) {
    AddTask({kTfAicpu, kTF_AiCpu, 0});
  }
  inited_ = true;
}

};
