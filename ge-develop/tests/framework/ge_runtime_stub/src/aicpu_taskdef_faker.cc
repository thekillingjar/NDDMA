/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "faker/aicpu_taskdef_faker.h"
#include <cinttypes>
#include <securec.h>
#include "faker/task_def_faker.h"
#include "aicpu_engine_struct.h"
#include "aicpu_task_struct.h"
#include "faker/aicpu_ext_info_faker.h"

namespace gert {
struct AicpuTaskStruct {
  aicpu::AicpuParamHead head;
  uint64_t io_addrp[6];
}__attribute__((packed));

void SetTfKernelDef(domi::KernelExDef *kernel_ex_def) {
  static std::shared_ptr<STR_FWK_OP_KERNEL> str_fwkop_kernel_ptr;
  str_fwkop_kernel_ptr = std::make_shared<STR_FWK_OP_KERNEL>();
  str_fwkop_kernel_ptr->fwkKernelType = 0;// FMK_KERNEL_TYPE_TF;
  aicpu::FWKAdapter::FWKOperateParam *str_tf_kernel = &(str_fwkop_kernel_ptr->fwkKernelBase.fwk_kernel);
  str_tf_kernel->opType = aicpu::FWKAdapter::FWK_ADPT_KERNEL_RUN;
  str_tf_kernel->sessionID = 2;
  str_tf_kernel->kernelID = 1;

  static std::string task_info;
  task_info.append("this is a test");

  auto ext_info = GetFakeExtInfo();
  kernel_ex_def->set_kernel_ext_info_size(ext_info.size());
  std::string *mutable_ext_info = kernel_ex_def->mutable_kernel_ext_info();
  (*mutable_ext_info) = ext_info;

  kernel_ex_def->set_args(reinterpret_cast<void*>(str_fwkop_kernel_ptr.get()), sizeof(STR_FWK_OP_KERNEL));
  kernel_ex_def->set_args_size(sizeof(STR_FWK_OP_KERNEL));
  kernel_ex_def->set_task_info(task_info);
  kernel_ex_def->set_task_info_size(task_info.size());
}

void SetCCKernelDef(domi::KernelDef *kernel_def, bool is_memcpy_task) {
  AicpuTaskStruct args;
  args.head.length = sizeof(args);
  if (is_memcpy_task) {
    args.head.ioAddrNum = 4;
  } else {
    args.head.ioAddrNum = 3;
  }
  
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_kernel_type(7); // cust aicpu
  kernel_def->set_args(reinterpret_cast<const char *>(&args), args.head.length);
  kernel_def->set_args_size(args.head.length);

  auto ext_info = GetFakeExtInfo();
  kernel_def->set_kernel_ext_info(ext_info.c_str(), ext_info.size());
  kernel_def->set_kernel_ext_info_size(ext_info.size());
}

AiCpuTfTaskDefFaker &AiCpuTfTaskDefFaker::SetNeedMemcpy(bool need_memcpy) {
  need_memcpy_ = need_memcpy;
  return *this;
}

vector<domi::TaskDef> AiCpuTfTaskDefFaker::CreateTaskDef(uint64_t op_index) {
  AddTask({kTfAicpu, kTF_AiCpu, 0});
  if (need_memcpy_) {
    AddTask({kTfAicpu, kTF_AiCpu, 0});
  }
  
  auto task_def = TaskDefFaker::CreateTaskDef(op_index);
  domi::KernelExDef *kernel_ex_def = task_def[0].mutable_kernel_ex();
  SetTfKernelDef(kernel_ex_def);
  if (need_memcpy_) {
    domi::KernelExDef *kernel_ex_def = task_def[1].mutable_kernel_ex();
    SetTfKernelDef(kernel_ex_def);
  }
  return task_def;
}

std::unique_ptr<TaskDefFaker> AiCpuTfTaskDefFaker::Clone()  const {
  return std::unique_ptr<AiCpuTfTaskDefFaker>(new AiCpuTfTaskDefFaker(*this));
}

AiCpuCCTaskDefFaker &AiCpuCCTaskDefFaker::SetNeedMemcpy(bool need_memcpy) {
  need_memcpy_ = need_memcpy;
  return *this;
}

vector<domi::TaskDef> AiCpuCCTaskDefFaker::CreateTaskDef(uint64_t op_index) {
  AddTask({kCCAicpu, kCC_AiCpu, 0});
  if (need_memcpy_) {
    AddTask({kCCAicpu, kCC_AiCpu, 0});
  }

  auto task_def = TaskDefFaker::CreateTaskDef(op_index);
  domi::KernelDef *kernel_def = task_def[0].mutable_kernel();
  SetCCKernelDef(kernel_def);
  if (need_memcpy_) {
    domi::KernelDef *kernel_def = task_def[1].mutable_kernel();
    SetCCKernelDef(kernel_def, true);
  }
  return task_def;
}

std::unique_ptr<TaskDefFaker> AiCpuCCTaskDefFaker::Clone()  const {
  return std::unique_ptr<AiCpuCCTaskDefFaker>(new AiCpuCCTaskDefFaker(*this));
}
} // gert
