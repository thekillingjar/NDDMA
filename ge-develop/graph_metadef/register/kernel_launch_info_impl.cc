/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_launch_info_impl.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "common/checker.h"
#include "runtime/rt_model.h"
#include "ge/framework/common/taskdown_common.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace ge {
namespace {
bool IsAllKernel(const domi::TaskDef &task_def) {
  return (static_cast<ModelTaskType>(task_def.type()) == ModelTaskType::MODEL_TASK_ALL_KERNEL) ||
      (static_cast<ModelTaskType>(task_def.type()) == ModelTaskType::MODEL_TASK_VECTOR_ALL_KERNEL);
}
}
KernelLaunchInfoImplPtr KernelLaunchInfoImpl::LoadFromData(const gert::ExeResGenerationContext *context,
    const std::vector<uint8_t> &data) {
  GE_ASSERT_NOTNULL(context);
  auto impl_ptr = ComGraphMakeUnique<KernelLaunchInfoImpl>();
  GE_ASSERT_NOTNULL(impl_ptr);
  GE_ASSERT_TRUE(impl_ptr->task_def_.ParseFromArray(data.data(), static_cast<int32_t>(data.size())));
  impl_ptr->context_ = const_cast<gert::ExeResGenerationContext *>(context);
  return impl_ptr;
}
KernelLaunchInfoImplPtr KernelLaunchInfoImpl::CreateAicpuKfcTask(const gert::ExeResGenerationContext *context,
    const char *so_name, const char *kernel_name) {
  GE_ASSERT_NOTNULL(context);
  auto impl_ptr = ComGraphMakeUnique<KernelLaunchInfoImpl>();
  GE_ASSERT_NOTNULL(impl_ptr);
  impl_ptr->context_ = const_cast<gert::ExeResGenerationContext *>(context);
  impl_ptr->task_def_.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  auto kernel_def = impl_ptr->task_def_.mutable_kernel();
  GE_ASSERT_NOTNULL(kernel_def);
  kernel_def->set_so_name(so_name);
  kernel_def->set_kernel_name(kernel_name);
  auto kernel_context = kernel_def->mutable_context();
  GE_ASSERT_NOTNULL(kernel_context);
  kernel_context->set_kernel_type(static_cast<uint32_t>(ccKernelType::AI_CPU_KFC));
  kernel_context->set_op_index(static_cast<uint32_t>(context->GetOpId()));
  return impl_ptr;
}

KernelLaunchInfoImplPtr KernelLaunchInfoImpl::CreateHcomRecordTask(const gert::ExeResGenerationContext *context,
    const char *group_name) {
  GE_ASSERT_NOTNULL(context);
  GE_ASSERT_NOTNULL(group_name);
  auto impl_ptr = ComGraphMakeUnique<KernelLaunchInfoImpl>();
  GE_ASSERT_NOTNULL(impl_ptr);
  impl_ptr->context_ = const_cast<gert::ExeResGenerationContext *>(context);
  impl_ptr->task_def_.set_id(static_cast<uint32_t>(context->GetOpId()));
  impl_ptr->task_def_.set_notify_id(UINT32_MAX);
  impl_ptr->task_def_.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NOTIFY_RECORD));
  impl_ptr->task_def_.set_private_def(group_name);
  return impl_ptr;
}

KernelLaunchInfoImplPtr KernelLaunchInfoImpl::CreateHcomWaitTask(const gert::ExeResGenerationContext *context,
    const char *group_name) {
  GE_ASSERT_NOTNULL(context);
  GE_ASSERT_NOTNULL(group_name);
  auto impl_ptr = ComGraphMakeUnique<KernelLaunchInfoImpl>();
  GE_ASSERT_NOTNULL(impl_ptr);
  impl_ptr->context_ = const_cast<gert::ExeResGenerationContext *>(context);
  impl_ptr->task_def_.set_id(static_cast<uint32_t>(context->GetOpId()));
  impl_ptr->task_def_.set_notify_id(UINT32_MAX);
  impl_ptr->task_def_.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NOTIFY_WAIT));
  impl_ptr->task_def_.set_private_def(group_name);
  return impl_ptr;
}

KernelLaunchInfoImplPtr KernelLaunchInfoImpl::CreateCcuTask(const gert::ExeResGenerationContext *context,
    const std::vector<std::string>& groups) {
  GE_ASSERT_NOTNULL(context);
  auto impl_ptr = ComGraphMakeUnique<KernelLaunchInfoImpl>();
  GE_ASSERT_NOTNULL(impl_ptr);
  impl_ptr->context_ = const_cast<gert::ExeResGenerationContext *>(context);
  impl_ptr->task_def_.set_id(static_cast<uint32_t>(context->GetOpId()));
  impl_ptr->task_def_.set_notify_id(UINT32_MAX);
  impl_ptr->task_def_.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CCU_KERNEL));
  impl_ptr->task_def_.set_sqe_num(TASK_CCU_KERNEL_SQE_NUM);

  auto *fusion_task_def = impl_ptr->task_def_.mutable_fusion_task();
  GE_ASSERT_NOTNULL(fusion_task_def);
  fusion_task_def->set_op_index(static_cast<uint32_t>(context->GetOpId()));

  auto *sub_task_info = fusion_task_def->add_fusion_sub_task_info();
  GE_ASSERT_NOTNULL(sub_task_info);
  sub_task_info->set_type(domi::FusionSubTaskInfo::CCU);

  auto* task = sub_task_info->mutable_task();
  GE_ASSERT_NOTNULL(task);
  auto *ccu_task_group = task->mutable_ccu_task_group();
  GE_ASSERT_NOTNULL(ccu_task_group);

  auto *ccu_task_info = ccu_task_group->add_ccu_task_info();
  GE_ASSERT_NOTNULL(ccu_task_info);

  for (const auto& group : groups) {
    ccu_task_group->add_group(group);
  }
  return impl_ptr;
}

KernelLaunchInfoImplPtr KernelLaunchInfoImpl::CreateFusionTask(const gert::ExeResGenerationContext *context,
    const std::vector<const KernelLaunchInfoImpl*>& sub_tasks) {
  GE_ASSERT_NOTNULL(context);

  auto impl_ptr = ComGraphMakeUnique<KernelLaunchInfoImpl>();
  GE_ASSERT_NOTNULL(impl_ptr);

  impl_ptr->context_ = const_cast<gert::ExeResGenerationContext*>(context);
  impl_ptr->task_def_.set_id(static_cast<uint32_t>(context->GetOpId()));
  impl_ptr->task_def_.set_notify_id(UINT32_MAX);
  impl_ptr->task_def_.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FUSION_KERNEL));

  if (!ProcessFusionTask(context, sub_tasks, impl_ptr.get())) {
      return impl_ptr;
  }

  return impl_ptr;
}

bool KernelLaunchInfoImpl::ProcessFusionTask(const gert::ExeResGenerationContext* context,
    const std::vector<const KernelLaunchInfoImpl*>& sub_tasks,
    KernelLaunchInfoImpl* impl_ptr) {
  auto* fusion_task_def = impl_ptr->task_def_.mutable_fusion_task();
  GE_ASSERT_NOTNULL(fusion_task_def);

  fusion_task_def->set_op_index(static_cast<uint32_t>(context->GetOpId()));

  uint32_t sqe_num = 0;
  for (const auto& sub_task : sub_tasks) {
    if (!ProcessFusionSubTask(sub_task, fusion_task_def)) {
      return false;
    }
    uint32_t task_sqe_num = sub_task->task_def_.sqe_num();
    if (task_sqe_num == 0) {
      sqe_num++;
    } else {
      sqe_num += task_sqe_num;
    }
  }
  impl_ptr->task_def_.set_sqe_num(sqe_num);

  return true;
}

bool KernelLaunchInfoImpl::ProcessFusionSubTask(const KernelLaunchInfoImpl* sub_task,
    domi::FusionTaskDef* fusion_task_def) {
  auto* sub_task_info = fusion_task_def->add_fusion_sub_task_info();

  const auto task_type = static_cast<ModelTaskType>(sub_task->task_def_.type());
  if ((task_type == ModelTaskType::MODEL_TASK_ALL_KERNEL) ||
      (task_type == ModelTaskType::MODEL_TASK_KERNEL)) {
    return ProcessAicoreFusionSubTask(sub_task, sub_task_info);
  } else if (task_type == ModelTaskType::MODEL_TASK_CCU_KERNEL) {
    return ProcessCcuFusionSubTask(sub_task, sub_task_info);
  } else {
    GELOGE(FAILED, "Unsupported task type in fusion sub task: %d", task_type);
    return false;
  }
}

bool KernelLaunchInfoImpl::ProcessCcuFusionSubTask(const KernelLaunchInfoImpl* sub_task,
    domi::FusionSubTaskInfo* sub_task_info) {
  sub_task_info->set_type(domi::FusionSubTaskInfo::CCU);

  auto* task = sub_task_info->mutable_task();
  GE_ASSERT_NOTNULL(task);

  auto* ccu_task_group = task->mutable_ccu_task_group();
  GE_ASSERT_NOTNULL(ccu_task_group);

  GE_ASSERT_TRUE(sub_task->task_def_.has_fusion_task());

  const auto& old_fusion_task = sub_task->task_def_.fusion_task();
  GE_ASSERT_TRUE((old_fusion_task.fusion_sub_task_info_size() > 0));

  const auto& old_sub_task = old_fusion_task.fusion_sub_task_info(0);
  GE_ASSERT_TRUE(old_sub_task.task().has_ccu_task_group());

  const auto& old_ccu_task_group = old_sub_task.task().ccu_task_group();
  ccu_task_group->CopyFrom(old_ccu_task_group);
  return true;
}

bool KernelLaunchInfoImpl::ProcessAicoreFusionSubTask(const KernelLaunchInfoImpl* sub_task, domi::FusionSubTaskInfo* sub_task_info) {
  auto* task = sub_task_info->mutable_task();
  GE_ASSERT_NOTNULL(task);
  auto* aicore_task_info = task->mutable_aicore_fusion_task_info();
  GE_ASSERT_NOTNULL(aicore_task_info);
  auto aicore_context = aicore_task_info->mutable_context();
  GE_ASSERT_NOTNULL(aicore_context);
  aicore_context->set_kernel_type(static_cast<uint32_t>(ccKernelType::MIX_AICORE));
  sub_task_info->set_type(domi::FusionSubTaskInfo::AICORE);
  aicore_task_info->set_is_all_kernel(IsAllKernel(sub_task->task_def_));

  SetBlockDimForAicoreFusionSubTask(sub_task, sub_task_info);
  return true;
}

void KernelLaunchInfoImpl::SetBlockDimForAicoreFusionSubTask(const KernelLaunchInfoImpl* sub_task,
    domi::FusionSubTaskInfo* sub_task_info) {
  const auto blockdim_value = sub_task->GetBlockDim();
  auto* aicore_task_info = sub_task_info->mutable_task()->mutable_aicore_fusion_task_info();
  auto* config = aicore_task_info->mutable_config();

  for (int32_t j = 0; j < config->launch_attribute_size(); ++j) {
    auto* attr = config->mutable_launch_attribute(j);
    if (attr->id() == domi::LaunchAttribute::BLOCKDIM) {
      attr->mutable_value()->set_block_dim(blockdim_value);
      return;
    }
  }

  domi::LaunchAttribute new_attr;
  new_attr.set_id(domi::LaunchAttribute::BLOCKDIM);
  new_attr.mutable_value()->set_block_dim(blockdim_value);
  *config->add_launch_attribute() = new_attr;
}


std::vector<uint8_t> KernelLaunchInfoImpl::Serialize() const {
  const auto buffer_size = task_def_.ByteSizeLong();
  std::vector<uint8_t> buffer(buffer_size, 0);
  GE_ASSERT_TRUE(task_def_.SerializeToArray(buffer.data(), static_cast<int32_t>(buffer_size)));
  return buffer;
}
uint32_t KernelLaunchInfoImpl::GetStreamId() const {
  return task_def_.stream_id();
}
void KernelLaunchInfoImpl::SetStreamId(uint32_t stream_id) {
  task_def_.set_stream_id(stream_id);
}

uint32_t KernelLaunchInfoImpl::GetBlockDim() const {
  uint32_t block_dim = 0;
  if (static_cast<ModelTaskType>(task_def_.type()) == ModelTaskType::MODEL_TASK_KERNEL) {
    block_dim = task_def_.kernel().block_dim();
  } else if (IsAllKernel(task_def_)) {
    block_dim = task_def_.kernel_with_handle().block_dim();
  } else {
    GELOGE(FAILED, "Only aicpu and aicore task has block_dim, but get[%d]",
        task_def_.type());
  }
  return block_dim;
}

graphStatus KernelLaunchInfoImpl::SetBlockDim(uint32_t block_dim) {
  if (static_cast<ModelTaskType>(task_def_.type()) == ModelTaskType::MODEL_TASK_KERNEL) {
    auto kernel_def = task_def_.mutable_kernel();
    GE_ASSERT_NOTNULL(kernel_def);
    kernel_def->set_block_dim(block_dim);
  } else if (IsAllKernel(task_def_)) {
    auto kernel_with_handle = task_def_.mutable_kernel_with_handle();
    GE_ASSERT_NOTNULL(kernel_with_handle);
    kernel_with_handle->set_block_dim(block_dim);
  } else {
    // 报错
    GE_ASSERT_TRUE(false, "Only aicpu and aicore task can set args format, but get[%d]",
        task_def_.type());
  }
  return SUCCESS;
}

const char *KernelLaunchInfoImpl::GetArgsFormat() const {
  if (static_cast<ModelTaskType>(task_def_.type()) == ModelTaskType::MODEL_TASK_KERNEL) {
    return task_def_.kernel().context().args_format().c_str();
  }
  if (IsAllKernel(task_def_)) {
    return task_def_.kernel_with_handle().context().args_format().c_str();
  }
  if (static_cast<ModelTaskType>(task_def_.type()) == ModelTaskType::MODEL_TASK_FUSION_KERNEL) {
    return task_def_.fusion_task().args_format().c_str();
  }
  GELOGE(FAILED, "Only aicpu, aicore and fusion task has args format, but get[%d]",
      task_def_.type());
  return nullptr;
}
graphStatus KernelLaunchInfoImpl::SetArgsFormat(const char *args_format) {
  GE_ASSERT_NOTNULL(args_format);
  if (static_cast<ModelTaskType>(task_def_.type()) == ModelTaskType::MODEL_TASK_FUSION_KERNEL) {
    task_def_.mutable_fusion_task()->set_args_format(args_format);
    return SUCCESS;
  }
  domi::KernelContext *kernel_context = nullptr;
  if (static_cast<ModelTaskType>(task_def_.type()) == ModelTaskType::MODEL_TASK_KERNEL) {
    auto kernel_def = task_def_.mutable_kernel();
    GE_ASSERT_NOTNULL(kernel_def);
    kernel_context = kernel_def->mutable_context();
  } else if (IsAllKernel(task_def_)) {
    auto kernel_with_handle = task_def_.mutable_kernel_with_handle();
    GE_ASSERT_NOTNULL(kernel_with_handle);
    kernel_context = kernel_with_handle->mutable_context();
  } else {
    GELOGE(FAILED, "Only aicpu and aicore task can set args format, but get[%d]",
        task_def_.type());
  }
  GE_ASSERT_NOTNULL(kernel_context);
  kernel_context->set_args_format(args_format);
  return SUCCESS;
}

const char *KernelLaunchInfoImpl::GetSoName() const {
  return task_def_.kernel().so_name().c_str();
}
const char *KernelLaunchInfoImpl::GetKernelName() const {
  return task_def_.kernel().kernel_name().c_str();
}
}