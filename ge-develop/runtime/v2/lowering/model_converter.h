/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_LOWERING_MODEL_CONVERTER_H_
#define AIR_CXX_RUNTIME_V2_LOWERING_MODEL_CONVERTER_H_
#include "base/registry/op_impl_space_registry_v2.h"
#include "common/model/ge_root_model.h"
#include "common/ge_visibility.h"
#include "runtime/gert_api.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace gert {
using OpImplSpaceRegistryArrayV2Ptr = std::shared_ptr<OpImplSpaceRegistryV2Array>;
class ModelDescHolder {
 public:
  explicit ModelDescHolder(int64_t reusable_stream_num = 1, int64_t event_num = 0, int64_t notify_num = 0,
                           int64_t attached_stream_num = 0) {
    model_desc_.SetReusableStreamNum(reusable_stream_num);
    model_desc_.SetReusableEventNum(event_num);
    model_desc_.SetReusableNotifyNum(notify_num);
    model_desc_.SetAttachedStreamNum(attached_stream_num);
  }
  const ModelDesc &GetModelDesc() const {
    return model_desc_;
  }

  ModelDesc &MutableModelDesc() {
    return model_desc_;
  }

  void SetSpaceRegistry(const OpImplSpaceRegistryV2Ptr &space_registry,
                        ge::OppImplVersion opp_impl_version = ge::OppImplVersion::kOpp) {
    if (space_registries_v2_ == nullptr) {
      space_registries_v2_ = ge::MakeShared<OpImplSpaceRegistryV2Array>();
    }
    if (space_registry != nullptr && opp_impl_version < ge::OppImplVersion::kVersionEnd) {
      space_registries_v2_->at(static_cast<size_t>(opp_impl_version)) = space_registry;
    }
  }

  void SetSpaceRegistries(const OpImplSpaceRegistryArrayV2Ptr &space_registry) {
    space_registries_v2_ = space_registry;
  }

  OpImplSpaceRegistryV2Ptr GetSpaceRegistry(OppImplVersionTag opp_impl_version = OppImplVersionTag::kOpp) {
    if (opp_impl_version < OppImplVersionTag::kVersionEnd) {
      return space_registries_v2_->at(static_cast<size_t>(opp_impl_version));
    }
    return nullptr;
  }
  const OpImplSpaceRegistryArrayV2Ptr &GetSpaceRegistries() const {
    return space_registries_v2_;
  }

  void SetFileConstantWeightDir(const std::string &file_constant_weight_dir) {
    file_constant_weight_dir_ = file_constant_weight_dir;
  }
  const string &GetFileConstantWeightDir() const {
    return file_constant_weight_dir_;
  }
  void SetOutputNodeName(const std::vector<std::string> &out_node_name) {
    out_node_name_ = out_node_name;
  }
  const std::vector<std::string> &GetOutputNodeName() const {
    return out_node_name_;
  }

 private:
  ModelDesc model_desc_;
  std::string file_constant_weight_dir_;
  OpImplSpaceRegistryArrayV2Ptr space_registries_v2_;
  std::vector<std::string> out_node_name_;
};

class VISIBILITY_EXPORT ModelConverter {
 public:
  struct Args {
    Args() = default;
    Args(const LoweringOption &opt, StreamAllocator *const strm_alloc, EventAllocator *const evnt_alloc,
         NotifyAllocator *const noti_alloc, const std::vector<ge::FileConstantMem> *const file_const_mems)
        : option(opt), stream_allocator(strm_alloc), event_allocator(evnt_alloc),
          notify_allocator(noti_alloc), file_constant_mems(file_const_mems){}
    LoweringOption option{};
    StreamAllocator *const stream_allocator = nullptr;
    EventAllocator *const event_allocator = nullptr;
    NotifyAllocator *const notify_allocator = nullptr;
    const std::vector<ge::FileConstantMem> *const file_constant_mems = nullptr;
  };
  ge::ExecuteGraphPtr ConvertGeModelToExecuteGraph(const ge::GeRootModelPtr &root_model, const Args &args);
  ge::ExecuteGraphPtr ConvertGeModelToExecuteGraph(const ge::GeRootModelPtr &root_model) {
    return ConvertGeModelToExecuteGraph(root_model, {});
  }
  ModelDescHolder &GetModelDescHolder() {
    return model_desc_holder_;
  }
 private:
  ge::graphStatus CreateModelDesc(const ge::GeRootModelPtr &root_model, StreamAllocator *const stream_allocator,
                                  EventAllocator *const event_allocator, NotifyAllocator *const notify_allocator);
  ModelDescHolder model_desc_holder_;
};
ge::graphStatus LoadSgtKernelBinToOpDesc(const ge::NodePtr &node, const ge::ComputeGraphPtr &graph,
                                         const ge::GeModelPtr &ge_model, const ge::ModelTaskType task_type);
} // gert
#endif  // AIR_CXX_RUNTIME_V2_LOWERING_MODEL_CONVERTER_H_
