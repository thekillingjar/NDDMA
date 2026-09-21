/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_UT_GE_RUNTIME_V2_MODEL_DESC_HOLDER_FAKER_H_
#define AIR_CXX_TESTS_UT_GE_RUNTIME_V2_MODEL_DESC_HOLDER_FAKER_H_
#include "lowering/model_converter.h"
#include "faker/space_registry_faker.h"

namespace gert {

class ModelDescHolderFaker {
 public:
  ModelDescHolder Build(int64_t stream_num = 1, int64_t event_num = 0, int64_t notify_num = 0,
                        int64_t attached_stream_num = 0) {
    ModelDescHolder model_desc_holder;
    model_desc_holder.MutableModelDesc().SetReusableStreamNum(stream_num);
    model_desc_holder.MutableModelDesc().SetReusableEventNum(event_num);
    model_desc_holder.MutableModelDesc().SetReusableNotifyNum(notify_num);
    model_desc_holder.MutableModelDesc().SetAttachedStreamNum(attached_stream_num);
    auto space_registry = DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    if (space_registry == nullptr) {
      space_registry = SpaceRegistryFaker().Build();
    }
    model_desc_holder.SetSpaceRegistry(space_registry);
    return model_desc_holder;
  }
};
}  // namespace gert
#endif  //AIR_CXX_TESTS_UT_GE_RUNTIME_V2_MODEL_DESC_HOLDER_FAKER_H_
