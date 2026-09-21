/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_SAMPLE_DEPENDS_FAKER_KERNEL_CONTEXT_HOLDER_BUILDER_H_
#define ATT_SAMPLE_DEPENDS_FAKER_KERNEL_CONTEXT_HOLDER_BUILDER_H_
#include "exe_graph/lowering/tiling_context_builder.h"
#include "platform/platform_infos_def.h"
namespace att {
struct InOutput {
  ge::GeShape shape;
  ge::Format format;
  ge::DataType dtype;
  InOutput(ge::GeShape ge_shape, ge::Format ge_format, ge::DataType ge_datatype)
      : shape(ge_shape), format(ge_format), dtype(ge_datatype) {}
};

class KernelContextHolderBuilder {
 public:
  KernelContextHolderBuilder() = default;
  KernelContextHolderBuilder &AddInput(const InOutput &input);
  KernelContextHolderBuilder &AddOutput(const InOutput &output);
  KernelContextHolderBuilder &AddPrivateAtt(const std::pair<ge::AscendString, ge::AnyValue> &attr);
  KernelContextHolderBuilder &SetTilingData(const size_t size);
  KernelContextHolderBuilder &SetWorkSpace(const size_t size);
  KernelContextHolderBuilder &SetCompileInfo(const size_t size);
  KernelContextHolderBuilder &SetPlatformInfo();
  gert::KernelContextHolder Build();
 private:
  std::unique_ptr<gert::TilingContextBuilder> tiling_ctx_builder_;
  std::vector<InOutput> inputs_;
  std::vector<InOutput> outputs_;
  std::unique_ptr<uint8_t[]> tiling_data_;
  std::unique_ptr<uint8_t[]> workspace_size_;
  std::unique_ptr<uint8_t[]> compile_info_;
  std::unique_ptr<fe::PlatFormInfos> platform_info_;
  std::vector<std::pair<ge::AscendString, ge::AnyValue>> private_attrs_;
};
}
#endif // ATT_SAMPLE_DEPENDS_FAKER_KERNEL_CONTEXT_HOLDER_BUILDER_H_