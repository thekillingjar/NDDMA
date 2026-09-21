/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "lowering/tiling_parse_context_builder.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "platform/platform_infos_def.h"
#include "graph_metadef/common/ge_common/util.h"
#include "register/op_impl_registry.h"
#include "faker/node_faker.h"

namespace gert {
class TilingParseContextBuilderUT : public testing::Test {};

TEST_F(TilingParseContextBuilderUT, CompileInfoNullptr) {
  fe::PlatFormInfos platform_infos;
  auto builder = TilingParseContextBuilder();

  auto node = ComputeNodeFaker().NameAndType("bar", "Bar").IoNum(2, 1).InputNames({"x", "y"}).Build();
  auto op = ge::OpDescUtils::CreateOperatorFromNode(node->shared_from_this());
  auto holder = builder.CompileJson(nullptr).PlatformInfo(&platform_infos).Build(op);
  EXPECT_EQ(holder.context_, nullptr);
}

TEST_F(TilingParseContextBuilderUT, PlatformInfosNullptr) {
  std::string op_compile_info_json = "{}";
  fe::PlatFormInfos platform_infos;
  auto builder = TilingParseContextBuilder();

  auto node = ComputeNodeFaker().NameAndType("bar", "Bar").IoNum(2, 1).InputNames({"x", "y"}).Build();
  auto op = ge::OpDescUtils::CreateOperatorFromNode(node->shared_from_this());
  auto holder = builder.CompileJson(op_compile_info_json.c_str()).PlatformInfo(nullptr).Build(op);
  EXPECT_EQ(holder.context_, nullptr);
}

TEST_F(TilingParseContextBuilderUT, TilingFuncNullptr) {
  std::string op_compile_info_json = "{}";
  fe::PlatFormInfos platform_infos;
  auto builder = TilingParseContextBuilder();

  // construct op
  auto node = ComputeNodeFaker().NameAndType("bar", "Bar").IoNum(2, 1).InputNames({"x", "y"}).Build();
  ge::OpDescPtr op_desc = node->GetOpDesc();
  ge::GeTensorDesc tensor_desc(ge::GeShape({1}));
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddInputDesc("y", tensor_desc);
  op_desc->AddOutputDesc("z", tensor_desc);
  op_desc->MutableInputDesc(1)->SetDataType(ge::DT_INT32);
  op_desc->MutableInputDesc(1)->SetShape(ge::GeShape({1}));
  op_desc->MutableInputDesc(1)->SetOriginShape(ge::GeShape({1}));
  auto op = ge::OpDescUtils::CreateOperatorFromNode(node->shared_from_this());

  auto holder = builder.CompileJson(op_compile_info_json.c_str()).PlatformInfo(&platform_infos).Build(op);
  EXPECT_NE(holder.context_, nullptr);
}

TEST_F(TilingParseContextBuilderUT, BuildSuccess) {
  std::string op_compile_info_json = "{123}";
  fe::PlatFormInfos platform_infos;
  auto builder = TilingParseContextBuilder();

  // construct op
  auto node = ComputeNodeFaker().NameAndType("bar", "Bar").IoNum(2, 1).InputNames({"x", "y"}).Build();
  ge::OpDescPtr op_desc = node->GetOpDesc();
  ge::GeTensorDesc tensor_desc(ge::GeShape({1}));
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddInputDesc("y", tensor_desc);
  op_desc->AddOutputDesc("z", tensor_desc);
  op_desc->MutableInputDesc(1)->SetDataType(ge::DT_INT32);
  op_desc->MutableInputDesc(1)->SetShape(ge::GeShape({1}));
  op_desc->MutableInputDesc(1)->SetOriginShape(ge::GeShape({1}));
  auto op = ge::OpDescUtils::CreateOperatorFromNode(node->shared_from_this());

  OpImplRegisterV2::CompileInfoCreatorFunc create_func = []() -> void * { return new int32_t(0); };

  OpImplRegisterV2::CompileInfoDeleterFunc delete_func = [](void *ptr) { delete reinterpret_cast<int32_t *>(ptr); };

  auto holder = builder.CompileJson(op_compile_info_json.c_str())
                    .PlatformInfo(&platform_infos)
                    .CompileInfoCreatorFunc(create_func)
                    .CompileInfoDeleterFunc(delete_func)
                    .Build(op);
  EXPECT_NE(holder.GetKernelContext(), nullptr);
  auto tiling_parse_context = reinterpret_cast<TilingParseContext *>(holder.context_);
  EXPECT_NE(tiling_parse_context->GetCompiledInfo<int32_t>(), nullptr);
  EXPECT_NE(tiling_parse_context->GetPlatformInfo(), nullptr);
  EXPECT_EQ(*tiling_parse_context->GetCompiledInfo<int32_t>(), 0);
  EXPECT_STREQ(tiling_parse_context->GetCompiledJson(), "{123}");
}
}  // namespace gert
