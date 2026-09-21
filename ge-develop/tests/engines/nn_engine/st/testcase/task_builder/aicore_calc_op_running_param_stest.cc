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
#include <nlohmann/json.hpp>
#include <iostream>

#include <list>

#define private public
#define protected public
#include "common/sgt_slice_type.h"
#include "inc/ffts_log.h"
#include "inc/ffts_error_codes.h"
#include "inc/ffts_type.h"
#include "ops_kernel_builder/aicore_ops_kernel_builder.h"
#include "graph/node.h"
#include "graph_builder_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/compute_graph.h"
#include "graph/op_kernel_bin.h"
#include "graph/debug/ge_attr_define.h"
#include "runtime/context.h"
#include "runtime/stream.h"
#include "runtime/rt_model.h"
#include "runtime/kernel.h"
#include "runtime/mem.h"
#include "../fe_test_utils.h"
#include "base/registry/op_impl_space_registry_v2.h"
using namespace std;
using namespace fe;
using namespace ffts;
using namespace ge;

class STEST_AiCoreCalcOpRunningParamTest : public testing::Test
{
protected:
	void SetUp()
	{
		node_ = CreateNode();
    nodedy_ = CreateDyNode();
    slice_info_ptr_ = node_->GetOpDesc()->TryGetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr_);
    aicore_ops_kernel_builder_ptr_ = make_shared<fe::AICoreOpsKernelBuilder>();
    auto space_registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
    gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);
	}
	void TearDown() {

	}
  static void SetSliceInfo(ffts::ThreadSliceMapPtr &slice_info_ptr) {
    slice_info_ptr->slice_instance_num = 2;
    slice_info_ptr->thread_mode = 1;
    slice_info_ptr->input_tensor_slice.clear();
    slice_info_ptr->output_tensor_slice.clear();

    vector<vector<DimRange>> range_achor;
    vector<DimRange> shape;
    DimRange range;
    range.lower = 0;
    range.higher = 1;
    shape.emplace_back(range);

    range.lower = 0;
    range.higher = 4;
    shape.emplace_back(range);
    range_achor.emplace_back(shape);
    slice_info_ptr->input_tensor_slice.emplace_back(range_achor);
    slice_info_ptr->output_tensor_slice.emplace_back(range_achor);

    shape.clear();
    range_achor.clear();
    range.lower = 1;
    range.higher = 2;
    shape.emplace_back(range);

    range.lower = 0;
    range.higher = 4;
    shape.emplace_back(range);
    range_achor.emplace_back(shape);
    slice_info_ptr->input_tensor_slice.emplace_back(range_achor);
    slice_info_ptr->output_tensor_slice.emplace_back(range_achor);
    slice_info_ptr->input_tensor_indexes.emplace_back(0);
    slice_info_ptr->output_tensor_indexes.emplace_back(0);
  }
  static NodePtr CreateNode()
	{
		FeTestOpDescBuilder builder;
		builder.SetName("test_tvm_");
		builder.SetType("conv");
		builder.SetInputs({ 1 });
		builder.SetOutputs({ 1 });
		builder.AddInputDesc({ 2, 4 }, ge::FORMAT_NCHW, ge::DT_FLOAT);
		builder.AddOutputDesc({ 2, 4 }, ge::FORMAT_NCHW, ge::DT_FLOAT);
		auto node = builder.Finish();

    ffts::ThreadSliceMapPtr slice_info_ptr = make_shared<ffts::ThreadSliceMap>();
		SetSliceInfo(slice_info_ptr);
		node->GetOpDesc()->SetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
    return node;
  }
  static NodePtr CreateDyNode()
	{
		FeTestOpDescBuilder builder;
		builder.SetName("test_tvm_dy");
		builder.SetType("conv");
		builder.SetInputs({ 1 });
		builder.SetOutputs({ 1 });
		builder.AddInputDesc({ -1, 4 }, ge::FORMAT_NCHW, ge::DT_FLOAT);
		builder.AddOutputDesc({ -1, 4 }, ge::FORMAT_NCHW, ge::DT_FLOAT);
		auto node = builder.Finish();
    ffts::ThreadSliceMapPtr slice_info_ptr = make_shared<ffts::ThreadSliceMap>();
		SetSliceInfo(slice_info_ptr);
		node->GetOpDesc()->SetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
    return node;
  }
public:
	ge::NodePtr node_{ nullptr };
  ge::NodePtr nodedy_{ nullptr };
	ffts::ThreadSliceMapPtr slice_info_ptr_{ nullptr };
  std::shared_ptr<fe::AICoreOpsKernelBuilder> aicore_ops_kernel_builder_ptr_{ nullptr };
};

TEST_F(STEST_AiCoreCalcOpRunningParamTest, CalcOpRunningParamSuccess)
{
  Status ret = aicore_ops_kernel_builder_ptr_->CalcOpRunningParam(*node_);
	EXPECT_EQ(ffts::SUCCESS, ret);

  for (auto const &anchor : node_->GetAllOutDataAnchors()) {
    const uint32_t anchor_index = static_cast<uint32_t>(anchor->GetIdx());
    const ge::GeTensorDescPtr &tensor_desc_ptr = node_->GetOpDesc()->MutableOutputDesc(anchor_index);
    EXPECT_NE(tensor_desc_ptr, nullptr);
    int64_t slice_tensor_size = 0;
    ge::AttrUtils::GetInt(tensor_desc_ptr, ge::ATTR_NAME_FFTS_SUB_TASK_TENSOR_SIZE, slice_tensor_size);
    EXPECT_EQ(16, slice_tensor_size);
  }
}

TEST_F(STEST_AiCoreCalcOpRunningParamTest, CalcOpRunningParamManualMode)
{
  slice_info_ptr_->thread_mode = 0;
  Status ret = aicore_ops_kernel_builder_ptr_->CalcOpRunningParam(*node_);
	EXPECT_EQ(ffts::SUCCESS, ret);

  for (auto const &anchor : node_->GetAllOutDataAnchors()) {
    const uint32_t anchor_index = static_cast<uint32_t>(anchor->GetIdx());
    const ge::GeTensorDescPtr &tensor_desc_ptr = node_->GetOpDesc()->MutableOutputDesc(anchor_index);
    EXPECT_NE(tensor_desc_ptr, nullptr);
    int64_t slice_tensor_size = 0;
    bool ret = ge::AttrUtils::GetInt(tensor_desc_ptr, ge::ATTR_NAME_FFTS_SUB_TASK_TENSOR_SIZE, slice_tensor_size);
    EXPECT_EQ(false, ret);
  }

  slice_info_ptr_->thread_mode = 1;
}

TEST_F(STEST_AiCoreCalcOpRunningParamTest, CalcOpRunningParamDynamicShape)
{
  Status ret = aicore_ops_kernel_builder_ptr_->CalcOpRunningParam(*nodedy_);
	EXPECT_EQ(ffts::SUCCESS, ret);

  for (auto const &anchor : node_->GetAllOutDataAnchors()) {
    const uint32_t anchor_index = static_cast<uint32_t>(anchor->GetIdx());
    const ge::GeTensorDescPtr &tensor_desc_ptr = node_->GetOpDesc()->MutableOutputDesc(anchor_index);
    EXPECT_NE(tensor_desc_ptr, nullptr);
    int64_t slice_tensor_size = 0;
    bool ret = ge::AttrUtils::GetInt(tensor_desc_ptr, ge::ATTR_NAME_FFTS_SUB_TASK_TENSOR_SIZE, slice_tensor_size);
    EXPECT_EQ(false, ret);
  }
}

TEST_F(STEST_AiCoreCalcOpRunningParamTest, CalcOpRunningParamSliceNumIs0Fail)
{
  slice_info_ptr_->slice_instance_num = 0;
  Status ret = aicore_ops_kernel_builder_ptr_->CalcOpRunningParam(*node_);
	EXPECT_EQ(ffts::FAILED, ret);
  slice_info_ptr_->slice_instance_num = 2;
}

TEST_F(STEST_AiCoreCalcOpRunningParamTest, CalcOpRunningParamSliceNullptrSuccess)
{
  ffts::ThreadSliceMapPtr slice_info_ptr = nullptr;
	node_->GetOpDesc()->SetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
  Status ret = aicore_ops_kernel_builder_ptr_->CalcOpRunningParam(*node_);
	EXPECT_EQ(ffts::SUCCESS, ret);

  node_->GetOpDesc()->SetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr_);
}

TEST_F(STEST_AiCoreCalcOpRunningParamTest, CalcOpRunningParamCalcOutputFail)
{
  slice_info_ptr_->output_tensor_slice.clear();
  Status ret = aicore_ops_kernel_builder_ptr_->CalcOpRunningParam(*node_);
	EXPECT_EQ(ffts::FAILED, ret);
  for (auto const &anchor : node_->GetAllOutDataAnchors()) {
    const uint32_t anchor_index = static_cast<uint32_t>(anchor->GetIdx());
    const ge::GeTensorDescPtr &tensor_desc_ptr = node_->GetOpDesc()->MutableOutputDesc(anchor_index);
    EXPECT_NE(tensor_desc_ptr, nullptr);
    int64_t slice_tensor_size = 0;
    bool ret = ge::AttrUtils::GetInt(tensor_desc_ptr, ge::ATTR_NAME_FFTS_SUB_TASK_TENSOR_SIZE, slice_tensor_size);
    EXPECT_EQ(false, ret);
  }
}

TEST_F(STEST_AiCoreCalcOpRunningParamTest, CalcOpRunningParamCalcInputFail)
{
  slice_info_ptr_->input_tensor_slice.clear();
  Status ret = aicore_ops_kernel_builder_ptr_->CalcOpRunningParam(*node_);
	EXPECT_EQ(ffts::SUCCESS, ret);
}
