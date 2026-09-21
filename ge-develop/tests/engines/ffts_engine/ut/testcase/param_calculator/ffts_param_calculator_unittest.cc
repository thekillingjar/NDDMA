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
 #include "inc/ffts_error_codes.h"
 #include "inc/ffts_type.h"
 #include "graph/node.h"
 #include "graph_builder_utils.h"
 #include "graph/compute_graph.h"
 #include "graph/op_kernel_bin.h"
 #include "runtime/context.h"
 #include "runtime/stream.h"
 #include "runtime/rt_model.h"
 #include "runtime/kernel.h"
 #include "runtime/mem.h"

 #include "inc/ffts_param_calculator.h"
 #include "inc/ffts_log.h"
 #include "inc/ffts_utils.h"
 #include "inc/memory_slice.h"
 #include "graph/utils/node_utils.h"
 #include "graph/debug/ge_attr_define.h"
 #include "graph/utils/op_desc_utils.h"
 #undef private
 #undef protected
 #include "graph/utils/graph_utils.h"
 #define private public
 #define protected public
 #include "graph/utils/tensor_utils.h"
 #include "../test_utils.h"

 using namespace std;
 using namespace fe;
 using namespace ffts;
 using namespace ge;
 
 class UTEST_FftsCalcParamTest : public testing::Test
 {
 protected:
    void SetUp()
    {
        node_ = CreateNode();
        nodedy_ = CreateDyNode();
        slice_info_ptr_ = node_->GetOpDesc()->TryGetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr_);
    }
    void TearDown() {
        cout << "a test Tear Down" << endl;
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
        builder.AddInputDesc({ -1, 4 }, ge::FORMAT_NCHW, ge::DT_FLOAT6_E3M2);
        builder.AddOutputDesc({ -1, 4 }, ge::FORMAT_NCHW, ge::DT_FLOAT6_E3M2);
        auto node = builder.Finish();
        ffts::ThreadSliceMapPtr slice_info_ptr = make_shared<ffts::ThreadSliceMap>();
        SetSliceInfo(slice_info_ptr);
        node->GetOpDesc()->SetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
        return node;
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
 public:
    ge::NodePtr node_{ nullptr };
    ge::NodePtr nodedy_{ nullptr };
    ffts::ThreadSliceMapPtr slice_info_ptr_{ nullptr };
 };
 
TEST_F(UTEST_FftsCalcParamTest, CalcParamAutoThreadInput)
{
    vector<vector<vector<int64_t>>> tensor_slice;
    vector<uint64_t> task_addr_offset;
    std::vector<uint32_t> input_tensor_indexes;
    ge::NodePtr node{ nullptr };
    Status ret = ParamCalculator::CalcAutoThreadInput(node, tensor_slice, task_addr_offset, input_tensor_indexes);
    EXPECT_EQ(ffts::FAILED, ret);
    ffts::ParamCalculator::ConvertTensorSlice(slice_info_ptr_->output_tensor_slice, tensor_slice);
    ret = ParamCalculator::CalcAutoThreadInput(node_, tensor_slice, task_addr_offset, input_tensor_indexes);
    EXPECT_EQ(ffts::SUCCESS, ret);
    ffts::ParamCalculator::ConvertTensorSlice(slice_info_ptr_->output_tensor_slice, tensor_slice);
    input_tensor_indexes.push_back((uint32_t)1);
    input_tensor_indexes.push_back((uint32_t)0);
    ret = ParamCalculator::CalcAutoThreadInput(node_, tensor_slice, task_addr_offset, input_tensor_indexes);
    EXPECT_EQ(ffts::SUCCESS, ret);
    ffts::ParamCalculator::ConvertTensorSlice(slice_info_ptr_->output_tensor_slice, tensor_slice);
    input_tensor_indexes.push_back((uint32_t)1);
    input_tensor_indexes.push_back((uint32_t)0);
    ret = ParamCalculator::CalcAutoThreadInput(nodedy_, tensor_slice, task_addr_offset, input_tensor_indexes);
    EXPECT_EQ(ffts::SUCCESS, ret);
    /* tensor_slice.clear();
    vector<int64_t> slice0 = {(int64_t)0};
    vector<vector<int64_t>> slice1 = {slice0};
    tensor_slice.emplace_back(slice1);
    ret = ParamCalculator::CalcAutoThreadInput(node, tensor_slice, task_addr_offset, input_tensor_indexes);
    EXPECT_EQ(ffts::FAILED, ret); */
}

TEST_F(UTEST_FftsCalcParamTest, CalcParamAutoThreadOutput)
{
    vector<vector<vector<int64_t>>> tensor_slice;
    vector<uint64_t> task_addr_offset;
    std::vector<uint32_t> input_tensor_indexes;
    ge::NodePtr node{ nullptr };
    Status ret = ParamCalculator::CalcAutoThreadOutput(node, tensor_slice, task_addr_offset, input_tensor_indexes);
    EXPECT_EQ(ffts::FAILED, ret);
    ffts::ParamCalculator::ConvertTensorSlice(slice_info_ptr_->output_tensor_slice, tensor_slice);
    ret = ParamCalculator::CalcAutoThreadOutput(node_, tensor_slice, task_addr_offset, input_tensor_indexes);
    EXPECT_EQ(ffts::SUCCESS, ret);
    ffts::ParamCalculator::ConvertTensorSlice(slice_info_ptr_->output_tensor_slice, tensor_slice);
    input_tensor_indexes.push_back((uint32_t)1);
    input_tensor_indexes.push_back((uint32_t)0);
    ret = ParamCalculator::CalcAutoThreadOutput(node_, tensor_slice, task_addr_offset, input_tensor_indexes);
    EXPECT_EQ(ffts::SUCCESS, ret);
}

 