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

#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/compute_graph.h"
#include "graph/op_kernel_bin.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/buffer.h"
#include "../fe_test_utils.h"
#include "securec.h"
#include "rt_error_codes.h"

#define protected public
#define private public
#include "common/fe_log.h"
#include "common/resource_def.h"
#include "common/large_bitmap.h"
#include "common/comm_error_codes.h"
#include "common/fe_inner_error_codes.h"
#include "adapter/adapter_itf/task_builder_adapter.h"
#undef private
#undef protected
#include "runtime/kernel.h"

using namespace std;
using namespace testing;
using namespace ge;
using namespace fe;

using NodePtr = std::shared_ptr<Node>;

namespace fe {
    class TestAdapter: public TaskBuilderAdapter {
    public:
        TestAdapter(const ge::Node& node, TaskBuilderContext& context)
                : TaskBuilderAdapter(node, context)
        {
        }
        ~TestAdapter()
        {
        }
        Status Run(domi::TaskDef &task_def) override
        {
            return fe::SUCCESS;
        }
        Status InitInput() override
        {
            return fe::SUCCESS;
        }
        void SetRunFunc(void *run_func)
        {
        }
    };
}

class STEST_TaskBuilderAdapter : public testing::Test
{
protected:
    void SetUp()
    {
        rtContext_t rtContext;
        assert(rtCtxCreate(&rtContext, RT_CTX_GEN_MODE, 0) == ACL_RT_SUCCESS);
        assert(rtCtxSetCurrent(rtContext) == ACL_RT_SUCCESS);


        node_ = CreateNode();
        context_ = CreateContext();
        adapter_ = shared_ptr<TestAdapter> (new (nothrow) TestAdapter(*node_, context_));
    }

    void TearDown()
    {
        adapter_.reset();
        DestroyContext(context_);
        node_.reset();

        rtContext_t rtContext;
        assert(rtCtxGetCurrent(&rtContext) == ACL_RT_SUCCESS);
        assert(rtCtxDestroy(rtContext) == ACL_RT_SUCCESS);


    }

    static NodePtr CreateNodeWithoutAttrs(bool has_weight = false)
    {
        FeTestOpDescBuilder builder;
        builder.SetName("test_tvm");
        builder.SetInputs({1});
        builder.SetOutputs({1});
        builder.AddInputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT);
        builder.AddOutputDesc({1,1,1,1}, ge::FORMAT_NCHW, ge::DT_FLOAT);
        if (has_weight) {
            size_t len = 10;
            unique_ptr<float[]> buf(new float[len]);
            auto weight = builder.AddWeight((uint8_t*) buf.get(), len * sizeof(float), { 1, 1, 2, 5 },
                    ge::FORMAT_NCHW, ge::DT_FLOAT);
            ge::TensorUtils::SetWeightSize(weight->MutableTensorDesc(), len * sizeof(float));
        }

        return builder.Finish();
    }

    static NodePtr CreateNode(bool has_weight = false)
    {
        NodePtr node = CreateNodeWithoutAttrs(has_weight);

        const char tbe_bin[] = "tbe_bin";
        vector<char> buffer(tbe_bin, tbe_bin+strlen(tbe_bin));
        OpKernelBinPtr tbe_kernel_ptr = std::make_shared<OpKernelBin>(node->GetName(), std::move(buffer));
        node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);

        ge::AttrUtils::SetStr(node->GetOpDesc(), "_kernelname", "my_kernel");
        ge::AttrUtils::SetInt(node->GetOpDesc(), "tvm_blockdim", 10);
        ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
        ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_metadata", "my_metadata");
        vector<int64_t> input_l1_flag = {1, 1, 1, 1};
        ge::AttrUtils::SetListInt(node->GetOpDesc(), ge::ATTR_NAME_OP_INPUT_L1_FLAG, input_l1_flag);
        vector<int64_t> input_l1_addr = {1, 2, 3, 4};
        ge::AttrUtils::SetListInt(node->GetOpDesc(), ge::ATTR_NAME_OP_INPUT_L1_ADDR, input_l1_addr);
        return node;
    }

    static TaskBuilderContext CreateContext()
    {
        assert(fe::SetFunctionState(FuncParamType::FUSION_L2, false));

        TaskBuilderContext context;
        context.dataMemSize = 100;
        context.dataMemBase = (uint8_t *) (intptr_t) 1000;
        context.weightMemSize = 200;
        context.weightMemBase = (uint8_t *) (intptr_t) 1100;
        context.weightBufferHost = Buffer(20);

        return context;
    }

    static void DestroyContext(TaskBuilderContext& context)
    {
    }

protected:
    NodePtr node_ { nullptr };
    TaskBuilderContext context_;
    std::shared_ptr<TestAdapter> adapter_;
};

TEST_F(STEST_TaskBuilderAdapter, case_all_success)
{
    EXPECT_EQ(adapter_->Init(), fe::SUCCESS);
}

TEST_F(STEST_TaskBuilderAdapter, case_has_weight_success)
{
    NodePtr node = CreateNode(true);
    TestAdapter adapter(*node, context_);

    EXPECT_EQ(adapter.Init(), fe::SUCCESS);
}

TEST_F(STEST_TaskBuilderAdapter, case_op_desc_error)
{
    adapter_->op_desc_ = nullptr;
    EXPECT_NE(adapter_->Init(), fe::SUCCESS);
}

TEST_F(STEST_TaskBuilderAdapter, case_Init_output_Output_offset_error)
{
    adapter_->op_desc_->SetOutputOffset({});
    EXPECT_EQ(adapter_->InitOutput(), fe::SUCCESS);
}

TEST_F(STEST_TaskBuilderAdapter, case_Init_output_option_output)
{
    adapter_->op_desc_->SetOutputOffset({});
    auto output_desc_ptr = adapter_->op_desc_->MutableOutputDesc(0);
    (void)ge::AttrUtils::SetInt(output_desc_ptr, "_memory_size_calc_type", 1);
    (void)ge::AttrUtils::SetStr(adapter_->op_desc_, "optionalOutputMode", "gen_placeholder");
    EXPECT_EQ(adapter_->InitOutput(), fe::SUCCESS);
}

TEST_F(STEST_TaskBuilderAdapter, case_Init_workspace_workspace_size_error)
{
    adapter_->op_desc_->SetWorkspaceBytes({1,2});
    adapter_->op_desc_->SetWorkspace({});
    EXPECT_NE(adapter_->InitWorkspace(), fe::SUCCESS);
}

TEST_F(STEST_TaskBuilderAdapter, case_Init_workspace_workspace_size_error2)
{
    adapter_->op_desc_->SetWorkspaceBytes({1,2});
    adapter_->op_desc_->SetWorkspace({1, 1});
    std::vector<int64_t> mem_type_list = {17, 17};
    ge::AttrUtils::SetListInt(adapter_->op_desc_, ge::ATTR_NAME_WORKSPACE_TYPE_LIST, mem_type_list);
    EXPECT_NE(adapter_->InitWorkspace(), fe::SUCCESS);
}

TEST_F(STEST_TaskBuilderAdapter, case_Init_workspace_workspace_size_error3)
{
    adapter_->op_desc_->SetWorkspaceBytes({1,2});
    adapter_->op_desc_->SetWorkspace({1, 1});
    adapter_->context_.mem_type_to_data_mem_size.insert(std::make_pair(17, 1));
    std::vector<int64_t> mem_type_list = {17, 17};
    ge::AttrUtils::SetListInt(adapter_->op_desc_, ge::ATTR_NAME_WORKSPACE_TYPE_LIST, mem_type_list);
    EXPECT_NE(adapter_->InitWorkspace(), fe::SUCCESS);
}

TEST_F(STEST_TaskBuilderAdapter, case_Init_workspace_success)
{
    adapter_->op_desc_->SetWorkspaceBytes({1, 1});
    adapter_->op_desc_->SetWorkspace({1, 1});
    EXPECT_EQ(adapter_->InitWorkspace(), fe::SUCCESS);
}

TEST_F(STEST_TaskBuilderAdapter, case_init_input_l1_addr_1)
{
    adapter_->input_l1_addrs_.clear();
    adapter_->op_desc_->DelAttr(ge::ATTR_NAME_OP_INPUT_L1_FLAG);
    adapter_->op_desc_->DelAttr(ge::ATTR_NAME_OP_INPUT_L1_ADDR);
    EXPECT_EQ(adapter_->InitInputL1Addr(), fe::SUCCESS);
    EXPECT_EQ(adapter_->input_l1_addrs_.size(), 0);

    adapter_->input_l1_addrs_.clear();
    vector<int64_t> input_l1_flag = {1, 1, 1, 1};
    ge::AttrUtils::SetListInt(adapter_->op_desc_, ge::ATTR_NAME_OP_INPUT_L1_FLAG, input_l1_flag);
    EXPECT_EQ(adapter_->InitInputL1Addr(), fe::SUCCESS);
    EXPECT_EQ(adapter_->input_l1_addrs_.size(), 0);

    adapter_->input_l1_addrs_.clear();
    vector<int64_t> input_l1_addr = {};
    ge::AttrUtils::SetListInt(adapter_->op_desc_, ge::ATTR_NAME_OP_INPUT_L1_ADDR, input_l1_addr);
    EXPECT_EQ(adapter_->InitInputL1Addr(), fe::SUCCESS);
    EXPECT_EQ(adapter_->input_l1_addrs_.size(), 0);

    adapter_->input_l1_addrs_.clear();
    input_l1_addr = {1, 2, 3};
    ge::AttrUtils::SetListInt(adapter_->op_desc_, ge::ATTR_NAME_OP_INPUT_L1_ADDR, input_l1_addr);
    EXPECT_EQ(adapter_->InitInputL1Addr(), fe::SUCCESS);
    EXPECT_EQ(adapter_->input_l1_addrs_.size(), 0);

    adapter_->input_l1_addrs_.clear();
    input_l1_addr = {1, 2, 3, 4};
    ge::AttrUtils::SetListInt(adapter_->op_desc_, ge::ATTR_NAME_OP_INPUT_L1_ADDR, input_l1_addr);
    EXPECT_EQ(adapter_->InitInputL1Addr(), fe::SUCCESS);
    EXPECT_EQ(adapter_->input_l1_addrs_.size(), 4);
}

TEST_F(STEST_TaskBuilderAdapter, case_Check_offset_and_size_data_mem_base_eq_weight_mem_base)
{
    adapter_->context_.dataMemBase = adapter_->context_.weightMemBase;
    EXPECT_EQ(adapter_->CheckOffsetAndSize(0, 1, 10), fe::SUCCESS);
}

TEST_F(STEST_TaskBuilderAdapter, case_Check_offset_and_size_offset_error)
{
    EXPECT_NE(adapter_->CheckOffsetAndSize(-1, 1, 10), fe::SUCCESS);
}

TEST_F(STEST_TaskBuilderAdapter, case_Check_offset_and_size_total_size_error)
{
    EXPECT_NE(adapter_->CheckOffsetAndSize(0, 1, 0), fe::SUCCESS);
}

TEST_F(STEST_TaskBuilderAdapter, set_func_state_test) {
  FuncParamType type = FuncParamType::MAX_NUM;
  bool isopen = false;
  bool ret = fe::SetFunctionState(type, false);
  EXPECT_EQ(ret, false);
}

TEST_F(STEST_TaskBuilderAdapter, get_bit_test) {
  size_t index = 0;
  auto largebitmap = fe::LargeBitMap(index);
  bool ret = largebitmap.GetBit(index);
  EXPECT_EQ(ret, false);
}

TEST_F(STEST_TaskBuilderAdapter, feed_args_info)
{
    adapter_->FeedArgsInfo(domi::ArgsInfo_ArgsType_OUTPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0U);
    EXPECT_EQ(adapter_->kernel_args_info_.size(), 1);
}
