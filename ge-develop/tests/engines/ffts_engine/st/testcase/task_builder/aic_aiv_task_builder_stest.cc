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
#include "task_builder/fftsplus_ops_kernel_builder.h"
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
#include "../test_utils.h"

using namespace std;
using namespace ffts;
using namespace ge;

#define SET_SIZE 1000

using AICAIVTaskBuilderPtr = shared_ptr<AICAIVTaskBuilder>;
//using FftsPlusTaskDefPtr = shared_ptr<domi::FftsPlusTaskDef>;
class FFTSPlusAICAIVTaskBuilderSTest : public testing::Test
{
protected:
	void SetUp()
	{
		aic_aiv_task_builder_ptr_ = make_shared<AICAIVTaskBuilder>();
    ffts_plus_def_ptr_ = make_shared<domi::FftsPlusCtxDef>();
	}
	void TearDown() {
	}
	static void SetOpDecSize(NodePtr& node) {
		OpDesc::Vistor<GeTensorDesc> tensors = node->GetOpDesc()->GetAllInputsDesc();
		for (int i = 0; i < node->GetOpDesc()->GetAllInputsSize(); i++) {
			ge::GeTensorDesc tensor = node->GetOpDesc()->GetAllInputsDesc().at(i);
			ge::TensorUtils::SetSize(tensor, SET_SIZE);
			node->GetOpDesc()->UpdateInputDesc(i, tensor);
		}
		OpDesc::Vistor<GeTensorDesc> tensorsOutput = node->GetOpDesc()->GetAllOutputsDesc();
		for (int i = 0; i < tensorsOutput.size(); i++) {
			ge::GeTensorDesc tensorOutput = tensorsOutput.at(i);
			ge::TensorUtils::SetSize(tensorOutput, SET_SIZE);
			node->GetOpDesc()->UpdateOutputDesc(i, tensorOutput);
		}
	}
	static NodePtr CreateNode()
	{
		FeTestOpDescBuilder builder;
		builder.SetName("test_tvm");
		builder.SetType("conv");
		builder.SetInputs({ 1 });
		builder.SetOutputs({ 1 });
		builder.AddInputDesc({ 2,4,4,4 }, ge::FORMAT_NCHW, ge::DT_FLOAT);
		builder.AddOutputDesc({ 2,4,4,4 }, ge::FORMAT_NCHW, ge::DT_FLOAT);
		auto node = builder.Finish();

		const char tbeBin[] = "tbe_bin";
		vector<char> buffer(tbeBin, tbeBin + strlen(tbeBin));
		OpKernelBinPtr tbeKernelPtr = std::make_shared<OpKernelBin>("test_tvm", std::move(buffer));
		node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbeKernelPtr);
		ge::AttrUtils::SetInt(node->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);

		ge::AttrUtils::SetInt(node->GetOpDesc(), "_fe_imply_type", (int64_t)2);
		ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
		ge::AttrUtils::SetBool(node->GetOpDesc(), "is_first_node", true);
		ge::AttrUtils::SetBool(node->GetOpDesc(), "is_last_node", true);
		ge::AttrUtils::SetStr(node->GetOpDesc(), "_kernelname", "kernelname");
    (void)ge::AttrUtils::SetStr(node->GetOpDesc(), "_cube_vector_core_type", "AIV");
		SetOpDecSize(node);
		ThreadSliceMapPtr tsmp_ptr = make_shared<ThreadSliceMap>();
		tsmp_ptr->slice_instance_num = 4;
		node->GetOpDesc()->SetExtAttr("_sgt_struct_info", tsmp_ptr);
		return node;
	}

  Status GenManualAICAIVCtxDef() {
    domi::FftsPlusAicAivCtxDef *aic_aiv_ctx_def = ffts_plus_def_ptr_->mutable_aic_aiv_ctx();
    FFTS_CHECK_NOTNULL(aic_aiv_ctx_def);

    // cache managemet will do at GenerateDataTaskDef()
    aic_aiv_ctx_def->set_prefetch_once_bitmap(0);
    aic_aiv_ctx_def->set_prefetch_enable_bitmap(0);
    aic_aiv_ctx_def->set_atm(ffts::kManualMode);
    aic_aiv_ctx_def->set_thread_dim(1);

    int32_t block_dim = 1;
    //(void) ge::AttrUtils::GetInt(op_desc, ge::TVM_ATTR_NAME_BLOCKDIM, block_dim);
    aic_aiv_ctx_def->set_tail_block_dim(static_cast<uint32_t>(block_dim));
    aic_aiv_ctx_def->set_non_tail_block_dim(static_cast<uint32_t>(block_dim));

    //input
    aic_aiv_ctx_def->add_task_addr(111);
    aic_aiv_ctx_def->add_task_addr(222);

    //output
    aic_aiv_ctx_def->add_task_addr(333);

    //workspace
    aic_aiv_ctx_def->add_task_addr(444);

    auto op_desc = node_->GetOpDesc();
    string attr_key_kernel_name = kKernelName;
    string attr_kernel_name;
    (void) ge::AttrUtils::GetStr(op_desc, attr_key_kernel_name, attr_kernel_name);
    aic_aiv_ctx_def->add_kernel_name(attr_kernel_name);
    (void)ge::AttrUtils::SetInt(op_desc, ffts::kAttrAICoreCtxType, static_cast<int64_t>(ffts::TaskBuilderType::EN_TASK_TYPE_AIC_AIV));
    (void)op_desc->SetExtAttr(ffts::kAttrAICoreCtxDef, ffts_plus_def_ptr_);
    return ffts::SUCCESS;
  }

public:
	AICAIVTaskBuilderPtr aic_aiv_task_builder_ptr_;
  std::shared_ptr<domi::FftsPlusCtxDef> ffts_plus_def_ptr_;
  NodePtr node_{ nullptr };

};

TEST_F(FFTSPlusAICAIVTaskBuilderSTest, GenContextDef_SUCCESS)
{
  node_ = CreateNode();
  (void)GenManualAICAIVCtxDef();
	domi::FftsPlusTaskDef *ffts_plus_def_ptr = new domi::FftsPlusTaskDef;
	Status ret = aic_aiv_task_builder_ptr_->GenContextDef(node_, ffts_plus_def_ptr);
	delete ffts_plus_def_ptr;
	EXPECT_EQ(ffts::SUCCESS, ret);
}
