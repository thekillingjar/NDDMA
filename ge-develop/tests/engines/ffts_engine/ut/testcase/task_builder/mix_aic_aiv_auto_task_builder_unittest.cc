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

using MixAICAIVAutoTaskBuilderPtr = shared_ptr<MixAICAIVAutoTaskBuilder>;
using MixAICAIVDynamicTaskBuilderPtr = shared_ptr<MixAICAIVDynamicTaskBuilder>;
class FFTSPlusMixAICAIVAutoTaskBuilderUTest : public testing::Test
{
protected:
	void SetUp()
	{
		mix_aic_aiv_auto_task_builder_ptr_ = make_shared<MixAICAIVAutoTaskBuilder>();
    mix_aic_aiv_dynamic_task_builder_ptr_ = make_shared<MixAICAIVDynamicTaskBuilder>();
		ffts_plus_def_ptr_ = new domi::FftsPlusTaskDef;
		node_ = CreateNode();
	}
	void TearDown() {
		delete ffts_plus_def_ptr_;
	}
	static void SetOpDecSize(NodePtr& node) {
		OpDesc::Vistor<GeTensorDesc> tensors = node->GetOpDesc()->GetAllInputsDesc();
		for (int i = 0; i < node->GetOpDesc()->GetAllInputsDesc().size(); i++) {
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
		builder.AddInputDesc({ 2, 4, 4, 4 }, ge::FORMAT_NCHW, ge::DT_FLOAT);
		builder.AddOutputDesc({ 2, 4, 4, 4 }, ge::FORMAT_NCHW, ge::DT_FLOAT);
		auto node = builder.Finish();

		const char tbeBin[] = "tbe_bin";
		vector<char> buffer(tbeBin, tbeBin + strlen(tbeBin));
		OpKernelBinPtr tbeKernelPtr = std::make_shared<OpKernelBin>("test_tvm", std::move(buffer));
		node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbeKernelPtr);
		ge::AttrUtils::SetInt(node->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
		ge::AttrUtils::SetInt(node->GetOpDesc(), "_fe_imply_type", 2);
		ge::AttrUtils::SetStr(node->GetOpDesc(), "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
		ge::AttrUtils::SetBool(node->GetOpDesc(), "is_first_node", true);
		ge::AttrUtils::SetBool(node->GetOpDesc(), "is_last_node", true);
		ge::AttrUtils::SetStr(node->GetOpDesc(), "_kernelname", "kernelname");

    vector<string> thread_core_type = {"MIX_AIC", "MIX_AIC"};
    (void)ge::AttrUtils::SetListStr(node->GetOpDesc(), "_thread_cube_vector_core_type", thread_core_type);
		vector<uint32_t> context_id_list = {3, 4};
		ge::AttrUtils::SetListInt(node->GetOpDesc(), kAutoCtxIdList, context_id_list);

		SetOpDecSize(node);
		ThreadSliceMapPtr tsmp_ptr = make_shared<ThreadSliceMap>();
		tsmp_ptr->slice_instance_num = 1;
    tsmp_ptr->parallel_window_size = 1;
    tsmp_ptr->thread_mode = 1;

    DimRange dim_rang;
		dim_rang.higher = 3;
		dim_rang.lower = 0;
		vector<DimRange> input_tensor_slice_v;
		input_tensor_slice_v.push_back(dim_rang);
		dim_rang.higher = 3;
		dim_rang.lower = 0;
		input_tensor_slice_v.push_back(dim_rang);
		input_tensor_slice_v.push_back(dim_rang);
		input_tensor_slice_v.push_back(dim_rang);
		vector<vector<DimRange>> input_tensor_slice_vv;
		input_tensor_slice_vv.push_back(input_tensor_slice_v);
		vector<vector<vector<DimRange>>> input_tensor_slice_vvv = { input_tensor_slice_vv };
    vector<vector<vector<DimRange>>> output_tensor_slice = { input_tensor_slice_vv };
		tsmp_ptr->input_tensor_slice = input_tensor_slice_vvv;
		tsmp_ptr->output_tensor_slice = output_tensor_slice;
		node->GetOpDesc()->SetExtAttr("_sgt_struct_info", tsmp_ptr);
		return node;
	}

  static Status GenAutoMixAICAIVCtxDef(NodePtr node) {
	  auto op_desc = node->GetOpDesc();
    std::shared_ptr<domi::FftsPlusCtxDef> ffts_plus_def_ptr = make_shared<domi::FftsPlusCtxDef>();
    domi::FftsPlusMixAicAivCtxDef *mix_aic_aiv_ctx_def = ffts_plus_def_ptr->mutable_mix_aic_aiv_ctx();
    FFTS_CHECK_NOTNULL(mix_aic_aiv_ctx_def);

    mix_aic_aiv_ctx_def->set_prefetch_once_bitmap(0);
    mix_aic_aiv_ctx_def->set_prefetch_enable_bitmap(0);
    mix_aic_aiv_ctx_def->set_aten(ffts::kAutoMode);
    mix_aic_aiv_ctx_def->set_atm(ffts::kAutoMode);

    ThreadSliceMapPtr slice_info_ptr = nullptr;
    slice_info_ptr = op_desc->TryGetExtAttr(kAttrSgtStructInfo, slice_info_ptr);
    mix_aic_aiv_ctx_def->set_thread_dim(slice_info_ptr->slice_instance_num);

    vector<int32_t> block_dims;
    (void)ge::AttrUtils::GetListInt(op_desc, ge::TVM_ATTR_NAME_THREAD_BLOCKDIM, block_dims);
    if (block_dims.size() > 1) {
      mix_aic_aiv_ctx_def->set_non_tail_block_dim(static_cast<uint32_t>(block_dims[0]));
      mix_aic_aiv_ctx_def->set_tail_block_dim(static_cast<uint32_t>(block_dims[1]));
    }

    // generate _register_stub_func
    vector<string> unique_ids;
    string session_graph_id = "";
    if (ge::AttrUtils::GetStr(op_desc, ge::ATTR_NAME_SESSION_GRAPH_ID, session_graph_id) && !session_graph_id.empty()) {
      unique_ids.push_back(session_graph_id + "_" + op_desc->GetName() + "_0");
      unique_ids.push_back(session_graph_id + "_" + op_desc->GetName() + "_1");
    } else {
      unique_ids.push_back(op_desc->GetName() + "_0");
      unique_ids.push_back(op_desc->GetName() + "_1");
    }
    (void)ge::AttrUtils::SetListStr(op_desc, "_register_stub_func", unique_ids);

    uint32_t input_output_num = 2;
    mix_aic_aiv_ctx_def->set_input_output_count(input_output_num);
    //input
    mix_aic_aiv_ctx_def->add_task_addr(222);
    //output
    mix_aic_aiv_ctx_def->add_task_addr(333);

    mix_aic_aiv_ctx_def->set_ns(1);

    vector<uint32_t> task_ratio_list;
    (void)ge::AttrUtils::GetListInt(op_desc, kThreadTaskRadio, task_ratio_list);
    if (task_ratio_list.size() > 1) {
      mix_aic_aiv_ctx_def->set_non_tail_block_ratio_n(task_ratio_list[0]);
      mix_aic_aiv_ctx_def->set_tail_block_ratio_n(task_ratio_list[1]);
    }

    // modeInArgsFirstField
    uint32_t mode = 0;
    uint64_t modeArgs = 0;
    (void)ge::AttrUtils::GetInt(op_desc, kModeInArgsFirstField, mode);
    // mode == 1 indicates we need reserve 8 Bytes for the args beginning
    if (mode == 1) {
      mix_aic_aiv_ctx_def->add_task_addr(modeArgs);
    }

    for (auto &prefix : kMixPrefixs) {
      string attr_key_kernel_name = prefix + kThreadKernelName;
      string attr_kernel_name;
      (void)ge::AttrUtils::GetStr(op_desc, attr_key_kernel_name, attr_kernel_name);
      mix_aic_aiv_ctx_def->add_kernel_name(attr_kernel_name);
    }
    (void)ge::AttrUtils::SetInt(op_desc, ffts::kAttrAICoreCtxType, static_cast<int64_t>(ffts::TaskBuilderType::EN_TASK_TYPE_MIX_AIC_AIV_AUTO));
    (void)op_desc->SetExtAttr(ffts::kAttrAICoreCtxDef, ffts_plus_def_ptr);
    return ffts::SUCCESS;
  }
public:
	MixAICAIVAutoTaskBuilderPtr mix_aic_aiv_auto_task_builder_ptr_;
  MixAICAIVDynamicTaskBuilderPtr mix_aic_aiv_dynamic_task_builder_ptr_;
	domi::FftsPlusTaskDef *ffts_plus_def_ptr_;
	NodePtr node_{ nullptr };
};

TEST_F(FFTSPlusMixAICAIVAutoTaskBuilderUTest, Gen_Mix_AICAIV_AUTO_ContextDef_SUCCESS)
{
  (void)GenAutoMixAICAIVCtxDef(node_);
	Status ret = mix_aic_aiv_auto_task_builder_ptr_->GenContextDef(node_, ffts_plus_def_ptr_);
	EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusMixAICAIVAutoTaskBuilderUTest, Add_Additional_Args_SUCCESS)
{
  (void)GenAutoMixAICAIVCtxDef(node_);
	ge::OpDescPtr op_desc = node_->GetOpDesc();
	ge::AttrUtils::SetInt(op_desc, kModeInArgsFirstField, 1);
  size_t ctx_num = 2;
	Status ret = mix_aic_aiv_auto_task_builder_ptr_->AddAdditionalArgs(op_desc, ffts_plus_def_ptr_, ctx_num);
	EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusMixAICAIVAutoTaskBuilderUTest, GenMixAicAivDynamicContextDefSUCCESS)
{
  (void)GenAutoMixAICAIVCtxDef(node_);
  ge::OpDescPtr op_desc = node_->GetOpDesc();
  ge::AttrUtils::SetInt(op_desc, kModeInArgsFirstField, 1);
  Status ret = mix_aic_aiv_dynamic_task_builder_ptr_->GenContextDef(node_, ffts_plus_def_ptr_);
  EXPECT_EQ(ffts::SUCCESS, ret);
}
