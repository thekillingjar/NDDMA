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
#include "rt_error_codes.h"

using namespace std;
using namespace ffts;
using namespace ge;

#define SET_SIZE 1000

using DSAManualTaskBuilderPtr = shared_ptr<DSAManualTaskBuilder>;
class FFTSPlusDsaTaskBuilderSTest : public testing::Test
{
private:
  static RunContext CreateContext()
  {
    RunContext context;
    context.dataMemSize = 100;
    context.dataMemBase = (uint8_t *) (intptr_t) 1000;
    context.weightMemSize = 200;
    context.weightMemBase = (uint8_t *) (intptr_t) 1100;
    context.weightsBuffer = Buffer(20);

    return context;
  }

  OpDescPtr GreateOpDesc() {
    vector<int64_t> dims = {1, 2, 3, 4};
    GeShape shape(dims);
    shared_ptr<ge::GeTensorDesc> tensor_desc_ptr = make_shared<ge::GeTensorDesc>();
    tensor_desc_ptr->SetShape(shape);
    tensor_desc_ptr->SetDataType(ge::DT_FLOAT);
    tensor_desc_ptr->SetFormat(ge::FORMAT_NCHW);

    OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("test_op_desc", "conv");
    op_desc_ptr->SetId(123456);
    op_desc_ptr->AddInputDesc(tensor_desc_ptr->Clone());
    op_desc_ptr->AddOutputDesc(tensor_desc_ptr->Clone());

    return op_desc_ptr;
  }

  uint64_t GetSeed(uint32_t seed0, uint32_t seed1) {
    uint64_t seed_value = seed1;
    seed_value = seed_value << 32 | seed0;
    return seed_value;
  }

  bool IsSameAddr(const std::string& addr, int64_t offset) {
    uint64_t real_addr = *reinterpret_cast<const uint64_t*>(addr.c_str());
    uint64_t expect_addr = reinterpret_cast<const uint64_t>(context_.dataMemBase) + offset;
    return real_addr == expect_addr;
  }

  void ExpectSameAddr(const std::string& addr, int64_t offset) {
    uint64_t real_addr = *reinterpret_cast<const uint64_t*>(addr.c_str());
    uint64_t expect_addr = reinterpret_cast<const uint64_t>(context_.dataMemBase) + offset;
    EXPECT_EQ(real_addr, expect_addr);
  }

  bool IsSameValue(const std::string& addr, float value) {
    float real_addr = *reinterpret_cast<const float*>(addr.c_str());
    EXPECT_FLOAT_EQ(real_addr, value);
    return real_addr == value;
  }

  bool IsSameValue(const std::string& addr, int64_t value) {
    int64_t real_addr = *reinterpret_cast<const int64_t*>(addr.c_str());
    EXPECT_EQ(real_addr, value);
    return real_addr == value;
  }

  bool IsSameAddr(uint64_t addr, int64_t offset) {
    uint64_t real_addr = addr;
    uint64_t expect_addr = reinterpret_cast<const uint64_t>(context_.dataMemBase) + offset;
    return real_addr == expect_addr;
  }
  static void SetConstInputsValue(const ge::NodePtr &node) {
    auto op = ge::OpDescUtils::CreateOperatorFromNode(node);
    std::vector<std::string> value_list;
    auto input_desc_size = node->GetOpDesc()->GetAllInputsSize();
    for (size_t input_idx = 0; input_idx < input_desc_size; input_idx++) {
      std::string value;
      auto const_tensor = ge::OpDescUtils::GetInputConstData(op, input_idx);
      if (const_tensor == nullptr) {
        value_list.emplace_back(value);
        continue;
      }
      value.assign(reinterpret_cast<const char *>(const_tensor->GetData().GetData()), const_tensor->GetData().GetSize());
      value_list.emplace_back(value);
    }
    (void)ge::AttrUtils::SetListStr(node->GetOpDesc(), "_const_value_list", value_list);
  }
  static void SetWeight(ge::NodePtr node, int index, float w) {
    float data[] = {w};
    ge::GeTensorDesc tensor_desc(ge::GeShape({1}), ge::FORMAT_ND, ge::DT_FLOAT16);
    ge::GeTensorPtr tensor = std::make_shared<ge::GeTensor>(tensor_desc, (uint8_t *) data, sizeof(data));
    map<int, ge::GeTensorPtr> weights = {{index, tensor}};
    ge::OpDescUtils::SetWeights(*node, weights);
  }

  static void SetWeight(ge::NodePtr node, int index, int64_t w) {
    int64_t data[] = {w};
    ge::GeTensorDesc tensor_desc(ge::GeShape({1}), ge::FORMAT_ND, ge::DT_INT64);
    ge::GeTensorPtr tensor = std::make_shared<ge::GeTensor>(tensor_desc, (uint8_t *) data, sizeof(data));
    map<int, ge::GeTensorPtr> weights = {{index, tensor}};
    ge::OpDescUtils::SetWeights(*node, weights);
  }

 private:
  RunContext context_;
  const uint32_t DSA_INT32 = 0b000;
  const uint32_t DSA_INT64 = 0b001;
  const uint32_t DSA_UINT32 = 0b010;
  const uint32_t DSA_UINT64 = 0b011;
  const uint32_t DSA_BF16 = 0b100;
  const uint32_t DSA_FLOAT16 = 0b101;
  const uint32_t DSA_FLOAT = 0b110;

  const uint32_t DSA_BITMASK = 0b000;
  const uint32_t DSA_UNIFORM = 0b001;
  const uint32_t DSA_NORMAL = 0b010;
  const uint32_t DSA_TRUNCATED_NORMAL = 0b011;

  const uint32_t VLD_TRUNCATED_NORMAL = 0b11000;
  const uint32_t VLD_NORMAL = 0b11000;
  const uint32_t VLD_BITMASK = 0b00001;
  const uint32_t VLD_UNIFORM = 0b00110;

  const uint32_t DSA_DATA_VALUE = 0b1;
  const uint32_t DSA_DATA_ADDR = 0b0;

protected:
	void SetUp()
	{
		dsa_task_builder_ptr_ = make_shared<DSAManualTaskBuilder>();
	}
	void TearDown() {

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

	using RunContextPtr = std::shared_ptr<ge::RunContext>;
	

	ge::NodePtr CreateDSARandomNormalNode()
	{
		ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
		OpDescPtr src_op = std::make_shared<OpDesc>("DSARandomNormal", "DSARandomNormal");
		GeTensorDesc src_tensor_desc(GeShape({100, 2, 3, 512, 4}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
		src_tensor_desc.SetOriginShape(GeShape({10, 11, 12, 13}));
		src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
		src_op->AddOutputDesc(src_tensor_desc);
		src_op->AddInputDesc(src_tensor_desc);
		src_op->AddInputDesc(src_tensor_desc);
		src_op->AddInputDesc(src_tensor_desc);
		src_op->AddInputDesc(src_tensor_desc);
		src_op->SetWorkspace({1024, 2048});
		src_op->SetOutputOffset({2048+1024});
		src_op->SetInputOffset({0, 32, 64, 128});
		ge::AttrUtils::SetInt(src_op, "_context_id", 1);
		auto src_node = graph->AddNode(src_op);


		RunContextPtr contextptr = nullptr;
    FFTS_MAKE_SHARED(contextptr = std::make_shared<ge::RunContext>(CreateContext()), return nullptr);
		(void)src_op->SetExtAttr("_ffts_runtime_context", contextptr);
		std::shared_ptr<domi::FftsPlusCtxDef> ffts_plus_def_ptr = make_shared<domi::FftsPlusCtxDef>();
		domi::FftsPlusDsaCtxDef * dsa_ctx_def = ffts_plus_def_ptr->mutable_dsa_ctx();
		if (dsa_ctx_def == nullptr) {
			return nullptr;
		}
    (void)src_op->SetExtAttr(ffts::kAttrDsaCtxDef, ffts_plus_def_ptr);
		return src_node;
	}

	ge::NodePtr CreateDSARandomTruncatedNormalNode()
	{
		ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
		OpDescPtr src_op = std::make_shared<OpDesc>("DSARandomTruncatedNormal", "DSARandomTruncatedNormal");
		GeTensorDesc src_tensor_desc(GeShape({100, 2, 3, 512, 4}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
		src_tensor_desc.SetOriginShape(GeShape({10, 11, 12, 13}));
		src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
		src_op->AddOutputDesc(src_tensor_desc);
		src_op->AddInputDesc(src_tensor_desc);
		src_op->AddInputDesc(src_tensor_desc);
		src_op->AddInputDesc(src_tensor_desc);
		src_op->AddInputDesc(src_tensor_desc);
		src_op->SetWorkspace({1024, 2048});
		src_op->SetOutputOffset({2048+1024});
		src_op->SetInputOffset({0, 32, 64, 128});
		src_op->SetAttr("seed0", GeAttrValue::CreateFrom<GeAttrValue::INT>(1));
		src_op->SetAttr("seed1", GeAttrValue::CreateFrom<GeAttrValue::INT>(2));
		auto src_node = graph->AddNode(src_op);

		
		RunContextPtr contextptr = nullptr;
    FFTS_MAKE_SHARED(contextptr = std::make_shared<ge::RunContext>(CreateContext()), return nullptr);
		(void)src_op->SetExtAttr("_ffts_runtime_context", contextptr);
		std::shared_ptr<domi::FftsPlusCtxDef> ffts_plus_def_ptr = make_shared<domi::FftsPlusCtxDef>();
		domi::FftsPlusDsaCtxDef * dsa_ctx_def = ffts_plus_def_ptr->mutable_dsa_ctx();
		if (dsa_ctx_def == nullptr) {
			return nullptr;
		}
    (void)src_op->SetExtAttr(ffts::kAttrDsaCtxDef, ffts_plus_def_ptr);
		return src_node;
	}

	ge::NodePtr CreateDSARandomUniformNode()
	{
		ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
		OpDescPtr src_op = std::make_shared<OpDesc>("DSARandomUniform", "DSARandomUniform");
		GeTensorDesc src_tensor_desc(GeShape({100, 2, 3, 512, 4}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
		src_tensor_desc.SetOriginShape(GeShape({10, 11, 12, 13}));
		src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
		src_op->AddOutputDesc(src_tensor_desc);
		src_op->AddInputDesc(src_tensor_desc);
		src_op->AddInputDesc(src_tensor_desc);
		src_op->AddInputDesc(src_tensor_desc);
		src_op->AddInputDesc(src_tensor_desc);
		src_op->SetWorkspace({1024, 2048});
		src_op->SetOutputOffset({2048+1024});
		src_op->SetInputOffset({0, 32, 64, 128});
		auto src_node = graph->AddNode(src_op);

		
		RunContextPtr contextptr = nullptr;
    FFTS_MAKE_SHARED(contextptr = std::make_shared<ge::RunContext>(CreateContext()), return nullptr);
		(void)src_op->SetExtAttr("_ffts_runtime_context", contextptr);
		std::shared_ptr<domi::FftsPlusCtxDef> ffts_plus_def_ptr = make_shared<domi::FftsPlusCtxDef>();
		domi::FftsPlusDsaCtxDef * dsa_ctx_def = ffts_plus_def_ptr->mutable_dsa_ctx();
		if (dsa_ctx_def == nullptr) {
			return nullptr;
		}
    (void)src_op->SetExtAttr(ffts::kAttrDsaCtxDef, ffts_plus_def_ptr);
		return src_node;
	}

	ge::NodePtr CreateDSARandomUniformconstant0Node()
	{
		ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
		OpDescPtr src_op = std::make_shared<OpDesc>("DSARandomUniform", "DSARandomUniform");
		GeTensorDesc src_tensor_desc(GeShape({1}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
		src_tensor_desc.SetOriginShape(GeShape({1}));
		src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
		src_op->AddOutputDesc(src_tensor_desc);
		src_op->AddInputDesc(src_tensor_desc);
		src_op->AddInputDesc(src_tensor_desc);
		src_op->AddInputDesc(src_tensor_desc);
		src_op->AddInputDesc(src_tensor_desc);
		src_op->SetWorkspace({1024, 2048});
		src_op->SetOutputOffset({2048+1024});
		src_op->SetInputOffset({0, 32, 64, 128});
		auto src_node = graph->AddNode(src_op);
		SetWeight(src_node, 0, 256L);
		SetWeight(src_node, 1, 512L);
		SetWeight(src_node, 2, 0.12f);
		SetWeight(src_node, 3, 0.24f);
		SetConstInputsValue(src_node);
		
		RunContextPtr contextptr = nullptr;
    FFTS_MAKE_SHARED(contextptr = std::make_shared<ge::RunContext>(CreateContext()), return nullptr);
		(void)src_op->SetExtAttr("_ffts_runtime_context", contextptr);
		std::shared_ptr<domi::FftsPlusCtxDef> ffts_plus_def_ptr = make_shared<domi::FftsPlusCtxDef>();
		domi::FftsPlusDsaCtxDef * dsa_ctx_def = ffts_plus_def_ptr->mutable_dsa_ctx();
		if (dsa_ctx_def == nullptr) {
			return nullptr;
		}
    (void)src_op->SetExtAttr(ffts::kAttrDsaCtxDef, ffts_plus_def_ptr);
		return src_node;
	}

	ge::NodePtr CreateDSARandomUniformconstant1Node()
	{
		ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
		OpDescPtr random_uniform = std::make_shared<OpDesc>("DSARandomUniform", "DSARandomUniform");
		OpDescPtr count = std::make_shared<OpDesc>("count", "Data");
		OpDescPtr seed = std::make_shared<OpDesc>("seed", "Data");
		OpDescPtr low = std::make_shared<OpDesc>("low", "Const");
		OpDescPtr high = std::make_shared<OpDesc>("DSARandomUniform", "Data");

		GeTensorDesc input_desc(GeShape({1}), ge::FORMAT_ND, ge::DT_FLOAT);
		input_desc.SetOriginShape(GeShape({1}));
		input_desc.SetOriginFormat(ge::FORMAT_ND);
		GeTensorDesc seed_desc(GeShape({1}), ge::FORMAT_ND, ge::DT_INT32);
		seed_desc.SetOriginShape(GeShape({1}));
		seed_desc.SetOriginFormat(ge::FORMAT_ND);

		random_uniform->AddInputDesc("count", seed_desc);
		random_uniform->AddInputDesc("seed", seed_desc);
		random_uniform->AddInputDesc("low", input_desc);
		random_uniform->AddInputDesc("high", input_desc);
		random_uniform->AddOutputDesc(input_desc);

		count->AddOutputDesc(seed_desc);
		seed->AddOutputDesc(seed_desc);
		low->AddOutputDesc(input_desc);
		high->AddOutputDesc(input_desc);

		random_uniform->SetWorkspace({1024, 2048});
		random_uniform->SetOutputOffset({2048+1024});
		random_uniform->SetInputOffset({0, 32, 64, 128});

		auto random_uniform_node = graph->AddNode(random_uniform);
		auto count_node = graph->AddNode(count);
		auto seed_node = graph->AddNode(seed);
		auto low_node = graph->AddNode(low);
		auto high_node = graph->AddNode(high);


		GraphUtils::AddEdge(count_node->GetOutDataAnchor(0), random_uniform_node->GetInDataAnchor(0));
		GraphUtils::AddEdge(seed_node->GetOutDataAnchor(0), random_uniform_node->GetInDataAnchor(1));
		GraphUtils::AddEdge(low_node->GetOutDataAnchor(0), random_uniform_node->GetInDataAnchor(2));
		GraphUtils::AddEdge(high_node->GetOutDataAnchor(0), random_uniform_node->GetInDataAnchor(3));

		float low_value = 0.24f;
		GeTensor const_low(input_desc, (uint8_t*)(&low_value), sizeof(low_value));
		ge::AttrUtils::SetTensor(low, "value", const_low);

		std::vector<domi::TaskDef> task_defs;
		SetConstInputsValue(random_uniform_node);
		
		RunContextPtr contextptr = nullptr;
    FFTS_MAKE_SHARED(contextptr = std::make_shared<ge::RunContext>(CreateContext()), return nullptr);
		(void)random_uniform->SetExtAttr("_ffts_runtime_context", contextptr);
		std::shared_ptr<domi::FftsPlusCtxDef> ffts_plus_def_ptr = make_shared<domi::FftsPlusCtxDef>();
		domi::FftsPlusDsaCtxDef * dsa_ctx_def = ffts_plus_def_ptr->mutable_dsa_ctx();
		if (dsa_ctx_def == nullptr) {
			return nullptr;
		}
    (void)random_uniform->SetExtAttr(ffts::kAttrDsaCtxDef, ffts_plus_def_ptr);
		return random_uniform_node;
	}

	ge::NodePtr CreateDSARandominputconstNode()
	{
		ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
		OpDescPtr random_uniform = std::make_shared<OpDesc>("DSARandomUniform", "DSARandomUniform");
		OpDescPtr low = std::make_shared<OpDesc>("low", "Const");

		GeTensorDesc input_desc(GeShape({1}), ge::FORMAT_ND, ge::DT_FLOAT);
		input_desc.SetOriginShape(GeShape({1}));
		input_desc.SetOriginFormat(ge::FORMAT_ND);
		GeTensorDesc seed_desc(GeShape({1}), ge::FORMAT_ND, ge::DT_INT32);
		seed_desc.SetOriginShape(GeShape({1}));
		seed_desc.SetOriginFormat(ge::FORMAT_ND);
		random_uniform->AddInputDesc("low", input_desc);
		random_uniform->AddOutputDesc(input_desc);
		low->AddOutputDesc(input_desc);
		auto random_uniform_node = graph->AddNode(random_uniform);
		auto low_node = graph->AddNode(low);
		
		float low_value = 0.24f;
		GeTensor const_low(input_desc, (uint8_t*)(&low_value), sizeof(low_value));
		ge::AttrUtils::SetTensor(low, "value", const_low);
		(void)random_uniform->SetExtAttr("_ffts_runtime_context", nullptr);
		GraphUtils::AddEdge(low_node->GetOutDataAnchor(0), random_uniform_node->GetInDataAnchor(0));
		return random_uniform_node;
	}

	ge::NodePtr CreateDSARandomUniformconstantallNode()
	{
		ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
		OpDescPtr src_op = std::make_shared<OpDesc>("DSARandomUniform", "DSARandomUniform");
		GeTensorDesc src_tensor_desc(GeShape({1}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
		src_tensor_desc.SetOriginShape(GeShape({1}));
		src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
		src_op->AddOutputDesc(src_tensor_desc);
		src_op->AddInputDesc(src_tensor_desc);
		src_op->AddInputDesc(src_tensor_desc);
		src_op->AddInputDesc(src_tensor_desc);
		src_op->AddInputDesc(src_tensor_desc);
		src_op->SetWorkspace({1024, 2048});
		src_op->SetOutputOffset({2048+1024});
		src_op->SetInputOffset({0, 32, 64, 128});
		auto src_node = graph->AddNode(src_op);
		SetWeight(src_node, 0, 256L);
		SetWeight(src_node, 1, 512L);
		SetWeight(src_node, 2, 0.12f);
		SetWeight(src_node, 3, 0.24f);
		SetConstInputsValue(src_node);


		RunContextPtr contextptr = nullptr;
    FFTS_MAKE_SHARED(contextptr = std::make_shared<ge::RunContext>(CreateContext()), return nullptr);
		(void)src_op->SetExtAttr("_ffts_runtime_context", contextptr);
		std::shared_ptr<domi::FftsPlusCtxDef> ffts_plus_def_ptr = make_shared<domi::FftsPlusCtxDef>();
		domi::FftsPlusDsaCtxDef * dsa_ctx_def = ffts_plus_def_ptr->mutable_dsa_ctx();
		if (dsa_ctx_def == nullptr) {
			return nullptr;
		}
    (void)src_op->SetExtAttr(ffts::kAttrDsaCtxDef, ffts_plus_def_ptr);
		return src_node;
	}

	ge::NodePtr CreateDSAGenBitMaskNode()
	{
		ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
		OpDescPtr src_op = std::make_shared<OpDesc>("DSAGenBitMask", "DSAGenBitMask");
		GeTensorDesc src_tensor_desc(GeShape({100, 2, 3, 512, 4}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
		src_tensor_desc.SetOriginShape(GeShape({10, 11, 12, 13}));
		src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
		src_op->AddOutputDesc(src_tensor_desc);
		src_op->AddInputDesc(src_tensor_desc);
		src_op->AddInputDesc(src_tensor_desc);
		src_op->AddInputDesc(src_tensor_desc);
		src_op->SetWorkspace({1024, 2048});
		src_op->SetOutputOffset({2048+1024});
		src_op->SetInputOffset({0, 32, 64});
		auto src_node = graph->AddNode(src_op);
		
		RunContextPtr contextptr = nullptr;
    FFTS_MAKE_SHARED(contextptr = std::make_shared<ge::RunContext>(CreateContext()), return nullptr);
		(void)src_op->SetExtAttr("_ffts_runtime_context", contextptr);
		std::shared_ptr<domi::FftsPlusCtxDef> ffts_plus_def_ptr = make_shared<domi::FftsPlusCtxDef>();
		domi::FftsPlusDsaCtxDef * dsa_ctx_def = ffts_plus_def_ptr->mutable_dsa_ctx();
		if (dsa_ctx_def == nullptr) {
			return nullptr;
		}
    (void)src_op->SetExtAttr(ffts::kAttrDsaCtxDef, ffts_plus_def_ptr);
		return src_node;
	}
public:
	DSAManualTaskBuilderPtr dsa_task_builder_ptr_;
};

TEST_F(FFTSPlusDsaTaskBuilderSTest, GenContextDef_DSARandomNormal)
{
	auto node = CreateDSARandomNormalNode();
  domi::FftsPlusTaskDef taskdef;
	Status ret = dsa_task_builder_ptr_->GenContextDef(node, &taskdef);
	EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusDsaTaskBuilderSTest, GenContextDef_DSARandomTruncatedNormal)
{
	auto node = CreateDSARandomTruncatedNormalNode();
  domi::FftsPlusTaskDef taskdef;
	Status ret = dsa_task_builder_ptr_->GenContextDef(node, &taskdef);
	EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusDsaTaskBuilderSTest, GenContextDef_DSARandomUniform)
{
	auto node = CreateDSARandomUniformNode();
  domi::FftsPlusTaskDef taskdef;
	Status ret = dsa_task_builder_ptr_->GenContextDef(node, &taskdef);
	EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusDsaTaskBuilderSTest, GenContextDef_DSARandomUniformconstant0)
{
	auto node = CreateDSARandomUniformconstant0Node();
  domi::FftsPlusTaskDef taskdef;
	Status ret = dsa_task_builder_ptr_->GenContextDef(node, &taskdef);
	EXPECT_EQ(ffts::SUCCESS, ret);
}


TEST_F(FFTSPlusDsaTaskBuilderSTest, GenContextDef_DSARandomUniformconstant1)
{
	auto node = CreateDSARandomUniformconstant1Node();
  domi::FftsPlusTaskDef taskdef;
	Status ret = dsa_task_builder_ptr_->GenContextDef(node, &taskdef);
	EXPECT_EQ(ffts::SUCCESS, ret);
}


TEST_F(FFTSPlusDsaTaskBuilderSTest, GenContextDef_DSARandomUniformconstantall)
{
	auto node = CreateDSARandomUniformconstantallNode();
  domi::FftsPlusTaskDef taskdef;
	Status ret = dsa_task_builder_ptr_->GenContextDef(node, &taskdef);
	EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusDsaTaskBuilderSTest, GenContextDef_DSAGenBitMask)
{
	auto node = CreateDSAGenBitMaskNode();
  domi::FftsPlusTaskDef taskdef;
	Status ret = dsa_task_builder_ptr_->GenContextDef(node, &taskdef);
	EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusDsaTaskBuilderSTest, GenContextDef_DSAGetConstInputValue)
{
	auto node = CreateDSARandomUniformconstant1Node();
	std::string value;
	bool ret = dsa_task_builder_ptr_->GetConstInputValue(*(node.get()), 2, value);
	EXPECT_EQ(true, ret);
}

TEST_F(FFTSPlusDsaTaskBuilderSTest, GenContextDef_IsConstInput)
{
	auto node = CreateDSARandominputconstNode();
	std::string value;
	bool ret = dsa_task_builder_ptr_->IsConstInput(*(node.get()), 0);
	EXPECT_EQ(false, ret);
}

TEST_F(FFTSPlusDsaTaskBuilderSTest, ExceptTest)
{
	auto node = CreateDSARandominputconstNode();
	std::string value;
	Status ret = dsa_task_builder_ptr_->FillDsaContextData(*(node.get()), nullptr, nullptr);
	EXPECT_EQ(ffts::FAILED, ret);
	domi::FftsPlusTaskDef taskdef;
	ret = dsa_task_builder_ptr_->GenContextDef(node, &taskdef);
	ge::DataType data_type = ge::DataType::DT_FLOAT;
	bool ret1 = dsa_task_builder_ptr_->GetInputDataType(*(node.get()), 5, data_type);
	EXPECT_EQ(false, ret1);
}
