/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/lowering/bg_kernel_context_extend.h"
#include "exe_graph/lowering/getcdim.h"
#include <gtest/gtest.h>
#include <memory>
#include "graph/compute_graph.h"
#include "exe_graph/runtime/context_extend.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/tensor.h"
#include "graph/debug/ge_attr_define.h"
#include "expand_dimension.h"
#include "graph/utils/graph_utils.h"
#include "runtime/runtime_attrs_def.h"
#include "graph/compute_graph.h"
#include "exe_graph/lowering/bg_kernel_context_extend.h"
#include "exe_graph/runtime/tiling_context.h"

#include "graph/compute_graph.h"
#include "exe_graph/lowering/bg_kernel_context_extend.h"
#include "exe_graph/runtime/tiling_context.h"
#include "lowering/kernel_run_context_builder.h"
namespace gert {

struct CDimFakeKernelContextHolder {
  template<typename T>
  T *GetContext() {
    return reinterpret_cast<T*>(holder.context_);
  }
  ComputeNodeInfo *MutableComputeNodeInfo() {
    return reinterpret_cast<ComputeNodeInfo *>(holder.compute_node_extend_holder_.get());
  }
  size_t kernel_input_num;
  size_t kernel_output_num;
  KernelContextHolder holder;
};
CDimFakeKernelContextHolder CDimBuildKernelRunContext(size_t input_num, size_t output_num, int64_t reshape_type);

class CDimKernelRunContextFaker {
 public:
  CDimKernelRunContextFaker() = default;
  CDimKernelRunContextFaker &KernelIONum(size_t input_num, size_t output_num);
  CDimKernelRunContextFaker &NodeIoNum(size_t input_num, size_t output_num);
  CDimKernelRunContextFaker &IrInputNum(size_t input_num);
  CDimKernelRunContextFaker &IrInstanceNum(std::vector<uint32_t> instance_num);
  CDimKernelRunContextFaker &NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                     ge::Format storage_format);
  CDimKernelRunContextFaker &NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                      ge::Format storage_format);
  CDimKernelRunContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value);
  CDimKernelRunContextFaker &Inputs(std::vector<void *> inputs);
  CDimKernelRunContextFaker &Outputs(std::vector<void *> outputs);

  CDimFakeKernelContextHolder Build(int64_t reshape_type) const;

 private:
  ge::OpDescPtr FakeOp(int64_t reshape_type) const;

 private:
  size_t kernel_input_num_;
  size_t kernel_output_num_;
  size_t node_input_num_;
  size_t node_output_num_;
  std::vector<uint32_t> ir_instance_num_;
  std::vector<CompileTimeTensorDesc> node_input_tds_;
  std::vector<CompileTimeTensorDesc> node_output_tds_;
  std::vector<void *> inputs_;
  std::vector<void *> outputs_;
  std::vector<std::pair<std::string, ge::AnyValue>> attrs_;
};


class CDimTilingContextFaker {
 public:
  CDimTilingContextFaker &NodeIoNum(size_t input_num, size_t output_num);
  CDimTilingContextFaker &IrInputNum(size_t input_num) {
    base_faker_.IrInputNum(input_num);
    return *this;
  }
  CDimTilingContextFaker &IrInstanceNum(std::vector<uint32_t> instance_num) {
    base_faker_.IrInstanceNum(std::move(instance_num));
    return *this;
  }
  CDimTilingContextFaker &NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format, ge::Format storage_format) {
    base_faker_.NodeInputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  CDimTilingContextFaker &NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                   ge::Format storage_format) {
    base_faker_.NodeOutputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  CDimTilingContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
    base_faker_.NodeAttrs(std::move(keys_to_value));
    return *this;
  }
  CDimTilingContextFaker &InputShapes(std::vector<gert::StorageShape *> input_shapes);
  CDimTilingContextFaker &OutputShapes(std::vector<gert::StorageShape *> output_shapes);
  CDimTilingContextFaker &CompileInfo(void *compile_info);
  CDimTilingContextFaker &TilingData(void *tiling_data);
  CDimTilingContextFaker &Workspace(ContinuousVector *workspace);

  CDimFakeKernelContextHolder Build(int64_t reshape_type) const;

 private:
  void UpdateInputs();
  void UpdateOutputs();
 private:
  enum InputsAppend { kInputsCompileInfo, kInputsTilingFunc, kInputsAppendEnd };

  CDimKernelRunContextFaker base_faker_;
  std::vector<gert::StorageShape *> input_shapes_;
  std::vector<gert::StorageShape *> output_shapes_;

  void *compile_info_;
};


CDimFakeKernelContextHolder CDimBuildKernelRunContext(size_t input_num, size_t output_num, int64_t reshape_type) {
  return CDimKernelRunContextFaker().KernelIONum(input_num, output_num).Build(reshape_type);
}
CDimKernelRunContextFaker &CDimKernelRunContextFaker::KernelIONum(size_t input_num, size_t output_num) {
  kernel_input_num_ = input_num;
  kernel_output_num_ = output_num;
  return *this;
}
CDimKernelRunContextFaker &CDimKernelRunContextFaker::NodeIoNum(size_t input_num, size_t output_num) {
  node_input_num_ = input_num;
  node_output_num_ = output_num;
  node_input_tds_.resize(input_num);
  node_output_tds_.resize(output_num);
  return *this;
}
CDimKernelRunContextFaker &CDimKernelRunContextFaker::IrInputNum(size_t input_num) {
  ir_instance_num_ = std::vector<uint32_t>(input_num, 1);
  return *this;
}
CDimKernelRunContextFaker &CDimKernelRunContextFaker::IrInstanceNum(std::vector<uint32_t> instance_num) {
  ir_instance_num_ = std::move(instance_num);
  return *this;
}

ge::OpDescPtr CDimKernelRunContextFaker::FakeOp(int64_t reshape_type) const {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  for (size_t i = 0; i < node_input_num_; ++i) {
    auto prefix = "x_" + std::to_string(i) + "_";
    op_desc->AppendIrInput(prefix, ge::kIrInputRequired);
    auto td = ge::GeTensorDesc();
    if (reshape_type != 0) {
      (void) ge::AttrUtils::SetInt(td, ge::ATTR_NAME_RESHAPE_TYPE_MASK, reshape_type);
      //td.SetExpandDimsType(ExpandDimsType(reshape_type));
    }
    
      td.SetOriginFormat(node_input_tds_[i].GetOriginFormat());
      td.SetFormat(node_input_tds_[i].GetStorageFormat());
      td.SetDataType(node_input_tds_[i].GetDataType());
      td.SetOriginDataType(node_input_tds_[i].GetDataType());
    op_desc->AddInputDesc(prefix, td);
  }
  for (size_t i = 0; i < node_output_num_; ++i) {
    auto prefix = "y_" + std::to_string(i) + "_";
    auto td = ge::GeTensorDesc();
    if (reshape_type != 0) {
      (void) ge::AttrUtils::SetInt(td, ge::ATTR_NAME_RESHAPE_TYPE_MASK, reshape_type);
      //td.SetExpandDimsType(ExpandDimsType(reshape_type));
    }
    
      td.SetOriginFormat(node_output_tds_[i].GetOriginFormat());
      td.SetFormat(node_output_tds_[i].GetStorageFormat());
      td.SetDataType(node_output_tds_[i].GetDataType());
      td.SetOriginDataType(node_output_tds_[i].GetDataType());
    
    op_desc->AddOutputDesc(prefix, td);
  }
  return op_desc;
}

CDimFakeKernelContextHolder CDimKernelRunContextFaker::Build(int64_t reshape_type) const {
  CDimFakeKernelContextHolder fake_holder;
  fake_holder.kernel_input_num = kernel_input_num_;
  fake_holder.kernel_output_num = kernel_output_num_;
  KernelRunContextBuilder kernel_context_builder;
  auto op_desc = FakeOp(reshape_type);
  std::cout << "kernel_input_num = " << kernel_input_num_ << "kernel_output_num = " << kernel_output_num_
            << " input.size = " << inputs_.size() << "  outputs.size = " << outputs_.size() << std::endl;
  if (inputs_.size() != kernel_input_num_ || outputs_.size() != kernel_output_num_) {
    std::vector<void *> inputs(kernel_input_num_, nullptr);
    std::vector<void *> outputs(kernel_output_num_, nullptr);
    fake_holder.holder = kernel_context_builder.Inputs(inputs).Outputs(outputs).Build(op_desc);
    return fake_holder;
  }
  fake_holder.holder = kernel_context_builder.Inputs(inputs_).Outputs(outputs_).Build(op_desc);
  return fake_holder;
}
CDimKernelRunContextFaker &CDimKernelRunContextFaker::NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                                          ge::Format storage_format) {
  node_input_tds_[index].SetDataType(dt);
  node_input_tds_[index].SetOriginFormat(origin_format);
  node_input_tds_[index].SetStorageFormat(storage_format);
  return *this;
}
CDimKernelRunContextFaker &CDimKernelRunContextFaker::NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                                           ge::Format storage_format) {
  node_output_tds_[index].SetDataType(dt);
  node_output_tds_[index].SetOriginFormat(origin_format);
  node_output_tds_[index].SetStorageFormat(storage_format);
  return *this;
}
CDimKernelRunContextFaker &CDimKernelRunContextFaker::Inputs(std::vector<void *> inputs) {
  inputs_ = std::move(inputs);
  return *this;
}
CDimKernelRunContextFaker &CDimKernelRunContextFaker::Outputs(std::vector<void *> outputs) {
  outputs_ = std::move(outputs);
  return *this;
}
CDimKernelRunContextFaker &
CDimKernelRunContextFaker::NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
  attrs_ = std::move(keys_to_value);
  return *this;
}

CDimTilingContextFaker &CDimTilingContextFaker::NodeIoNum(size_t input_num, size_t output_num) {
  base_faker_.KernelIONum(input_num, output_num);
  base_faker_.NodeIoNum(input_num, output_num);
  return *this;
}
CDimTilingContextFaker &CDimTilingContextFaker::InputShapes(std::vector<gert::StorageShape *> input_shapes) {
  input_shapes_ = std::move(input_shapes);
  UpdateInputs();
  return *this;
}
CDimTilingContextFaker &CDimTilingContextFaker::OutputShapes(std::vector<gert::StorageShape *> output_shapes) {
  output_shapes_ = std::move(output_shapes);
  UpdateOutputs();
  return *this;
}
CDimTilingContextFaker &CDimTilingContextFaker::CompileInfo(void *compile_info) {
  compile_info_ = compile_info;
  UpdateInputs();
  UpdateOutputs();
  return *this;
}
CDimTilingContextFaker &CDimTilingContextFaker::TilingData(void *tiling_data) {
  return *this;
}
CDimTilingContextFaker &CDimTilingContextFaker::Workspace(ContinuousVector *workspace) {
  return *this;
}
CDimFakeKernelContextHolder CDimTilingContextFaker::Build(int64_t reshape_type) const {
  return base_faker_.Build(reshape_type);
}
void CDimTilingContextFaker::UpdateInputs() {
  std::vector<void *> inputs;
  for (const auto input_shape : input_shapes_) {
    inputs.push_back(input_shape);
  }
  inputs.push_back(nullptr);        // kInputsTilingFunc
  base_faker_.Inputs(std::move(inputs));
}

void CDimTilingContextFaker::UpdateOutputs() {
  std::vector<void *> outputs;
  for (const auto output_shape : output_shapes_) {
    outputs.push_back(output_shape);
  }
  base_faker_.Outputs(std::move(outputs));
}


namespace {
struct CDimTestTilingData {
  int64_t a;
};
struct CDimTestCompileInfo {
  int64_t a;
  int64_t b;
  std::vector<int64_t> c;
};
}
class GetCDimTestUT : public testing::Test {};

// 测试构造kernel context的时候从tensor desc上获取ATTR_NAME_RESHAPE_TYPE_MASK并设置到compile time tensor desc 上
// 同时测试调用Expand是否能够得到正确的扩维shape
TEST_F(GetCDimTestUT, BuildRequiredInputWithExpandDimsType01) {
  gert::StorageShape in_shape = {{5,2,3,4}, {5, 1, 1, 1, 1}};
  gert::StorageShape out_shape = {{5,2,3,4}, {5, 1,1, 1, 1}};
   vector<int64_t> origin_shape = {5,2,3,4};
  int64_t int_reshape_type =  0;
  // tiling data
  CDimTestCompileInfo compile_info_holder = {10, 200, {10, 20, 30}};
  auto param = gert::TilingData::CreateCap(2048);
  auto holder = gert::CDimTilingContextFaker()
                    .NodeIoNum(2, 1)
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                    .InputShapes({&in_shape})
                    .OutputShapes({&out_shape})
                    .Build(int_reshape_type);

  auto context = holder.GetContext<TilingContext>();
  ASSERT_NE(context, nullptr);
  int64_t cdim = GetInputCDim(context, 0);
  EXPECT_EQ(2, cdim);
  cdim = GetOutputCDim(context, 0);
  EXPECT_EQ(2, cdim);
}

TEST_F(GetCDimTestUT, BuildRequiredInputWithExpandDimsType02) {
  gert::StorageShape in_shape = {{5,2,3,4, 1}, {5, 1, 1, 1, 1}};
  gert::StorageShape out_shape = {{5,2,3,4, 1}, {5, 1,1, 1, 1}};
   vector<int64_t> origin_shape = {5,2,3,4};
  int64_t int_reshape_type = transformer::ExpandDimension::GenerateReshapeType(ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0,
                                                                               origin_shape.size(), "CHW");
  int_reshape_type  = 0;
  // tiling data
  CDimTestCompileInfo compile_info_holder = {10, 200, {10, 20, 30}};
  auto param = gert::TilingData::CreateCap(2048);
  auto holder = gert::CDimTilingContextFaker()
                    .NodeIoNum(2, 1)
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0)
                    .InputShapes({&in_shape})
                    .OutputShapes({&out_shape})
                    .Build(int_reshape_type);

  auto context = holder.GetContext<TilingContext>();
  ASSERT_NE(context, nullptr);
  int64_t cdim = GetInputCDim(context, 0);
  EXPECT_EQ(-1, cdim);
  cdim = GetOutputCDim(context, 0);
  EXPECT_EQ(-1, cdim);
}

TEST_F(GetCDimTestUT, BuildRequiredInputWithExpandDimsType03) {
  gert::StorageShape in_shape = {{5, 6, 7}, {5, 6, 7, 1}};
  gert::StorageShape out_shape = {{5, 6, 7}, {5, 6, 7, 1}};
  vector<int64_t> origin_shape = {5, 6, 7};
  int64_t int_reshape_type = transformer::ExpandDimension::GenerateReshapeType(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0,
                                                                               origin_shape.size(), "NCH");
  
  // tiling data
  CDimTestCompileInfo compile_info_holder = {10, 200, {10, 20, 30}};
  auto param = gert::TilingData::CreateCap(2048);
  auto holder = gert::CDimTilingContextFaker()
                    .NodeIoNum(2, 1)
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                    .InputShapes({&in_shape})
                    .OutputShapes({&out_shape})
                    .Build(int_reshape_type);

  auto context = holder.GetContext<TilingContext>();
  ASSERT_NE(context, nullptr);
  int64_t cdim = GetInputCDim(context, 0);
  EXPECT_EQ(6, cdim);
  cdim = GetOutputCDim(context, 0);
  EXPECT_EQ(6, cdim);
}

TEST_F(GetCDimTestUT, GetAxisIndexByName01) {
  int32_t c_ax_index = transformer::ExpandDimension::GetAxisIndexByName('C', ge::FORMAT_NCHW);
  EXPECT_EQ(1, c_ax_index);
}

TEST_F(GetCDimTestUT, GetReshapeAxicValueByName01) {
  vector<int64_t> origin_shape = {5, 6, 7};
  vector<int64_t> storage_shape = {5, 6, 7, 1};
  ge::GeShape inshape{storage_shape};
  int64_t int_reshape_type = transformer::ExpandDimension::GenerateReshapeType(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0,
                                                                               origin_shape.size(), "NCH");
  int64_t c_ax_value = transformer::ExpandDimension::GetReshapeAxicValueByName(int_reshape_type, 'W', inshape, ge::FORMAT_NCHW);
  EXPECT_EQ(1, c_ax_value);
}

}  // namespace gert
