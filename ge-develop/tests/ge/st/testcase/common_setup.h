/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_GRAPH_ENGINE_JIT_EXECUTION_COMMON_SETUP_H
#define CANN_GRAPH_ENGINE_JIT_EXECUTION_COMMON_SETUP_H
#include "ge/ge_api.h"
#include "ge_running_env/fake_engine.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"
#include "ge/framework/common/taskdown_common.h"
#include "common/env_path.h"
#include "graph/operator_factory_impl.h"
#include "graph/utils/op_desc_utils.h"
#include "faker/space_registry_faker.h"
#include "ge/ut/ge/jit_execution/jit_share_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include <gtest/gtest.h>

using namespace testing;
namespace ge {
namespace {
// 单输入单输出算子，使用输入tensor的shape和dtype作为输出tensor的shape和dtype，模拟infershape函数
const auto SingleIOForwardInfer = [](Operator &op) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto in_td = op_desc->GetInputDescPtr(0);
  auto td = op_desc->MutableOutputDesc(0);
  td->SetShape(in_td->GetShape());
  td->SetOriginShape(in_td->GetOriginShape());
  td->SetDataType(in_td->GetDataType());
  td->SetOriginDataType(in_td->GetOriginDataType());
  return GRAPH_SUCCESS;
};
// 该infershape仅用于动态输入构图
// 输入为动态shape表示为编译态，输出shape为动态shape
// 输入为静态shape表示为运行态，输出shape为静态shape
auto UniqueInferFun = [](Operator &op) -> graphStatus {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  const auto &input_desc = op_desc->GetInputDesc(0);
  const auto input_shape = input_desc.GetShape().GetDims();

  auto output0_desc = op_desc->MutableOutputDesc(0);
  input_desc.GetShape().IsUnknownShape() ? output0_desc->SetShape(GeShape({-1})) : output0_desc->SetShape(GeShape({16}));

  output0_desc->SetShapeRange({
      {1, input_shape[0]}
  });
  return GRAPH_SUCCESS;
};
const auto ReshapeInferFun = [](Operator &op) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends({"shape"});
  auto x_desc = op_desc->MutableInputDesc("x");
  auto y_desc = op_desc->MutableOutputDesc("y");
  auto x_shape = vector<int64_t>(x_desc->GetShape().GetDims());
  std::string shape_input_name = "shape";

  Tensor shape_tensor;
  op.GetInputConstData(shape_input_name.c_str(), shape_tensor);

  GeShape output_shape;
  auto shape_tensor_desc = op_desc->MutableInputDesc("shape");
  auto &shape_ref = shape_tensor_desc->MutableShape();
  auto shape_dims = shape_ref.GetDims();

  int64_t dim_num = shape_dims[0];
  const int64_t *shape_data = const_cast<int64_t *>(reinterpret_cast<const int64_t *>(shape_tensor.GetData()));
  std::vector<int64_t> out_dims;
  int64_t product = 1;
  for (int64_t i = 0; i < dim_num; i++) {
      auto dim = shape_data[i];
      if (dim != 0 && product > (INT64_MAX / dim)) { 
          return GRAPH_PARAM_INVALID;
      }
      out_dims.push_back(dim);
      product *= dim;
  }

  auto td = op_desc->MutableOutputDesc("y");
  td->SetShape(GeShape(out_dims));
  td->SetOriginShape(GeShape(out_dims));
  if (out_dims.size() == 0) {
    td->SetShape(GeShape({-1, -1, -1, -1}));
    td->SetOriginShape(GeShape({-1, -1, -1, -1}));
  }
  td->SetDataType(op_desc->MutableInputDesc("x")->GetDataType());
  td->SetOriginDataType(op_desc->MutableInputDesc("x")->GetDataType());
  return GRAPH_SUCCESS;
};
auto infer_fun = [](Operator &op) -> graphStatus {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  *op_desc->MutableOutputDesc(0) = *op_desc->GetInputDescPtr(0);
  return GRAPH_SUCCESS;
};

auto shape_infer_fun = [](Operator &op) -> graphStatus {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto x_desc = op_desc->MutableInputDesc("x");
  const auto x_dim_num = x_desc->GetShape().GetDimNum();

  auto td = op_desc->MutableOutputDesc("y");
  td->SetShape(ge::GeShape({static_cast<int64_t>(x_dim_num)}));
  td->SetOriginShape(ge::GeShape({static_cast<int64_t>(x_dim_num)}));
  td->SetDataType(DT_INT64);
  td->SetOriginDataType(DT_INT64);
  return GRAPH_SUCCESS;
};

auto reshape_infer_fun = [](Operator &op) -> graphStatus {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends({"shape"});
  auto x_desc = op_desc->MutableInputDesc("x");
  auto y_desc = op_desc->MutableOutputDesc("y");
  auto x_shape = vector<int64_t>(x_desc->GetShape().GetDims());
  std::string shape_input_name = "shape";

  ge::Tensor shape_tensor;
  op.GetInputConstData(shape_input_name.c_str(), shape_tensor);

  ge::GeShape output_shape;
  auto shape_tensor_desc = op_desc->MutableInputDesc("shape");
  auto &shape_ref = shape_tensor_desc->MutableShape();
  auto shape_dims = shape_ref.GetDims();

  int64_t dim_num = shape_dims[0];
  const int64_t *shape_data = const_cast<int64_t *>(reinterpret_cast<const int64_t *>(shape_tensor.GetData()));
  std::vector<int64_t> out_dims;
  int64_t product = 1;
  for (int64_t i = 0; i < dim_num; i++) {
    auto dim = shape_data[i];
    if (dim != 0 && product > (INT64_MAX / dim)) {
      return ge::GRAPH_PARAM_INVALID;
    }
    out_dims.push_back(dim);
    product *= dim;
  }

  auto td = op_desc->MutableOutputDesc("y");
  td->SetShape(ge::GeShape(out_dims));
  td->SetOriginShape(ge::GeShape(out_dims));
  td->SetDataType(op_desc->MutableInputDesc("x")->GetDataType());
  td->SetOriginDataType(op_desc->MutableInputDesc("x")->GetDataType());
  return ge::GRAPH_SUCCESS;
};
auto type2_infer = [](Operator &op) -> graphStatus {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends({"__input0"});
  Tensor tensor;
  auto output_desc = op_desc->MutableOutputDesc(0);
  if (op.GetInputConstData("__input0", tensor) == GRAPH_SUCCESS) {
    output_desc->SetShape(GeShape({4}));
  } else {
    output_desc->SetShape(GeShape({-1}));
    output_desc->SetShapeRange({{1, 16}});
  }
  return GRAPH_SUCCESS;
};
class FakeAiCoreEngineOptimizer : public FakeGraphOptimizer {
 public:
  // 为code gen生成的node添加编译结果
  Status OptimizeWholeGraph(ComputeGraph &graph) override {
    for (const auto &node : graph.GetAllNodes()) {
      if (node->GetInDataNodes().empty() || node->GetOutDataNodes().empty()) {
        continue;
      }
      std::cout << "mark node compile result " << node->GetName() << std::endl;
      JitShareGraph::AddCompileResult(
          node, true,
          "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", "
          "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}");
    }
    return SUCCESS;
  }
};
class FakeAiCoreOpsKernelBuilder : public FakeOpsKernelBuilder {
 public:
  explicit FakeAiCoreOpsKernelBuilder(const std::string &engine_name) : FakeOpsKernelBuilder(engine_name) {}
  Status GenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) override {
    std::cout << "generate task for node " << node.GetName() << std::endl;
    auto op_desc = node.GetOpDesc();
    op_desc->SetOpKernelLibName("AIcoreEngine");

    EnvPath env_path;
    std::string cmake_binary_dir = env_path.GetBinRootPath();
    std::string autofuse_stub_so_path = cmake_binary_dir + "/tests/depends/op_stub/libslice_autofuse_stub.so";
    (void)ge::AttrUtils::SetStr(op_desc, "bin_file_path", autofuse_stub_so_path);

    size_t arg_size = 100;
    std::vector<uint8_t> args(arg_size, 0);
    domi::TaskDef task_def;
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    auto kernel_info = task_def.mutable_kernel(); // auto fuse kernel
    kernel_info->set_args(args.data(), args.size());
    kernel_info->set_args_size(arg_size);
    kernel_info->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    //kernel_info->set_node_info(node.GetName());
    kernel_info->set_block_dim(1);
    uint16_t args_offset[2] = {0};
    kernel_info->mutable_context()->set_args_offset(args_offset, 2 * sizeof(uint16_t));
    kernel_info->mutable_context()->set_op_index(node.GetOpDesc()->GetId());

    auto kernel_with_handle_info = task_def.mutable_kernel_with_handle(); // auto fuse kernel
    kernel_with_handle_info->set_args(args.data(), args.size());
    kernel_with_handle_info->set_args_size(arg_size);
    kernel_with_handle_info->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    //kernel_info->set_node_info(node.GetName());
    kernel_with_handle_info->set_block_dim(1);
    kernel_with_handle_info->mutable_context()->set_args_offset(args_offset, 2 * sizeof(uint16_t));
    kernel_with_handle_info->mutable_context()->set_op_index(node.GetOpDesc()->GetId());

    tasks.emplace_back(task_def);
    std::cout << "emplace_back task for node " << node.GetName() << std::endl;
    return SUCCESS;
  }
};
} // namespace
struct CommonSetupUtil {
  static void inline CommonSetup(bool with_ge_init = true) {
    backup_operator_creators_v2_ = ge::OperatorFactoryImpl::operator_creators_v2_;
    backup_operator_creators_ = ge::OperatorFactoryImpl::operator_creators_;
    // 3 初始化GE
    if (with_ge_init) {
      std::map<AscendString, AscendString> global_options;
      EXPECT_EQ(GEInitialize(global_options), SUCCESS);
    }

    // 1. set up space registry
    gert::LoadDefaultSpaceRegistry();
    gert::SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();

    // 2. fake engine and ops
    auto fe_optimizer = MakeShared<FakeAiCoreEngineOptimizer>();
    auto fe_ops_kernel_builder = MakeShared<FakeAiCoreOpsKernelBuilder>("AIcoreEngine");
    GeRunningEnvFaker().Reset()
        .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeEngine("AIcoreEngine")
                     .KernelInfoStore("AIcoreEngine")
                     .GraphOptimizer("fe", fe_optimizer)
                     .KernelBuilder(fe_ops_kernel_builder))
        .Install(FakeEngine("DNN_VM_RTS").KernelInfoStore("DNN_VM_RTS_OP_STORE"))
        .Install(FakeOp(IF).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(RESHAPE).Inputs({"x", "shape"}).Outputs({"y"}).AttrsDef("axis", 0).AttrsDef("num_axes", -1).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE").InferShape(ReshapeInferFun))
        .Install(FakeOp("Relu").Inputs({"x"}).Outputs({"y"}).InfoStoreAndBuilder("AIcoreEngine").InferShape(SingleIOForwardInfer))
        .Install(FakeOp(ADD).Inputs({"x1", "x2"}).Outputs({"y"}).InfoStoreAndBuilder("AIcoreEngine").InferShape(SingleIOForwardInfer))
        .Install(FakeOp("AscBackend").Inputs({"x"}).Outputs({"y"}).InfoStoreAndBuilder("AIcoreEngine").InferShape(SingleIOForwardInfer))
        .Install(FakeOp("Add").Inputs({"x1", "x2"}).Outputs({"y"}).InfoStoreAndBuilder("AIcoreEngine").InferShape(SingleIOForwardInfer))
        .Install(FakeOp("Unique").Inputs({"x"}).Outputs({"y", "idx"}).InfoStoreAndBuilder("AIcoreEngine").InferShape(UniqueInferFun))

        .Install(FakeOp("FakeType2Op").InfoStoreAndBuilder("AIcoreEngine").InferShape(type2_infer))
        .Install(FakeOp(MUL).InfoStoreAndBuilder("AIcoreEngine").InferShape(infer_fun))
        .Install(FakeOp(SHAPE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE").InferShape(shape_infer_fun))
        .Install(FakeOp(VARIABLE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(CONSTANTOP).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(IDENTITY).InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE"))
        .Install(FakeOp(EXIT).InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE"))
        .Install(FakeOp(RECV).InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE"))
        .Install(FakeOp(SEND).InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE"));

    // 4. 开启自动融合、编译guard so 所需的环境变量
    auto ascend_install_path = EnvPath().GetAscendInstallPath();
    setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
    setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
    //setenv("ASCEND_OPP_PATH", "/usr/local/Ascend/latest/opp", 1);
    auto work_path = EnvPath().GetAirBasePath() + "/output";
    setenv("ASCEND_WORK_PATH", work_path.c_str(), 1);

    mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1); // 开启自动融合
  }
  static void inline CommonTearDown(bool with_ge_finalize = true) {
    if (with_ge_finalize) {
      GEFinalize();
    }
    GeRunningEnvFaker().InstallDefault();
    gert::UnLoadDefaultSpaceRegistry();

    unsetenv("ASCEND_OPP_PATH");
    unsetenv("LD_LIBRARY_PATH");
    unsetenv("AUTOFUSE_FLAGS");
    ge::OperatorFactoryImpl::operator_creators_v2_ = std::move(backup_operator_creators_v2_);
    ge::OperatorFactoryImpl::operator_creators_ = std::move(backup_operator_creators_);
  }
 private:
  static std::shared_ptr<std::map<std::string, ge::OpCreatorV2>> backup_operator_creators_v2_;
  static std::shared_ptr<std::map<std::string, OpCreator>> backup_operator_creators_;
};
}


#endif  // CANN_GRAPH_ENGINE_JIT_EXECUTION_COMMON_SETUP_H
