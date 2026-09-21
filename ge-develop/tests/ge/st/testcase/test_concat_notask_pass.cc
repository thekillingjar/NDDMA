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
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/build/graph_builder.h"
#include "ge/ge_api.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/op_desc_utils.h"
#include "faker/global_data_faker.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"
#include "utils/mock_ops_kernel_builder.h"
#include "ge_running_env/tensor_utils.h"
#include "ge_running_env/fake_graph_optimizer.h"
#include "ge_running_env/fake_ops_kernel_builder.h"
#include "utils/taskdef_builder.h"
#include "graph/utils/tensor_adapter.h"
#include "register/op_tiling_registry.h"
#include "register/node_converter_registry.h"
#include "exe_graph/lowering/lowering_global_data.h"
#include "graph/passes/memory_optimize/concat_notask_pass.h"
#include "ge_running_env/op_reg.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace ge {
FakeOpsKernelInfoStore g_fake_ops_kernel_info_store;
static void MockGenerateTask() {
  auto aicore_func = [](const ge::Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
    auto op_desc = node.GetOpDesc();
    op_desc->SetOpKernelLibName("AiCoreLib");
    size_t arg_size = 100;
    std::vector<uint8_t> args(arg_size, 0);
    domi::TaskDef task_def;
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    auto kernel_info = task_def.mutable_kernel();
    kernel_info->set_args(args.data(), args.size());
    kernel_info->set_args_size(arg_size);
    kernel_info->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    kernel_info->set_kernel_name(node.GetName());
    kernel_info->set_block_dim(1);
    uint16_t args_offset[2] = {0};
    kernel_info->mutable_context()->set_args_offset(args_offset, 2 * sizeof(uint16_t));
    kernel_info->mutable_context()->set_op_index(node.GetOpDesc()->GetId());

    tasks.emplace_back(task_def);
    return SUCCESS;
  };

  auto hccl_func = [](const ge::Node &node, RunContext &run_context, std::vector<domi::TaskDef> &tasks) -> Status {
    std::cout << "======node.GetType():" << node.GetType()  << std::endl;
    if (node.GetType() != "HcomAllGather") {
        std::cout << "*****return***"<< std::endl;
        return SUCCESS;
    }

    domi::TaskDef task_def;
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_HCCL));
    task_def.set_stream_id(0);
    const auto op_desc = node.GetOpDesc();
    OpsKernelInfoStore *ptr = &g_fake_ops_kernel_info_store;
    op_desc->SetExtAttr("OpsKernelInfoStorePtr", ptr);
    int32_t root_id = 0;
    (void)ge::AttrUtils::SetInt(op_desc, HCOM_ATTR_ROOT_RANK, root_id);
    auto &kernel_hccl_def = *task_def.mutable_kernel_hccl();
    kernel_hccl_def.set_op_index(op_desc->GetId());
    kernel_hccl_def.set_hccl_type("HcomAllGather");
    tasks.emplace_back(task_def);
    return SUCCESS;
  };

  MockForGenerateTask("AiCoreLib", aicore_func);
  MockForGenerateTask("ops_kernel_info_hccl", hccl_func);
}
class ConcatNotaskPassTest : public testing::Test {
 protected:
  void SetUp() {
    MockGenerateTask();
  }
  void TearDown() {
    OpsKernelBuilderRegistry::GetInstance().Unregister("AiCoreLib");
    OpsKernelBuilderRegistry::GetInstance().Unregister("ops_kernel_info_hccl");
  }
};
graphStatus infer_fun(Operator &op){
      // info shape实现
      auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
      auto output_desc = *op_desc->MutableOutputDesc(0);
      *op_desc->MutableOutputDesc(0) = *op_desc->GetInputDescPtr(0);
      bool reuse_flag = false;
      (void)TensorUtils::GetReuseInput(output_desc, reuse_flag);
      if (reuse_flag) {
        uint32_t idx = 100;
        TensorUtils::GetReuseInputIndex(output_desc, idx);
        TensorUtils::SetReuseInput(*op_desc->MutableOutputDesc(0), true);
        TensorUtils::SetReuseInputIndex(*op_desc->MutableOutputDesc(0), idx);
      }
      return GRAPH_SUCCESS;
}
graphStatus infer_fun_concat(Operator &op){
  // info shape实现
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  *op_desc->MutableOutputDesc(0) = *op_desc->GetInputDescPtr(0);

  int64_t dim_0 = 0;
  for (size_t i = 0U; i < op_desc->GetInputsSize(); ++i) {
    const auto input_shape = op_desc->GetInputDescPtr(i)->GetShape();
    if (!input_shape.GetDims().empty()) {
      dim_0 += input_shape.GetDim(0);
    }
  }
  auto out_shape = op_desc->MutableOutputDesc(0)->GetShape().GetDims();
  if (!out_shape.empty()) {
    out_shape[0] = dim_0;
  }

  op_desc->MutableOutputDesc(0)->SetShape(GeShape(out_shape));
  return GRAPH_SUCCESS;
}
class MockConcatNotaskPass {
 public:
  explicit MockConcatNotaskPass() {
    auto allgather_infer_fun = [](Operator &op) -> graphStatus {
      auto output_desc = op.GetInputDesc(0);
      int64_t rank_size = 8;
      auto shape = output_desc.GetShape();
      shape.SetDim(0, shape.GetDim(0) * rank_size);
      output_desc.SetShape(shape);
      GE_CHK_GRAPH_STATUS_RET(op.UpdateOutputDesc("y", output_desc));
      return GRAPH_SUCCESS;
    };
    auto InferShapeFuncForReshape = [](Operator &op) -> graphStatus  {
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

    auto ge_env = GeRunningEnvFaker();
    ge_env.Reset()
         .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AiCoreLib").GraphOptimizer("AIcoreEngine"))
         .Install(FakeEngine(kEngineNameAiCpu).KernelInfoStore(kEngineNameAiCpu))
         .Install(FakeEngine("DNN_HCCL").KernelInfoStore(kEngineNameHccl))
         .Install(FakeEngine("DNN_VM_RTS").KernelInfoStore(kEngineNameRts).GraphOptimizer("DNN_VM_RTS_GRAPH_OPTIMIZER_STORE"))
         .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(VARIABLE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(SHAPE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(IF).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE").InferShape(infer_fun))
         .Install(FakeOp("PhonyConcat").InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE").InferShape(infer_fun))
         .Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(CONSTANTOP).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp("Identity").InfoStoreAndBuilder("AiCoreLib"))
         .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(ADD).InfoStoreAndBuilder("AiCoreLib").InferShape(infer_fun))
         .Install(FakeOp("ConcatD").InfoStoreAndBuilder("AiCoreLib").InferShape(infer_fun_concat))
         .Install(FakeOp(HCOMALLGATHER).InfoStoreAndBuilder(kEngineNameHccl).InferShape(allgather_infer_fun))
         .Install(FakeOp("Send").InfoStoreAndBuilder(kEngineNameRts))
         .Install(FakeOp("Recv").InfoStoreAndBuilder(kEngineNameRts))
         .Install(FakeOp(RELU).InfoStoreAndBuilder("AiCoreLib").InferShape(infer_fun))
         .Install(FakeOp(CONV2D).InfoStoreAndBuilder("AiCoreLib").InferShape(infer_fun))
         .Install(FakeOp(RESHAPE).InfoStoreAndBuilder("AiCoreLib").InferShape(InferShapeFuncForReshape))
         .Install(FakeOp(STREAMACTIVE).InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE"));
      optiling::OpTilingFuncV2 tilingfun = [](const ge::Operator &op,
                                          const optiling::OpCompileInfoV2 &compile_info,
                                          optiling::OpRunInfoV2 &run_info) -> bool {
        run_info.SetWorkspaces({1024});
        return true;
      };
      optiling::OpTilingRegistryInterf_V2(RELU, tilingfun);
      REGISTER_OP_TILING_UNIQ_V2(ReLU, tilingfun, 1);
  }

  virtual ~MockConcatNotaskPass() {
    auto ge_env = GeRunningEnvFaker();
    ge_env.InstallDefault();
  }
};

struct FakeAicoreLibOpsKernelBuilder : FakeOpsKernelBuilder {
 public:
  FakeAicoreLibOpsKernelBuilder(const std::string &kernel_lib_name) : FakeOpsKernelBuilder(kernel_lib_name) {}
  FakeAicoreLibOpsKernelBuilder() : FakeOpsKernelBuilder() {}

 protected:
  Status Initialize(const map<std::string, std::string> &options) override {
    return SUCCESS;
  }
  Status Finalize() override {
    return SUCCESS;
  }
  Status GenerateTask(const ge::Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) override {
    GELOGI("Start gen task for %s", node.GetName().c_str());
    tasks.emplace_back(AiCoreTaskDefBuilder(node).BuildTask());
    return SUCCESS;
  }
};

template <typename T>
void Fake5DNodeEngine(GeRunningEnvFaker &ge_env, const std::vector<int64_t> &origin_shape) {
  auto ffo = MakeShared<T>();
  auto ops_kernel_builder = MakeShared<FakeAicoreLibOpsKernelBuilder>("AiCoreLib");
  auto aicore_engine_builder = MakeShared<FakeAicoreLibOpsKernelBuilder>("AIcoreEngine");
  // {c0_value, bit_value}: c0_value = 2 ^ (bit_value - 1)
  // {1, 1}, {2, 2}, {4, 3}, {8, 4}, {16, 5}, {32, 6}, {64, 7}, {128, 8}, {256, 9}
  // 5 indicates that cube size is 16
  const Format src_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_NC1HWC0, FORMAT_RESERVED, 5));
  const Format dst_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_FRACTAL_Z, FORMAT_NHWC, 5));
  ffo->OpFormatByType(
      CONV2D, {
          .input_formats = {
              {src_format, GeShape(std::vector<int64_t>(origin_shape))},
              {dst_format, GeShape(std::vector<int64_t>({4,1,16,16}))},
          },
          .output_formats = {
              {src_format, GeShape(std::vector<int64_t>(origin_shape))}
          }
      });
  ffo->OpFormatByType(
      "ConcatD", {
          .input_formats = {
              {src_format, GeShape(std::vector<int64_t>(origin_shape))},
              {src_format, GeShape(std::vector<int64_t>(origin_shape))},
          },
          .output_formats = {
              {src_format, GeShape(std::vector<int64_t>(origin_shape))}
          }
      });
      ge_env.Reset()
         .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AiCoreLib").GraphOptimizer("AIcoreEngine"))
         .Install(FakeEngine("DNN_HCCL").KernelInfoStore(kEngineNameHccl))
         .Install(FakeEngine("DNN_VM_RTS").KernelInfoStore(kEngineNameRts).GraphOptimizer("DNN_VM_RTS_GRAPH_OPTIMIZER_STORE"))
         .Install(FakeEngine("AiCoreLib").GraphOptimizer("FormatOp", ffo).KernelBuilder(ops_kernel_builder).KernelBuilder(aicore_engine_builder))
         .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(VARIABLE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(SHAPE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(IF).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp("PhonyConcat").InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE").InferShape(infer_fun))
         .Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(CONSTANTOP).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp("Identity").InfoStoreAndBuilder("AiCoreLib"))
         .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(ADD).InfoStoreAndBuilder("AiCoreLib").InferShape(infer_fun))
         .Install(FakeOp("ConcatD").InfoStoreAndBuilder("AiCoreLib"))
         .Install(FakeOp("Send").InfoStoreAndBuilder(kEngineNameRts))
         .Install(FakeOp("Recv").InfoStoreAndBuilder(kEngineNameRts))
         .Install(FakeOp(RELU).InfoStoreAndBuilder("AiCoreLib").InferShape(infer_fun))
         .Install(FakeOp(CONV2D).InfoStoreAndBuilder("AiCoreLib").InferShape(infer_fun))
         .Install(FakeOp(RESHAPE).InfoStoreAndBuilder("AiCoreLib"))
         .Install(FakeOp("RefData").InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(TRANSDATA).InfoStoreAndBuilder("AiCoreLib"));
}

template <typename T>
void Fake5DNodeEngine2(GeRunningEnvFaker &ge_env) {
  auto ffo = MakeShared<T>();
  auto ops_kernel_builder = MakeShared<FakeAicoreLibOpsKernelBuilder>("AiCoreLib");
  auto aicore_engine_builder = MakeShared<FakeAicoreLibOpsKernelBuilder>("AIcoreEngine");
  // {c0_value, bit_value}: c0_value = 2 ^ (bit_value - 1)
  // {1, 1}, {2, 2}, {4, 3}, {8, 4}, {16, 5}, {32, 6}, {64, 7}, {128, 8}, {256, 9}
  // 5 indicates that cube size is 16
  const Format src_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_NC1HWC0, FORMAT_RESERVED, 5));
  const Format dst_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_FRACTAL_Z, FORMAT_NHWC, 5));
  ffo->OpFormatByType(
      CONV2D, {
          .input_formats = {
              {src_format, GeShape(std::vector<int64_t>({1,2,16,16,16}))},
              {dst_format, GeShape(std::vector<int64_t>({4,1,16,16}))},
          },
          .output_formats = {
              {src_format, GeShape(std::vector<int64_t>({1,2,16,16,16}))}
          }
      });
      ge_env.Reset()
         .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AiCoreLib").GraphOptimizer("AIcoreEngine"))
         .Install(FakeEngine("DNN_HCCL").KernelInfoStore(kEngineNameHccl))
         .Install(FakeEngine("DNN_VM_RTS").KernelInfoStore(kEngineNameRts).GraphOptimizer("DNN_VM_RTS_GRAPH_OPTIMIZER_STORE"))
         .Install(FakeEngine("AiCoreLib").GraphOptimizer("FormatOp", ffo).KernelBuilder(ops_kernel_builder).KernelBuilder(aicore_engine_builder))
         .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(VARIABLE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(SHAPE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(IF).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp("PhonyConcat").InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE").InferShape(infer_fun))
         .Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(CONSTANTOP).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp("Identity").InfoStoreAndBuilder("AiCoreLib"))
         .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(ADD).InfoStoreAndBuilder("AiCoreLib").InferShape(infer_fun))
         .Install(FakeOp("ConcatD").InfoStoreAndBuilder("AiCoreLib"))
         .Install(FakeOp("Send").InfoStoreAndBuilder(kEngineNameRts))
         .Install(FakeOp("Recv").InfoStoreAndBuilder(kEngineNameRts))
         .Install(FakeOp(RELU).InfoStoreAndBuilder("AiCoreLib").InferShape(infer_fun))
         .Install(FakeOp(CONV2D).InfoStoreAndBuilder("AiCoreLib").InferShape(infer_fun))
         .Install(FakeOp(RESHAPE).InfoStoreAndBuilder("AiCoreLib"))
         .Install(FakeOp("RefData").InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(TRANSDATA).InfoStoreAndBuilder("AiCoreLib"));
}
/**
 *                    data2
 *     data1            |
 *       |         HcomAllGather2
 *  HcomAllGather1  /
 *       \         /
 *          concat
 *            |
 *         netoutput
 */
TEST_F(ConcatNotaskPassTest, allgather_connect_to_concat) {
  MockConcatNotaskPass mock_concat_notask_pass;
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto data1 = OP_CFG(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288});
  auto data2 = OP_CFG(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288});
  auto HcomAllGather1 = OP_CFG(HCOMALLGATHER)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {2048, 12288});
  auto HcomAllGather2 = OP_CFG(HCOMALLGATHER)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {2048, 12288});
  auto concat = OP_CFG("ConcatD")
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {4096, 12288});
  auto netoutput = OP_CFG(NETOUTPUT)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {4096, 12288});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->NODE("HcomAllGather_1", HcomAllGather1)->EDGE(0, 0)->NODE("concat", concat)
              ->NODE(NODE_NAME_NET_OUTPUT, netoutput));
    CHAIN(NODE("data_2", data1)->NODE("HcomAllGather_2", HcomAllGather2)->EDGE(0, 1)->NODE("concat", concat));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 0);
  op_desc->SetOpEngineName("AIcoreEngine");

  auto allgather_node1 = compute_graph->FindNode("HcomAllGather_1");
  auto allgather_desc1 = allgather_node1->GetOpDesc();
  allgather_desc1->SetOpEngineName("DNN_HCCL");

  auto allgather_node2 = compute_graph->FindNode("HcomAllGather_2");
  auto allgather_desc2 = allgather_node2->GetOpDesc();
  allgather_desc2->SetOpEngineName("DNN_HCCL");

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  
  EXPECT_EQ(ret, SUCCESS);
  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_EQ(notask, true);

  bool can_reuse = true;
  auto output_tensor_desc = allgather_desc1->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_EQ(can_reuse, false);

  can_reuse = true;
  auto output_tensor_desc2 = allgather_desc2->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc2, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_EQ(can_reuse, false);
}

TEST_F(ConcatNotaskPassTest, scalar_input_connect_to_concat) {
  MockConcatNotaskPass mock_concat_notask_pass;
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto data1 = OP_CFG(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {});
  auto data2 = OP_CFG(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {});
  auto relu1 = OP_CFG(RELU)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {});
  auto relu2 = OP_CFG(RELU)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {});
  auto concat = OP_CFG("ConcatD")
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {});
  auto netoutput = OP_CFG(NETOUTPUT)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->NODE("relu1", relu1)->EDGE(0, 0)->NODE("concat", concat)
              ->NODE(NODE_NAME_NET_OUTPUT, netoutput));
    CHAIN(NODE("data_2", data2)->NODE("relu2", relu2)->EDGE(0, 1)->NODE("concat", concat));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 0);
  op_desc->SetOpEngineName("AIcoreEngine");

  auto relu_node1 = compute_graph->FindNode("relu1");
  auto relu_desc1 = relu_node1->GetOpDesc();
  relu_desc1->SetOpEngineName("AIcoreEngine");

  auto relu_node2 = compute_graph->FindNode("relu2");
  auto relu_desc2 = relu_node2->GetOpDesc();
  relu_desc2->SetOpEngineName("AIcoreEngine");

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  
  EXPECT_EQ(ret, SUCCESS);
  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);
}

TEST_F(ConcatNotaskPassTest, single_scalar_input_connect_to_concat) {
  MockConcatNotaskPass mock_concat_notask_pass;
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto data1 = OP_CFG(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {});
  auto relu1 = OP_CFG(RELU)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {});
  auto concat = OP_CFG("ConcatD")
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {});
  auto netoutput = OP_CFG(NETOUTPUT)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->NODE("relu1", relu1)->EDGE(0, 0)->NODE("concat", concat)
              ->NODE(NODE_NAME_NET_OUTPUT, netoutput));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", -1);
  op_desc->SetOpEngineName("AIcoreEngine");

  auto relu_node1 = compute_graph->FindNode("relu1");
  auto relu_desc1 = relu_node1->GetOpDesc();
  relu_desc1->SetOpEngineName("AIcoreEngine");

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  
  EXPECT_EQ(ret, SUCCESS);
  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);
}

TEST_F(ConcatNotaskPassTest, concat_notask_lxfusion_op) {
  MockConcatNotaskPass mock_concat_notask_pass;
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto data1 = OP_CFG(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288});
  auto data2 = OP_CFG(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288});
  auto HcomAllGather1 = OP_CFG(RELU)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {2048, 12288});
  auto HcomAllGather2 = OP_CFG(RELU)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {2048, 12288});
  auto concat = OP_CFG("ConcatD")
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {4096, 12288});
  auto netoutput = OP_CFG(NETOUTPUT)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {4096, 12288});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->NODE("HcomAllGather_1", HcomAllGather1)->EDGE(0, 0)->NODE("concat_lxslice", concat)
              ->NODE(NODE_NAME_NET_OUTPUT, netoutput));
    CHAIN(NODE("data_2", data1)->NODE("HcomAllGather_2", HcomAllGather2)->EDGE(0, 1)->NODE("concat_lxslice", concat));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto concat_node = compute_graph->FindNode("concat_lxslice");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 0);
  op_desc->SetOpEngineName("AIcoreEngine");

  auto allgather_node1 = compute_graph->FindNode("HcomAllGather_1");
  auto allgather_desc1 = allgather_node1->GetOpDesc();
  allgather_desc1->SetOpEngineName("AIcoreEngine");

  auto allgather_node2 = compute_graph->FindNode("HcomAllGather_2");
  auto allgather_desc2 = allgather_node2->GetOpDesc();
  allgather_desc2->SetOpEngineName("AIcoreEngine");

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  
  EXPECT_EQ(ret, SUCCESS);
  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);

  bool can_reuse = true;
  auto output_tensor_desc = allgather_desc1->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);

  can_reuse = true;
  auto output_tensor_desc2 = allgather_desc2->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc2, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);
}

TEST_F(ConcatNotaskPassTest, input_mem_type_invalid) {
  MockConcatNotaskPass mock_concat_notask_pass;
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto data1 = OP_CFG(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288});
  auto data2 = OP_CFG(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288});
  auto HcomAllGather1 = OP_CFG(RELU)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {2048, 12288});
  auto HcomAllGather2 = OP_CFG(RELU)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {2048, 12288});
  auto concat = OP_CFG("ConcatD")
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {4096, 12288});
  auto netoutput = OP_CFG(NETOUTPUT)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {4096, 12288});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->NODE("HcomAllGather_1", HcomAllGather1)->EDGE(0, 0)->NODE("concat", concat)
              ->NODE(NODE_NAME_NET_OUTPUT, netoutput));
    CHAIN(NODE("data_2", data1)->NODE("HcomAllGather_2", HcomAllGather2)->EDGE(0, 1)->NODE("concat", concat));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 0);
  op_desc->SetOpEngineName("AIcoreEngine");
  std::vector<int64_t> in_mem_type {RT_MEMORY_L1, RT_MEMORY_L1};
  (void)ge::AttrUtils::SetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST,
                                in_mem_type);

  auto allgather_node1 = compute_graph->FindNode("HcomAllGather_1");
  auto allgather_desc1 = allgather_node1->GetOpDesc();
  allgather_desc1->SetOpEngineName("AIcoreEngine");

  auto allgather_node2 = compute_graph->FindNode("HcomAllGather_2");
  auto allgather_desc2 = allgather_node2->GetOpDesc();
  allgather_desc2->SetOpEngineName("AIcoreEngine");

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  
  EXPECT_EQ(ret, SUCCESS);
  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);

  bool can_reuse = true;
  auto output_tensor_desc = allgather_desc1->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);

  can_reuse = true;
  auto output_tensor_desc2 = allgather_desc2->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc2, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);
}

TEST_F(ConcatNotaskPassTest, output_mem_type_invalid) {
  MockConcatNotaskPass mock_concat_notask_pass;
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto data1 = OP_CFG(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288});
  auto data2 = OP_CFG(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288});
  auto HcomAllGather1 = OP_CFG(RELU)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {2048, 12288});
  auto HcomAllGather2 = OP_CFG(RELU)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {2048, 12288});
  auto concat = OP_CFG("ConcatD")
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {4096, 12288});
  auto netoutput = OP_CFG(NETOUTPUT)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {4096, 12288});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->NODE("HcomAllGather_1", HcomAllGather1)->EDGE(0, 0)->NODE("concat", concat)
              ->NODE(NODE_NAME_NET_OUTPUT, netoutput));
    CHAIN(NODE("data_2", data1)->NODE("HcomAllGather_2", HcomAllGather2)->EDGE(0, 1)->NODE("concat", concat));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 0);
  op_desc->SetOpEngineName("AIcoreEngine");
  std::vector<int64_t> in_mem_type {RT_MEMORY_L1};
  (void)ge::AttrUtils::SetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST,
                                in_mem_type);

  auto allgather_node1 = compute_graph->FindNode("HcomAllGather_1");
  auto allgather_desc1 = allgather_node1->GetOpDesc();
  allgather_desc1->SetOpEngineName("AIcoreEngine");

  auto allgather_node2 = compute_graph->FindNode("HcomAllGather_2");
  auto allgather_desc2 = allgather_node2->GetOpDesc();
  allgather_desc2->SetOpEngineName("AIcoreEngine");

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  session.BuildGraph(1, inputs);
  
  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);

  bool can_reuse = true;
  auto output_tensor_desc = allgather_desc1->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);

  can_reuse = true;
  auto output_tensor_desc2 = allgather_desc2->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc2, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);
}

TEST_F(ConcatNotaskPassTest, concat_notask_output_invalid) {
  MockConcatNotaskPass mock_concat_notask_pass;
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto data1 = OP_CFG(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288});
  auto data2 = OP_CFG(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288});
  auto HcomAllGather1 = OP_CFG(RELU)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {2048, 12288});
  auto HcomAllGather2 = OP_CFG(RELU)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {2048, 12288});
  auto concat = OP_CFG("ConcatD")
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {4096, 12288});
  auto netoutput = OP_CFG(NETOUTPUT)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {4096, 12288});
  auto relu1 = OP_CFG(RELU)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {4096, 12288});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->NODE("HcomAllGather_1", HcomAllGather1)->EDGE(0, 0)->NODE("concat", concat)
              ->NODE("relu_1", relu1)->NODE(NODE_NAME_NET_OUTPUT, netoutput));
    CHAIN(NODE("data_2", data1)->NODE("HcomAllGather_2", HcomAllGather2)->EDGE(0, 1)->NODE("concat", concat));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 0);
  op_desc->SetOpEngineName("AIcoreEngine");

  auto allgather_node1 = compute_graph->FindNode("HcomAllGather_1");
  auto allgather_desc1 = allgather_node1->GetOpDesc();
  allgather_desc1->SetOpEngineName("AIcoreEngine");

  auto allgather_node2 = compute_graph->FindNode("HcomAllGather_2");
  auto allgather_desc2 = allgather_node2->GetOpDesc();
  allgather_desc2->SetOpEngineName("AIcoreEngine");

  auto relu1_node = compute_graph->FindNode("relu_1");
  auto relu1_desc = relu1_node->GetOpDesc();
  relu1_desc->SetOpEngineName("AIcoreEngine");
  (void)ge::AttrUtils::SetBool(relu1_desc, ge::ATTR_NAME_NOTASK, true);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  
  EXPECT_EQ(ret, SUCCESS);
  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);

  bool can_reuse = true;
  auto output_tensor_desc = allgather_desc1->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);

  can_reuse = true;
  auto output_tensor_desc2 = allgather_desc2->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc2, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);
}

TEST_F(ConcatNotaskPassTest, concat_notask_pre_node_attr_invalid) {
  MockConcatNotaskPass mock_concat_notask_pass;
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto data1 = OP_CFG(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288});
  auto data2 = OP_CFG(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288});
  auto HcomAllGather1 = OP_CFG(RELU)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {2048, 12288});
  auto HcomAllGather2 = OP_CFG(RELU)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {2048, 12288});
  auto concat = OP_CFG("ConcatD")
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {4096, 12288});
  auto netoutput = OP_CFG(NETOUTPUT)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {4096, 12288});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->NODE("HcomAllGather_1", HcomAllGather1)->EDGE(0, 0)->NODE("concat", concat)
              ->NODE(NODE_NAME_NET_OUTPUT, netoutput));
    CHAIN(NODE("data_2", data1)->NODE("HcomAllGather_2", HcomAllGather2)->EDGE(0, 1)->NODE("concat", concat));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 0);
  op_desc->SetOpEngineName("AIcoreEngine");

  auto allgather_node1 = compute_graph->FindNode("HcomAllGather_1");
  auto allgather_desc1 = allgather_node1->GetOpDesc();
  allgather_desc1->SetOpEngineName("AIcoreEngine");

  auto allgather_node2 = compute_graph->FindNode("HcomAllGather_2");
  auto allgather_desc2 = allgather_node2->GetOpDesc();
  allgather_desc2->SetOpEngineName("AIcoreEngine");
  (void)ge::AttrUtils::SetBool(allgather_desc2, ge::ATTR_NAME_CONTINUOUS_INPUT, true);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  
  EXPECT_EQ(ret, SUCCESS);
  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);

  bool can_reuse = true;
  auto output_tensor_desc = allgather_desc1->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);

  can_reuse = true;
  auto output_tensor_desc2 = allgather_desc2->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc2, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);
}

TEST_F(ConcatNotaskPassTest, concat_notask_pre_node_attr_notask) {
  MockConcatNotaskPass mock_concat_notask_pass;
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto data1 = OP_CFG(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288});
  auto data2 = OP_CFG(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288});
  auto HcomAllGather1 = OP_CFG(RELU)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {2048, 12288});
  auto HcomAllGather2 = OP_CFG(RELU)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {2048, 12288});
  auto concat = OP_CFG("ConcatD")
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {4096, 12288});
  auto netoutput = OP_CFG(NETOUTPUT)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {4096, 12288});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->NODE("HcomAllGather_1", HcomAllGather1)->EDGE(0, 0)->NODE("concat", concat)
              ->NODE(NODE_NAME_NET_OUTPUT, netoutput));
    CHAIN(NODE("data_2", data1)->NODE("HcomAllGather_2", HcomAllGather2)->EDGE(0, 1)->NODE("concat", concat));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 0);
  op_desc->SetOpEngineName("AIcoreEngine");

  auto allgather_node1 = compute_graph->FindNode("HcomAllGather_1");
  auto allgather_desc1 = allgather_node1->GetOpDesc();
  allgather_desc1->SetOpEngineName("AIcoreEngine");

  auto allgather_node2 = compute_graph->FindNode("HcomAllGather_2");
  auto allgather_desc2 = allgather_node2->GetOpDesc();
  allgather_desc2->SetOpEngineName("AIcoreEngine");
  (void)ge::AttrUtils::SetBool(allgather_desc2, ge::ATTR_NAME_NOTASK, true);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  
  EXPECT_EQ(ret, SUCCESS);
  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);

  bool can_reuse = true;
  auto output_tensor_desc = allgather_desc1->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);

  can_reuse = true;
  auto output_tensor_desc2 = allgather_desc2->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc2, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);
}

TEST_F(ConcatNotaskPassTest, concat_notask_input_mem_type_not_same) {
  MockConcatNotaskPass mock_concat_notask_pass;
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto data1 = OP_CFG(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288});
  auto data2 = OP_CFG(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288});
  auto HcomAllGather1 = OP_CFG(RELU)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {2048, 12288});
  auto HcomAllGather2 = OP_CFG(RELU)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {2048, 12288});
  auto concat = OP_CFG("ConcatD")
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {4096, 12288});
  auto netoutput = OP_CFG(NETOUTPUT)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {4096, 12288});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->NODE("HcomAllGather_1", HcomAllGather1)->EDGE(0, 0)->NODE("concat", concat)
              ->NODE(NODE_NAME_NET_OUTPUT, netoutput));
    CHAIN(NODE("data_2", data1)->NODE("HcomAllGather_2", HcomAllGather2)->EDGE(0, 1)->NODE("concat", concat));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 0);
  op_desc->SetOpEngineName("AIcoreEngine");

  auto allgather_node1 = compute_graph->FindNode("HcomAllGather_1");
  auto allgather_desc1 = allgather_node1->GetOpDesc();
  allgather_desc1->SetOpEngineName("AIcoreEngine");

  auto allgather_node2 = compute_graph->FindNode("HcomAllGather_2");
  auto allgather_desc2 = allgather_node2->GetOpDesc();
  allgather_desc2->SetOpEngineName("AIcoreEngine");
  std::vector<int64_t> in_mem_type {RT_MEMORY_L1};
  (void)ge::AttrUtils::SetListInt(allgather_desc2, ATTR_NAME_OUTPUT_MEM_TYPE_LIST,
                                in_mem_type);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  session.BuildGraph(1, inputs);
  
  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);

  bool can_reuse = true;
  auto output_tensor_desc = allgather_desc1->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);

  can_reuse = true;
  auto output_tensor_desc2 = allgather_desc2->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc2, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);
}

TEST_F(ConcatNotaskPassTest, data_connect_to_concat) {
  MockConcatNotaskPass mock_concat_notask_pass;
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto data1 = OP_CFG(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {16384, 12288});
  auto data2 = OP_CFG(DATA)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288});
  auto HcomAllGather1 = OP_CFG(RELU)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {2048, 12288});
  auto HcomAllGather2 = OP_CFG(RELU)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {2048, 12288});
  auto concat = OP_CFG("ConcatD")
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {4096, 12288});
  auto netoutput = OP_CFG(NETOUTPUT)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {4096, 12288});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->NODE("concat", concat)
              ->NODE(NODE_NAME_NET_OUTPUT, netoutput));
    CHAIN(NODE("data_2", data1)->NODE("HcomAllGather_2", HcomAllGather2)->EDGE(0, 1)->NODE("concat", concat));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 0);
  op_desc->SetOpEngineName("AIcoreEngine");

  auto allgather_node2 = compute_graph->FindNode("HcomAllGather_2");
  auto allgather_desc2 = allgather_node2->GetOpDesc();
  allgather_desc2->SetOpEngineName("AIcoreEngine");

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  session.BuildGraph(1, inputs);
  
  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);

  bool can_reuse = true;
  auto output_tensor_desc2 = allgather_desc2->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc2, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);
}

gert::LowerResult LoweringFoo(const ge::NodePtr &node, const gert::LowerInput &lower_input) {
   return {HyperStatus::Success(),
                              {gert::bg::ValueHolder::CreateFeed(2)},
                              {gert::bg::ValueHolder::CreateFeed(0)},
                              {gert::bg::DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};
}
REGISTER_NODE_CONVERTER("_lower_foo", LoweringFoo);

TEST_F(ConcatNotaskPassTest, allgather_connect_to_concat_unknownop) {
  MockConcatNotaskPass mock_concat_notask_pass;
  vector<std::string> engine_list = {"AIcoreEngine"};
  auto data1 = OP_CFG(DATA)
      .Attr(ATTR_NAME_INDEX, 0)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, -1});
  auto data2 = OP_CFG(DATA)
      .Attr(ATTR_NAME_INDEX, 1)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288});
  auto data3 = OP_CFG(DATA)
      .Attr(ATTR_NAME_INDEX, 2)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288});
  auto data4 = OP_CFG(DATA)
      .Attr(ATTR_NAME_INDEX, 3)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {256, 12288});
  auto relu1 = OP_CFG(RELU)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {2048, -1});
  auto relu2 = OP_CFG(RELU)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {2048, -1});
  auto relu3 = OP_CFG(RELU)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {2048, 12288});
  auto relu4 = OP_CFG(RELU)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {2048, 12288});
  auto concat1 = OP_CFG("ConcatD")
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {4096, -1});
  auto concat2 = OP_CFG("ConcatD")
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {4096, -1});
  auto netoutput = OP_CFG(NETOUTPUT)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, {4096, 12288});

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->NODE("relu_1", relu1)->EDGE(0, 0)->NODE("concat_1", concat1)
              ->NODE(NODE_NAME_NET_OUTPUT, netoutput));
    CHAIN(NODE("data_2", data2)->NODE("relu_2", relu2)->EDGE(0, 1)->NODE("concat_1", concat1));
    CHAIN(NODE("data_3", data3)->NODE("relu_3", relu3)->EDGE(0, 0)->NODE("concat_2", concat2)
          ->NODE(NODE_NAME_NET_OUTPUT, netoutput));
    CHAIN(NODE("data_4", data4)->NODE("relu_4", relu4)->EDGE(0, 1)->NODE("concat_2", concat2));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto concat_node = compute_graph->FindNode("concat_1");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 0);
  op_desc->SetOpEngineName("AIcoreEngine");
  ge::AttrUtils::SetStr(op_desc, "_ge_attr_lowering_func", "_lower_foo");

  auto allgather_node1 = compute_graph->FindNode("relu_1");
  auto allgather_desc1 = allgather_node1->GetOpDesc();
  allgather_desc1->SetOpEngineName("AIcoreEngine");
  ge::AttrUtils::SetStr(allgather_desc1, "_ge_attr_lowering_func", "_lower_foo");

  auto allgather_node2 = compute_graph->FindNode("relu_2");
  auto allgather_desc2 = allgather_node2->GetOpDesc();
  allgather_desc2->SetOpEngineName("AIcoreEngine");
  ge::AttrUtils::SetStr(allgather_desc2, "_ge_attr_lowering_func", "_lower_foo");
  ConcatNotaskPass pass;
  op_desc->MutableInputDesc(0)->SetOriginShape(GeShape({4096, -1}));
  op_desc->MutableInputDesc(0)->SetShape(GeShape({4096, -1}));
  pass.Run(compute_graph);
  op_desc->MutableInputDesc(0)->SetOriginShape(GeShape({4096, 12288}));
  op_desc->MutableInputDesc(0)->SetShape(GeShape({4096, 12288}));
  op_desc->MutableOutputDesc(0)->SetOriginShape(GeShape({4096, -1}));
  op_desc->MutableOutputDesc(0)->SetShape(GeShape({4096, -1}));
  pass.Run(compute_graph);
  op_desc->MutableOutputDesc(0)->SetOriginShape(GeShape({4096, 12288}));
  op_desc->MutableOutputDesc(0)->SetShape(GeShape({4096, 12288}));
  compute_graph->SetGraphUnknownFlag(true);
  pass.Run(compute_graph);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);

  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);
  
  for (const auto &graph : compute_graph->GetAllSubgraphs()) {
    pass.Run(graph);
  }
  
  EXPECT_NE(ret, SUCCESS);
  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);

  bool can_reuse = true;
  auto output_tensor_desc = allgather_desc1->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);

  can_reuse = true;
  auto output_tensor_desc2 = allgather_desc2->MutableOutputDesc(0);
  ge::AttrUtils::GetBool(output_tensor_desc2, "can_reused_for_concat_optimize", can_reuse);
  EXPECT_TRUE(can_reuse == true);
}

static Graph BuildRefDataWithStroageFormatTrainGraph1(Format storage_format, Format origin_format,
                                                             const std::vector<int64_t> &origin_shape,
                                                             const std::string &expand_dims_rule) {
  std::vector<int64_t> shape = {2,2,3,2};  // HWCN
  DEF_GRAPH(graph) {
    auto data1 = OP_DATA(0).TensorDesc(FORMAT_NCHW, DT_FLOAT16, origin_shape).Attr(ATTR_NAME_INDEX, 0).Build("data1");
    auto refdata1 = OP_CFG("RefData")
                        .TensorDesc(FORMAT_NC1HWC0, DT_FLOAT16, {})
                .InCnt(1)
                .OutCnt(1)
              .Attr(ATTR_NAME_INDEX, 1)
                        .InNames({"x"})
                        .OutNames({"y"})
                        .Build("refdata1");
    refdata1->MutableInputDesc(0)->SetOriginFormat(origin_format);
    (void)AttrUtils::SetBool(refdata1->MutableInputDesc(0), ATTR_NAME_ORIGIN_FORMAT_IS_SET, true);
    refdata1->MutableInputDesc(0)->SetOriginShape(GeShape({1, 2, 4, 5}));
    refdata1->MutableInputDesc(0)->SetFormat(storage_format);
    refdata1->MutableInputDesc(0)->SetShape(GeShape());
    refdata1->MutableInputDesc(0)->SetExpandDimsRule(expand_dims_rule);
    refdata1->UpdateOutputDesc(0, refdata1->GetInputDesc(0));

    auto conv2d1 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).Build("conv2d1");
    conv2d1->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d1->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
    conv2d1->MutableInputDesc(1)->SetOriginFormat(FORMAT_HWCN);
    conv2d1->MutableInputDesc(1)->SetFormat(FORMAT_HWCN);
    conv2d1->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d1->MutableOutputDesc(0)->SetFormat(FORMAT_NCHW);

    auto data2 = OP_DATA(0).TensorDesc(FORMAT_NCHW, DT_FLOAT16, origin_shape).Attr(ATTR_NAME_INDEX, 0).Build("data2");
    auto refdata2 = OP_CFG("RefData")
                        .TensorDesc(FORMAT_NC1HWC0, DT_FLOAT16, {})
              .InCnt(1)
              .OutCnt(1)
                        .Attr(ATTR_NAME_INDEX, 1)
                        .InNames({"x"})
                        .OutNames({"y"})
                        .Build("refdata2");
    refdata2->MutableInputDesc(0)->SetOriginFormat(origin_format);
    (void)AttrUtils::SetBool(refdata2->MutableInputDesc(0), ATTR_NAME_ORIGIN_FORMAT_IS_SET, true);
    refdata2->MutableInputDesc(0)->SetOriginShape(GeShape({1, 2, 4, 5}));
    refdata2->MutableInputDesc(0)->SetFormat(storage_format);
    refdata2->MutableInputDesc(0)->SetShape(GeShape());
    refdata2->MutableInputDesc(0)->SetExpandDimsRule(expand_dims_rule);
    refdata2->UpdateOutputDesc(0, refdata2->GetInputDesc(0));

    auto conv2d2 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).Build("conv2d2");
    conv2d2->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d2->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
    conv2d2->MutableInputDesc(1)->SetOriginFormat(FORMAT_HWCN);
    conv2d2->MutableInputDesc(1)->SetFormat(FORMAT_HWCN);
    conv2d2->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d2->MutableOutputDesc(0)->SetFormat(FORMAT_NCHW);

    auto concat = OP_CFG("ConcatD").InCnt(2).OutCnt(1).Build("concat");
    concat->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    concat->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
    concat->MutableInputDesc(1)->SetOriginFormat(FORMAT_NCHW);
    concat->MutableInputDesc(1)->SetFormat(FORMAT_NCHW);
    concat->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    concat->MutableOutputDesc(0)->SetFormat(FORMAT_NCHW);
    concat->SetOpEngineName("AIcoreEngine");

    CHAIN(NODE(data1)->NODE(conv2d1)->NODE(concat));
    CHAIN(NODE(refdata1)->EDGE(0, 1)->NODE(conv2d1));
    CHAIN(NODE(data2)->NODE(conv2d2));
    CHAIN(NODE(refdata2)->EDGE(0, 1)->NODE(conv2d2)->EDGE(0, 1)->NODE(concat));

    ADD_OUTPUT(concat, 0);
  };

  return ToGeGraph(graph);
}

static Graph BuildRefDataWithStroageFormatTrainGraphNHWC(Format storage_format, Format origin_format,
                                                             const std::vector<int64_t> &origin_shape,
                                                             const std::string &expand_dims_rule) {
  std::vector<int64_t> shape = {2,2,3,2};  // HWCN
  DEF_GRAPH(graph) {
    auto data1 = OP_DATA(0).TensorDesc(origin_format, DT_FLOAT16, origin_shape).Attr(ATTR_NAME_INDEX, 0).Build("data1");
    auto refdata1 = OP_CFG("RefData")
                        .TensorDesc(FORMAT_NC1HWC0, DT_FLOAT16, {})
                        .InCnt(1)
                        .OutCnt(1)
                        .Attr(ATTR_NAME_INDEX, 1)
                        .InNames({"x"})
                        .OutNames({"y"})
                        .Build("refdata1");
    refdata1->MutableInputDesc(0)->SetOriginFormat(origin_format);
    (void)AttrUtils::SetBool(refdata1->MutableInputDesc(0), ATTR_NAME_ORIGIN_FORMAT_IS_SET, true);
    refdata1->MutableInputDesc(0)->SetOriginShape(GeShape({1, 2, 4, 5}));
    refdata1->MutableInputDesc(0)->SetFormat(storage_format);
    refdata1->MutableInputDesc(0)->SetShape(GeShape());
    refdata1->MutableInputDesc(0)->SetExpandDimsRule(expand_dims_rule);
    refdata1->UpdateOutputDesc(0, refdata1->GetInputDesc(0));

    auto conv2d1 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).Build("conv2d1");
    conv2d1->MutableInputDesc(0)->SetOriginFormat(origin_format);
    conv2d1->MutableInputDesc(0)->SetFormat(origin_format);
    conv2d1->MutableInputDesc(1)->SetOriginFormat(origin_format);
    conv2d1->MutableInputDesc(1)->SetFormat(origin_format);
    conv2d1->MutableOutputDesc(0)->SetOriginFormat(origin_format);
    conv2d1->MutableOutputDesc(0)->SetFormat(origin_format);

    auto data2 = OP_DATA(0).TensorDesc(origin_format, DT_FLOAT16, origin_shape).Attr(ATTR_NAME_INDEX, 0).Build("data2");
    auto refdata2 = OP_CFG("RefData")
                        .TensorDesc(FORMAT_NC1HWC0, DT_FLOAT16, {})
                        .InCnt(1)
                        .OutCnt(1)
                        .Attr(ATTR_NAME_INDEX, 1)
                        .InNames({"x"})
                        .OutNames({"y"})
                        .Build("refdata2");
    refdata2->MutableInputDesc(0)->SetOriginFormat(origin_format);
    (void)AttrUtils::SetBool(refdata2->MutableInputDesc(0), ATTR_NAME_ORIGIN_FORMAT_IS_SET, true);
    refdata2->MutableInputDesc(0)->SetOriginShape(GeShape({1, 2, 4, 5}));
    refdata2->MutableInputDesc(0)->SetFormat(storage_format);
    refdata2->MutableInputDesc(0)->SetShape(GeShape());
    refdata2->MutableInputDesc(0)->SetExpandDimsRule(expand_dims_rule);
    refdata2->UpdateOutputDesc(0, refdata2->GetInputDesc(0));

    auto conv2d2 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).Build("conv2d2");
    conv2d2->MutableInputDesc(0)->SetOriginFormat(origin_format);
    conv2d2->MutableInputDesc(0)->SetFormat(origin_format);
    conv2d2->MutableInputDesc(1)->SetOriginFormat(origin_format);
    conv2d2->MutableInputDesc(1)->SetFormat(origin_format);
    conv2d2->MutableOutputDesc(0)->SetOriginFormat(origin_format);
    conv2d2->MutableOutputDesc(0)->SetFormat(origin_format);

    auto concat = OP_CFG("ConcatD").InCnt(2).OutCnt(1).Build("concat");
    concat->MutableInputDesc(0)->SetOriginFormat(origin_format);
    concat->MutableInputDesc(0)->SetFormat(origin_format);
    concat->MutableInputDesc(1)->SetOriginFormat(origin_format);
    concat->MutableInputDesc(1)->SetFormat(origin_format);
    concat->MutableOutputDesc(0)->SetOriginFormat(origin_format);
    concat->MutableOutputDesc(0)->SetFormat(origin_format);
    concat->SetOpEngineName("AIcoreEngine");

    CHAIN(NODE(data1)->NODE(conv2d1)->NODE(concat));
    CHAIN(NODE(refdata1)->EDGE(0, 1)->NODE(conv2d1));
    CHAIN(NODE(data2)->NODE(conv2d2));
    CHAIN(NODE(refdata2)->EDGE(0, 1)->NODE(conv2d2)->EDGE(0, 1)->NODE(concat));

    ADD_OUTPUT(concat, 0);
  };

  return ToGeGraph(graph);
}

static Graph BuildRefDataWithStroageFormatTrainGraph2(Format storage_format, Format origin_format,
                                                             const std::vector<int64_t> &origin_shape,
                                                             const std::string &expand_dims_rule) {
  std::vector<int64_t> shape = {2,2,3,2};  // HWCN
  DEF_GRAPH(graph) {
    auto data1 = OP_DATA(0).TensorDesc(FORMAT_NCHW, DT_FLOAT16, {1, 32, 16, 16}).Attr(ATTR_NAME_INDEX, 0).Build("data1");
    auto refdata1 = OP_CFG("RefData")
                        .TensorDesc(FORMAT_NC1HWC0, DT_FLOAT16, {})
                        .InCnt(1)
                        .OutCnt(1)
                        .Attr(ATTR_NAME_INDEX, 1)
                        .InNames({"x"})
                        .OutNames({"y"})
                        .Build("refdata1");
    refdata1->MutableInputDesc(0)->SetOriginFormat(origin_format);
    (void)AttrUtils::SetBool(refdata1->MutableInputDesc(0), ATTR_NAME_ORIGIN_FORMAT_IS_SET, true);
    refdata1->MutableInputDesc(0)->SetOriginShape(GeShape(origin_shape));
    refdata1->MutableInputDesc(0)->SetFormat(storage_format);
    refdata1->MutableInputDesc(0)->SetShape(GeShape());
    refdata1->MutableInputDesc(0)->SetExpandDimsRule(expand_dims_rule);
    refdata1->UpdateOutputDesc(0, refdata1->GetInputDesc(0));

    auto conv2d1 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).Build("conv2d1");
    conv2d1->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d1->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
    conv2d1->MutableInputDesc(1)->SetOriginFormat(FORMAT_HWCN);
    conv2d1->MutableInputDesc(1)->SetFormat(FORMAT_HWCN);
    conv2d1->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d1->MutableOutputDesc(0)->SetFormat(FORMAT_HWCN);

    auto data2 = OP_DATA(0).TensorDesc(FORMAT_NCHW, DT_FLOAT16, {1, 32, 16, 16}).Attr(ATTR_NAME_INDEX, 0).Build("data2");
    auto refdata2 = OP_CFG("RefData")
                        .TensorDesc(FORMAT_NC1HWC0, DT_FLOAT16, {})
                        .InCnt(1)
                        .OutCnt(1)
                        .Attr(ATTR_NAME_INDEX, 1)
                        .InNames({"x"})
                        .OutNames({"y"})
                        .Build("refdata2");
    refdata2->MutableInputDesc(0)->SetOriginFormat(origin_format);
    (void)AttrUtils::SetBool(refdata2->MutableInputDesc(0), ATTR_NAME_ORIGIN_FORMAT_IS_SET, true);
    refdata2->MutableInputDesc(0)->SetOriginShape(GeShape(origin_shape));
    refdata2->MutableInputDesc(0)->SetFormat(storage_format);
    refdata2->MutableInputDesc(0)->SetShape(GeShape());
    refdata2->MutableInputDesc(0)->SetExpandDimsRule(expand_dims_rule);
    refdata2->UpdateOutputDesc(0, refdata2->GetInputDesc(0));

    auto conv2d2 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).Build("conv2d2");
    conv2d2->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d2->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
    conv2d2->MutableInputDesc(1)->SetOriginFormat(FORMAT_HWCN);
    conv2d2->MutableInputDesc(1)->SetFormat(FORMAT_HWCN);
    conv2d2->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d2->MutableOutputDesc(0)->SetFormat(FORMAT_HWCN);

    auto concat = OP_CFG("ConcatD").InCnt(2).OutCnt(1).Build("concat");
    concat->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    concat->MutableInputDesc(0)->SetFormat(FORMAT_HWCN);
    concat->MutableInputDesc(1)->SetOriginFormat(FORMAT_NCHW);
    concat->MutableInputDesc(1)->SetFormat(FORMAT_HWCN);
    concat->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    concat->MutableOutputDesc(0)->SetFormat(FORMAT_NCHW);
    concat->SetOpEngineName("AIcoreEngine");

    CHAIN(NODE(data1)->NODE(conv2d1)->NODE(concat));
    CHAIN(NODE(refdata1)->EDGE(0, 1)->NODE(conv2d1));
    CHAIN(NODE(data2)->NODE(conv2d2));
    CHAIN(NODE(refdata2)->EDGE(0, 1)->NODE(conv2d2)->EDGE(0, 1)->NODE(concat));

    ADD_OUTPUT(concat, 0);
  };

  return ToGeGraph(graph);
}

static Graph BuildRefDataWithStroageFormatTrainGraphSameAnchor(Format storage_format, Format origin_format,
                                                             const std::vector<int64_t> &origin_shape,
                                                             const std::string &expand_dims_rule) {
  std::vector<int64_t> shape = {2,2,3,2};  // HWCN
  DEF_GRAPH(graph) {
    auto data1 = OP_DATA(0).TensorDesc(FORMAT_NCHW, DT_FLOAT16, origin_shape).Attr(ATTR_NAME_INDEX, 0).Build("data1");
    auto refdata1 = OP_CFG("RefData")
                        .TensorDesc(FORMAT_NC1HWC0, DT_FLOAT16, {})
                        .InCnt(1)
                        .OutCnt(1)
                        .Attr(ATTR_NAME_INDEX, 1)
                        .InNames({"x"})
                        .OutNames({"y"})
                        .Build("refdata1");
    refdata1->MutableInputDesc(0)->SetOriginFormat(origin_format);
    (void)AttrUtils::SetBool(refdata1->MutableInputDesc(0), ATTR_NAME_ORIGIN_FORMAT_IS_SET, true);
    refdata1->MutableInputDesc(0)->SetOriginShape(GeShape({1, 2, 4, 5}));
    refdata1->MutableInputDesc(0)->SetFormat(storage_format);
    refdata1->MutableInputDesc(0)->SetShape(GeShape());
    refdata1->MutableInputDesc(0)->SetExpandDimsRule(expand_dims_rule);
    refdata1->UpdateOutputDesc(0, refdata1->GetInputDesc(0));

    auto conv2d1 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).Build("conv2d1");
    conv2d1->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d1->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
    conv2d1->MutableInputDesc(1)->SetOriginFormat(FORMAT_HWCN);
    conv2d1->MutableInputDesc(1)->SetFormat(FORMAT_HWCN);
    conv2d1->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d1->MutableOutputDesc(0)->SetFormat(FORMAT_NCHW);

    auto data2 = OP_DATA(0).TensorDesc(FORMAT_NCHW, DT_FLOAT16, origin_shape).Attr(ATTR_NAME_INDEX, 0).Build("data2");
    auto refdata2 = OP_CFG("RefData")
                        .TensorDesc(FORMAT_NC1HWC0, DT_FLOAT16, {})
                        .InCnt(1)
                        .OutCnt(1)
                        .Attr(ATTR_NAME_INDEX, 1)
                        .InNames({"x"})
                        .OutNames({"y"})
                        .Build("refdata2");
    refdata2->MutableInputDesc(0)->SetOriginFormat(origin_format);
    (void)AttrUtils::SetBool(refdata2->MutableInputDesc(0), ATTR_NAME_ORIGIN_FORMAT_IS_SET, true);
    refdata2->MutableInputDesc(0)->SetOriginShape(GeShape({1, 2, 4, 5}));
    refdata2->MutableInputDesc(0)->SetFormat(storage_format);
    refdata2->MutableInputDesc(0)->SetShape(GeShape());
    refdata2->MutableInputDesc(0)->SetExpandDimsRule(expand_dims_rule);
    refdata2->UpdateOutputDesc(0, refdata2->GetInputDesc(0));

    auto conv2d2 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).Build("conv2d2");
    conv2d2->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d2->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
    conv2d2->MutableInputDesc(1)->SetOriginFormat(FORMAT_HWCN);
    conv2d2->MutableInputDesc(1)->SetFormat(FORMAT_HWCN);
    conv2d2->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d2->MutableOutputDesc(0)->SetFormat(FORMAT_NCHW);

    auto concat = OP_CFG("ConcatD").InCnt(2).OutCnt(1).Build("concat");
    concat->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    concat->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
    concat->MutableInputDesc(1)->SetOriginFormat(FORMAT_NCHW);
    concat->MutableInputDesc(1)->SetFormat(FORMAT_NCHW);
    concat->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    concat->MutableOutputDesc(0)->SetFormat(FORMAT_NCHW);
    concat->SetOpEngineName("AIcoreEngine");

    CHAIN(NODE(data1)->NODE(conv2d1)->NODE(concat));
    CHAIN(NODE(refdata1)->EDGE(0, 1)->NODE(conv2d1));
    CHAIN(NODE(conv2d1)->EDGE(0, 1)->NODE(concat));

    ADD_OUTPUT(concat, 0);
  };

  return ToGeGraph(graph);
}

/**
 *                   data2   refdata2
 *  data1   refdata2     \    /  
 *      \    /          conv2d2
 *     conv2d1            /
 *         \             /
 *             concat
 *               |
 *            netoutput
 */
TEST_F(ConcatNotaskPassTest, conv_5hd_connect_to_concat) {
  GeRunningEnvFaker ge_env;
  Fake5DNodeEngine<FakeFormatsOptimizer>(ge_env, {1,2,16,16,16});
  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
    // 3.构图
  auto train_graph =
      BuildRefDataWithStroageFormatTrainGraph1(FORMAT_NC1HWC0, FORMAT_NCHW, {1, 32, 16, 16}, "");

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 1);
  // 4.add graph
  auto ret = session.AddGraph(3, train_graph);
  EXPECT_EQ(ret, SUCCESS);
  
  // 5.run graph with stream async
  std::vector<ge::Tensor> g1_inputs;
  std::vector<ge::Tensor> g1_outputs(1);
  auto input_tensor = GenerateTensor({1, 2, 4, 5});
  input_tensor->MutableTensorDesc().SetOriginFormat(FORMAT_NCHW);
  input_tensor->MutableTensorDesc().SetFormat(FORMAT_NC1HWC0);
  input_tensor->MutableTensorDesc().SetOriginShape(GeShape({1, 2, 4, 5}));
  input_tensor->MutableTensorDesc().SetShape(GeShape());
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*input_tensor)); // data1
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*input_tensor)); // refdata1
  ret = session.CompileGraph(3);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_EQ(notask, true);
}

TEST_F(ConcatNotaskPassTest, conv_5hd_connect_to_concat_pre_outanchor_can_reuse) {
  GeRunningEnvFaker ge_env;
  Fake5DNodeEngine<FakeFormatsOptimizer>(ge_env, {1,2,16,16,16});
  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
    // 3.构图
  auto train_graph =
      BuildRefDataWithStroageFormatTrainGraph1(FORMAT_NC1HWC0, FORMAT_NCHW, {1, 32, 16, 16}, "");

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 1);
  auto conv2d1_node2 = compute_graph->FindNode("conv2d1");
  auto op_desc2 = conv2d1_node2->GetOpDesc();
  auto input_tensor_desc = op_desc2->MutableInputDesc(0);
  (void)ge::AttrUtils::SetBool(input_tensor_desc, "can_reused_for_concat_optimize", false);
  // 4.add graph
  auto ret = session.AddGraph(3, train_graph);
  EXPECT_EQ(ret, SUCCESS);

  // 5.run graph with stream async
  std::vector<ge::Tensor> g1_inputs;
  std::vector<ge::Tensor> g1_outputs(1);
  auto input_tensor = GenerateTensor({1, 2, 4, 5});
  input_tensor->MutableTensorDesc().SetOriginFormat(FORMAT_NCHW);
  input_tensor->MutableTensorDesc().SetFormat(FORMAT_NC1HWC0);
  input_tensor->MutableTensorDesc().SetOriginShape(GeShape({1, 2, 4, 5}));
  input_tensor->MutableTensorDesc().SetShape(GeShape());
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*input_tensor)); // data1
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*input_tensor)); // refdata1
  ret = session.CompileGraph(3);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);
}

TEST_F(ConcatNotaskPassTest, conv_5hd_connect_to_concat_has_same_anchor) {
  GeRunningEnvFaker ge_env;
  Fake5DNodeEngine<FakeFormatsOptimizer>(ge_env, {1,2,16,16,16});
  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
    // 3.构图
  auto train_graph =
      BuildRefDataWithStroageFormatTrainGraphSameAnchor(FORMAT_NC1HWC0, FORMAT_NCHW, {1, 32, 16, 16}, "");

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 1);
  // 4.add graph
  auto ret = session.AddGraph(3, train_graph);
  EXPECT_EQ(ret, SUCCESS);

  // 5.run graph with stream async
  std::vector<ge::Tensor> g1_inputs;
  std::vector<ge::Tensor> g1_outputs(1);
  auto input_tensor = GenerateTensor({1, 2, 4, 5});
  input_tensor->MutableTensorDesc().SetOriginFormat(FORMAT_NCHW);
  input_tensor->MutableTensorDesc().SetFormat(FORMAT_NC1HWC0);
  input_tensor->MutableTensorDesc().SetOriginShape(GeShape({1, 2, 4, 5}));
  input_tensor->MutableTensorDesc().SetShape(GeShape());
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*input_tensor)); // data1
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*input_tensor)); // refdata1
  ret = session.CompileGraph(3);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);
}

TEST_F(ConcatNotaskPassTest, conv_5hd_connect_to_concat_c1_not_align) {
  GeRunningEnvFaker ge_env;
  Fake5DNodeEngine<FakeFormatsOptimizer>(ge_env, {1,2,16,16,16});
  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
    // 3.构图
  auto train_graph =
      BuildRefDataWithStroageFormatTrainGraph1(FORMAT_NC1HWC0, FORMAT_NCHW, {1, 31, 16, 16}, "");

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 1);
  // 4.add graph
  auto ret = session.AddGraph(3, train_graph);
  EXPECT_EQ(ret, SUCCESS);

  // 5.run graph with stream async
  std::vector<ge::Tensor> g1_inputs;
  std::vector<ge::Tensor> g1_outputs(1);
  auto input_tensor = GenerateTensor({1, 2, 4, 5});
  input_tensor->MutableTensorDesc().SetOriginFormat(FORMAT_NCHW);
  input_tensor->MutableTensorDesc().SetFormat(FORMAT_NC1HWC0);
  input_tensor->MutableTensorDesc().SetOriginShape(GeShape({1, 2, 4, 5}));
  input_tensor->MutableTensorDesc().SetShape(GeShape());
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*input_tensor)); // data1
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*input_tensor)); // refdata1
  ret = session.CompileGraph(3);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);
}

TEST_F(ConcatNotaskPassTest, conv_5hd_connect_to_concat_tensor_not_align) {
  GeRunningEnvFaker ge_env;
  Fake5DNodeEngine<FakeFormatsOptimizer>(ge_env, {1,2,15,15,15});
  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
    // 3.构图
  auto train_graph =
      BuildRefDataWithStroageFormatTrainGraph1(FORMAT_NC1HWC0, FORMAT_NCHW, {1, 32, 16, 16}, "");

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 1);
  // 4.add graph
  auto ret = session.AddGraph(3, train_graph);
  EXPECT_EQ(ret, SUCCESS);

  // 5.run graph with stream async
  std::vector<ge::Tensor> g1_inputs;
  std::vector<ge::Tensor> g1_outputs(1);
  auto input_tensor = GenerateTensor({1, 2, 4, 5});
  input_tensor->MutableTensorDesc().SetOriginFormat(FORMAT_NCHW);
  input_tensor->MutableTensorDesc().SetFormat(FORMAT_NC1HWC0);
  input_tensor->MutableTensorDesc().SetOriginShape(GeShape({1, 2, 4, 5}));
  input_tensor->MutableTensorDesc().SetShape(GeShape());
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*input_tensor)); // data1
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*input_tensor)); // refdata1
  ret = session.CompileGraph(3);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);
}

TEST_F(ConcatNotaskPassTest, conv_5hd_connect_to_concat_dim_minus3) {
  GeRunningEnvFaker ge_env;
  Fake5DNodeEngine<FakeFormatsOptimizer>(ge_env, {1,2,16,16,16});
  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
    // 3.构图
  auto train_graph =
      BuildRefDataWithStroageFormatTrainGraph1(FORMAT_NC1HWC0, FORMAT_NCHW, {1, 32, 16, 16}, "");

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", -3);
  // 4.add graph
  auto ret = session.AddGraph(3, train_graph);
  EXPECT_EQ(ret, SUCCESS);

  // 5.run graph with stream async
  std::vector<ge::Tensor> g1_inputs;
  std::vector<ge::Tensor> g1_outputs(1);
  auto input_tensor = GenerateTensor({1, 2, 4, 5});
  input_tensor->MutableTensorDesc().SetOriginFormat(FORMAT_NCHW);
  input_tensor->MutableTensorDesc().SetFormat(FORMAT_NC1HWC0);
  input_tensor->MutableTensorDesc().SetOriginShape(GeShape({1, 2, 4, 5}));
  input_tensor->MutableTensorDesc().SetShape(GeShape());
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*input_tensor)); // data1
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*input_tensor)); // refdata1
  ret = session.CompileGraph(3);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_EQ(notask, true);
}

TEST_F(ConcatNotaskPassTest, conv_5hd_connect_to_concat_dim_minus1) {
  GeRunningEnvFaker ge_env;
  Fake5DNodeEngine<FakeFormatsOptimizer>(ge_env, {1,1,256,256,16});
  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
    // 3.构图
  auto train_graph =
      BuildRefDataWithStroageFormatTrainGraphNHWC(FORMAT_NC1HWC0, FORMAT_NHWC, {1, 256, 256, 2}, "");

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", -1);
  // 4.add graph
  auto ret = session.AddGraph(3, train_graph);
  EXPECT_EQ(ret, SUCCESS);

  // 5.run graph with stream async
  std::vector<ge::Tensor> g1_inputs;
  std::vector<ge::Tensor> g1_outputs(1);
  auto input_tensor = GenerateTensor({1, 2, 4, 5});
  input_tensor->MutableTensorDesc().SetOriginFormat(FORMAT_NHWC);
  input_tensor->MutableTensorDesc().SetFormat(FORMAT_NC1HWC0);
  input_tensor->MutableTensorDesc().SetOriginShape(GeShape({1, 2, 4, 5}));
  input_tensor->MutableTensorDesc().SetOriginShape(GeShape({1, 2, 4, 5}));
  input_tensor->MutableTensorDesc().SetShape(GeShape());
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*input_tensor)); // data1
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*input_tensor)); // refdata1
  ret = session.CompileGraph(3);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);
}

TEST_F(ConcatNotaskPassTest, nchw_to_hwcn_concat_dim0_fail) {
  GeRunningEnvFaker ge_env;
  Fake5DNodeEngine2<FakeFormatsOptimizer>(ge_env);
  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
    // 3.构图
  auto train_graph =
      BuildRefDataWithStroageFormatTrainGraph2(FORMAT_NC1HWC0, FORMAT_NCHW, {1, 2, 4, 5}, "");

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 0);
  // 4.add graph
  auto ret = session.AddGraph(3, train_graph);
  EXPECT_EQ(ret, SUCCESS);

  // 5.run graph with stream async
  std::vector<ge::Tensor> g1_inputs;
  std::vector<ge::Tensor> g1_outputs(1);
  auto input_tensor = GenerateTensor({1, 2, 4, 5});
  input_tensor->MutableTensorDesc().SetOriginFormat(FORMAT_NCHW);
  input_tensor->MutableTensorDesc().SetFormat(FORMAT_NC1HWC0);
  input_tensor->MutableTensorDesc().SetOriginShape(GeShape({1, 2, 4, 5}));
  input_tensor->MutableTensorDesc().SetShape(GeShape());
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*input_tensor)); // data1
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*input_tensor)); // refdata1
  ret = session.CompileGraph(3);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);
}

/**
 *                   data2   data3 data4
 *                       \     /     /
 *                           concat
 *          data1             /
 *            \              /
 *                 conv2d1
 *                    |
 *                netoutput
 */
static Graph BuildStroageFormatFzTrainGraph(Format origin_format,
  const std::vector<int64_t> &concat_input_shape,
  const std::vector<int64_t> &conv_input_shape) {
  DEF_GRAPH(graph) {
    auto data1 = OP_DATA(0).TensorDesc(FORMAT_ND, DT_FLOAT16, conv_input_shape).Attr(ATTR_NAME_INDEX, 0).Build("data1");

    auto conv2d1 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).Build("conv2d1");
    conv2d1->MutableInputDesc(0)->SetOriginFormat(FORMAT_NHWC);
    conv2d1->MutableInputDesc(0)->SetFormat(FORMAT_NHWC);
    conv2d1->MutableInputDesc(1)->SetOriginFormat(FORMAT_HWCN);
    conv2d1->MutableInputDesc(1)->SetFormat(FORMAT_HWCN);
    conv2d1->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NHWC);
    conv2d1->MutableOutputDesc(0)->SetFormat(FORMAT_NHWC);

    auto data2 = OP_DATA(0).TensorDesc(FORMAT_ND, DT_FLOAT16, concat_input_shape).Attr(ATTR_NAME_INDEX, 0).Build("data2");
    auto data3 = OP_DATA(0).TensorDesc(FORMAT_ND, DT_FLOAT16, concat_input_shape).Attr(ATTR_NAME_INDEX, 0).Build("data3");
    auto data4 = OP_DATA(0).TensorDesc(FORMAT_ND, DT_FLOAT16, concat_input_shape).Attr(ATTR_NAME_INDEX, 0).Build("data4");

    auto concat = OP_CFG("ConcatD").InCnt(3).OutCnt(1).Build("concat");
    concat->MutableInputDesc(0)->SetOriginFormat(FORMAT_HWCN);
    concat->MutableInputDesc(0)->SetFormat(FORMAT_HWCN);
    concat->MutableInputDesc(1)->SetOriginFormat(FORMAT_HWCN);
    concat->MutableInputDesc(1)->SetFormat(FORMAT_HWCN);
    concat->MutableInputDesc(2)->SetOriginFormat(FORMAT_HWCN);
    concat->MutableInputDesc(2)->SetFormat(FORMAT_HWCN);
    concat->MutableOutputDesc(0)->SetOriginFormat(FORMAT_HWCN);
    concat->MutableOutputDesc(0)->SetFormat(FORMAT_HWCN);
    concat->SetOpEngineName("AIcoreEngine");

    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(conv2d1));
    CHAIN(NODE(data2)->EDGE(0, 0)->NODE(concat));
    CHAIN(NODE(data3)->EDGE(0, 1)->NODE(concat)->EDGE(0, 1)->NODE(conv2d1));
    CHAIN(NODE(data4)->EDGE(0, 2)->NODE(concat));
    ADD_OUTPUT(conv2d1, 0);
  };

  return ToGeGraph(graph);
}

template <typename T>
void FakeFzNodeEngine(GeRunningEnvFaker &ge_env, const std::vector<int64_t> &concat_input_shape,
  const std::vector<int64_t> &concat_output_shape, const std::vector<int64_t> &conv_input) {
  auto ffo = MakeShared<T>();
  auto ops_kernel_builder = MakeShared<FakeAicoreLibOpsKernelBuilder>("AiCoreLib");
  auto aicore_engine_builder = MakeShared<FakeAicoreLibOpsKernelBuilder>("AIcoreEngine");
  // {c0_value, bit_value}: c0_value = 2 ^ (bit_value - 1)
  // {1, 1}, {2, 2}, {4, 3}, {8, 4}, {16, 5}, {32, 6}, {64, 7}, {128, 8}, {256, 9}
  // 5 indicates that cube size is 16
  const Format src_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_NC1HWC0, FORMAT_RESERVED, 5));
  const Format dst_format = static_cast<Format>(GetFormatFromSubAndC0(FORMAT_FRACTAL_Z, FORMAT_NHWC, 5));
  ffo->OpFormatByType(
      CONV2D, {
          .input_formats = {
              {src_format, GeShape(std::vector<int64_t>(conv_input))},
              {dst_format, GeShape(std::vector<int64_t>(concat_output_shape))},
          },
          .output_formats = {
              {src_format, GeShape(std::vector<int64_t>({2,60,16,16}))}
          }
      });
  ffo->OpFormatByType(
      "ConcatD", {
          .input_formats = {
              {dst_format, GeShape(std::vector<int64_t>(concat_input_shape))},
              {dst_format, GeShape(std::vector<int64_t>(concat_input_shape))},
              {dst_format, GeShape(std::vector<int64_t>(concat_input_shape))},
          },
          .output_formats = {
              {dst_format, GeShape(std::vector<int64_t>(concat_output_shape))}
          }
      });
      ge_env.Reset()
         .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AiCoreLib").GraphOptimizer("AIcoreEngine"))
         .Install(FakeEngine("DNN_HCCL").KernelInfoStore(kEngineNameHccl))
         .Install(FakeEngine("DNN_VM_RTS").KernelInfoStore(kEngineNameRts).GraphOptimizer("DNN_VM_RTS_GRAPH_OPTIMIZER_STORE"))
         .Install(FakeEngine("AiCoreLib").GraphOptimizer("FormatOp", ffo).KernelBuilder(ops_kernel_builder).KernelBuilder(aicore_engine_builder))
         .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(VARIABLE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(SHAPE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(IF).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp("PhonyConcat").InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE").InferShape(infer_fun))
         .Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(CONSTANTOP).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp("Identity").InfoStoreAndBuilder("AiCoreLib"))
         .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(ADD).InfoStoreAndBuilder("AiCoreLib").InferShape(infer_fun))
         .Install(FakeOp("ConcatD").InfoStoreAndBuilder("AiCoreLib"))
         .Install(FakeOp("Send").InfoStoreAndBuilder(kEngineNameRts))
         .Install(FakeOp("Recv").InfoStoreAndBuilder(kEngineNameRts))
         .Install(FakeOp(RELU).InfoStoreAndBuilder("AiCoreLib").InferShape(infer_fun))
         .Install(FakeOp(CONV2D).InfoStoreAndBuilder("AiCoreLib").InferShape(infer_fun))
         .Install(FakeOp(RESHAPE).InfoStoreAndBuilder("AiCoreLib"))
         .Install(FakeOp("RefData").InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
         .Install(FakeOp(TRANSDATA).InfoStoreAndBuilder("AiCoreLib"));
}
/*HWCN 转FZ
*HWCN =>(C1,H,W),N1,N0,C0*
*{1,2,3,16} =>{1*1*2,1,16,16}
* src_to_dst_transfer_dims: {{0}, {0}, {0,3}, {1,2}}
* dst_to_src_transfer_dims: {{2,0,1},{3},{3},{2}}*/
TEST_F(ConcatNotaskPassTest, conv_fz_connect_to_concat_H) {
  GeRunningEnvFaker ge_env;
  FakeFzNodeEngine<FakeFormatsOptimizer>(ge_env, {2,1,16,16}, {6,1,16,16}, {2,1,60,16,16});
  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
    // 3.构图
  auto train_graph =
      BuildStroageFormatFzTrainGraph(FORMAT_NCHW, {1,2,3,16}, {2,60,16,3});

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 0);
  // 4.add graph
  auto ret = session.AddGraph(3, train_graph);
  EXPECT_EQ(ret, SUCCESS);
  
  // 5.compile graph
  ret = session.CompileGraph(3);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_EQ(notask, true);
}

/*HWCN 转FZ
*HWCN =>(C1,H,W),N1,N0,C0*
*{1,2,3,16} =>{1*1*2,1,16,16}
* src_to_dst_transfer_dims: {{0}, {0}, {0,3}, {1,2}}
* dst_to_src_transfer_dims: {{2,0,1},{3},{3},{2}}*/
TEST_F(ConcatNotaskPassTest, conv_fz_connect_to_concat_W_pass) {
  GeRunningEnvFaker ge_env;
  FakeFzNodeEngine<FakeFormatsOptimizer>(ge_env, {2,1,16,16}, {6,1,16,16}, {2,1,60,16,16});
  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
    // 3.构图
  auto train_graph =
      BuildStroageFormatFzTrainGraph(FORMAT_NCHW, {1,2,3,16}, {2,60,16,3});

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 1);
  // 4.add graph
  auto ret = session.AddGraph(3, train_graph);
  EXPECT_EQ(ret, SUCCESS);
  
  // 5.compile graph
  ret = session.CompileGraph(3);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_EQ(notask, true);
}

/*HWCN 转FZ
*HWCN =>(C1,H,W),N1,N0,C0*
*{2,2,3,16} =>{1*2*2,1,16,16}
* src_to_dst_transfer_dims: {{0}, {0}, {0,3}, {1,2}}
* dst_to_src_transfer_dims: {{2,0,1},{3},{3},{2}}*/
TEST_F(ConcatNotaskPassTest, conv_fz_connect_to_concat_W_fail) {
  GeRunningEnvFaker ge_env;
  FakeFzNodeEngine<FakeFormatsOptimizer>(ge_env, {4,1,16,16}, {12,1,16,16}, {2,1,60,16,16});
  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
    // 3.构图 
  auto train_graph =
      BuildStroageFormatFzTrainGraph(FORMAT_NCHW, {2,2,3,16}, {2,60,16,3});

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 1);
  // 4.add graph
  auto ret = session.AddGraph(3, train_graph);
  EXPECT_EQ(ret, SUCCESS);
  
  // 5.compile graph
  ret = session.CompileGraph(3);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);
}

/*HWCN 转FZ
*HWCN =>(C1,H,W),N1,N0,C0*
*{2,2,3,16} =>{1*2*2,1,16,16}
* src_to_dst_transfer_dims: {{0}, {0}, {0,3}, {1,2}}
* dst_to_src_transfer_dims: {{2,0,1},{3},{3},{2}}*/
TEST_F(ConcatNotaskPassTest, conv_fz_connect_to_concat_C_fail) {
  GeRunningEnvFaker ge_env;
  FakeFzNodeEngine<FakeFormatsOptimizer>(ge_env, {4,1,16,16}, {12,1,16,16}, {2,1,60,16,16});
  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
    // 3.构图 
  auto train_graph =
      BuildStroageFormatFzTrainGraph(FORMAT_NCHW, {2,2,3,16}, {2,60,16,3});

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 2);
  // 4.add graph
  auto ret = session.AddGraph(3, train_graph);
  EXPECT_EQ(ret, SUCCESS);
  
  // 5.compile graph
  ret = session.CompileGraph(3);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);
}

/*HWCN 转FZ
*HWCN =>(C1,H,W),N1,N0,C0*
*{2,2,32,16} =>{2*2*2,1,16,16}
* src_to_dst_transfer_dims: {{0}, {0}, {0,3}, {1,2}}
* dst_to_src_transfer_dims: {{2,0,1},{3},{3},{2}}*/
TEST_F(ConcatNotaskPassTest, conv_fz_connect_to_concat_C_pass) {
  GeRunningEnvFaker ge_env;
  FakeFzNodeEngine<FakeFormatsOptimizer>(ge_env, {8,1,16,16}, {24,1,16,16}, {2,1,60,16,16});
  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
    // 3.构图 
  auto train_graph =
      BuildStroageFormatFzTrainGraph(FORMAT_NCHW, {2,2,32,16}, {2,60,16,3});

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 2);
  // 4.add graph
  auto ret = session.AddGraph(3, train_graph);
  EXPECT_EQ(ret, SUCCESS);
  
  // 5.compile graph
  ret = session.CompileGraph(3);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_EQ(notask, true);
}

/*HWCN 转FZ
*HWCN =>(C1,H,W),N1,N0,C0*
*{1,1,3,32} =>{1*1*1,2,16,16}
* src_to_dst_transfer_dims: {{0}, {0}, {0,3}, {1,2}}
* dst_to_src_transfer_dims: {{2,0,1},{3},{3},{2}}*/
TEST_F(ConcatNotaskPassTest, conv_fz_connect_to_concat_N_pass) {
  GeRunningEnvFaker ge_env;
  FakeFzNodeEngine<FakeFormatsOptimizer>(ge_env, {1,1,3,32}, {1,6,16,16}, {2,1,60,16,16});
  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
    // 3.构图
  auto train_graph =
      BuildStroageFormatFzTrainGraph(FORMAT_NCHW, {1,1,3,32}, {2,60,16,3});

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 3);
  // 4.add graph
  auto ret = session.AddGraph(3, train_graph);
  EXPECT_EQ(ret, SUCCESS);
  
  // 5.compile graph
  ret = session.CompileGraph(3);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_EQ(notask, true);
}

/*HWCN 转FZ
*HWCN =>(C1,H,W),N1,N0,C0*
*{2,1,3,32} =>{1*2*1,2,16,16}
* src_to_dst_transfer_dims: {{0}, {0}, {0,3}, {1,2}}
* dst_to_src_transfer_dims: {{2,0,1},{3},{3},{2}}*/
TEST_F(ConcatNotaskPassTest, conv_fz_connect_to_concat_N_fail) {
  GeRunningEnvFaker ge_env;
  FakeFzNodeEngine<FakeFormatsOptimizer>(ge_env, {2,1,3,32}, {2,6,16,16}, {2,1,60,16,16});
  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
    // 3.构图
  auto train_graph =
      BuildStroageFormatFzTrainGraph(FORMAT_NCHW, {2,1,3,32}, {2,60,16,3});

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 3);
  // 4.add graph
  auto ret = session.AddGraph(3, train_graph);
  EXPECT_EQ(ret, SUCCESS);
  
  // 5.compile graph
  ret = session.CompileGraph(3);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);
}

/*HWCN 转FZ
*HWCN =>(C1,H,W),N1,N0,C0*
*{1,1,32,32} =>{2*1*1,2,16,16}
* src_to_dst_transfer_dims: {{0}, {0}, {0,3}, {1,2}}
* dst_to_src_transfer_dims: {{2,0,1},{3},{3},{2}}*/
TEST_F(ConcatNotaskPassTest, conv_fz_connect_to_concat_W_fail_02) {
  GeRunningEnvFaker ge_env;
  FakeFzNodeEngine<FakeFormatsOptimizer>(ge_env, {2,1,32,32}, {2,6,16,16}, {2,1,60,16,16});
  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
    // 3.构图
  auto train_graph =
      BuildStroageFormatFzTrainGraph(FORMAT_NCHW, {2,1,32,32}, {4,60,16,3});

  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 1);
  // 4.add graph
  auto ret = session.AddGraph(3, train_graph);
  EXPECT_EQ(ret, SUCCESS);
  
  // 5.compile graph
  ret = session.CompileGraph(3);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);
}

void SetSubGraph(ComputeGraphPtr graph, ComputeGraphPtr subgraph, OpDesc &op_desc, const std::string &name) {
  subgraph->SetParentGraph(graph);
  subgraph->SetParentNode(graph->FindNode(op_desc.GetName()));
  op_desc.AddSubgraphName(name);
  op_desc.SetSubgraphInstanceName(0, name);
  graph->AddSubgraph(name, subgraph);
}

static ComputeGraphPtr BuildControlOpIfGraph(const ComputeGraphPtr &root_graph, const NodePtr &if_parent
  ,const std::vector<int64_t> &input_shape) {
  DEF_GRAPH(then_branch) {
     auto data = OP_CFG(DATA)
         .InCnt(1)
         .OutCnt(1)
         .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
         .TensorDesc(FORMAT_ND, DT_FLOAT16, input_shape);
     auto net_output = OP_CFG(NETOUTPUT)
         .InCnt(1)
         .OutCnt(1)
         .TensorDesc(FORMAT_ND, DT_FLOAT16, input_shape).Build("then_Node_Output");
     AttrUtils::SetInt(net_output->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
     CHAIN(NODE("then_arg_0", data)->NODE(net_output));
 };

  DEF_GRAPH(else_branch) {
     auto data = OP_CFG(DATA)
         .InCnt(1)
         .OutCnt(1)
         .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
         .TensorDesc(FORMAT_ND, DT_FLOAT16, input_shape);
     auto neg_op = OP_CFG(RELU)
         .InCnt(1)
         .OutCnt(1)
         .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
         .TensorDesc(FORMAT_ND, DT_FLOAT16, input_shape);
     auto net_output = OP_CFG(NETOUTPUT)
         .InCnt(1)
         .OutCnt(1)
         .TensorDesc(FORMAT_ND, DT_FLOAT16, input_shape).Build("else_Node_Output");
    AttrUtils::SetInt(net_output->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
     CHAIN(NODE("else_arg_0", data)->NODE("neg_else", neg_op)->NODE(net_output));
  };

  auto then_graph = ToComputeGraph(then_branch);
  auto else_graph = ToComputeGraph(else_branch);

  DEF_GRAPH(if_graph) {
    auto pred_data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT16, input_shape);

    auto value_data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 1)
        .TensorDesc(FORMAT_ND, DT_FLOAT16, input_shape);

    auto if_op = OP_CFG(IF)
        .InCnt(2)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT16, input_shape)
        .Build("if");

    if_op->MutableOutputDesc(0)->SetShape(GeShape(input_shape));
    if_op->RegisterSubgraphIrName("then_branch", SubgraphType::kStatic);
    if_op->RegisterSubgraphIrName("else_branch", SubgraphType::kStatic);
    if_op->AddSubgraphName(then_graph->GetName());
    if_op->SetSubgraphInstanceName(0, then_graph->GetName());
    if_op->AddSubgraphName(else_graph->GetName());
    if_op->SetSubgraphInstanceName(1, else_graph->GetName());

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, input_shape);

    CHAIN(NODE("arg_pred", pred_data)->NODE(if_op)->NODE("If_Node_Output", net_output));
    CHAIN(NODE("arg_value", value_data)->NODE(if_op));
  };

  auto if_ge_graph = ToComputeGraph(if_graph);
  if_ge_graph->SetParentGraph(root_graph);
  if_ge_graph->SetParentNode(if_parent);
  auto if_node = if_ge_graph->FindFirstNodeMatchType(IF);
  EXPECT_TRUE(if_node != nullptr);
  then_graph->SetParentNode(if_node);
  then_graph->SetParentGraph(if_ge_graph);
  else_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(if_ge_graph);
  if_ge_graph->TopologicalSorting();
  root_graph->AddSubgraph(then_graph);
  root_graph->AddSubgraph(else_graph);
  return if_ge_graph;
}

/**
 *                              
 *       data1   data2         data3   
 *          \     /             /
 *     PartitionedCall       RELU
 *            \               /
 *                 concat
 *                    |
 *                netoutput
 */
static Graph BuildGraphWithSubGraph(Format origin_format,
  const std::vector<int64_t> &concat_input_shape) {
  DEF_GRAPH(graph) {
    auto data1 = OP_DATA(0).TensorDesc(FORMAT_ND, DT_FLOAT16, concat_input_shape).Attr(ATTR_NAME_INDEX, 0).Build("data1");
    auto data2 = OP_DATA(1).TensorDesc(FORMAT_ND, DT_FLOAT16, concat_input_shape).Attr(ATTR_NAME_INDEX, 1).Build("data2");
    auto data3 = OP_DATA(2).TensorDesc(FORMAT_ND, DT_FLOAT16, concat_input_shape).Attr(ATTR_NAME_INDEX, 2).Build("data3");

    auto partitioned_call_1 = OP_CFG(PARTITIONEDCALL)
      .InCnt(2)
      .OutCnt(1)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, concat_input_shape).Build("partitioned_call_1");
    auto relu = OP_CFG(RELU)
      .InCnt(1)
      .OutCnt(1)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, concat_input_shape).Build("relu");
    auto concat = OP_CFG("ConcatD").InCnt(2).OutCnt(1).Build("concat");
    concat->MutableInputDesc(0)->SetOriginFormat(FORMAT_HWCN);
    concat->MutableInputDesc(0)->SetFormat(FORMAT_HWCN);
    concat->MutableInputDesc(1)->SetOriginFormat(FORMAT_HWCN);
    concat->MutableInputDesc(1)->SetFormat(FORMAT_HWCN);
    concat->MutableOutputDesc(0)->SetOriginFormat(FORMAT_HWCN);
    concat->MutableOutputDesc(0)->SetFormat(FORMAT_HWCN);
    concat->SetOpEngineName("AIcoreEngine");

    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(partitioned_call_1)->EDGE(0, 0)->NODE(concat));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(partitioned_call_1));
    CHAIN(NODE(data3)->NODE(relu)->EDGE(0, 1)->NODE(concat));
    ADD_OUTPUT(concat, 0);
  };

  return ToGeGraph(graph);
}

TEST_F(ConcatNotaskPassTest, partitioncall_connect_to_concat) {
  MockConcatNotaskPass mock_concat_notask_pass;
  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
    // 3.构图
  auto train_graph =
      BuildGraphWithSubGraph(FORMAT_ND, {1,2,32,32});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto partitioned_call_1 = compute_graph->FindNode("partitioned_call_1");
  auto if_graph_1 = BuildControlOpIfGraph(compute_graph, partitioned_call_1, {1,2,32,32});
  SetSubGraph(compute_graph, if_graph_1, *partitioned_call_1->GetOpDesc(), "if_graph");
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 1);
  // 4.add graph
  auto ret = session.AddGraph(1, train_graph);
  EXPECT_EQ(ret, SUCCESS);
  
  // 5.compile graph
  ret = session.CompileGraph(1);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);
}

/**
 *                              
 *       data1   data2             data3   
 *          \     /                  /
 *     PartitionedCall    data4   RELU
 *            \            /       /
 *                add     
 *                    \          /
 *                       concat
 *                         |
 *                      netoutput
 */
static Graph BuildGraphWithSubGraphAndRefNode(Format origin_format,
  const std::vector<int64_t> &concat_input_shape) {
  DEF_GRAPH(graph) {
    auto data1 = OP_DATA(0).TensorDesc(FORMAT_ND, DT_FLOAT16, concat_input_shape).Attr(ATTR_NAME_INDEX, 0).Build("data1");
    auto data2 = OP_DATA(1).TensorDesc(FORMAT_ND, DT_FLOAT16, concat_input_shape).Attr(ATTR_NAME_INDEX, 1).Build("data2");
    auto data3 = OP_DATA(2).TensorDesc(FORMAT_ND, DT_FLOAT16, concat_input_shape).Attr(ATTR_NAME_INDEX, 2).Build("data3");
    auto data4 = OP_DATA(3).TensorDesc(FORMAT_ND, DT_FLOAT16, concat_input_shape).Attr(ATTR_NAME_INDEX, 3).Build("data4");

    auto add = OP_CFG(ADD).InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, concat_input_shape).Build("add");
    add->SetOpEngineName("AIcoreEngine");
    auto output_tensor = add->MutableOutputDesc(0);
    TensorUtils::SetReuseInput(*output_tensor, true);
    TensorUtils::SetReuseInputIndex(*output_tensor, 0U);
    auto partitioned_call_1 = OP_CFG(PARTITIONEDCALL)
      .InCnt(2)
      .OutCnt(1)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, concat_input_shape).Build("partitioned_call_1");
    auto relu = OP_CFG(RELU)
      .InCnt(1)
      .OutCnt(1)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, concat_input_shape).Build("relu");
    auto concat = OP_CFG("ConcatD").InCnt(2).OutCnt(1).Build("concat");
    concat->MutableInputDesc(0)->SetOriginFormat(FORMAT_ND);
    concat->MutableInputDesc(0)->SetFormat(FORMAT_ND);
    concat->MutableInputDesc(0)->SetOriginShape(GeShape(concat_input_shape));
    concat->MutableInputDesc(1)->SetOriginFormat(FORMAT_ND);
    concat->MutableInputDesc(1)->SetFormat(FORMAT_ND);
    concat->MutableOutputDesc(0)->SetOriginFormat(FORMAT_ND);
    concat->MutableOutputDesc(0)->SetFormat(FORMAT_ND);
    concat->SetOpEngineName("AIcoreEngine");

    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(partitioned_call_1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(partitioned_call_1)->EDGE(0, 0)->NODE(add)->EDGE(0, 0)->NODE(concat));
    CHAIN(NODE(data4)->EDGE(0, 1)->NODE(add));
    CHAIN(NODE(data3)->NODE(relu)->EDGE(0, 1)->NODE(concat));
    ADD_OUTPUT(concat, 0);
  };

  return ToGeGraph(graph);
}

TEST_F(ConcatNotaskPassTest, partitioncall_refnode_connect_to_concat) {
  MockConcatNotaskPass mock_concat_notask_pass;
  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
    // 3.构图
  auto train_graph =
      BuildGraphWithSubGraphAndRefNode(FORMAT_ND, {1,2,32,32});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto partitioned_call_1 = compute_graph->FindNode("partitioned_call_1");
  auto if_graph_1 = BuildControlOpIfGraph(compute_graph, partitioned_call_1, {1,2,32,32});
  SetSubGraph(compute_graph, if_graph_1, *partitioned_call_1->GetOpDesc(), "if_graph");
  auto concat_node = compute_graph->FindNode("concat");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 1);
  // 4.add graph
  auto ret = session.AddGraph(1, train_graph);
  EXPECT_EQ(ret, SUCCESS);
  
  // 5.compile graph
  ret = session.CompileGraph(1);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);
}

static ComputeGraphPtr BuildControlOpIfGraphWithConcat(const ComputeGraphPtr &root_graph, const NodePtr &if_parent
  ,const std::vector<int64_t> &input_shape) {
  DEF_GRAPH(then_branch) {
     auto data = OP_CFG(DATA)
         .InCnt(1)
         .OutCnt(1)
         .Attr(ATTR_NAME_PARENT_NODE_INDEX, 1)
         .TensorDesc(FORMAT_ND, DT_FLOAT16, input_shape);
     auto data2 = OP_CFG(DATA)
         .InCnt(1)
         .OutCnt(1)
         .Attr(ATTR_NAME_PARENT_NODE_INDEX, 2)
         .TensorDesc(FORMAT_ND, DT_FLOAT16, input_shape);
     auto relu_op = OP_CFG(RELU)
         .InCnt(1)
         .OutCnt(1)
         .TensorDesc(FORMAT_ND, DT_FLOAT16, input_shape);
     auto relu_op2 = OP_CFG(RELU)
         .InCnt(1)
         .OutCnt(1)
         .TensorDesc(FORMAT_ND, DT_FLOAT16, input_shape);
     auto concat = OP_CFG("ConcatD").InCnt(2).OutCnt(1).Build("concat_then");
     (void)ge::AttrUtils::SetInt(concat, "concat_dim", 1);
     auto net_output = OP_CFG(NETOUTPUT)
         .InCnt(1)
         .OutCnt(1)
         .TensorDesc(FORMAT_ND, DT_FLOAT16, input_shape).Build("then_Node_Output");
     AttrUtils::SetInt(net_output->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
     AttrUtils::SetInt(net_output->MutableInputDesc(1), ATTR_NAME_PARENT_NODE_INDEX, 1);
     CHAIN(NODE("then_arg_0", data)->NODE("relu_op_then", relu_op)->EDGE(0, 0)->NODE(concat)->NODE(net_output));
     CHAIN(NODE("then_arg_1", data2)->NODE("relu_op2_then", relu_op2)->EDGE(0, 1)->NODE(concat));
 };

  DEF_GRAPH(else_branch) {
     auto data = OP_CFG(DATA)
         .InCnt(1)
         .OutCnt(1)
         .Attr(ATTR_NAME_PARENT_NODE_INDEX, 1)
         .TensorDesc(FORMAT_ND, DT_FLOAT16, input_shape);
    auto data2 = OP_CFG(DATA)
         .InCnt(1)
         .OutCnt(1)
         .Attr(ATTR_NAME_PARENT_NODE_INDEX, 2)
         .TensorDesc(FORMAT_ND, DT_FLOAT16, input_shape);
     auto relu_op = OP_CFG(RELU)
         .InCnt(1)
         .OutCnt(1)
         .TensorDesc(FORMAT_ND, DT_FLOAT16, input_shape);
     auto relu_op2 = OP_CFG(RELU)
         .InCnt(1)
         .OutCnt(1)
         .TensorDesc(FORMAT_ND, DT_FLOAT16, input_shape);
     auto concat = OP_CFG("ConcatD").InCnt(2).OutCnt(1).Build("concat_else");
     (void)ge::AttrUtils::SetInt(concat, "concat_dim", 1);
     auto net_output = OP_CFG(NETOUTPUT)
         .InCnt(1)
         .OutCnt(1)
         .TensorDesc(FORMAT_ND, DT_FLOAT16, input_shape).Build("else_Node_Output");
    AttrUtils::SetInt(net_output->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
    AttrUtils::SetInt(net_output->MutableInputDesc(1), ATTR_NAME_PARENT_NODE_INDEX, 1);
     CHAIN(NODE("else_arg_0", data)->NODE("relu_op_else", relu_op)->EDGE(0, 0)->NODE(concat)->NODE(net_output));
     CHAIN(NODE("else_arg_1", data2)->NODE("relu_op2_else", relu_op2)->EDGE(0, 1)->NODE(concat));
  };

  auto then_graph = ToComputeGraph(then_branch);
  auto else_graph = ToComputeGraph(else_branch);

  DEF_GRAPH(if_graph) {
    auto pred_data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT16, input_shape);

    auto value_data = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 1)
        .TensorDesc(FORMAT_ND, DT_FLOAT16, input_shape);
    auto value_data2 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 2)
        .TensorDesc(FORMAT_ND, DT_FLOAT16, input_shape);

    auto if_op = OP_CFG(IF)
        .InCnt(3)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT16, input_shape)
        .Build("if");

    if_op->MutableOutputDesc(0)->SetShape(GeShape(input_shape));
    if_op->RegisterSubgraphIrName("then_branch", SubgraphType::kStatic);
    if_op->RegisterSubgraphIrName("else_branch", SubgraphType::kStatic);
    if_op->AddSubgraphName(then_graph->GetName());
    if_op->SetSubgraphInstanceName(0, then_graph->GetName());
    if_op->AddSubgraphName(else_graph->GetName());
    if_op->SetSubgraphInstanceName(1, else_graph->GetName());

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, input_shape);

    CHAIN(NODE("arg_pred", pred_data)->EDGE(0, 0)->NODE(if_op)->NODE("If_Node_Output", net_output));
    CHAIN(NODE("arg_value", value_data)->EDGE(0, 1)->NODE(if_op));
    CHAIN(NODE("arg_value2", value_data2)->EDGE(0, 2)->NODE(if_op));
  };

  auto if_ge_graph = ToComputeGraph(if_graph);
  if_ge_graph->SetParentGraph(root_graph);
  if_ge_graph->SetParentNode(if_parent);
  auto if_node = if_ge_graph->FindFirstNodeMatchType(IF);
  EXPECT_TRUE(if_node != nullptr);
  then_graph->SetParentNode(if_node);
  then_graph->SetParentGraph(if_ge_graph);
  else_graph->SetParentNode(if_node);
  else_graph->SetParentGraph(if_ge_graph);
  if_ge_graph->TopologicalSorting();
  root_graph->AddSubgraph(then_graph);
  root_graph->AddSubgraph(else_graph);
  return if_ge_graph;
}

/**
 *                              
 * data0  data1   data2             data3   
 *     \    \     /                  /
 *     PartitionedCall    data4   RELU
 *            \            /       /
 *                add     
 *                    \          /
 *                       concat
 *                         |
 *                      netoutput
 */
static Graph BuildGraphWithSubGraphWithConcat(Format origin_format,
  const std::vector<int64_t> &concat_input_shape) {
  DEF_GRAPH(graph) {
    auto data0 = OP_DATA(0).TensorDesc(FORMAT_ND, DT_FLOAT16, concat_input_shape).Attr(ATTR_NAME_INDEX, 0).Build("data0");
    auto data1 = OP_DATA(1).TensorDesc(FORMAT_ND, DT_FLOAT16, concat_input_shape).Attr(ATTR_NAME_INDEX, 1).Build("data1");
    auto data2 = OP_DATA(2).TensorDesc(FORMAT_ND, DT_FLOAT16, concat_input_shape).Attr(ATTR_NAME_INDEX, 2).Build("data2");
    auto data3 = OP_DATA(3).TensorDesc(FORMAT_ND, DT_FLOAT16, concat_input_shape).Attr(ATTR_NAME_INDEX, 3).Build("data3");
    auto data4 = OP_DATA(4).TensorDesc(FORMAT_ND, DT_FLOAT16, concat_input_shape).Attr(ATTR_NAME_INDEX, 4).Build("data4");

    auto add = OP_CFG(ADD).InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, concat_input_shape).Build("add");
    add->SetOpEngineName("AIcoreEngine");
    auto output_tensor = add->MutableOutputDesc(0);
    TensorUtils::SetReuseInput(*output_tensor, true);
    TensorUtils::SetReuseInputIndex(*output_tensor, 0U);
    auto partitioned_call_1 = OP_CFG(PARTITIONEDCALL)
      .InCnt(3)
      .OutCnt(1)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, concat_input_shape).Build("partitioned_call_1");
    auto relu = OP_CFG(RELU)
      .InCnt(1)
      .OutCnt(1)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, concat_input_shape).Build("relu");
    auto concat = OP_CFG("ConcatD").InCnt(2).OutCnt(1).Build("concat_main");
    concat->MutableInputDesc(0)->SetOriginFormat(FORMAT_ND);
    concat->MutableInputDesc(0)->SetFormat(FORMAT_ND);
    concat->MutableInputDesc(0)->SetOriginShape(GeShape(concat_input_shape));
    concat->MutableInputDesc(1)->SetOriginFormat(FORMAT_ND);
    concat->MutableInputDesc(1)->SetFormat(FORMAT_ND);
    concat->MutableOutputDesc(0)->SetOriginFormat(FORMAT_ND);
    concat->MutableOutputDesc(0)->SetFormat(FORMAT_ND);
    concat->SetOpEngineName("AIcoreEngine");

    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(partitioned_call_1));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(partitioned_call_1));
    CHAIN(NODE(data2)->EDGE(0, 2)->NODE(partitioned_call_1)->EDGE(0, 0)->NODE(add)->EDGE(0, 0)->NODE(concat));
    CHAIN(NODE(data4)->EDGE(0, 1)->NODE(add));
    CHAIN(NODE(data3)->NODE(relu)->EDGE(0, 1)->NODE(concat));
    ADD_OUTPUT(concat, 0);
  };

  return ToGeGraph(graph);
}

TEST_F(ConcatNotaskPassTest, static_graph_with_concat_in_static_subgraph_success) {
  MockConcatNotaskPass mock_concat_notask_pass;
  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
    // 3.构图
  auto train_graph =
      BuildGraphWithSubGraphWithConcat(FORMAT_ND, {1,2,32,32});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto partitioned_call_1 = compute_graph->FindNode("partitioned_call_1");
  auto if_graph_1 = BuildControlOpIfGraphWithConcat(compute_graph, partitioned_call_1, {1,2,32,32});
  SetSubGraph(compute_graph, if_graph_1, *partitioned_call_1->GetOpDesc(), "if_graph");
  auto concat_node = compute_graph->FindNode("concat_main");
  auto op_desc = concat_node->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "concat_dim", 1);
  auto then_branch = compute_graph->GetSubgraph("then_branch");
  auto concat_then = then_branch->FindNode("concat_then");
  auto concat_then_desc = concat_then->GetOpDesc();
  (void)ge::AttrUtils::SetInt(concat_then_desc, "concat_dim", 1);
  
  // 4.add graph
  auto ret = session.AddGraph(1, train_graph);
  EXPECT_EQ(ret, SUCCESS);
  
  // 5.compile graph
  ret = session.CompileGraph(1);
  EXPECT_EQ(ret, SUCCESS);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == false);
  (void)ge::AttrUtils::GetBool(concat_then_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == true);
}

static Graph BuildDynamicGraphWithSubGraphWithConcat(Format origin_format,
  const std::vector<int64_t> &concat_input_shape) {
  DEF_GRAPH(graph) {
    auto data0 = OP_DATA(0).TensorDesc(FORMAT_ND, DT_FLOAT16, concat_input_shape).Attr(ATTR_NAME_INDEX, 0).Build("data0");
    auto data1 = OP_DATA(1).TensorDesc(FORMAT_ND, DT_FLOAT16, concat_input_shape).Attr(ATTR_NAME_INDEX, 1).Build("data1");
    auto data2 = OP_DATA(2).TensorDesc(FORMAT_ND, DT_FLOAT16, concat_input_shape).Attr(ATTR_NAME_INDEX, 2).Build("data2");
    auto data3 = OP_DATA(3).TensorDesc(FORMAT_ND, DT_FLOAT16, {1,-1,32,32}).Attr(ATTR_NAME_INDEX, 3).Build("data3");
    auto data4 = OP_DATA(4).TensorDesc(FORMAT_ND, DT_FLOAT16, concat_input_shape).Attr(ATTR_NAME_INDEX, 4).Build("data4");

    auto add = OP_CFG(ADD).InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, concat_input_shape).Build("add");
    add->SetOpEngineName("AIcoreEngine");
    auto output_tensor = add->MutableOutputDesc(0);
    TensorUtils::SetReuseInput(*output_tensor, true);
    TensorUtils::SetReuseInputIndex(*output_tensor, 0U);
    auto partitioned_call_1 = OP_CFG(PARTITIONEDCALL)
      .InCnt(3)
      .OutCnt(1)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, concat_input_shape).Build("partitioned_call_1");
    auto relu = OP_CFG(RELU)
      .InCnt(1)
      .OutCnt(1)
      .TensorDesc(FORMAT_ND, DT_FLOAT16, concat_input_shape).Build("relu");
    auto concat = OP_CFG("ConcatD").InCnt(2).OutCnt(1).Build("concat_main");
    concat->MutableInputDesc(0)->SetOriginFormat(FORMAT_ND);
    concat->MutableInputDesc(0)->SetFormat(FORMAT_ND);
    concat->MutableInputDesc(0)->SetOriginShape(GeShape(concat_input_shape));
    concat->MutableInputDesc(1)->SetOriginFormat(FORMAT_ND);
    concat->MutableInputDesc(1)->SetFormat(FORMAT_ND);
    concat->MutableOutputDesc(0)->SetOriginFormat(FORMAT_ND);
    concat->MutableOutputDesc(0)->SetFormat(FORMAT_ND);
    concat->SetOpEngineName("AIcoreEngine");

    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(partitioned_call_1));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(partitioned_call_1));
    CHAIN(NODE(data2)->EDGE(0, 2)->NODE(partitioned_call_1)->EDGE(0, 0)->NODE(add)->EDGE(0, 0)->NODE(concat));
    CHAIN(NODE(data4)->EDGE(0, 1)->NODE(add));
    CHAIN(NODE(data3)->NODE(relu)->EDGE(0, 1)->NODE(concat));
    ADD_OUTPUT(concat, 0);
  };

  return ToGeGraph(graph);
}

TEST_F(ConcatNotaskPassTest, dynamic_graph_with_concat_in_static_subgraph_success) {
  MockConcatNotaskPass mock_concat_notask_pass;
  std::map<AscendString, AscendString> options;
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  Session session(options);
    // 3.构图
  auto train_graph =
      BuildDynamicGraphWithSubGraphWithConcat(FORMAT_ND, {1,2,32,32});
  auto compute_graph = GraphUtilsEx::GetComputeGraph(train_graph);
  auto partitioned_call_1 = compute_graph->FindNode("partitioned_call_1");
  auto if_graph_1 = BuildControlOpIfGraphWithConcat(compute_graph, partitioned_call_1, {1,2,32,32});
  SetSubGraph(compute_graph, if_graph_1, *partitioned_call_1->GetOpDesc(), "if_graph");
  auto then_branch = compute_graph->GetSubgraph("then_branch");
  auto concat_then = then_branch->FindNode("concat_then");
  auto concat_then_desc = concat_then->GetOpDesc();
  (void)ge::AttrUtils::SetInt(concat_then_desc, "concat_dim", 1);

  // 4.add graph
  auto ret = session.AddGraph(1, train_graph);
  EXPECT_EQ(ret, SUCCESS);
  
  // 5.compile graph
  ret = session.CompileGraph(1);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(compute_graph->GetGraphUnknownFlag(), true);

  bool notask = false;
  (void)ge::AttrUtils::GetBool(concat_then_desc, ge::ATTR_NAME_NOTASK, notask);
  EXPECT_TRUE(notask == true);
}
}
