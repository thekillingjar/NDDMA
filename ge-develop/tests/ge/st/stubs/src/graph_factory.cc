/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "utils/graph_factory.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/debug/ge_attr_define.h"
#include "common/types.h"
#include "common/tbe_handle_store/kernel_store.h"
#include "ge_running_env/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"

FAKE_NS_BEGIN
namespace {
void SetMemTypeForHostMemInput(ComputeGraphPtr &graph) {
  for (const auto& node : graph->GetAllNodes()) {
    if (node == nullptr || node->GetOpDesc() == nullptr) {
      continue;
    }
    for (size_t i = 0U; i < node->GetOpDesc()->GetAllInputsSize(); ++i) {
      const GeTensorDescPtr &input_desc = node->GetOpDesc()->MutableInputDesc(static_cast<uint32_t>(i));
      if (input_desc == nullptr) {
        continue;
      }
      (void)ge::AttrUtils::SetInt(*input_desc, ge::ATTR_NAME_PLACEMENT, 2);
    }
  }
}
} // namespace

Graph GraphFactory::SingeOpGraph(){
  DEF_GRAPH(dynamic_op) {
    auto op_ptr = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
        .Build("data1");

    auto transdata = OP_CFG("MyTransdata")
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "AIcoreEngine")
        .Attr("op_para_size", 1)
        .Attr("compile_info_key", "ddd")
        .Attr("compile_info_json", "cccc")
        .Attr("support_dynamicshape", true)
        .Attr("tvm_magic", "RT_DEV_BINARY_MAGIC_ELF")
        .Build("transdata1");

    auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).Build("netoutput");
    CHAIN(NODE(op_ptr)->NODE(transdata)->NODE(netoutput));
  };
  return ToGeGraph(dynamic_op);
}

Graph GraphFactory::SingeOpGraph2(){
  DEF_GRAPH(dynamic_op) {
    auto op_ptr = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
        .Build("data1");

    auto transdata = OP_CFG(TRANSDATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "aicpu_tf_kernel")
        .Attr("op_para_size", 1)
        .Build("transdata1");

    auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).Build("netoutput");
    CHAIN(NODE(op_ptr)->NODE(transdata)->NODE(netoutput));
  };
  return ToGeGraph(dynamic_op);
}

Graph GraphFactory::VariableAddGraph() {
  DEF_GRAPH(dynamic_op) {
    auto op_ptr =
        OP_CFG(DATA).InCnt(1).OutCnt(1).Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE").Build("data1");
    auto var1 = OP_CFG(VARIABLE)
                    .InCnt(0)
                    .OutCnt(1)
                    .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
                    .Build("var1");
    auto var2 = OP_CFG(VARIABLE)
                    .InCnt(0)
                    .OutCnt(1)
                    .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
                    .Build("var2");
    auto var3 = OP_CFG(VARIABLE)
                    .InCnt(0)
                    .OutCnt(1)
                    .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
                    .Build("var3");
    auto var4 = OP_CFG(VARIABLE)
                    .InCnt(0)
                    .OutCnt(1)
                    .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
                    .Build("var4");
    auto add1 = OP_CFG(ADDN)
                    .InCnt(5)
                    .OutCnt(1)
                    .Attr("_ge_attr_op_kernel_lib_name", "aicpu_tf_kernel")
                    .Attr("op_para_size", 1)
                    .Build("add1");

    auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).Build("netoutput");
    CHAIN(NODE(op_ptr)->NODE(add1)->NODE(netoutput));
    CHAIN(NODE(var1)->NODE(add1));
    CHAIN(NODE(var2)->NODE(add1));
    CHAIN(NODE(var3)->NODE(add1));
    CHAIN(NODE(var4)->NODE(add1));
  };
  return ToGeGraph(dynamic_op);
}

Graph GraphFactory::SingeOpGraph3(){
  DEF_GRAPH(dynamic_op) {
    auto op_ptr = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
        .Build("data1");

    auto op_ptr2 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
        .Build("data2");

    auto reduce_sum = OP_CFG(REDUCESUM)
        .InCnt(2)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "AIcoreEngine")
        .Attr("_op_infer_depends", std::vector<std::string>({"axes"}))
        .Attr("support_dynamicshape", true)
        .Attr("op_para_size", 1)
        .Build("reducesum");
    reduce_sum->UpdateInputName({{"axes", 1}, {"x", 0}});
    reduce_sum->UpdateOutputName({{"y", 0}});
    reduce_sum->SetOpInferDepends({"axes"});
    reduce_sum->AppendIrInput("x", kIrInputRequired);
    reduce_sum->AppendIrInput("axes", kIrInputRequired);
    reduce_sum->AppendIrOutput("y", kIrOutputRequired);
    auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).Build("netoutput");
    CHAIN(NODE(op_ptr)->NODE(reduce_sum));
    CHAIN(NODE(op_ptr2)->EDGE(0,1)->NODE(reduce_sum)->NODE(netoutput));
  };
  return ToGeGraph(dynamic_op);
}

Graph GraphFactory::BuildRefreshWeight() {
  std::vector<int64_t> shape = {2,2,3,2};
  auto data_tensor = GenerateTensor(shape);
  DEF_GRAPH(const_refresh) {
    auto const1 = OP_CFG(CONSTANT)
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .Weight(data_tensor)
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
        .Build("const1");
    auto const2 = OP_CFG(CONSTANT)
        .Weight(data_tensor).TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
        .Build("const2");
    auto cast1 = OP_CFG(CAST)
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "AIcoreEngine")
        .Attr("dst_type", DT_FLOAT16)
        .Build("cast1");
    cast1->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
    auto cast2 = OP_CFG(CAST)
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "AIcoreEngine")
        .Attr("dst_type", DT_FLOAT16)
        .Build("cast2");
    cast2->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);

     auto trans1 = OP_CFG(TRANSDATA)
         .TensorDesc(FORMAT_ND, DT_FLOAT16, shape)
         .InCnt(1)
         .OutCnt(1)
         .Attr("_ge_attr_op_kernel_lib_name", "AIcoreEngine")
         .Build("trans1");
     trans1->MutableOutputDesc(0)->SetFormat(FORMAT_FRACTAL_NZ);
     auto trans2 = OP_CFG(TRANSDATA)
         .TensorDesc(FORMAT_ND, DT_FLOAT16, shape)
         .InCnt(1)
         .OutCnt(1)
         .Attr("_ge_attr_op_kernel_lib_name", "AIcoreEngine")
         .Build("trans2");
     trans2->MutableOutputDesc(0)->SetFormat(FORMAT_FRACTAL_NZ);
     auto matmul = OP_CFG(MATMUL)
         .TensorDesc(FORMAT_ND, DT_FLOAT16, shape)
         .InCnt(2)
         .OutCnt(1)
         .Attr("_ge_attr_op_kernel_lib_name", "AIcoreEngine")
         .Build("matmul");

    CHAIN(NODE(const1)->NODE(cast1));
    CHAIN(NODE(const2)->NODE(cast2));
    CHAIN(NODE(cast1)->NODE(trans1));
    CHAIN(NODE(cast2)->NODE(trans2));
    CHAIN(NODE(trans1)->EDGE(0, 0)->NODE(matmul));
    CHAIN(NODE(trans2)->EDGE(0, 1)->NODE(matmul));

    ADD_OUTPUT(matmul, 0);
  };

  return ToGeGraph(const_refresh);
}

/**
 *             data
 *              |
 *            netouput
 */
Graph GraphFactory::GraphDataToNetoutput() {
  DEF_GRAPH(dynamic_op) {
    ge::OpDescPtr data1;
    data1 = OP_CFG(DATA)
                .Attr(ATTR_NAME_INDEX, 0)
                .InCnt(1)
                .TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 3, 16, 16})
                .OutCnt(1)
                .TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 3, 16, 16})
                .Build("data1");

    auto netoutput1 = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).Build("netoutput1");
    CHAIN(NODE(data1)->NODE(netoutput1));
  };

  return ToGeGraph(dynamic_op);
}

Graph GraphFactory::SingeOpGraphWithSkipAttr(){
  DEF_GRAPH(dynamic_op) {
    auto op_ptr = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
        .Build("sub_data1");

    auto transdata = OP_CFG(TRANSDATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "aicpu_tf_kernel")
        .Attr("op_para_size", 1)
        .Build("transdata1");

    auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).Build("netoutput");
    CHAIN(NODE(op_ptr)->NODE(transdata)->NODE(netoutput));
  };
  DEF_GRAPH(graph1) {
    auto op_ptr = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
        .Build("data1");

    auto partition_ptr = OP_CFG(PARTITIONEDCALL)
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
        .Attr("op_para_size", 1)
        .Build("PartitionedCall_0");

    auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).Build("netoutput");
    CHAIN(NODE(op_ptr)->NODE(partition_ptr)->NODE(netoutput));
  };
  auto sub_graph = ToGeGraph(dynamic_op);
  auto root_graph = ToGeGraph(graph1);
  auto compute_root_graph = ge::GraphUtilsEx::GetComputeGraph(root_graph);
  auto compute_sub_graph = ge::GraphUtilsEx::GetComputeGraph(sub_graph);

  auto data_node = compute_sub_graph->FindFirstNodeMatchType(DATA);
  auto op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  data_node = compute_root_graph->FindFirstNodeMatchType(DATA);
  op_desc = data_node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);

  auto trans_node = compute_sub_graph->FindFirstNodeMatchType(TRANSDATA);
  op_desc = data_node->GetOpDesc();
  AttrUtils::SetBool(op_desc, ATTR_NAME_OUT_SHAPE_LOCKED, true);

  auto partion_node = compute_root_graph->FindFirstNodeMatchType(PARTITIONEDCALL);
  compute_sub_graph->SetParentNode(partion_node);
  partion_node->GetOpDesc()->AddSubgraphName("dynamic_op");
  partion_node->GetOpDesc()->SetSubgraphInstanceName(0, "dynamic_op");
  compute_sub_graph->SetParentGraph(compute_root_graph);
  compute_root_graph->AddSubgraph("dynamic_op", compute_sub_graph);
  return root_graph;
}


Graph GraphFactory::HybridSingeOpGraphForHostMemInput() {
  DEF_GRAPH(dynamic_op) {
    auto op_ptr = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
        .Attr("compile_info_key", "ddd")
        .Attr("compile_info_json", "cccc")
        .Attr("_force_unknown_shape", true)
        .Attr(ATTR_NAME_PLACEMENT, 2)
        .TensorDesc(FORMAT_ND, DT_UINT64, {4})
        .Build("data1");

    auto op_ptr2 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 1)
        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
        .Attr("compile_info_key", "ddd")
        .Attr("compile_info_json", "cccc")
        .Attr(ATTR_NAME_PLACEMENT, 2)
        .TensorDesc(FORMAT_ND, DT_UINT64, {4})
        .Attr("_force_unknown_shape", true)
        .Build("data2");

    auto transdata1 = OP_CFG(TRANSDATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "AIcoreEngine")
        .Attr("op_para_size", 1)
        .Attr("compile_info_key", "ddd")
        .Attr("compile_info_json", "cccc")
        .Attr("_force_unknown_shape", true)
        .Attr("_force_infershape_when_running", true)
        .Attr("support_dynamicshape", true)
        .Attr(ATTR_NAME_PLACEMENT, 2)
        .Build("transdata1");

    auto transdata2 = OP_CFG(TRANSDATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "AIcoreEngine")
        .Attr("op_para_size", 1)
        .Attr("compile_info_key", "ddd")
        .Attr("compile_info_json", "cccc")
        .Attr("_force_infershape_when_running", true)
        .Attr("_force_unknown_shape", true)
        .Attr("support_dynamicshape", true)
        .Attr(ATTR_NAME_PLACEMENT, 2)
        .Build("transdata2");

    auto matmul = OP_CFG(MATMUL)
        .InCnt(2)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "AIcoreEngine")
        .Attr("op_para_size", 1)
        .Attr("compile_info_key", "ddd")
        .Attr("compile_info_json", "cccc")
        .Attr("_force_infershape_when_running", true)
        .Attr("_force_unknown_shape", true)
        .Attr("support_dynamicshape", true)
        .Attr(ATTR_NAME_PLACEMENT, 2)
        .TensorDesc(FORMAT_ND, DT_UINT64, {4})
        .Build("matmul");

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
        .TensorDesc(FORMAT_ND, DT_UINT64, {4})
        .Build("net_output");

    CHAIN(NODE(op_ptr)->NODE(transdata1)->NODE(matmul)->NODE(net_output));
    CHAIN(NODE(op_ptr2)->NODE(transdata2)->NODE(matmul));
  };
  ComputeGraphPtr graph_ptr = ToComputeGraph(dynamic_op);
  SetMemTypeForHostMemInput(graph_ptr);
  return ToGeGraph(dynamic_op);
}

/**
 *          net_output
 *              |
 *            matmul
 *        /           \
 *       /             \
 *   transdata1    transdata2
 *      |              |
 *    op_ptr         op_ptr2
 */
Graph GraphFactory::HybridSingeOpGraphAicpuForHostMemInput(){
  DEF_GRAPH(dynamic_op) {
    auto op_ptr = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1})
        .Attr(ATTR_NAME_INDEX, 0)
        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
        .Attr("compile_info_key", "ddd")
        .Attr("compile_info_json", "cccc")
        .Attr("_force_unknown_shape", true)
        .Attr(ATTR_NAME_PLACEMENT, 2)
        .Build("data1");

    auto op_ptr2 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1})
        .Attr(ATTR_NAME_INDEX, 1)
        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
        .Attr("compile_info_key", "ddd")
        .Attr("compile_info_json", "cccc")
        .Attr("_force_unknown_shape", true)
        .Attr(ATTR_NAME_PLACEMENT, 2)
        .Build("data2");

    auto transdata1 = OP_CFG(TRANSDATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1})
        .Attr("_ge_attr_op_kernel_lib_name", "aicpu_ascend_kernel")
        .Attr("op_para_size", 1)
        .Attr("compile_info_key", "ddd")
        .Attr("compile_info_json", "cccc")
        .Attr("_force_unknown_shape", true)
        .Attr("_force_infershape_when_running", true)
        .Attr("support_dynamicshape", true)
        .Attr("_support_blockdim_flag", true)
        .Attr(ATTR_NAME_PLACEMENT, 2)
        .Build("transdata1");

    auto transdata2 = OP_CFG(TRANSDATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1})
        .Attr("_ge_attr_op_kernel_lib_name", "aicpu_ascend_kernel")
        .Attr("op_para_size", 1)
        .Attr("compile_info_key", "ddd")
        .Attr("compile_info_json", "cccc")
        .Attr("_force_infershape_when_running", true)
        .Attr("_force_unknown_shape", true)
        .Attr("support_dynamicshape", true)
        .Attr("_support_blockdim_flag", true)
        .Attr(ATTR_NAME_PLACEMENT, 2)
        .Build("transdata2");

    auto matmul = OP_CFG(MATMUL)
        .InCnt(2)
        .OutCnt(1)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1})
        .Attr("_ge_attr_op_kernel_lib_name", "AIcoreEngine")
        .Attr("op_para_size", 1)
        .Attr("compile_info_key", "ddd")
        .Attr("compile_info_json", "cccc")
        .Attr("_force_infershape_when_running", true)
        .Attr("_force_unknown_shape", true)
        .Attr("support_dynamicshape", true)
        .Attr(ATTR_NAME_PLACEMENT, 2)
        .Build("matmul");

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1})
        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
        .Build("net_output");

    CHAIN(NODE(op_ptr)->NODE(transdata1)->NODE(matmul)->NODE(net_output));
    CHAIN(NODE(op_ptr2)->NODE(transdata2)->NODE(matmul));
  };
  ComputeGraphPtr graph_ptr = ToComputeGraph(dynamic_op);
  SetMemTypeForHostMemInput(graph_ptr);
  return ToGeGraph(dynamic_op);
}

Graph GraphFactory::HybridSingeOpGraph(){
  DEF_GRAPH(dynamic_op) {
    auto op_ptr = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
        .Attr("compile_info_key", "ddd")
        .Attr("compile_info_json", "cccc")
        .Attr("_force_unknown_shape", true)
        .Build("data1");

    auto op_ptr2 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
        .Attr("compile_info_key", "ddd")
        .Attr("compile_info_json", "cccc")
        .Attr("_force_unknown_shape", true)
        .Build("data2");

    auto transdata1 = OP_CFG("MyTransdata")
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "AIcoreEngine")
        .Attr("op_para_size", 1)
        .Attr("compile_info_key", "ddd")
        .Attr("compile_info_json", "cccc")
        .Attr("_force_unknown_shape", true)
        .Attr("_force_infershape_when_running", true)
        .Attr("support_dynamicshape", true)
        .Build("transdata1");

    auto transdata2 = OP_CFG(TRANSDATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "AIcoreEngine")
        .Attr("op_para_size", 1)
        .Attr("compile_info_key", "ddd")
        .Attr("compile_info_json", "cccc")
        .Attr("_force_infershape_when_running", true)
        .Attr("_force_unknown_shape", true)
        .Attr("support_dynamicshape", true)
        .Build("transdata2");

    auto matmul = OP_CFG(MATMUL)
        .InCnt(2)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "AIcoreEngine")
        .Attr("op_para_size", 1)
        .Attr("compile_info_key", "ddd")
        .Attr("compile_info_json", "cccc")
        .Attr("_force_infershape_when_running", true)
        .Attr("_force_unknown_shape", true)
        .Attr("support_dynamicshape", true)
        .TensorDesc(FORMAT_ND, DT_UINT64, {4})
        .Build("matmul");

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
        .Build("net_output");

    CHAIN(NODE(op_ptr)->NODE(transdata1)->NODE(matmul)->NODE(net_output));
    CHAIN(NODE(op_ptr2)->NODE(transdata2)->NODE(matmul));
  };

  return ToGeGraph(dynamic_op);
}

Graph GraphFactory::HybridSingeOpGraph2(){
  DEF_GRAPH(dynamic_op) {
    auto op_ptr = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
        .Attr("compile_info_key", "ddd")
        .Attr("compile_info_json", "cccc")
        .Attr("_force_unknown_shape", true)
        .Build("data1");

    auto op_ptr2 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
        .Attr("compile_info_key", "ddd")
        .Attr("compile_info_json", "cccc")
        .Attr("_force_unknown_shape", true)
        .Build("data2");

    auto transdata1 = OP_CFG(TRANSDATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "AIcoreEngine")
        .Attr("op_para_size", 1)
        .Attr("compile_info_key", "ddd")
        .Attr("compile_info_json", "cccc")
        .Attr("_force_unknown_shape", true)
        .Attr("_force_infershape_when_running", true)
        .Attr("support_dynamicshape", true)
        .Build("transdata1");

    auto transdata2 = OP_CFG(TRANSDATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "AIcoreEngine")
        .Attr("op_para_size", 1)
        .Attr("compile_info_key", "ddd")
        .Attr("compile_info_json", "cccc")
        .Attr("_force_infershape_when_running", true)
        .Attr("_force_unknown_shape", true)
        .Attr("support_dynamicshape", true)
        .Build("transdata2");

    auto matmul = OP_CFG(MATMUL)
        .InCnt(2)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "AIcoreEngine")
        .Attr("op_para_size", 1)
        .Attr("compile_info_key", "ddd")
        .Attr("compile_info_json", "cccc")
        .Attr("_force_infershape_when_running", true)
        .Attr("_force_unknown_shape", true)
        .Attr("support_dynamicshape", true)
        .Build("matmul");

    auto reshape = OP_CFG("Reshape")
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
        .Attr("_force_infershape_when_running", true)
        .Attr("_force_unknown_shape", true)
        .Build("reshape");

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
        .Build("net_output");

    CHAIN(NODE(op_ptr)->NODE(transdata1)->NODE(matmul)->NODE(reshape)->NODE(net_output));
    CHAIN(NODE(op_ptr2)->NODE(transdata2)->NODE(matmul));
  };

  return ToGeGraph(dynamic_op);
}

/**
 *          net_output
 *              |
 *            matmul
 *        /           \
 *       /             \
 *   transdata1    transdata2
 *      |              |
 *    op_ptr         op_ptr2
 */
Graph GraphFactory::BuildAicpuSingeOpGraph(){
  DEF_GRAPH(dynamic_op) {
    auto op_ptr = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, -1, -1, -1})
        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
        .Attr("compile_info_key", "ddd")
        .Attr("compile_info_json", "cccc")
        .Attr("_force_unknown_shape", true)
        .Build("data1");

    auto op_ptr2 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, -1, -1, -1})
        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
        .Attr("compile_info_key", "ddd")
        .Attr("compile_info_json", "cccc")
        .Attr("_force_unknown_shape", true)
        .Build("data2");

    auto transdata1 = OP_CFG(TRANSDATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, -1, -1, -1})
        .Attr("_ge_attr_op_kernel_lib_name", "aicpu_ascend_kernel")
        .Attr("op_para_size", 1)
        .Attr("compile_info_key", "ddd")
        .Attr("compile_info_json", "cccc")
        .Attr("_force_unknown_shape", true)
        .Attr("_force_infershape_when_running", true)
        .Attr("support_dynamicshape", true)
        .Attr("_support_blockdim_flag", true)
        .Build("transdata1");

    auto transdata2 = OP_CFG(TRANSDATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, -1, -1, -1})
        .Attr("_ge_attr_op_kernel_lib_name", "aicpu_ascend_kernel")
        .Attr("op_para_size", 1)
        .Attr("compile_info_key", "ddd")
        .Attr("compile_info_json", "cccc")
        .Attr("_force_infershape_when_running", true)
        .Attr("_force_unknown_shape", true)
        .Attr("support_dynamicshape", true)
        .Attr("_support_blockdim_flag", true)
        .Build("transdata2");

    auto matmul = OP_CFG(MATMUL)
        .InCnt(2)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "AIcoreEngine")
        .Attr("op_para_size", 1)
        .Attr("compile_info_key", "ddd")
        .Attr("compile_info_json", "cccc")
        .Attr("_force_infershape_when_running", true)
        .Attr("_force_unknown_shape", true)
        .Attr("support_dynamicshape", true)
        .Build("matmul");

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
        .Build("net_output");

    CHAIN(NODE(op_ptr)->NODE(transdata1)->NODE(matmul)->NODE(net_output));
    CHAIN(NODE(op_ptr2)->NODE(transdata2)->NODE(matmul));
  };

  return ToGeGraph(dynamic_op);
}

/*
 *    out0            out1
 *     |               |
 *   assign1        assign2
 *    /  \           /  \
 * var1  const1   var2  const2
 */
Graph GraphFactory::BuildVarInitGraph1() {
  std::vector<int64_t> shape = {2,2,3,2};  // HWCN
  auto data_tensor = GenerateTensor(shape);
  DEF_GRAPH(var_init) {
    auto var1 = OP_CFG(VARIABLE)
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .InCnt(1)
        .OutCnt(1)
        .Build("var1");
    auto var2 = OP_CFG(VARIABLE)
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .InCnt(1)
        .OutCnt(1)
        .Build("var2");
    auto assign1 = OP_CFG(ASSIGN)
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .InNames({"ref", "value"})
        .OutNames({"ref"})
        .Build("assign1");
    auto assign2 = OP_CFG(ASSIGN)
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .InNames({"ref", "value"})
        .OutNames({"ref"})
        .Build("assign2");
    auto const1 = OP_CFG(CONSTANT)
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .Weight(data_tensor)
        .InCnt(1)
        .OutCnt(1)
        .Build("const1");
    auto const2 = OP_CFG(CONSTANT)
        .Weight(data_tensor).TensorDesc(FORMAT_HWCN, DT_FLOAT, shape)
        .InCnt(1)
        .OutCnt(1)
        .Build("const2");

    CHAIN(NODE(var1)->NODE(assign1));
    CHAIN(NODE(const1)->EDGE(0, 1)->NODE(assign1));
    CHAIN(NODE(var2)->NODE(assign2));
    CHAIN(NODE(const2)->EDGE(0, 1)->NODE(assign2));

    ADD_OUTPUT(assign1, 0);
    ADD_OUTPUT(assign2, 0);
  };

  return ToGeGraph(var_init);
}

/*
 *    out0                out1
 *     |                   |
 *   assign1             assign2
 *    /     \           /       \
 * refdata1  const1   refdata2  const2
 */
Graph GraphFactory::BuildRefDataInitGraph1() {
  std::vector<int64_t> shape = {2,2,3,2};  // HWCN
  auto data_tensor = GenerateTensor(shape);
  DEF_GRAPH(var_init) {
    auto refdata1 = OP_CFG("RefData")
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .InNames({"x"})
        .OutNames({"y"})
        .Build("refdata1");
    auto refdata2 = OP_CFG("RefData")
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 1)
        .InNames({"x"})
        .OutNames({"y"})
        .Build("refdata2");
    auto assign1 = OP_CFG(ASSIGN)
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .InNames({"ref", "value"})
        .OutNames({"ref"})
        .Build("assign1");
    auto assign2 = OP_CFG(ASSIGN)
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .InNames({"ref", "value"})
        .OutNames({"ref"})
        .Build("assign2");
    auto const1 = OP_CFG(CONSTANT)
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .Weight(data_tensor)
        .InCnt(1)
        .OutCnt(1)
        .Build("const1");
    auto const2 = OP_CFG(CONSTANT)
        .Weight(data_tensor).TensorDesc(FORMAT_HWCN, DT_FLOAT, shape)
        .InCnt(1)
        .OutCnt(1)
        .Build("const2");
    auto cast1 = OP_CFG(CAST)
                         .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
                         .Weight(data_tensor)
                         .InCnt(1)
                         .OutCnt(1)
                         .Build("cast1");
    auto cast2 = OP_CFG(CAST)
                         .Weight(data_tensor)
                         .TensorDesc(FORMAT_HWCN, DT_FLOAT, shape)
                         .InCnt(1)
                         .OutCnt(1)
                         .Build("cast2");

    CHAIN(NODE(refdata1)->NODE(assign1)->NODE(cast1));
    CHAIN(NODE(const1)->EDGE(0, 1)->NODE(assign1));
    CHAIN(NODE(refdata2)->NODE(assign2)->NODE(cast2));
    CHAIN(NODE(const2)->EDGE(0, 1)->NODE(assign2));

    ADD_OUTPUT(cast1, 0);
    ADD_OUTPUT(cast2, 0);
  };

  return ToGeGraph(var_init);
}

/*
 * out0   out1
 *  |      |
 * var1   var2
 */
Graph GraphFactory::BuildCheckpointGraph1() {
  std::vector<int64_t> shape = {2,2,3,2};  // HWCN
  DEF_GRAPH(graph) {
    auto var1 = OP_CFG(VARIABLE)
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .InCnt(1)
        .OutCnt(1)
        .Build("var1");
    auto var2 = OP_CFG(VARIABLE)
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .InCnt(1)
        .OutCnt(1)
        .Build("var2");

    ADD_OUTPUT(var1, 0);
    ADD_OUTPUT(var2, 0);
  };
  return ToGeGraph(graph);
}

/*
 *     out0       out1
 *      |          |
 *   conv2d1      add1
 *    /  \        /  \
 * data1 var1  data2 var2
 */
Graph GraphFactory::BuildVarTrainGraph1() {
  std::vector<int64_t> shape = {2,2,3,2};  // HWCN
  DEF_GRAPH(graph) {
    auto data1 = OP_DATA(0)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, {8,3,16,16})
        .Build("data1");
    auto data2 = OP_DATA(1)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, {})
        .Build("data2");

    auto var1 = OP_CFG(VARIABLE)
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .InCnt(1)
        .OutCnt(1)
        .Build("var1");
    auto var2 = OP_CFG(VARIABLE)
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .InCnt(1)
        .OutCnt(1)
        .Build("var2");

    auto conv2d1 = OP_CFG(CONV2D)
        .InCnt(2)
        .OutCnt(1)
        .Build("conv2d1");
    conv2d1->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d1->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
    conv2d1->MutableInputDesc(1)->SetOriginFormat(FORMAT_HWCN);
    conv2d1->MutableInputDesc(1)->SetFormat(FORMAT_HWCN);
    conv2d1->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d1->MutableOutputDesc(0)->SetFormat(FORMAT_NCHW);

    auto add1 = OP_CFG(ADD)
        .InCnt(2)
        .OutCnt(1)
        .Build("add1");

    CHAIN(NODE(data1)->NODE(conv2d1));
    CHAIN(NODE(var1)->EDGE(0, 1)->NODE(conv2d1));
    CHAIN(NODE(data2)->NODE(add1));
    CHAIN(NODE(var2)->EDGE(0, 1)->NODE(add1));

    ADD_OUTPUT(conv2d1, 0);
    ADD_OUTPUT(add1, 0);
  };

  return ToGeGraph(graph);
}

/*
 *           out1
 *            |
 *           add
 *          /    \
 *      assign   data2
 *      /     \
 *   conv2d  refdata2
 *    /   \
 *  data1  refdata1
 */
Graph GraphFactory::BuildRefDataTrainGraph1() {
  std::vector<int64_t> shape = {2,2,3,2};  // HWCN
  DEF_GRAPH(graph) {
    auto data1 = OP_DATA(0)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, {8,3,16,16})
        .Build("data1");
    auto data2 = OP_DATA(1)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, {8,3,16,16})
        .Build("data2");

    auto refdata1 = OP_CFG("RefData")
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 2)
        .InNames({"x"})
        .OutNames({"y"})
        .Build("refdata1");
    auto refdata2 = OP_CFG("RefData")
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 3)
        .InNames({"x"})
        .OutNames({"y"})
        .Build("refdata2");

    auto conv2d1 = OP_CFG(CONV2D)
        .InCnt(2)
        .OutCnt(1)
        .Build("conv2d1");
    conv2d1->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d1->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
    conv2d1->MutableInputDesc(1)->SetOriginFormat(FORMAT_HWCN);
    conv2d1->MutableInputDesc(1)->SetFormat(FORMAT_HWCN);
    conv2d1->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d1->MutableOutputDesc(0)->SetFormat(FORMAT_NCHW);

    auto assign = OP_CFG(ASSIGN)
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .InNames({"ref", "value"})
        .OutNames({"ref"})
        .Build("assign");

    auto add = OP_CFG(ADD)
        .InCnt(2)
        .OutCnt(1)
        .Build("add");

    CHAIN(NODE(data1)->NODE(conv2d1));
    CHAIN(NODE(refdata1)->EDGE(0, 1)->NODE(conv2d1)->EDGE(0, 1)->NODE(assign)->NODE(add));
    CHAIN(NODE(refdata2)->EDGE(0, 0)->NODE(assign));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(add));

    ADD_OUTPUT(add, 0);
  };

  auto ge_graph = ToGeGraph(graph);
  return ge_graph;
}

/*
 *     out1
 *       |
 *   conv2d
 *    /   \
 *  data1  refdata1
 */
Graph GraphFactory::BuildRefDataWithStroageFormatTrainGraph1(Format storage_format, Format origin_format,
                                                             const std::vector<int64_t> &origin_shape,
                                                             const std::string &expand_dims_rule) {
  std::vector<int64_t> shape = {2,2,3,2};  // HWCN
  DEF_GRAPH(graph) {
    auto data1 = OP_DATA(0).TensorDesc(FORMAT_NCHW, DT_FLOAT, {8, 3, 16, 16}).Attr(ATTR_NAME_INDEX, 0).Build("data1");
    auto refdata1 = OP_CFG("RefData")
                        .TensorDesc(FORMAT_NC1HWC0, DT_FLOAT, {})
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
    conv2d1->MutableOutputDesc(0)->SetFormat(FORMAT_NCHW);

    CHAIN(NODE(data1)->NODE(conv2d1));
    CHAIN(NODE(refdata1)->EDGE(0, 1)->NODE(conv2d1));

    ADD_OUTPUT(conv2d1, 0);
  };

  return ToGeGraph(graph);
}
/*
 *          netoutput
 *         /c      c\
 *   assign1        assign2
 *    /  \           /  \
 * var1  const1   var2  const2
 */
Graph GraphFactory::BuildVarWriteNoOutputRefGraph1() {
  std::vector<int64_t> shape = {2,2,3,2};  // HWCN
  auto data_tensor = GenerateTensor(shape);
  DEF_GRAPH(var_init) {
    auto var1 = OP_CFG(VARIABLE)
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .InCnt(1)
        .OutCnt(1)
        .Build("var1");
    auto var2 = OP_CFG(VARIABLE)
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .InCnt(1)
        .OutCnt(1)
        .Build("var2");
    auto assign1 = OP_CFG(ASSIGNVARIABLEOP)
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .InCnt(2)
        .Build("assign1");
    auto assign2 = OP_CFG(ASSIGNVARIABLEOP)
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .InCnt(2)
        .Build("assign2");
    auto const1 = OP_CFG(CONSTANT)
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .Weight(data_tensor)
        .InCnt(1)
        .OutCnt(1)
        .Build("const1");
    auto const2 = OP_CFG(CONSTANT)
        .Weight(data_tensor).TensorDesc(FORMAT_HWCN, DT_FLOAT, shape)
        .InCnt(1)
        .OutCnt(1)
        .Build("const2");


    assign1->MutableInputDesc(0)->SetRefPortByIndex(std::vector<uint32_t>({0}));
    assign2->MutableInputDesc(0)->SetRefPortByIndex(std::vector<uint32_t>({0}));

    CHAIN(NODE(var1)->NODE(assign1));
    CHAIN(NODE(const1)->EDGE(0, 1)->NODE(assign1));
    CHAIN(NODE(var2)->NODE(assign2));
    CHAIN(NODE(const2)->EDGE(0, 1)->NODE(assign2));

    ADD_TARGET(assign1);
    ADD_TARGET(assign2);
  };

  return ToGeGraph(var_init);
}

/*
 *                     (out1)   (out2)       +---------------------+
 *                      |         |          |                     |
 * +------- add1       exit1    exit2      mul1                    |
 * |       /    \T    /F          \F     /T    \                   |
 * | const2       switch1         switch2     data2                |
 * |                \    \       /      \                          |
 * |                 \    loopcond     merge2 <-- nextiteration2 --+
 * |                  \        |           \
 * |                   \     less1        enter2
 * |                    \   /    \          |
 * +-> nextiteration1 -> merge1  data1    const3
 *                        |
 *                     enter1
 *                       |
 *                     const1
 */
Graph GraphFactory::BuildV1LoopGraph1() {
  DEF_GRAPH(g) {
    auto const_data1 = GenerateTensor(DT_INT32, {});
    auto const1 = OP_CFG(CONSTANT)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .Weight(const_data1)
        .OutCnt(1)
        .Build("const1");
    auto enter1 = OP_CFG(ENTER)
        .Attr("is_constant", false)
        .Attr("frame_name", "While/While_context")
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(1)
        .OutCnt(1)
        .Build("enter1");
    auto merge1 = OP_CFG(MERGE)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(2)
        .OutCnt(2)
        .Build("merge1");
    auto nextiteration1 = OP_CFG(NEXTITERATION)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(1)
        .OutCnt(1)
        .Build("nextiteration1");
    auto data1 = OP_DATA(0)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .Build("data1");
    auto const_data3 = GenerateTensor({8,3,24,1});
    auto const3 = OP_CFG(CONSTANT)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,24,1})
        .Weight(const_data3)
        .OutCnt(1)
        .Build("const3");
    auto less1 = OP_CFG(LESS)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(2)
        .OutCnt(1)
        .Build("less1");
    auto enter2 = OP_CFG(ENTER)
        .Attr("is_constant", false)
        .Attr("frame_name", "While/While_context")
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,24,1})
        .InCnt(1).OutCnt(1)
        .Build("enter2");
    auto loopcond1 = OP_CFG(LOOPCOND)
        .TensorDesc(FORMAT_ND, DT_BOOL, {})
        .InCnt(1)
        .OutCnt(1)
        .Build("loopcond1");
    auto merge2 = OP_CFG(MERGE)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,24,1})
        .InCnt(2).OutCnt(2)
        .Build("merge2");
    auto nextiteration2 = OP_CFG(NEXTITERATION)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,24,1})
        .InCnt(1).OutCnt(1)
        .Build("nextiteration2");
    auto switch1 = OP_CFG(SWITCH)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(2)
        .OutCnt(2)
        .Build("switch1");
    auto const_data2 = GenerateTensor(DT_INT32, {});
    auto const2 = OP_CFG(CONSTANT)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .Weight(const_data2)
        .OutCnt(1)
        .Build("const2");
    auto switch2 = OP_CFG(SWITCH)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,24,1})
        .InCnt(2).OutCnt(2).Build("switch2");
    auto data2 = OP_DATA(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,24,1})
        .Build("data2");
    auto mul1 = OP_CFG(MUL)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,24,1})
        .InCnt(2).OutCnt(1).Build("mul1");
    auto add1 = OP_CFG(ADD)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(2)
        .OutCnt(1)
        .Build("add1");
    auto exit1 = OP_CFG(EXIT)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(1)
        .OutCnt(1)
        .Build("exit1");
    auto exit2 = OP_CFG(EXIT)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,24,1})
        .InCnt(1)
        .OutCnt(1)
        .Build("exit2");

    less1->MutableOutputDesc(0)->SetDataType(DT_BOOL);
    less1->MutableOutputDesc(0)->SetOriginDataType(DT_BOOL);

    CHAIN(NODE(const1)->NODE(enter1)->NODE(merge1)->NODE(switch1)->EDGE(1, 1)->NODE(add1)->NODE(nextiteration1)->EDGE(0, 1)->NODE(merge1));
    CHAIN(NODE(const2)->EDGE(0, 0)->NODE(add1));
    CHAIN(NODE(merge1)->NODE(less1)->NODE(loopcond1)->EDGE(0, 1)->NODE(switch1)->NODE(exit1));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(less1));
    CHAIN(NODE(loopcond1)->EDGE(0, 1)->NODE(switch2)->NODE(exit2));
    CHAIN(NODE(const3)->NODE(enter2)->NODE(merge2)->EDGE(0, 0)->NODE(switch2)->EDGE(1, 0)->NODE(mul1)->NODE(nextiteration2)->EDGE(0, 1)->NODE(merge2));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(mul1));

    ADD_OUTPUT(exit1, 0);
    ADD_OUTPUT(exit2, 0);
  };
  return ToGeGraph(g);
}

/*
 *                     (out1)   (out2)       +------------------------------+
 *                      |         |          |     c                        |
 * +------- add1       exit1    exit2      mul1 <------ enter3 <---- const4 |
 * |       /    \T    /F          \F     /T    \                            |
 * | const2       switch1         switch2     data2                         |
 * |                \    \       /      \                                   |
 * |                 \    loopcond     merge2 <-- nextiteration2 -----------+
 * |                  \        |           \
 * |                   \     less1        enter2
 * |                    \   /    \          |
 * +-> nextiteration1 -> merge1  data1    const3
 *                        |
 *                     enter1
 *                       |
 *                     const1
 */
Graph GraphFactory::BuildV1LoopGraph2_CtrlEnterIn() {
  DEF_GRAPH(g) {
    auto const_data1 = GenerateTensor(DT_INT32, {});
    auto const1 = OP_CFG(CONSTANT)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .Weight(const_data1)
        .OutCnt(1)
        .Build("const1");
    auto enter1 = OP_CFG(ENTER)
        .Attr("is_constant", false)
        .Attr("frame_name", "While/While_context")
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(1)
        .OutCnt(1)
        .Build("enter1");
    auto merge1 = OP_CFG(MERGE)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(2)
        .OutCnt(2)
        .Build("merge1");
    auto nextiteration1 = OP_CFG(NEXTITERATION)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(1)
        .OutCnt(1)
        .Build("nextiteration1");
    auto data1 = OP_DATA(0)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .Build("data1");
    auto const_data3 = GenerateTensor({8,3,24,1});
    auto const3 = OP_CFG(CONSTANT)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,24,1})
        .Weight(const_data3)
        .OutCnt(1)
        .Build("const3");
    auto less1 = OP_CFG(LESS)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(2)
        .OutCnt(1)
        .Build("less1");
    auto enter2 = OP_CFG(ENTER)
        .Attr("is_constant", false)
        .Attr("frame_name", "While/While_context")
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,24,1})
        .InCnt(1).OutCnt(1)
        .Build("enter2");
    auto loopcond1 = OP_CFG(LOOPCOND)
        .TensorDesc(FORMAT_ND, DT_BOOL, {})
        .InCnt(1)
        .OutCnt(1)
        .Build("loopcond1");
    auto merge2 = OP_CFG(MERGE)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,24,1})
        .InCnt(2).OutCnt(2)
        .Build("merge2");
    auto nextiteration2 = OP_CFG(NEXTITERATION)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,24,1})
        .InCnt(1).OutCnt(1)
        .Build("nextiteration2");
    auto switch1 = OP_CFG(SWITCH)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(2)
        .OutCnt(2)
        .Build("switch1");
    auto const_data2 = GenerateTensor(DT_INT32, {});
    auto const2 = OP_CFG(CONSTANT)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .Weight(const_data2)
        .OutCnt(1)
        .Build("const2");
    auto switch2 = OP_CFG(SWITCH)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,24,1})
        .InCnt(2).OutCnt(2).Build("switch2");
    auto data2 = OP_CFG(DATA)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,24,1})
        .InCnt(1).OutCnt(1).Build("data2");
    auto mul1 = OP_CFG(MUL)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,24,1})
        .InCnt(2).OutCnt(1).Build("mul1");
    auto add1 = OP_CFG(ADD)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(2)
        .OutCnt(1)
        .Build("add1");
    auto exit1 = OP_CFG(EXIT)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(1)
        .OutCnt(1)
        .Build("exit1");
    auto exit2 = OP_CFG(EXIT)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,24,1})
        .InCnt(1)
        .OutCnt(1)
        .Build("exit2");
    auto const_data4 = GenerateTensor({8,3,24,1});
    auto const4 = OP_CFG(CONSTANT)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,24,1})
        .Weight(const_data4)
        .OutCnt(1)
        .Build("const4");
    auto enter3 = OP_CFG(ENTER)
        .Attr("is_constant", true)
        .Attr("frame_name", "While/While_context")
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(1)
        .OutCnt(1)
        .Build("enter3");

    less1->MutableOutputDesc(0)->SetDataType(DT_BOOL);
    less1->MutableOutputDesc(0)->SetOriginDataType(DT_BOOL);

    CHAIN(NODE(const1)->NODE(enter1)->NODE(merge1)->NODE(switch1)->EDGE(1, 1)->NODE(add1)->NODE(nextiteration1)->EDGE(0, 1)->NODE(merge1));
    CHAIN(NODE(const2)->EDGE(0, 0)->NODE(add1));
    CHAIN(NODE(merge1)->NODE(less1)->NODE(loopcond1)->EDGE(0, 1)->NODE(switch1)->NODE(exit1));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(less1));
    CHAIN(NODE(loopcond1)->EDGE(0, 1)->NODE(switch2)->NODE(exit2));
    CHAIN(NODE(const3)->NODE(enter2)->NODE(merge2)->EDGE(0, 0)->NODE(switch2)->EDGE(1, 0)->NODE(mul1)->NODE(nextiteration2)->EDGE(0, 1)->NODE(merge2));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(mul1));
    CHAIN(NODE(const4)->NODE(enter3));
    CTRL_CHAIN(NODE(enter3)->NODE(mul1));

    ADD_OUTPUT(exit1, 0);
    ADD_OUTPUT(exit2, 0);
  };
  return ToGeGraph(g);
}

/*
 *                     (out1)   (out2)       +------------------------------+
 *                      |         |          |                              |
 * +------- add1       exit1    exit2      mul1 <------ enter3 <---- const4 |
 * |       /    \T    /F          \F     /T                                 |
 * | const2       switch1         switch2                                   |
 * |                \    \       /      \                                   |
 * |                 \    loopcond     merge2 <-- nextiteration2 -----------+
 * |                  \        |           \
 * |                   \     less1        enter2
 * |                    \   /    \          |
 * +-> nextiteration1 -> merge1  data1    const3
 *                        |
 *                     enter1
 *                       |
 *                     const1
 */
Graph GraphFactory::BuildV1LoopGraph4_DataEnterInByPassMerge() {
  DEF_GRAPH(g) {
    auto const_data1 = GenerateTensor(DT_INT32, {});
    auto const1 = OP_CFG(CONSTANT)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .Weight(const_data1)
        .OutCnt(1)
        .Build("const1");
    auto enter1 = OP_CFG(ENTER)
        .Attr("is_constant", false)
        .Attr("frame_name", "While/While_context")
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(1)
        .OutCnt(1)
        .Build("enter1");
    auto merge1 = OP_CFG(MERGE)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(2)
        .OutCnt(2)
        .Build("merge1");
    auto nextiteration1 = OP_CFG(NEXTITERATION)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(1)
        .OutCnt(1)
        .Build("nextiteration1");
    auto data1 = OP_DATA(0)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .Build("data1");
    auto const_data3 = GenerateTensor({8,3,224,224});
    auto const3 = OP_CFG(CONSTANT)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,224,224})
        .Weight(const_data3)
        .OutCnt(1)
        .Build("const3");
    auto less1 = OP_CFG(LESS)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(2)
        .OutCnt(1)
        .Build("less1");
    auto enter2 = OP_CFG(ENTER)
        .Attr("is_constant", false)
        .Attr("frame_name", "While/While_context")
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,224,224})
        .InCnt(1).OutCnt(1)
        .Build("enter2");
    auto loopcond1 = OP_CFG(LOOPCOND)
        .TensorDesc(FORMAT_ND, DT_BOOL, {})
        .InCnt(1)
        .OutCnt(1)
        .Build("loopcond1");
    auto merge2 = OP_CFG(MERGE)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,224,224})
        .InCnt(2).OutCnt(2)
        .Build("merge2");
    auto nextiteration2 = OP_CFG(NEXTITERATION)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,224,224})
        .InCnt(1).OutCnt(1)
        .Build("nextiteration2");
    auto switch1 = OP_CFG(SWITCH)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(2)
        .OutCnt(2)
        .Build("switch1");
    auto const_data2 = GenerateTensor(DT_INT32, {});
    auto const2 = OP_CFG(CONSTANT)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .Weight(const_data2)
        .OutCnt(1)
        .Build("const2");
    auto switch2 = OP_CFG(SWITCH)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,224,224})
        .InCnt(2).OutCnt(2).Build("switch2");
    auto mul1 = OP_CFG(MUL)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,224,224})
        .InCnt(2).OutCnt(1).Build("mul1");
    auto add1 = OP_CFG(ADD)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(2)
        .OutCnt(1)
        .Build("add1");
    auto exit1 = OP_CFG(EXIT)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(1)
        .OutCnt(1)
        .Build("exit1");
    auto exit2 = OP_CFG(EXIT)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,224,224})
        .InCnt(1)
        .OutCnt(1)
        .Build("exit2");
    auto const_data4 = GenerateTensor({8,3,224,224});
    auto const4 = OP_CFG(CONSTANT)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,224,224})
        .Weight(const_data4)
        .OutCnt(1)
        .Build("const4");
    auto enter3 = OP_CFG(ENTER)
        .Attr("is_constant", true)
        .Attr("frame_name", "While/While_context")
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(1)
        .OutCnt(1)
        .Build("enter3");

    less1->MutableOutputDesc(0)->SetDataType(DT_BOOL);
    less1->MutableOutputDesc(0)->SetOriginDataType(DT_BOOL);

    CHAIN(NODE(const1)->NODE(enter1)->NODE(merge1)->NODE(switch1)->EDGE(1, 1)->NODE(add1)->NODE(nextiteration1)->EDGE(0, 1)->NODE(merge1));
    CHAIN(NODE(const2)->EDGE(0, 0)->NODE(add1));
    CHAIN(NODE(merge1)->NODE(less1)->NODE(loopcond1)->EDGE(0, 1)->NODE(switch1)->NODE(exit1));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(less1));
    CHAIN(NODE(loopcond1)->EDGE(0, 1)->NODE(switch2)->NODE(exit2));
    CHAIN(NODE(const3)->NODE(enter2)->NODE(merge2)->EDGE(0, 0)->NODE(switch2)->EDGE(1, 0)->NODE(mul1)->NODE(nextiteration2)->EDGE(0, 1)->NODE(merge2));
    CHAIN(NODE(const4)->NODE(enter3)->EDGE(0, 1)->NODE(mul1));

    ADD_OUTPUT(exit1, 0);
    ADD_OUTPUT(exit2, 0);
  };
  return ToGeGraph(g);
}

Graph GraphFactory::BuildGraphForMergeShapeNPass() {
  DEF_GRAPH(g) {
    auto data1 = OP_CFG(DATA).InCnt(1).OutCnt(1).Build("data1");
    auto data2 = OP_CFG(DATA).InCnt(1).OutCnt(1).Build("data2");
    auto relu1 = OP_CFG(RELU).InCnt(1).InCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {}).Build("relu1");
    auto relu2 = OP_CFG(RELU).InCnt(1).InCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {}).Build("relu2");
    auto shapeN1 = OP_CFG(SHAPEN).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {}).Build("shapeN1");
    auto shapeN2 = OP_CFG(SHAPEN).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {}).Build("shapeN2");
    auto relu3 = OP_CFG(RELU).InCnt(2).InCnt(1).Build("relu3");
    CHAIN(NODE(data1)->NODE(relu1)->NODE(shapeN1)->EDGE(0,0)->NODE(relu3)->NODE("netoutput1", NETOUTPUT));
    CHAIN(NODE(data2)->NODE(relu2)->NODE(shapeN2)->EDGE(0,1)->NODE(relu3));
    CTRL_CHAIN(NODE(relu1)->NODE(shapeN1));
    CTRL_CHAIN(NODE(relu2)->NODE(shapeN2));
    CTRL_CHAIN(NODE(shapeN1)->NODE(relu3));
    CTRL_CHAIN(NODE(shapeN2)->NODE(relu3));
  };
  auto graph = ToGeGraph(g);
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  auto node1 = compute_graph->FindNode("data1");
  auto input_desc_data1 = node1->GetOpDesc()->MutableOutputDesc(0);
  input_desc_data1->SetShape(GeShape({-1, -1, 2, 3}));

  auto node2 = compute_graph->FindNode("relu1");
  auto input_desc_relu1 = node2->GetOpDesc()->MutableInputDesc(0);
  input_desc_relu1->SetShape(GeShape({-1, -1, 2, 3}));
  auto output_desc_relu1 = node2->GetOpDesc()->MutableOutputDesc(0);
  output_desc_relu1->SetShape(GeShape({-1, -1, 2, 3}));

  auto node3 = compute_graph->FindNode("shapeN1");
  OpDescPtr op_desc_ptr = node3->GetOpDesc();
  ge::AttrUtils::SetStr(op_desc_ptr, "_split_shapen_origin_name", "ShapeN_unknown");
  auto input_desc_shapeN1 = node3->GetOpDesc()->MutableInputDesc(0);
  input_desc_shapeN1->SetShape(GeShape({-1, -1, 16, 16}));
  auto output_desc_shapeN1 = node3->GetOpDesc()->MutableOutputDesc(0);
  output_desc_shapeN1->SetShape(GeShape({-1, -1, 16, 16}));

  auto node4 = compute_graph->FindNode("relu3");
  auto input_desc_relu3 = node4->GetOpDesc()->MutableInputDesc(0);
  auto input_desc_relu3_2 = node4->GetOpDesc()->MutableInputDesc(1);
  input_desc_relu3->SetShape(GeShape({-1, -1, 2, 3}));
  input_desc_relu3_2->SetShape(GeShape({-1, -1, 2, 3}));
  auto output_desc_relu3 = node4->GetOpDesc()->MutableOutputDesc(0);
  output_desc_relu3->SetShape(GeShape({-1, -1, 2, 3}));

  auto node5 = compute_graph->FindNode("netoutput1");
  auto input_desc_netoutput1 = node5->GetOpDesc()->MutableInputDesc(0);
  input_desc_netoutput1->SetShape(GeShape({-1, 1, 2, 3}));

  auto node6 = compute_graph->FindNode("data2");
  auto input_desc_data2 = node6->GetOpDesc()->MutableOutputDesc(0);
  input_desc_data2->SetShape(GeShape({-1, -1, 2, 3}));

  auto node7 = compute_graph->FindNode("relu2");
  auto input_desc_relu2 = node7->GetOpDesc()->MutableInputDesc(0);
  input_desc_relu2->SetShape(GeShape({-1, -1, 2, 3}));
  auto output_desc_relu2 = node7->GetOpDesc()->MutableOutputDesc(0);
  output_desc_relu2->SetShape(GeShape({-1, -1, 2, 3}));

  auto node8 = compute_graph->FindNode("shapeN2");
  OpDescPtr op_desc_ptr1 = node8->GetOpDesc();
  ge::AttrUtils::SetStr(op_desc_ptr1, "_split_shapen_origin_name", "ShapeN_unknown");
  auto input_desc_shapeN2 = node8->GetOpDesc()->MutableInputDesc(0);
  input_desc_shapeN2->SetShape(GeShape({-1, -1, 16, 16}));
  auto output_desc_shapeN2 = node8->GetOpDesc()->MutableOutputDesc(0);
  output_desc_shapeN2->SetShape(GeShape({-1, -1, 16, 16}));

  return graph;
}

/*
 *                     (out1)   (out2)              sub1 -------------------------+
 *                      |         |             /         \                       |
 * +------- add1       exit1    exit2      mul1            \                      |
 * |       /    \T    /F          \F     /T    \            \                     |
 * | const2       switch1         switch2     const4 <--c--- enter3 <---- const5  |
 * |                \    \       /      \                                         |
 * |                 \    loopcond     merge2 <-- nextiteration2 -----------------+
 * |                  \        |           \
 * |                   \     less1        enter2
 * |                    \   /    \          |
 * +-> nextiteration1 -> merge1  data1    const3
 *                        |
 *                     enter1
 *                       |
 *                     const1
 */
Graph GraphFactory::BuildV1LoopGraph3_CtrlEnterIn2() {
  DEF_GRAPH(g) {
    auto const_data1 = GenerateTensor(DT_INT32, {});
    auto const1 = OP_CFG(CONSTANT)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .Weight(const_data1)
        .OutCnt(1)
        .Build("const1");
    auto enter1 = OP_CFG(ENTER)
        .Attr("is_constant", false)
        .Attr("frame_name", "While/While_context")
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(1)
        .OutCnt(1)
        .Build("enter1");
    auto merge1 = OP_CFG(MERGE)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(2)
        .OutCnt(2)
        .Build("merge1");
    auto nextiteration1 = OP_CFG(NEXTITERATION)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(1)
        .OutCnt(1)
        .Build("nextiteration1");
    auto data1 = OP_DATA(0)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .Build("data1");
    auto const_data3 = GenerateTensor({8,3,24,1});
    auto const3 = OP_CFG(CONSTANT)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,24,1})
        .Weight(const_data3)
        .OutCnt(1)
        .Build("const3");
    auto less1 = OP_CFG(LESS)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(2)
        .OutCnt(1)
        .Build("less1");
    auto enter2 = OP_CFG(ENTER)
        .Attr("is_constant", false)
        .Attr("frame_name", "While/While_context")
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,24,1})
        .InCnt(1).OutCnt(1)
        .Build("enter2");
    auto loopcond1 = OP_CFG(LOOPCOND)
        .TensorDesc(FORMAT_ND, DT_BOOL, {})
        .InCnt(1)
        .OutCnt(1)
        .Build("loopcond1");
    auto merge2 = OP_CFG(MERGE)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,24,1})
        .InCnt(2).OutCnt(2)
        .Build("merge2");
    auto nextiteration2 = OP_CFG(NEXTITERATION)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,24,1})
        .InCnt(1).OutCnt(1)
        .Build("nextiteration2");
    auto switch1 = OP_CFG(SWITCH)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(2)
        .OutCnt(2)
        .Build("switch1");
    auto const_data2 = GenerateTensor(DT_INT32, {});
    auto const2 = OP_CFG(CONSTANT)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .Weight(const_data2)
        .OutCnt(1)
        .Build("const2");
    auto switch2 = OP_CFG(SWITCH)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,24,1})
        .InCnt(2).OutCnt(2).Build("switch2");
    auto mul1 = OP_CFG(MUL)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,24,1})
        .InCnt(2).OutCnt(1).Build("mul1");
    auto add1 = OP_CFG(ADD)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(2)
        .OutCnt(1)
        .Build("add1");
    auto exit1 = OP_CFG(EXIT)
        .TensorDesc(FORMAT_ND, DT_INT32, {})
        .InCnt(1)
        .OutCnt(1)
        .Build("exit1");
    auto exit2 = OP_CFG(EXIT)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,24,1})
        .InCnt(1)
        .OutCnt(1)
        .Build("exit2");
    auto const_data4 = GenerateTensor({8,3,24,1});
    auto const4 = OP_CFG(CONSTANT)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,24,1})
        .Weight(const_data4)
        .OutCnt(1)
        .Build("const4");
    auto const_data5 = GenerateTensor({8,3,24,1});
    auto const5 = OP_CFG(CONSTANT)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,24,1})
        .Weight(const_data5)
        .OutCnt(1)
        .Build("const5");
    auto enter3 = OP_CFG(ENTER)
        .Attr("is_constant", true)
        .Attr("frame_name", "While/While_context")
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,24,1})
        .InCnt(1)
        .OutCnt(1)
        .Build("enter3");
    auto sub1 = OP_CFG(SUB)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {8,3,24,1})
        .InCnt(2)
        .OutCnt(1)
        .Build("sub1");

    less1->MutableOutputDesc(0)->SetDataType(DT_BOOL);
    less1->MutableOutputDesc(0)->SetOriginDataType(DT_BOOL);

    CHAIN(NODE(const1)->NODE(enter1)->NODE(merge1)->NODE(switch1)->EDGE(1, 1)->NODE(add1)->NODE(nextiteration1)->EDGE(0, 1)->NODE(merge1));
    CHAIN(NODE(const2)->EDGE(0, 0)->NODE(add1));
    CHAIN(NODE(merge1)->NODE(less1)->NODE(loopcond1)->EDGE(0, 1)->NODE(switch1)->NODE(exit1));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(less1));
    CHAIN(NODE(loopcond1)->EDGE(0, 1)->NODE(switch2)->NODE(exit2));
    CHAIN(NODE(const3)->NODE(enter2)->NODE(merge2)->EDGE(0, 0)->NODE(switch2)->EDGE(1, 0)->NODE(mul1)->NODE(sub1)->NODE(nextiteration2)->EDGE(0, 1)->NODE(merge2));
    CHAIN(NODE(const4)->EDGE(0, 1)->NODE(mul1));
    CHAIN(NODE(const5)->NODE(enter3)->EDGE(0, 1)->NODE(sub1));

    CTRL_CHAIN(NODE(enter3)->NODE(const4));

    ADD_OUTPUT(exit1, 0);
    ADD_OUTPUT(exit2, 0);
  };
  return ToGeGraph(g);
}

/*
 *         out1
 *          |
 *         add
 *       /    \
 *    assign  data2
 *      /   \
 *  conv2d1  refdata1
 *    /   \
 *  data0  data1
 */
Graph GraphFactory::BuildInputMergeCopyGraph() {
  std::vector<int64_t> shape = {2,2,3,2};  // HWCN
  DEF_GRAPH(graph) {
    auto data0 = OP_DATA(0)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, {8,3,16,16})
        .Build("data0");
    auto data1 = OP_DATA(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .Build("data1");
    auto data2 = OP_DATA(2)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, {8,3,16,16})
        .Build("data2");

    auto refdata1 = OP_CFG("RefData")
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 3)
        .InNames({"x"})
        .OutNames({"y"})
        .Build("refdata1");

    auto conv2d1 = OP_CFG(CONV2D)
        .InCnt(2)
        .OutCnt(1)
        .Build("conv2d1");
    conv2d1->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d1->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
    conv2d1->MutableInputDesc(1)->SetOriginFormat(FORMAT_HWCN);
    conv2d1->MutableInputDesc(1)->SetFormat(FORMAT_HWCN);
    conv2d1->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
    conv2d1->MutableOutputDesc(0)->SetFormat(FORMAT_NCHW);

    auto assign = OP_CFG(ASSIGN)
        .TensorDesc(FORMAT_ND, DT_FLOAT, shape)
        .InNames({"ref", "value"})
        .OutNames({"ref"})
        .Build("assign");

    auto add = OP_CFG(ADD)
        .InCnt(2)
        .OutCnt(1)
        .Build("add");

    CHAIN(NODE(data0)->NODE(conv2d1));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(conv2d1)->EDGE(0, 1)->NODE(assign)->NODE(add));
    CHAIN(NODE(refdata1)->EDGE(0, 0)->NODE(assign));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(add));

    ADD_OUTPUT(add, 0);
  };

  return ToGeGraph(graph);
}

/*

┌────────┐  (0,0)   ┌────────┐  (0,0)   ┌────────┐  (0,0)   ┌─────────┐  (0,0)   ┌─────────────┐
│ _arg_0 │ ───────> │  add   │ ───────> │ unique │ ───────> │   mul   │ ───────> │ Node_Output │
└────────┘          └────────┘          └────────┘          └─────────┘          └─────────────┘
                      ∧                                       ∧
                      │ (0,1)                                 │ (0,1)
                      │                                       │
                    ┌────────┐                              ┌─────────┐
                    │ _arg_1 │                              │ const_0 │
                    └────────┘                              └─────────┘

*/
Graph GraphFactory::BuildDynamicInputGraph() {
    DEF_GRAPH(dynamic_graph2) {
        // GeTensorDesc tensor_desc(GeShape({2, 16}));
        GeTensorDesc tensor_desc(GeShape({1}), FORMAT_ND, DT_FLOAT);
        GeTensor tensor(tensor_desc);
        int32_t value = 2;
        tensor.SetData((uint8_t *) &value, sizeof(value));
    
        auto data_0 = OP_CFG(DATA)
                            .InCnt(1)
                            .OutCnt(1)
                            .Attr(ATTR_NAME_INDEX, 0)
                            .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
    
        auto data_1 = OP_CFG(DATA)
                            .InCnt(1)
                            .OutCnt(1)
                            .Attr(ATTR_NAME_INDEX, 1)
                            .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
    
        auto add = OP_CFG(ADD)
                        .InCnt(2)
                        .OutCnt(1)
                        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
                
        auto unique_op = OP_CFG("Unique")
                            .InCnt(1)
                            .OutCnt(1)
                            .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

        auto const_0 = OP_CFG(CONSTANTOP)
                            .OutCnt(1)
                            .Attr(ATTR_NAME_WEIGHTS, tensor)
                            .Attr(ATTR_VARIABLE_PLACEMENT, "host")
                            .TensorDesc(FORMAT_ND, DT_FLOAT, {});

        auto mul = OP_CFG(MUL)
                            .InCnt(2)
                            .OutCnt(1)
                            .TensorDesc(FORMAT_ND, DT_FLOAT, {});
        auto net_output = OP_CFG(NETOUTPUT)
                                .InCnt(1)
                                .OutCnt(1)
                                .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

        CHAIN(NODE("_arg_0", data_0)->NODE("add", add));
        CHAIN(NODE("_arg_1", data_1)->NODE("add", add));

        CHAIN(NODE("add", add)->NODE("unique", unique_op));
        CHAIN(NODE("unique", unique_op)->NODE("mul", mul));    
        CHAIN(NODE("const_0", const_0)->NODE("mul", mul));    

        CHAIN(NODE("mul", mul)->NODE("Node_Output", net_output));
    };

return ToGeGraph(dynamic_graph2);
}
/*

┌────────┐  (0,0)   ┌──────────┐  (0,0)   ┌─────────────┐  (0,0)   ┌──────────┐  (0,0)   ┌─────────────┐  (0,0)   ┌──────────┐  (0,0)   ┌─────────────┐  (0,0)   ┌──────────┐  (0,0)   ┌────────────┐
│ _arg_0 │ ───────> │ unique_1 │ ───────> │ add_bridge1 │ ───────> │ unique_2 │ ───────> │ add_bridge2 │ ───────> │ unique_3 │ ───────> │ add_bridge3 │ ───────> │ unique_4 │ ───────> │ net_output │
└────────┘          └──────────┘          └─────────────┘          └──────────┘          └─────────────┘          └──────────┘          └─────────────┘          └──────────┘          └────────────┘
                                        ∧                                              ∧                                              ∧
                                        │ (0,1)                                        │ (0,1)                                        │
                                        │                                              │                                              │
                                      ┌─────────────┐                                ┌─────────────┐  (1,1)                           │
                                      │ variable_1  │                                │   const_0   │ ─────────────────────────────────┘
                                      └─────────────┘                                └─────────────┘
*/ 
Graph GraphFactory::BuildDynamicInputGraphWithVarAndConst() {
DEF_GRAPH(dynamic_graph_4) {
    // 初始化常量Tensor（用于ADD的第二个输入）
    GeTensorDesc const_tensor_desc(GeShape({1}), FORMAT_ND, DT_FLOAT);
    GeTensor const_tensor(const_tensor_desc);
    int32_t const_value = 2;
    const_tensor.SetData((uint8_t*)&const_value, sizeof(const_value));

    auto data_0 = OP_CFG(DATA)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_INDEX, 0)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto variable_1 = OP_CFG(VARIABLE)
                    .OutCnt(1)
                    //.Attr(ATTR_NAME_WEIGHTS, const_tensor)
                    .Attr(ATTR_VARIABLE_PLACEMENT, "host")  
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto unique_1 = OP_CFG("Unique")
                    .InCnt(1)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto unique_2 = OP_CFG("Unique")
                    .InCnt(1)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto unique_3 = OP_CFG("Unique")
                    .InCnt(1)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto unique_4 = OP_CFG("Unique")
                    .InCnt(1)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    // 定义3个ADD算子作为桥接
    auto add_bridge1 = OP_CFG(ADD)
                    .InCnt(2)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto add_bridge2 = OP_CFG(ADD)
                    .InCnt(2)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto add_bridge3 = OP_CFG(ADD)
                    .InCnt(2)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    // 常量节点（为ADD提供第二个输入）
    auto const_0 = OP_CFG(CONSTANTOP)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_WEIGHTS, const_tensor)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto net_output = OP_CFG(NETOUTPUT)
                    .InCnt(1)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    /******************** 构图逻辑 ********************/
    // 主数据流
    CHAIN(NODE("_arg_0", data_0) 
        -> NODE("unique_1", unique_1) 
        -> NODE("add_bridge1", add_bridge1) 
        -> NODE("unique_2", unique_2)
        -> NODE("add_bridge2", add_bridge2)
        -> NODE("unique_3", unique_3)
        -> NODE("add_bridge3", add_bridge3)
        -> NODE("unique_4", unique_4)
        -> NODE("net_output", net_output));

    // 修改连接：将data_1替换为variable_1
    CHAIN(NODE("variable_1", variable_1) -> NODE("add_bridge1", add_bridge1)); // 变量输入
    CHAIN(NODE("const_0", const_0) -> NODE("add_bridge2", add_bridge2));
    CHAIN(NODE("const_0", const_0) ->EDGE(0, 0)-> NODE("add_bridge3", add_bridge3));
};

return ToGeGraph(dynamic_graph_4);
}
/*
                                                                ┌──────────┐  (0,1)   ┌────────┐  (1,1)
                                                                │  add1_3  │ <─────── │  var1  │ ─────────────────────────────────┐
                                                                └──────────┘          └────────┘                                  │
                                                                  │                                                               │
                                                                  │                                                               │
                                                                  │                                                               │
                                                                  │                                                               │
                                                                  │ (0,2)                                                         │
                                                                  │                                                               │
                                                                  │                              (0,1)                            │                              (0,2)
             ┌────────────────────────────────────────────────────┼─────────────────────────────────────────┐                     │                   ┌───────────────────────────────────┐
             │                                                    ∨                                         ∨                     ∨                   ∨                                   │
             │  ┌──────────────┐  (0,1)   ┌──────────┐  (0,0)   ┌──────────┐  (0,0)   ┌────────┐  (0,0)   ┌──────────┐  (0,0)   ┌────────┐  (0,0)   ┌──────────┐  (0,0)   ┌────────────┐  │
             │  │ const_layer1 │ ───────> │  add1_1  │ ───────> │          │ ───────> │ add2_1 │ ───────> │          │ ───────> │ add3_1 │ ───────> │ unique_4 │ ───────> │ net_output │  │
             │  └──────────────┘          └──────────┘          │          │          └────────┘          │          │          └────────┘          └──────────┘          └────────────┘  │
             │    │                         ∧                   │          │            ∧                 │          │                                ∧                                   │
┌──────────────┼────┘                         │ (0,0)             │ unique_2 │            │ (0,1)           │ unique_3 │                                │                                   │
│              │                              │                   │          │            │                 │          │                                │                                   │
│              │  ┌──────────────┐  (0,0)   ┌──────────┐          │          │          ┌────────┐          │          │  (2,0)   ┌────────┐            │                                   │
│              │  │    data_0    │ ───────> │ unique_1 │          │          │ ─┐       │  var2  │          │          │ ───────> │ add3_3 │ ───────────┼───────────────────────────────────┘
│              │  └──────────────┘          └──────────┘          └──────────┘  │       └────────┘          └──────────┘          └────────┘            │
│              │                              │                     ∧           │                             │                     ∧        (1,1)      │
│              │                              │ (1,0)               │ (0,1)     │                             │                     └───────────────────┼─────────────────────┐
│              │                              ∨                     │           │                             │                                         │                     │
│              │  ┌──────────────┐  (0,1)   ┌──────────┐            │           │                             │                                         │                     │
│              │  │    data_1    │ ───────> │  add1_2  │ ───────────┘           │                             │                                         │                     │
│              │  └──────────────┘          └──────────┘                        │                             │                                         │                     │
│              │    │                                                           │                             │                                         │                     │
│              │    │ (1,1)                                                     │                             │                                         │                     │
│              │    ∨                                                           │                             │                                         │                     │
│              │  ┌──────────────┐  (0,1)                                       │                             │                                         │                     │
│    ┌─────────┼> │    add3_2    │ ─────────────────────────────────────────────┼─────────────────────────────┼─────────────────────────────────────────┘                     │
│    │         │  └──────────────┘                                              │                             │                                                               │
│    │         │                                                                │                             │                                                               │
│    │ (1,0)   └──────────────────────────────┐                                 │                             │                                                               │
│    │                                        │                                 │                             │                                                               │
│    │            ┌──────────────┐  (0,1)   ┌──────────┐                        │                             │                                                               │
│    │            │ const_layer2 │ ───────> │  add2_2  │                        │                             │                                                               │
│    │            └──────────────┘          └──────────┘                        │                             │                                                               │
│    │                                        ∧          (1,0)                  │                             │                                                               │
│    │                                        └─────────────────────────────────┘                             │                                                               │
│    │                                                                                                        │                                                               │
│    │                                                                                                        │                                                               │
│    └────────────────────────────────────────────────────────────────────────────────────────────────────────┘                                                               │
│                                                                                                                                                                             │
│                                                                                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

*/
Graph GraphFactory::BuildEnhancedUniqueGraph() {
    DEF_GRAPH(enhanced_graph) {
        // ================== 初始化常量/VARIABLE ==================
        // 常量组1 (固定形状)
        GeTensorDesc const_desc1(GeShape({1}), FORMAT_ND, DT_FLOAT);
        GeTensor const_tensor1(const_desc1);
        int32_t const_val1 = 2;
        const_tensor1.SetData((uint8_t*)&const_val1, sizeof(const_val1));

        GeTensorDesc const_desc2(GeShape({1}), FORMAT_ND, DT_INT64);
        GeTensor const_tensor2(const_desc2);
        int64_t  const_val2 = 2;
        const_tensor2.SetData((uint8_t*)&const_val2, sizeof(const_val2));

        // VARIABLE组 (动态兼容形状)
        auto var1 = OP_CFG(VARIABLE)
                .OutCnt(1)
                .Attr(ATTR_NAME_WEIGHTS, const_tensor1)
                .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

        auto var2 = OP_CFG(VARIABLE)
                .OutCnt(1)
                .Attr(ATTR_NAME_WEIGHTS, const_tensor1)
                .TensorDesc(FORMAT_ND, DT_FLOAT, {8});

        // ================== 输入节点 ======================
        auto data_0 = OP_CFG(DATA)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_INDEX, 0)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

        auto data_1 = OP_CFG(DATA)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_INDEX, 1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

        // ================== Unique层次结构 ==================
        // Layer 1: Unique1
        auto unique_1 = OP_CFG("Unique")
                    .InCnt(1)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

        // Layer 2: Unique2
        auto unique_2 = OP_CFG("Unique")
                    .InCnt(1)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

        // Layer 3: Unique3
        auto unique_3 = OP_CFG("Unique")
                    .InCnt(1)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

        // ================== 过渡算子组 ==================
        // Layer1→Layer2过渡算子 (3个ADD + 1个CONST)
        auto add1_1 = OP_CFG(ADD).InCnt(2).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
        auto add1_2 = OP_CFG(ADD).InCnt(2).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
        auto add1_3 = OP_CFG(ADD).InCnt(2).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
        auto const_layer1 = OP_CFG(CONSTANTOP)
                        .Attr(ATTR_NAME_WEIGHTS, const_tensor1)
                        .TensorDesc(FORMAT_ND, DT_FLOAT, {1});
        auto const_layer2 = OP_CFG(CONSTANTOP)
                        .Attr(ATTR_NAME_WEIGHTS, const_tensor1)
                        .TensorDesc(FORMAT_ND, DT_FLOAT, {1});

        // Layer2→Layer3过渡算子 (2个ADD + 1个VARIABLE)
        auto add2_1 = OP_CFG(ADD).InCnt(2).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
        auto add2_2 = OP_CFG(ADD).InCnt(2).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

        // Layer3→Layer4过渡算子 (3个ADD + 混合输入)
        auto add3_1 = OP_CFG(ADD).InCnt(2).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
        auto add3_2 = OP_CFG(ADD).InCnt(2).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
        auto add3_3 = OP_CFG(ADD).InCnt(2).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

        // 新增reshape和shape算子,以及variable和const分别连接add和断点unique
        auto shape = OP_CFG(SHAPE)
                .InCnt(1)
                .OutCnt(1)
                .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
                .Attr("_force_infershape_when_running", true)
                .TensorDesc(FORMAT_ND, DT_FLOAT, {-1}) 
                .Build("shape");

        auto reshape = OP_CFG(RESHAPE)
                .InCnt(2)
                .OutCnt(1)
                .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
                .Attr("_force_infershape_when_running", true)
                .TensorDesc(FORMAT_ND, DT_FLOAT, {-1}) 
                .Build("reshape");

        auto data_2 = OP_CFG(DATA)
                .InCnt(1)
                .OutCnt(1)
                .Attr(ATTR_NAME_INDEX, 2)
                .TensorDesc(FORMAT_ND, DT_INT64, {-1})  
                .Build("data_2");

        auto var3 = OP_CFG(VARIABLE)
                .OutCnt(1)
                .Attr(ATTR_NAME_WEIGHTS, const_tensor2)
                .TensorDesc(FORMAT_ND, DT_FLOAT, {16})
                .Build("var3");

        auto const_layer3 = OP_CFG(CONSTANTOP)
                .OutCnt(1)
                .Attr(ATTR_NAME_WEIGHTS, const_tensor2)
                .TensorDesc(FORMAT_ND, DT_FLOAT, {16})
                .Build("const_layer3");

        auto add4_1 = OP_CFG(ADD)
                .InCnt(2)
                .OutCnt(1)
                .TensorDesc(FORMAT_ND, DT_FLOAT, {-1})
                .Build("add4_1");

        auto unique_4 = OP_CFG("Unique")
                .InCnt(1)
                .OutCnt(1)
                .TensorDesc(FORMAT_ND, DT_FLOAT, {-1})
                .Build("unique_4");

        auto unique_5 = OP_CFG("Unique")
                .InCnt(1)
                .OutCnt(1)
                .TensorDesc(FORMAT_ND, DT_FLOAT, {-1})
                .Build("unique_5");

        // ================== 网络输出 ==================
        auto net_output = OP_CFG(NETOUTPUT)
                        .InCnt(6)
                        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1})
                        .Build("net_output");

        // ================== 分层连接逻辑 ==================
        // Layer1: data_0 → unique_1
        CHAIN(NODE("data_0", data_0) -> NODE("unique_1", unique_1));

        // Layer1→Layer2过渡 (多路径ADD)
        CHAIN(NODE("unique_1", unique_1) -> NODE("add1_1", add1_1));
        CHAIN(NODE("unique_1", unique_1) -> NODE("add1_2", add1_2));
        CHAIN(NODE("unique_1", unique_1) -> NODE("add1_3", add1_3));
        // 注入外部输入到ADD
        CHAIN(NODE("const_layer1", const_layer1) -> NODE("add1_1", add1_1));
        CHAIN(NODE("data_1", data_1) -> NODE("add1_2", add1_2));
        CHAIN(NODE("var1", var1) -> NODE("add1_3", add1_3));
        // 聚合到Layer2的Unique
        CHAIN(NODE("add1_1", add1_1) -> NODE("unique_2", unique_2));
        CHAIN(NODE("add1_2", add1_2) -> NODE("unique_2", unique_2));  // 多输入到同一Unique
        CHAIN(NODE("add1_3", add1_3) -> NODE("unique_2", unique_2));

        // Layer2→Layer3过渡
        CHAIN(NODE("unique_2", unique_2) -> NODE("add2_1", add2_1));
        CHAIN(NODE("unique_2", unique_2) -> NODE("add2_2", add2_2));
        // 注入变量和常量
        CHAIN(NODE("var2", var2) -> NODE("add2_1", add2_1));
        CHAIN(NODE("const_layer2", const_layer2) -> NODE("add2_2", add2_2));
        // 连接到Layer3
        CHAIN(NODE("add2_1", add2_1) -> NODE("unique_3", unique_3));
        CHAIN(NODE("add2_2", add2_2) -> NODE("unique_3", unique_3));

        // Layer3→Layer4过渡
        CHAIN(NODE("unique_3", unique_3) -> NODE("add3_1", add3_1));
        CHAIN(NODE("unique_3", unique_3) -> NODE("add3_2", add3_2));
        CHAIN(NODE("unique_3", unique_3) -> NODE("add3_3", add3_3));
        // 混合输入源
        CHAIN(NODE("var1", var1) -> NODE("add3_1", add3_1));
        CHAIN(NODE("data_1", data_1) -> NODE("add3_2", add3_2));
        CHAIN(NODE("const_layer1", const_layer1) -> NODE("add3_3", add3_3));
        // 二类算子特殊验证层
        CHAIN(NODE(var3) -> EDGE(0, 0) -> NODE(add4_1));
        CHAIN(NODE(const_layer3) -> EDGE(0, 1) -> NODE(add4_1));
        CHAIN(NODE(add4_1) -> EDGE(0, 0) -> NODE(reshape));
        CHAIN(NODE(data_2) -> EDGE(0, 0) -> NODE(shape) -> EDGE(0, 1) -> NODE(reshape));
        CHAIN(NODE(var3) -> EDGE(0, 0) -> NODE(unique_4) -> EDGE(0, 1) -> NODE(net_output));
        CHAIN(NODE(const_layer3) -> EDGE(0, 0) -> NODE(unique_5) -> EDGE(0, 2) -> NODE(net_output));
        CHAIN(NODE(reshape) -> EDGE(0, 0) -> NODE(net_output));
        // 最终输出
        CHAIN(NODE("add3_1", add3_1) -> NODE(net_output));
        CHAIN(NODE("add3_2", add3_2) -> NODE(net_output));
        CHAIN(NODE("add3_3", add3_3) -> NODE(net_output));

    };
    return ToGeGraph(enhanced_graph);
}

/*

                                                                                                               (2,2)
                                                                                                   ┌────────────────────────────────────┐
                                                                                                   ∨                                    │
 ┌────────────┐  (0,0)   ┌─────────────┐  (0,0)   ┌─────────────┐  (0,0)   ┌──────────┐  (0,0)   ┌───────────┐  (0,0)   ┌────────────┐  │
 │   _arg_0   │ ───────> │  unique_1   │ ───────> │ add_bridge1 │ ───────> │ unique_2 │ ───────> │ add_merge │ ───────> │ net_output │  │
 └────────────┘          └─────────────┘          └─────────────┘          └──────────┘          └───────────┘          └────────────┘  │
                                                    ∧                                              ∧                                    │
┌───────────────────────────────────────────┐         │ (0,1)                                        │                                    │
│                                           │         │                                              │                                    │
│  ┌────────────┐  (0,0)   ┌─────────────┐  │       ┌─────────────┐                                  │                                    │
│  │ variable_1 │ ───────> │  unique_3   │  └────── │   const_0   │ ─────────────────────────────────┼────────────────────────────────────┘
│  └────────────┘          └─────────────┘          └─────────────┘                                  │
│                            │                                                                       │
│                            │ (0,0)                                                                 │
│                            ∨                                                                       │
│                 (1,1)    ┌─────────────┐  (0,0)   ┌─────────────┐  (0,1)                           │
└────────────────────────> │ add_branch1 │ ───────> │  unique_4   │ ─────────────────────────────────┘
                         └─────────────┘          └─────────────┘

*/
Graph GraphFactory::BuildDynamicInputGraphWithVarAndConst2() {
DEF_GRAPH(parallel_graph) {
    GeTensorDesc const_tensor_desc(GeShape({1}), FORMAT_ND, DT_FLOAT);
    GeTensor const_tensor(const_tensor_desc);
    int32_t const_value = 2;
    const_tensor.SetData((uint8_t*)&const_value, sizeof(const_value));

    auto data_0 = OP_CFG(DATA)
                    .InCnt(1)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_INDEX, 0)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto variable_1 = OP_CFG(VARIABLE)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_WEIGHTS, const_tensor)
                    .Attr(ATTR_VARIABLE_PLACEMENT, "host")
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {16}); 

    auto unique_1 = OP_CFG("Unique")
                    .InCnt(1)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto unique_2 = OP_CFG("Unique")
                    .InCnt(1)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto unique_3 = OP_CFG("Unique")
                    .InCnt(1)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto unique_4 = OP_CFG("Unique")
                    .InCnt(1)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto add_bridge1 = OP_CFG(ADD)
                    .InCnt(2)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto add_branch1 = OP_CFG(ADD)
                    .InCnt(2)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto add_bridge2 = OP_CFG(ADD)
                    .InCnt(2)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto add_merge = OP_CFG(ADD)
                    .InCnt(2)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto const_0 = OP_CFG(CONSTANTOP)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_WEIGHTS, const_tensor)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    auto net_output = OP_CFG(NETOUTPUT)
                    .InCnt(1)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    CHAIN(NODE("_arg_0", data_0)
        -> NODE("unique_1", unique_1)
        -> NODE("add_bridge1", add_bridge1)
        -> NODE("unique_2", unique_2)
        -> NODE("add_merge", add_merge));

    CHAIN(NODE("variable_1", variable_1)
        -> NODE("unique_3", unique_3)
        -> NODE("add_branch1", add_branch1)
        -> NODE("unique_4", unique_4)
        -> NODE("add_merge", add_merge));

    CHAIN(NODE("const_0", const_0) ->  EDGE(0, 0)->NODE("add_bridge1", add_bridge1));
    CHAIN(NODE("const_0", const_0) -> EDGE(0, 0)->NODE("add_branch1", add_branch1));
    CHAIN(NODE("const_0", const_0) -> EDGE(0, 0)->NODE("add_merge", add_merge));

    CHAIN(NODE("add_merge", add_merge) -> NODE("net_output", net_output));
};

return ToGeGraph(parallel_graph);
}

/*
data_0
├─ (data_0 + variable_0) → Add1 → Unique1 → net_output
├─ (data_0 + variable_1) → Add2 → (Add2 + variable_2) → Add3 → Unique2 → net_output
├─ (data_0 + variable_1) → Add4 → (Add4 + variable_3) → Add5 → (Add5 + variable_4) → Add6 → Unique3 → net_output
├─ (data_0 + variable_0) → Add7 → (Add7 + variable_5) → Add8 → Unique4 → net_output
└─ (data_0 + variable_0) → Add7 → (Add7 + variable_6) → Add9 → Unique5 → net_output
*/
Graph GraphFactory::BuildDynamicAndBoarderInputGraph() {
DEF_GRAPH(dynamic_boarder_graph) {
    GeTensorDesc tensor_desc(GeShape({16}), FORMAT_ND, DT_FLOAT);
    GeTensor const_tensor(tensor_desc);
    int32_t const_value = 2;
    const_tensor.SetData((uint8_t*)&const_value, sizeof(const_value));
    
    // ================== 输入节点 ==================
    auto data_0 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1}); 

    auto data_1 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1}); 

    auto data_2 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1}); 

    auto data_3 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1}); 

    auto data_4 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1}); 

    // ================== Variable 节点 ==================
    auto variable_0 = OP_CFG(VARIABLE).OutCnt(1)
        .Attr(ATTR_NAME_WEIGHTS, const_tensor)
        .Attr(ATTR_VARIABLE_PLACEMENT, "host")
        .TensorDesc(FORMAT_ND, DT_FLOAT, {64}); 

        auto variable_1 = OP_CFG(VARIABLE).OutCnt(1)
        .Attr(ATTR_NAME_WEIGHTS, const_tensor)
        .Attr(ATTR_VARIABLE_PLACEMENT, "host")
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});  

        auto variable_2 = OP_CFG(VARIABLE).OutCnt(1)
        .Attr(ATTR_NAME_WEIGHTS, const_tensor)
        .Attr(ATTR_VARIABLE_PLACEMENT, "host")
        .TensorDesc(FORMAT_ND, DT_FLOAT, {32});  

        auto variable_3 = OP_CFG(VARIABLE).OutCnt(1)
        .Attr(ATTR_NAME_WEIGHTS, const_tensor)
        .Attr(ATTR_VARIABLE_PLACEMENT, "host")
        .TensorDesc(FORMAT_ND, DT_FLOAT, {64});  

        auto variable_4 = OP_CFG(VARIABLE).OutCnt(1)
        .Attr(ATTR_NAME_WEIGHTS, const_tensor)
        .Attr(ATTR_VARIABLE_PLACEMENT, "host")
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});  

    auto variable_5 = OP_CFG(VARIABLE).OutCnt(1)
        .Attr(ATTR_NAME_WEIGHTS, const_tensor)
        .Attr(ATTR_VARIABLE_PLACEMENT, "host")
        .TensorDesc(FORMAT_ND, DT_FLOAT, {64});  

    auto variable_6 = OP_CFG(VARIABLE).OutCnt(1)
        .Attr(ATTR_NAME_WEIGHTS, const_tensor)
        .Attr(ATTR_VARIABLE_PLACEMENT, "host")
        .TensorDesc(FORMAT_ND, DT_FLOAT, {32});   

    auto variable_7 = OP_CFG(VARIABLE).OutCnt(1)
        .Attr(ATTR_NAME_WEIGHTS, const_tensor)
        .Attr(ATTR_VARIABLE_PLACEMENT, "host")
        .TensorDesc(FORMAT_ND, DT_FLOAT, {16});  

    auto variable_8 = OP_CFG(VARIABLE)
        .OutCnt(1)
        .Attr(ATTR_NAME_WEIGHTS, const_tensor)
        .Attr(ATTR_VARIABLE_PLACEMENT, "host")
        .TensorDesc(FORMAT_ND, DT_FLOAT, {32});  

    auto variable_9 = OP_CFG(VARIABLE).OutCnt(1)
        .Attr(ATTR_NAME_WEIGHTS, const_tensor)
        .Attr(ATTR_VARIABLE_PLACEMENT, "host")
        .TensorDesc(FORMAT_ND, DT_FLOAT, {64}); 

    // ================== Unique 节点 ==================
    auto unique_1 = OP_CFG("Unique").InCnt(1).OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto unique_2 = OP_CFG("Unique").InCnt(1).OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto unique_3 = OP_CFG("Unique").InCnt(1).OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto unique_4 = OP_CFG("Unique").InCnt(1).OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    auto unique_5 = OP_CFG("Unique").InCnt(1).OutCnt(1) 
        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    // ADD节点集群（与拓扑图严格对应）
    auto add_1 = OP_CFG(ADD).InCnt(2).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
    auto add_2 = OP_CFG(ADD).InCnt(2).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
    auto add_3 = OP_CFG(ADD).InCnt(2).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
    auto add_4 = OP_CFG(ADD).InCnt(2).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
    auto add_5 = OP_CFG(ADD).InCnt(2).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
    auto add_6 = OP_CFG(ADD).InCnt(2).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
    auto add_7 = OP_CFG(ADD).InCnt(2).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
    auto add_8 = OP_CFG(ADD).InCnt(2).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
    auto add_9 = OP_CFG(ADD).InCnt(2).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
    auto add_10 = OP_CFG(ADD).InCnt(2).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
    
    auto net_output = OP_CFG(NETOUTPUT)
                    .InCnt(5)
                    .OutCnt(1)
                    .TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    // 分支1: (data_0 + variable_0) → Add1 → Unique1 → net_output
    CHAIN(NODE("_arg_0", data_0) -> NODE("add_1", add_1));
    CHAIN(
        NODE("variable_0", variable_0)
        -> NODE("add_1", add_1)
        -> NODE("unique_1", unique_1)
        -> NODE("net_output", net_output)
    );

    // 分支2: (data_0 + variable_1) → Add2 → (Add2 + variable_5) → Add3 → Unique2 → net_output
    CHAIN(NODE("_arg_1", data_1) -> NODE("add_2", add_2));
    CHAIN(NODE("variable_5", variable_5) -> NODE("add_3", add_3));
    CHAIN(
        NODE("variable_1", variable_1)
        -> NODE("add_2", add_2)
        -> NODE("add_3", add_3)
        -> NODE("unique_2", unique_2)
        -> NODE("net_output", net_output)
    );

    // 分支3:  (data_0 + variable_1) → Add4 → (Add4 + variable_6) → Add5 → (Add5 + variable_7) → Add6 → Unique3 → net_output
    CHAIN(NODE("_arg_2", data_2) -> NODE("add_4", add_4));
    CHAIN(NODE("variable_6", variable_6) -> NODE("add_5", add_5));
    CHAIN(NODE("variable_7", variable_7) -> NODE("add_6", add_6));
    CHAIN(
        NODE("variable_2", variable_2)
        -> NODE("add_4", add_4)
        -> NODE("add_5", add_5)
        -> NODE("add_6", add_6)
        -> NODE("unique_3", unique_3)
        -> NODE("net_output", net_output)
    );

    // 分支4: (data_0 + variable_0) → Add7 → (Add7 + variable_8) → Add8 → Unique4 → net_output
    CHAIN(NODE("_arg_3", data_3) -> NODE("add_7", add_7));
    CHAIN(NODE("variable_8", variable_8) -> NODE("add_8", add_8));
    CHAIN(
        NODE("variable_3", variable_3)
        -> NODE("add_7", add_7)
        -> NODE("add_8", add_8)
        -> NODE("unique_4", unique_4)
        -> NODE("net_output", net_output)
    );

    // 分支5: (data_0 + variable_0) → Add9 → (Add9 + variable_9) → Add10 → Unique5 → net_output
    CHAIN(NODE("_arg_4", data_4) -> NODE("add_9", add_9));
    CHAIN(NODE("variable_9", variable_9) -> NODE("add_10", add_9));
    CHAIN(
        NODE("variable_4", variable_4)
        -> NODE("add_9", add_9)
        -> NODE("add_10", add_9)
        -> NODE("unique_5", unique_5)
        -> NODE("net_output", net_output)
    ); 
};
return ToGeGraph(dynamic_boarder_graph);
}

/*
┌────────┐  (0,0)   ┌───────┐  (0,1)   ┌────────┐
│ data_1 │ ───────> │ add_1 │ <─────── │ data_2 │
└────────┘          └───────┘          └────────┘
*/
Graph GraphFactory::BuildStaticInputGraph() {
DEF_GRAPH(slice_scheduler_static_graph) {
    auto data1 = OP_CFG(DATA)
                   .InCnt(1)
                   .OutCnt(1)
                   .Attr(ATTR_NAME_INDEX, 0)
                   .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 24, 24});
    auto data2 = OP_CFG(DATA).InCnt(1)
                     .OutCnt(1)
                     .Attr(ATTR_NAME_INDEX, 1)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 24, 24});
    auto add1 = OP_CFG(ADD)
        .InCnt(2)
        .OutCnt(1)
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 24, 24});
    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 1, 24, 24});

    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add1)->NODE("neoutput", net_output));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add1));        
};
return ToGeGraph(slice_scheduler_static_graph);
}

Graph GraphFactory::BuildOutPutGraph() {
    DEF_GRAPH(outputTest) {    
        GeTensorDesc const_desc(GeShape({16, 16}), FORMAT_ND, DT_INT64);
        GeTensor const_tensor(const_desc);
        int64_t  const_val = 2;
        const_tensor.SetData((uint8_t*)&const_val, sizeof(const_val));
                // 新增reshape和shape算子,以及variable和const分别连接add和断点unique
        auto shape = OP_CFG(SHAPE)
                .InCnt(1)
                .OutCnt(1)
                .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
                .Attr("_force_infershape_when_running", true)
                .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1}) 
                .Build("shape");

        auto reshape = OP_CFG(RESHAPE)
                .InCnt(2)
                .OutCnt(1)
                .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
                .Attr("_force_infershape_when_running", true)
                .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1}) 
                .Build("reshape");

        auto data = OP_CFG(DATA)
                .InCnt(1)
                .OutCnt(1)
                .Attr(ATTR_NAME_INDEX, 2)
                .TensorDesc(FORMAT_ND, DT_INT64, {-1, -1})  
                .Build("data");

        auto var = OP_CFG(VARIABLE)
                .OutCnt(1)
                .Attr(ATTR_NAME_WEIGHTS, const_tensor)
                .TensorDesc(FORMAT_ND, DT_FLOAT, {16, 16})
                .Build("var");

        auto const_layer = OP_CFG(CONSTANTOP)
                .OutCnt(1)
                .Attr(ATTR_NAME_WEIGHTS, const_tensor)
                .TensorDesc(FORMAT_ND, DT_FLOAT, {16, 16})
                .Build("const_layer");

        auto add = OP_CFG(ADD)
                .InCnt(2)
                .OutCnt(1)
                .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1})
                .Build("add");

        // ================== 网络输出 ==================
        auto net_output = OP_CFG(NETOUTPUT)
                        .InCnt(1)
                        .TensorDesc(FORMAT_ND, DT_FLOAT, {-1, -1})
                        .Build("net_output");
        // 二类算子特殊验证层
        CHAIN(NODE(var) -> EDGE(0, 0) -> NODE(add));
        CHAIN(NODE(const_layer) -> EDGE(0, 1) -> NODE(add));
        CHAIN(NODE(add) -> EDGE(0, 0) -> NODE(reshape));
        CHAIN(NODE(data) -> EDGE(0, 0) -> NODE(shape) -> EDGE(0, 1) -> NODE(reshape));
        CHAIN(NODE(reshape) -> EDGE(0, 0) -> NODE(net_output));
    };
    return ToGeGraph(outputTest);
}



FAKE_NS_END
