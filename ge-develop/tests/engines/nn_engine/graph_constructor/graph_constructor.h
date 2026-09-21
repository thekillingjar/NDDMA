/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * @brief construct a graph using a simple way like:
 * Function definition: AddOpDesc(op's name, op's type);
 *                         SetInput(destination op's name , source op's name)
 * GraphConstructor graph_constructor("graph1");

 * graph_constructor.AddOpDesc("OutputName1", "OutputType1")
 *                 .AddOpDesc("OutputName2", "OutputType1")
 *                 .SetOutput("OutputName1")
 *                 .SetInput("OutputName2:0", {"InputType_1:0"})
 *                 .SetInput("OutputName2:1", {"InputType_1:0"})
 *                 .SetInput("OutputName1:1", {"InputType_2:0"});
 *                 .SetInput("OutputName1:0", {"InputType_3:0"});
 *
 *
 * or (In this case, if there are multiple inputs, we will consider it as the
 * input0, input1, input2... of the left parameter. Function definition:
 * SetInput(dst_node_name, {src_node_name0, src_node_name_1...})):
 * graph_constructor.AddOpDesc("OutputName1", "OutputType1")
 *                 .AddOpDesc("OutputName2", "OutputType1")
 *                 .SetOutput("OutputName1")
 *                 .SetInput("OutputName2", {"InputType_1:0", "InputType_1:0"})
 *                 .SetInput("OutputName1", {"InputType_3, InputType_2"});
 *
 *
 * or (In the following case, the SetInput is its second version, which is
 * defined as:
 * SetInput({dst_node_name0, dst_node_name1}, src_node_name). That is the output
 * of src node will be passed to multiple dst node.).
 * graph_constructor.AddOpDesc("OutputName1", "OutputType1")
 *                 .AddOpDesc("OutputName2", "OutputType1")
 *                 .SetOutput("OutputName1")
 *                 .SetInput({OutputName2:0, OutputName2:1}, "InputType_1:0")
 *                 .SetInput("OutputName1", {"InputType_3, InputType_2"});
 * p.s.:OutputName2 can be any node name.
 *
 *
 * or :
 * graph_constructor.SetOutput({"OutputType1_1", "OutputType1_2"})
 *                 .SetInput("OutputName2:0", {"InputType_1:0", "InputType_1:0"})
 *                 .SetInput("OutputName1", {"InputType_3, InputType_2"});
 *
 *
 * The user can either use:
 *
 * AddOpDesc(op's name, op's type) to set the mapping relation of
 * output op name and output op type
 *
 * or use:
 *
 * SetInput(destination op's name , source op's name) to set the connection
 * between the source and destination in which the program will automatically
 * extract the op type from the name which is in a correct format of:
 * OpType + "_" + Integer1(node index) + ":" + Integer2(input or output index).
 * If Integer1 is missing or "_" is missing, for example: Conv2D_:1 or
 * Conv2D:1, we will consider it as Conv2D_0.
 * And if Interger2 is missing or : is missing, for example: Conv2D: or Conv2D
 * or Conv2D_2, we will consider it as Con2D_0:0, Con2D_0:0, Con2D_2:0
 * seperately. *
 *
 * And for convenience, we will ignore the case of the node name.
 */

#ifndef LLT_FUSION_ENGINE_GRAPH_CONSTRUCTOR_GRAPH_CONSTRUCTOR_H_
#define LLT_FUSION_ENGINE_GRAPH_CONSTRUCTOR_GRAPH_CONSTRUCTOR_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "common/util/op_info_util.h"
#include "common/graph_comm.h"
#include "string.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"

using std::initializer_list;
using std::map;
using std::string;
using std::vector;

namespace fe {
#define GC_PRINT_SW (false)

#define GC_LOGD(...) \
  if (GC_PRINT_SW) { \
    D_FE_LOGD(FE_MODULE_NAME, __VA_ARGS__); \
  } \

#define GC_LOGI(...) \
  if (GC_PRINT_SW) { \
    D_FE_LOGI(FE_MODULE_NAME, __VA_ARGS__); \
  } \

enum ENUM_SRC_OR_DST {
  SOURCE = 0,
  DESTINATION = 1,
  SOURCE_AND_DESTINATION = 2,
};
static ge::Format DEFAULT_FORMAT = ge::FORMAT_NCHW;

static ge::DataType DEFAULT_DTYPE = ge::DT_FLOAT;

static ge::GeShape DEFAULT_SHAPE = ge::GeShape({17,33,35,45});

/**
   * @ingroup fe
   * @brief This struct contains the name, format, datatype,shape
   * We assume the original data_type is the same as the current datatype,
   * because we will not use that in FE. And the user only need to set the
   * original shape because the current shape will be inferred by the original
   * shape and format relation.
   */
ge::GeTensorDesc GetTensorDesc(const ge::OpDescPtr &op_desc_ptr,
                               const uint32_t &index, bool is_input);

void SetTensorDescIntAttr(const ge::OpDescPtr &op_desc_ptr,
                          const uint32_t &index, bool is_input,
                          const std::string &attr, const int64_t &attr_value);

struct DetailedNodeAndTensorInformation {
  string name_; // input string which consists of name + op index + tensor index
  ge::Format format_ = DEFAULT_FORMAT;   // the Op types of Ops
  ge::DataType data_type_ = DEFAULT_DTYPE;
  ge::GeShape original_shape_ = DEFAULT_SHAPE;
  ge::Format original_format_ = DEFAULT_FORMAT;
  DetailedNodeAndTensorInformation (const string& name_param) {
    name_ = name_param;
    format_ = DEFAULT_FORMAT;   // the Op types of Ops
    original_format_ = DEFAULT_FORMAT;
    data_type_ = DEFAULT_DTYPE;
    original_shape_ = DEFAULT_SHAPE;
  }
  /* Shape is inferred by current format original format and  original shape*/
  DetailedNodeAndTensorInformation (const string& name_param,
                                    const ge::Format& format,
                                    const ge::Format& original_format = DEFAULT_FORMAT,
                                    const ge::DataType& dtype = DEFAULT_DTYPE,
                                    const ge::GeShape& original_shape = DEFAULT_SHAPE) {
    name_ = name_param;
    format_ = format;   // the Op types of Ops
    original_format_ = original_format; // original format of tensor
    data_type_ = dtype;
    original_shape_ = original_shape;
  }
};

static const std::map<ge::Format, uint32_t> FORMAT_NAME_MAP {
    {ge::FORMAT_NDHWC, 5},
    {ge::FORMAT_DHWCN, 5},
    {ge::FORMAT_DHWNC, 5}
};
using ComputeGraphPtr = std::shared_ptr<ge::ComputeGraph>;
/** Graph Constructor
 *  Constructor a compute graph in a easy way
 */
class GraphConstructor {
 public:
  /**
   * @ingroup fe
   * @brief description of Ops
   */
  struct OpDesc {
    string op_name;                              // Identifier
    ge::NodePtr node;
    string type;                   // the Op types of Ops
  };

  /**
   * @ingroup fe
   * @brief It shows how to add data edges between nodes to nodes
   */
  struct ConnectionInfo {
    string op_name; // unique
    string type;   // the Op types of Ops
    ge::NodePtr node;
    int32_t starting_tensor_index;
  };

  using DetailedTensor = struct DetailedNodeAndTensorInformation;


  GraphConstructor(ComputeGraphPtr& graph, const string& name = "",
                   const ge::Format& default_format = DEFAULT_FORMAT,
                   const ge::DataType& default_dtype = DEFAULT_DTYPE,
                   const ge::GeShape& default_shape = DEFAULT_SHAPE);

  ~GraphConstructor();

  /* Get Opdesc by op_name */
  ge::NodePtr GetOp(const string &op_name);

  /** Add op description and the mapping relation between op_name and op_type into
   *  constructor.
   *  This step is not necessary, we can directly set the input and in the step
   *  of setting input, the program will automatically parse the op name and
   *  get the corresponding op type.
   *  @return GraphConstructor&
   */
  GraphConstructor& AddOpDesc(const string& op_name,
                              const string& op_type,
                              const size_t& inputs_size = 0,
                              const size_t& outputs_size = 0);

  /* Set the op impl type. */
  GraphConstructor& AddOpDesc(const OpImplType& impl_type,
                              const string& pattern,
                              const string& op_name,
                              const string& op_type,
                              const size_t& inputs_size = 0,
                              const size_t& outputs_size = 0);

  /** The input op_name is the destination node and the input_names is the source
   * nodes' name of this op. We will check whether the op_name is already in the
   * op desc vector(ops_, each OpDesc struct contains a node pointer). If it's
   * already in the op desc, we will create a new node and substitute the
   * existing node with the new one and if not, we'll just create a new node.
   * Then, add edge between source and dst. If the user specify the input or the
   * output index, we will add enough input or output tensor and then link the
   * specific anchor. Otherwise, the default tensor(anchor) index of output is 0
   * and the default tensor(anchor) index of input is increasing by 1 from 0.
   * op_name
   * @return GraphConstructor&
   */
  GraphConstructor& SetInputs(const DetailedTensor& dst_tensor,
                              const DetailedTensor& src_tensor);

  GraphConstructor& SetInputs(const DetailedTensor& dst_tensor,
                              const vector<DetailedTensor>& multiple_src_tensors);

  GraphConstructor& SetInputs(const string& dst_name,
                              const vector<string>& multiple_src_names);

  /* Set the src of last node */
  GraphConstructor& SetInputs(const vector<string>& multiple_src_names);
  /*1111111111111111111111 Set Input First Version Start 111111111111111111111*/
  GraphConstructor& SetInput(const string &dst_name, const string &src_name);

  /** For specific cases, we want to set the input and output format of specific
   * tensor. The following function provides an ability to set the format */
  GraphConstructor& SetInput(const string &dst_name, const string &src_name,
                             const ge::Format &format);

  GraphConstructor& SetInput(const string &dst_name, const string &src_name,
                             const ge::Format &format,
                             const ge::Format &original_format);

  GraphConstructor& SetInput(const string &dst_name, const string &src_name,
                             const ge::Format &format,
                             const ge::Format &original_format,
                             const vector<int64_t>& original_dims);

  GraphConstructor& SetInput(const string &dst_name,
                             const ge::Format &dst_format,
                             const string &src_name,
                             const ge::Format &src_format);

  GraphConstructor& SetInput(const string &dst_name,
                             const ge::Format &dst_format,
                             const ge::DataType &dst_dtype,
                             const string &src_name,
                             const ge::Format &src_format,
                             const ge::DataType &src_dtype);

  GraphConstructor& SetInput(const string &dst_name,
                             const ge::Format &dst_format,
                             const string &src_name,
                             const ge::Format &src_format,
                             const ge::Format &dst_original_format);

  GraphConstructor& SetInput(const string &dst_name,
                             const ge::Format &dst_format,
                             const string &src_name,
                             const ge::Format &src_format,
                             const ge::Format &dst_original_format,
                             const ge::Format &src_original_format);

  GraphConstructor& SetInput(const string &dst_name,
                             const ge::Format &dst_format,
                             const string &src_name,
                             const ge::Format &src_format,
                             const ge::Format &dst_original_format,
                             const ge::Format &src_original_format,
                             const vector<int64_t>& dst_original_dims,
                             const vector<int64_t>& src_original_dims);
  /*11111111111111111111111 Set Input First Version End 1111111111111111111111*/

  /*22222222222222222222222 Set Input Second Version Start 2222222222222222222*/
  /** For specific cases, we want to set the input and output format and shape
   * of specific tensor. The following function provides an ability to set the
   * format and shape*/
  GraphConstructor& SetInput(const string &dst_name, const string &src_name,
                             const vector<int64_t>& dims,
                             const uint32_t& dst_or_src = SOURCE_AND_DESTINATION);

  GraphConstructor& SetInput(const string &dst_name, const string &src_name,
                             const vector<int64_t>& dims,
                             const ge::Format &format,
                             const uint32_t& dst_or_src = SOURCE_AND_DESTINATION);
  /*22222222222222222222222 Set Input Second Version End 222222222222222222222*/

  /* Set Attr */
  GraphConstructor &Attr(const string &node_name, const std::string &name, bool value);
  GraphConstructor &Attr(const string &node_name, const std::string &name, int32_t value);
  GraphConstructor &Attr(const string &node_name, const std::string &name, uint32_t value);
  GraphConstructor &Attr(const string &node_name, const std::string &name, float value);
  GraphConstructor &Attr(const string &node_name, const std::string &name, const std::vector<int32_t> &value);
  GraphConstructor &Attr(const string &node_name, const std::string &name, const std::vector<uint32_t> &value);
  GraphConstructor &Attr(const string &node_name, const std::string &name, const std::vector<int64_t> &value);
  GraphConstructor &Attr(const string &node_name, const std::string &name, const char *value);

  GraphConstructor &Attr(const std::string &name, bool value);
  GraphConstructor &Attr(const std::string &name, int32_t value);
  GraphConstructor &Attr(const std::string &name, uint32_t value);
  GraphConstructor &Attr(const std::string &name, const std::vector<int32_t> &value);
  GraphConstructor &Attr(const std::string &name, const std::vector<uint32_t> &value);
  GraphConstructor &Attr(const std::string &name, const std::vector<int64_t> &value);
  GraphConstructor &Attr(const std::string &name, const char *value);

  static Status DumpGraph(const ge::ComputeGraphPtr& graph);

  /* If each edge and node of graph1 and graph2 are the same, return true,
   * otherwise return false. */
  static bool CompareGraph(const ComputeGraphPtr& graph1, const ComputeGraphPtr& graph2);

  static bool CompareNode(const ge::NodePtr & node1, const ge::NodePtr & node2);

  static string GetInputString(const ge::NodePtr& node);

  void GetNodeByName(const string& name, ge::NodePtr &node_out);

  static ge::NodePtr GetNodeByName(const string& name, const ComputeGraphPtr& graph);
  /* Set pattern for the last added node. */
  GraphConstructor& SetPattern(const string &optype);

  GraphConstructor& SetFeImPlType(OpImplType impl_type);

  /* Set last added node as tvm op. */
  GraphConstructor& SetTvmType();

  template <class T>
  GraphConstructor& SetExtAttr(string &&attr_name, const T &value);

 private:
  /* Set Tensor Desc information such as original shstatic ape and original format. */
  Status SetTensorDescInfo(ge::GeTensorDesc& tensor,
      const ge::Format& original_format, const ge::DataType& data_type,
      const ge::GeShape& shape, const ge::Format& format);

  Status AddNewNodeIntoGraph(const string& op_type,
                             const string& op_real_name,
                             const size_t& size_of_new_tensors,
                             const DetailedTensor& tensor_info,
                             const bool& is_dst_node,
                             int32_t& tensor_index,
                             ge::NodePtr& new_node);

  /* The node is existing and we */
  Status AddTensorIntoExistingNodes(
      const size_t& size_of_new_tensors,
      const DetailedTensor& tensor_info,
      const bool& is_dst_node,
      map<string, std::shared_ptr<OpDesc>>::iterator& iter,
      int32_t& input_index);

  /*Parse the name and update op and node info. Add the node
   * into connection info*/
  Status ParseNodeNameAndAddNodeIntoGraph(
      const DetailedTensor& dst_tensor,
      const size_t& size_of_new_tensors,
      bool is_dst_node, vector<ConnectionInfo>& connection_info_of_all_nodes);


  Status AddEdges(
      const vector<ConnectionInfo>& src_connection_info_of_all_nodes,
      const vector<ConnectionInfo>& dst_connection_info_of_all_nodes);



  /** Parse the input or output name, the name is as the following format:
   *
   * OpType_Integer1:Interer2
   * Integer1 is the index of the node with same type.
   * Integer2 is the index of the inputs or outputs.
   *
   * @return Status
   */
  Status NodeNameParser(const string &name, string& op_type,
                         string& op_real_name, int32_t& input_or_output_index);


  Status ReplaceNodeWithNewBode(ge::NodePtr& old_node, ge::NodePtr& new_node);

  void SetGraph(ComputeGraphPtr graph);

  ge::OpDescPtr CreateFunctionOp(vector<ge::NodePtr> &node_vec) const ;
  Status AddFunctionNodeInputDesc(ge::OpDescPtr fus_op, vector<fe::FusionDataFlow> &fus_input_edge_list);
  Status AddFunctionNodeOutputDesc(ge::OpDescPtr fus_op, vector<fe::FusionDataFlow> &fus_output_edge_list);
  Status CreateFunctionOpSubGraph(ge::NodePtr &function_node,
                                  std::vector<ge::NodePtr> &node_vec,
                                  vector<fe::FusionDataFlow> &input_edge_list,
                                  vector<fe::FusionDataFlow> &output_edge_list,
                                  vector<fe::FusionDataFlow> &output_ctrl_edge_list);
  Status TransSingleSubGraph(ge::ComputeGraph &graph, std::vector<ge::NodePtr> &node_vec);

 private:
  GraphConstructor(const GraphConstructor &) = default;

  GraphConstructor &operator=(const GraphConstructor &) = default;


 private:
  ComputeGraphPtr graph_;

  std::string graph_name_;

  vector<std::shared_ptr<OpDesc>> ops_;

  map<string, std::shared_ptr<OpDesc>> op_map_;

  ge::NodePtr last_added_node_;

  std::shared_ptr<fe::GraphComm> graph_comm_ptr_;

  size_t sgt_graph_index_{0};
};
}

#endif  // LLT_FUSION_ENGINE_GRAPH_CONSTRUCTOR_GRAPH_CONSTRUCTOR_H_
