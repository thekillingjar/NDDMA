/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include "flow_graph/data_flow.h"
#include "dflow/compiler/session/dflow_api.h"
#include "parser/onnx_parser.h"
#include "parser/tensorflow_parser.h"
#include "dlog_pub.h"

void DlogErrorInner(int moduleId, const char *fmt, ...) {}
void DlogRecord(int moduleId, int level, const char *fmt, ...) {}
namespace ge {
namespace {
constexpr size_t kMaxUserDataSize = 64U;
int8_t g_userData[kMaxUserDataSize] = {};
Status CheckParamsForUserData(const void *data, size_t size, size_t offset) {
  if (data == nullptr) {
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  if (size == 0U) {
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  if ((offset >= kMaxUserDataSize) || (kMaxUserDataSize - offset) < size) {
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  return SUCCESS;
}
}  // namespace

Status GEInitializeV2(const std::map<ge::AscendString, ge::AscendString> &options) {
  return SUCCESS;
}

Status GEFinalizeV2() {
  return SUCCESS;
}

struct MbufHeadMsg {
  uint64_t trans_id;
  uint16_t version;
  uint16_t msg_type;
  int32_t ret_code;
  uint64_t start_time;
  uint64_t end_time;
  uint32_t flags;
  uint8_t data_flag;
  uint8_t rsv[23];
  uint32_t route_label;  // will be delete when new design given.
};

class FlowMsgStub : public FlowMsg {
 public:
  FlowMsgStub() {
    head_msg_.msg_type = static_cast<uint16_t>(MsgType::MSG_TYPE_TENSOR_DATA);
  };

  FlowMsgStub(const std::vector<int64_t> &shape, DataType dt) {
    tensor_ = std::make_shared<Tensor>();
    head_msg_.msg_type = static_cast<uint16_t>(MsgType::MSG_TYPE_TENSOR_DATA);
  }

  explicit FlowMsgStub(int64_t size) {
    raw_data_.resize(size);
    head_msg_.msg_type = static_cast<uint16_t>(MsgType::MSG_TYPE_RAW_MSG);
  }

  explicit FlowMsgStub(const RawData &raw_data) {
    raw_data_.resize(raw_data.len);
    head_msg_.msg_type = static_cast<uint16_t>(MsgType::MSG_TYPE_RAW_MSG);
  }

  explicit FlowMsgStub(std::shared_ptr<Tensor> tensor) : tensor_(tensor) {
    head_msg_.msg_type = static_cast<uint16_t>(MsgType::MSG_TYPE_TENSOR_DATA);
  }

  ~FlowMsgStub() override {}

  MsgType GetMsgType() const override {
    return static_cast<MsgType>(head_msg_.msg_type);
  }

  void SetMsgType(MsgType msg_type) override {
    head_msg_.msg_type = static_cast<uint16_t>(msg_type);
  }

  Tensor *GetTensor() const override {
    return const_cast<Tensor *>(tensor_.get());
  }

  int32_t GetRetCode() const override {
    return head_msg_.ret_code;
  }

  void SetRetCode(int32_t ret_code) override {
    head_msg_.ret_code = ret_code;
  }

  void SetStartTime(uint64_t start_time) override {
    head_msg_.start_time = start_time;
  }

  uint64_t GetStartTime() const override {
    return head_msg_.start_time;
  }

  void SetEndTime(uint64_t end_time) override {
    head_msg_.end_time = end_time;
  }

  uint64_t GetEndTime() const override {
    return head_msg_.end_time;
  }

  void SetFlowFlags(uint32_t flags) override {
    head_msg_.flags = flags;
  }

  uint32_t GetFlowFlags() const override {
    return head_msg_.flags;
  }

  uint64_t GetTransactionId() const override {
    return UINT64_MAX;
  }

  void SetTransactionId(uint64_t transaction_id) {}

  Status GetRawData(void *&data_ptr, uint64_t &data_size) const override {
    data_ptr = reinterpret_cast<void *>(const_cast<uint8_t *>(&raw_data_[0]));
    data_size = raw_data_.size();
    return 0;
  }

  Status SetUserData(const void *data, size_t size, size_t offset = 0U) override {
    return 0;
  }

  Status GetUserData(void *data, size_t size, size_t offset = 0U) const override {
    return 0;
  }

 private:
  std::shared_ptr<Tensor> tensor_;
  MbufHeadMsg head_msg_ = {};
  std::vector<uint8_t> raw_data_;
};

namespace dflow {
Status DFlowInitialize(const std::map<AscendString, AscendString> &options) {
  return SUCCESS;
}
Status DFlowFinalize() {
  return SUCCESS;
}
DFlowSession::DFlowSession(const std::map<ge::AscendString, ge::AscendString> &options) {
}

DFlowSession::~DFlowSession() {
}

Status DFlowSession::AddGraph(uint32_t graph_id, const FlowGraph &graph,
                              const std::map<ge::AscendString, ge::AscendString> &options) {
  return SUCCESS;
}

Status DFlowSession::FeedDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes,
                                       const std::vector<Tensor> &inputs, const DataFlowInfo &info, int32_t timeout) {
  return SUCCESS;
}

Status DFlowSession::FetchDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes,
                                        std::vector<Tensor> &outputs, DataFlowInfo &info, int32_t timeout) {
  outputs.emplace_back(Tensor());
  return SUCCESS;
}

Status DFlowSession::FeedDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes,
                                       const std::vector<FlowMsgPtr> &inputs, int32_t timeout) {
  return SUCCESS;
}

Status DFlowSession::FetchDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes,
                                        std::vector<FlowMsgPtr> &outputs, int32_t timeout) {
  outputs.emplace_back(std::make_shared<FlowMsgStub>(std::make_shared<Tensor>()));
  return SUCCESS;
}
}

Operator::Operator(const char_t *name, const char_t *type) {}

Operator &Operator::SetAttr(const char_t *name, int64_t attr_value) {
  return *this;
}

Operator &Operator::SetAttr(const std::string &name, const std::string &attr_value) {
  return *this;
}

Operator &Operator::SetAttr(const char_t *, bool attr_value) {
  return *this;
}

Tensor::Tensor() {}

graphStatus Tensor::SetTensorDesc(const TensorDesc &tensor_desc) {
  return GRAPH_SUCCESS;
}

TensorDesc Tensor::GetTensorDesc() const {
  return TensorDesc();
}

const uint8_t *Tensor::GetData() const {
  return nullptr;
}

uint8_t *Tensor::GetData() {
  return nullptr;
}

graphStatus Tensor::SetData(const uint8_t *data, size_t size) {
  return GRAPH_SUCCESS;
}

graphStatus Tensor::SetData(const std::string &data) {
  return GRAPH_SUCCESS;
}

graphStatus Tensor::SetData(const std::vector<AscendString> &datas) {
  return GRAPH_SUCCESS;
}

Tensor Tensor::Clone() const {
  return Tensor();
}

size_t Tensor::GetSize() const {
  return 0U;
}

TensorDesc::TensorDesc() {}

TensorDesc::TensorDesc(Shape shape, Format format, DataType dt) {}

Shape::Shape() {}

Shape::Shape(const std::vector<int64_t> &dims) {}

size_t Shape::GetDimNum() const {
  return 0U;
}

std::vector<int64_t> Shape::GetDims() const {
  return {};
}

int64_t Shape::GetShapeSize() const {
  return 0;
}

AscendString::AscendString(const char_t *const name) {}

bool AscendString::operator<(const AscendString &d) const {
  return true;
}

Shape TensorDesc::GetShape() const {
  return Shape();
}

DataType TensorDesc::GetDataType() const {
  return DT_INT64;
}

graphStatus aclgrphParseONNX(const char *model_file, const std::map<AscendString, AscendString> &parser_params,
                             ge::Graph &graph) {
  return 10001;
}

graphStatus aclgrphParseTensorFlow(const char *model_file, const std::map<AscendString, AscendString> &parser_params,
                                   ge::Graph &graph) {
  return GRAPH_SUCCESS;
}

graphStatus Graph::LoadFromFile(const char_t *file_name) {
  return GRAPH_SUCCESS;
}

DataFlowInfo::DataFlowInfo() {}

DataFlowInfo::~DataFlowInfo() {}

void DataFlowInfo::SetStartTime(const uint64_t start_time) {}

uint64_t DataFlowInfo::GetStartTime() const {
  return 0UL;
}

void DataFlowInfo::SetEndTime(const uint64_t end_time) {}

uint64_t DataFlowInfo::GetEndTime() const {
  return 0UL;
}

void DataFlowInfo::SetFlowFlags(const uint32_t flow_flags) {}

uint32_t DataFlowInfo::GetFlowFlags() const {
  return 0U;
}

uint64_t DataFlowInfo::GetTransactionId() const {
  return 0UL;
}

void DataFlowInfo::SetTransactionId(uint64_t transaction_id) {}

Status DataFlowInfo::SetUserData(const void *data, size_t size, size_t offset) {
  memset(g_userData, 0, kMaxUserDataSize);
  if (CheckParamsForUserData(data, size, offset) != SUCCESS) {
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  memcpy(g_userData + offset, data, size);
  return SUCCESS;
}

Status DataFlowInfo::GetUserData(void *data, size_t size, size_t offset) const {
  if (CheckParamsForUserData(data, size, offset) != SUCCESS) {
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  memcpy(data, g_userData + offset, size);
  return SUCCESS;
}

std::shared_ptr<Tensor> FlowBufferFactory::AllocTensor(const std::vector<int64_t> &shape, DataType data_type,
                                                       uint32_t align) {
  return std::make_shared<Tensor>();
}

FlowMsgPtr FlowBufferFactory::AllocTensorMsg(const std::vector<int64_t> &shape, DataType data_type, uint32_t align) {
  return std::make_shared<FlowMsgStub>(shape, data_type);
}

FlowMsgPtr FlowBufferFactory::AllocRawDataMsg(size_t size, uint32_t align) {
  return std::make_shared<FlowMsgStub>(size);
}

FlowMsgPtr FlowBufferFactory::AllocEmptyDataMsg(MsgType type) {
  std::vector<int64_t> shape;
  return std::make_shared<FlowMsgStub>(shape, DataType::DT_FLOAT);
}

FlowMsgPtr FlowBufferFactory::ToFlowMsg(const Tensor &tensor) {
  return std::make_shared<FlowMsgStub>(std::make_shared<Tensor>(tensor));
}

FlowMsgPtr FlowBufferFactory::ToFlowMsg(const RawData &raw_data) {
  return std::make_shared<FlowMsgStub>(raw_data);
}

namespace dflow {
ProcessPoint::ProcessPoint(const char_t *pp_name, ProcessPointType pp_type) {}

ProcessPoint::~ProcessPoint() {}

GraphPp::GraphPp(const char_t *pp_name, const GraphBuilder &builder) : ProcessPoint(pp_name, ProcessPointType::GRAPH) {}

GraphPp &GraphPp::SetCompileConfig(const char_t *json_file_path) {
  return *this;
}

GraphPp::~GraphPp() = default;

void GraphPp::Serialize(ge::AscendString &str) const {}

FlowGraphPp::FlowGraphPp(const char_t *pp_name, const FlowGraphBuilder &builder)
    : ProcessPoint(pp_name, ProcessPointType::FLOW_GRAPH) {}

FlowGraphPp &FlowGraphPp::SetCompileConfig(const char_t *json_file_path) {
  return *this;
}

FlowGraphPp::~FlowGraphPp() = default;

void FlowGraphPp::Serialize(ge::AscendString &str) const {}

FunctionPp::FunctionPp(const char_t *pp_name) : ProcessPoint(pp_name, ProcessPointType::FUNCTION) {}

FunctionPp::~FunctionPp() = default;

FunctionPp &FunctionPp::SetCompileConfig(const char_t *json_file_path) {
  return *this;
}

void FunctionPp::Serialize(ge::AscendString &str) const {}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const ge::AscendString &value) {
  printf("=====AscendString====\n");
  return *this;
}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const char_t *value) {
  printf("=====char_t====\n");
  return *this;
}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const std::vector<ge::AscendString> &value) {
  printf("=====vector<ge::AscendString>====\n");
  return *this;
}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const int64_t &value) {
  printf("=====int64_t====\n");
  return *this;
}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const std::vector<int64_t> &value) {
  printf("=====vector<int64_t>====\n");
  return *this;
}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const std::vector<std::vector<int64_t>> &value) {
  printf("=====std::vector<std::vector<int64_t>>====\n");
  return *this;
}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const float &value) {
  printf("=====float====\n");
  return *this;
}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const std::vector<float> &value) {
  printf("=====std::vector<float>====\n");
  return *this;
}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const bool &value) {
  printf("=====bool====\n");
  return *this;
}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const std::vector<bool> &value) {
  printf("=====vector<bool>====\n");
  return *this;
}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const ge::DataType &value) {
  printf("=====DataType====\n");
  return *this;
}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const std::vector<ge::DataType> &value) {
  printf("=====std::vector<ge::DataType>====\n");
  return *this;
}

FunctionPp &FunctionPp::AddInvokedClosure(const char_t *name, const GraphPp &graph_pp) {
  return *this;
}

FunctionPp &FunctionPp::AddInvokedClosure(const char_t *name, const FlowGraphPp &flow_graph_pp) {
  return *this;
}

FlowOperator::FlowOperator(const char *name, const char *type) : ge::Operator(name, type) {}

FlowOperator::~FlowOperator() = default;

FlowData::FlowData(const char *name, int64_t index) : FlowOperator(name, "Data") {}

FlowData::~FlowData() = default;

FlowNode::FlowNode(const char *name, uint32_t input_num, uint32_t output_num) : FlowOperator(name, "FlowNode") {}

FlowNode::~FlowNode() = default;

FlowNode &FlowNode::SetInput(uint32_t dst_index, const FlowOperator &src_op, uint32_t src_index) {
  return *this;
}

FlowNode &FlowNode::MapInput(uint32_t node_input_index, const ProcessPoint &pp, uint32_t pp_input_index,
                             const std::vector<DataFlowInputAttr> &attrs) {
  return *this;
}

FlowNode &FlowNode::MapOutput(uint32_t node_output_index, const ProcessPoint &pp, uint32_t pp_output_index) {
  return *this;
}

FlowNode &FlowNode::AddPp(const ProcessPoint &pp) {
  return *this;
}

FlowNode &FlowNode::SetBalanceScatter() {
  return *this;
}
FlowNode &FlowNode::SetBalanceGather() {
  return *this;
}

FlowGraph::FlowGraph(const char *name) {}

FlowGraph::~FlowGraph() = default;

const ge::Graph &FlowGraph::ToGeGraph() const {
  static ge::Graph graph;
  return graph;
}

void FlowGraph::SetGraphPpBuilderAsync(bool graphpp_builder_async) {}

FlowGraph &FlowGraph::SetInputs(const std::vector<FlowOperator> &inputs) {
  return *this;
}

FlowGraph &FlowGraph::SetOutputs(const std::vector<FlowOperator> &outputs) {
  return *this;
}

FlowGraph &FlowGraph::SetOutputs(const std::vector<std::pair<FlowOperator, std::vector<size_t>>> &output_indexes) {
  return *this;
}

FlowGraph &FlowGraph::SetContainsNMappingNode(bool contains_n_mapping_node) {
  return *this;
}

FlowGraph &FlowGraph::SetInputsAlignAttrs(uint32_t align_max_cache_num, int32_t align_timeout,
                                          bool dropout_when_not_align) {
  return *this;
}

const char *FlowGraph::GetName() const {
  return "flow_graph";
}
FlowGraph &FlowGraph::SetExceptionCatch(bool enable_exception_catch) {
  return *this;
}
}  // namespace dflow
}  // namespace ge