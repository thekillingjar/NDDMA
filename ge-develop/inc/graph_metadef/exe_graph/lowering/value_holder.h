/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_VALUE_HOLDER_H_
#define AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_VALUE_HOLDER_H_
#include <cstdint>
#include <string>
#include <memory>
#include <atomic>

#include "graph/buffer.h"
#include "graph/any_value.h"
#include "graph/compute_graph.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/fast_node_utils.h"
#include "graph/node.h"
#include "common/hyper_status.h"
#include "graph_frame.h"
#include "exe_graph/runtime/tensor.h"
#include "common/checker.h"
#include "common/util/mem_utils.h"
#include "graph/fast_graph/execute_graph.h"

namespace gert {
namespace bg {
class ValueHolder;
using ValueHolderPtr = std::shared_ptr<ValueHolder>;
class ValueHolder {
 public:
  enum class ValueHolderType {
    kConst,      // 常量，执行时不变
    kFeed,       // 执行时外部指定
    kOutput,     // 由node产生，包含数据输出与控制输出
    kConstData,  // 常量Const，执行时由外部指定，执行时不变
    // Add new type definitions here
    kValueHolderTypeEnd
  };

  class CurrentComputeNodeGuarder {
   public:
    explicit CurrentComputeNodeGuarder(ge::NodePtr old_node) : old_node_(std::move(old_node)) {}
    ~CurrentComputeNodeGuarder() {
      try {
        ValueHolder::SetCurrentComputeNode(old_node_);
      } catch (...) {}
    }

   private:
    ge::NodePtr old_node_;
  };

  ValueHolder(const ValueHolder &other) = delete;
  ValueHolder &operator=(const ValueHolder &other) = delete;
  virtual ~ValueHolder();

  bool IsOk() const noexcept;

  HyperStatus AddInnerDataToKVMap(int32_t index) const noexcept;

  int64_t GetId() const noexcept;
  ValueHolderType GetType() const noexcept;

  ge::FastNode *GetFastNode() const noexcept;
  ge::ExecuteGraph *GetExecuteGraph() const noexcept;

  ValueHolderPtr GetGuarder() const noexcept;
  void SetGuarder(const bg::ValueHolderPtr &guarder) noexcept;

  int32_t GetOutIndex() const noexcept;

  // ref-from other的含义是，本value指向了other（本value没有独立的内存）
  ge::graphStatus RefFrom(const ValueHolderPtr &other);

  // 在other产生后，本holder的生命周期才结束
  void ReleaseAfter(const ValueHolderPtr &other);

  const int32_t &GetPlacement() const;
  void SetPlacement(const int32_t &placement);

  template<typename T, typename... Args>
  std::vector<std::shared_ptr<T>> AppendOutputs(size_t append_count, Args... args) {
    auto start_index = fast_node_->GetDataOutNum();
    auto ret = ge::FastNodeUtils::AppendOutputEdgeInfo(fast_node_, start_index + append_count);
    if (ret != ge::GRAPH_SUCCESS) {
      return {};
    }
    return CreateFromNode<T>(fast_node_, start_index, append_count, args...);
  }
  // src nodes may come from different graph from current node and can add data edges to current node
  // currently only support to pass through parent nodes with only one subgraph
  ge::graphStatus AppendInputs(const std::vector<ValueHolderPtr> &src);

  static ValueHolderPtr CreateError(const ge::char_t *fmt, ...);
  static ValueHolderPtr CreateError(const ge::char_t *fmt, va_list arg);

  static ValueHolderPtr CreateConst(const void *data, size_t size, bool is_string = false);

  static ValueHolderPtr CreateFeed(int64_t index);

  static ValueHolderPtr CreateConstData(int64_t index);

  static ValueHolderPtr CreateSingleDataOutput(const ge::char_t *node_type, const std::vector<ValueHolderPtr> &inputs);

  static std::vector<ValueHolderPtr> CreateDataOutput(const ge::char_t *node_type,
                                                      const std::vector<ValueHolderPtr> &inputs, size_t out_count);

  template<typename T, typename... Args>
  static std::shared_ptr<T> CreateVoid(const ge::char_t *node_type, const std::vector<ValueHolderPtr> &inputs,
                                       Args... args) {
    auto node = CreateNode(node_type, inputs, 0);
    GE_ASSERT_NOTNULL(node);
    return CreateFromNode<T>(node, -1, ValueHolderType::kOutput, args...);
  }

  static ValueHolderPtr CreateVoidGuarder(const ge::char_t *node_type, const ValueHolderPtr &resource,
                                          const std::vector<ValueHolderPtr> &args);

  static HyperStatus AddDependency(const ValueHolderPtr &src, const ValueHolderPtr &dst);

  /**
   * 压栈一个Root GraphFrame，只有栈底的GraphFrame才被称为ROOT GraphFrame，因此调用此借口前，需要保证栈内不存在GraphFrame，否则会失败
   * @return 成功后，返回创建好的GraphFrame指针，失败时返回空指针
   */
  static GraphFrame *PushGraphFrame();
  /**
   * 压栈一个非root的GraphFrame
   * @param belongs 新加入的GraphFrame所归属的ValueHolder，新压栈的GraphFrame会被挂在该ValueHolder所归属的Node上
   * @param graph_name 挂接GraphFrame到Node时，使用的name
   * @return 创建且挂接成功后，返回创建好的GraphFrame指针，失败时返回空指针
   */
  static GraphFrame *PushGraphFrame(const ValueHolderPtr &belongs, const ge::char_t *graph_name);
  /**
   * 压栈一个GraphFrame, 若该graph frame非root frame，需要保证栈顶frame为其父frame
   * @return 成功后，返回该GraphFrame指针，失败时返回空指针
   */
  static GraphFrame *PushGraphFrame(GraphFrame *graph_frame);

  static std::unique_ptr<GraphFrame> PopGraphFrame();
  static std::unique_ptr<GraphFrame> PopGraphFrame(const std::vector<ValueHolderPtr> &outputs,
                                                   const std::vector<ValueHolderPtr> &targets);

  static std::unique_ptr<GraphFrame> PopGraphFrame(const std::vector<ValueHolderPtr> &outputs,
                                                   const std::vector<ValueHolderPtr> &targets,
                                                   const ge::char_t *out_node_type);

  static GraphFrame *GetCurrentFrame();

  static void ClearGraphFrameResource();

  static ge::ExecuteGraph *GetCurrentExecuteGraph();

  static void SetCurrentComputeNode(const ge::NodePtr &node);
  static void AddRelevantInputNode(const ge::NodePtr &node);
  static std::unique_ptr<CurrentComputeNodeGuarder> SetScopedCurrentComputeNode(const ge::NodePtr &node);

  static ge::FastNode *AddNode(const ge::char_t *node_type, size_t input_count, size_t output_count,
                               const GraphFrame &frame);

  template<typename T, typename... Args>
  static std::vector<std::shared_ptr<T>> CreateFromNode(ge::FastNode *node, size_t start_index,
                                                        size_t create_count, Args... args) {
    if (node == nullptr) {
      return {create_count, nullptr};
    }
    std::vector<std::shared_ptr<T>> holders;
    for (size_t i = 0; i < create_count; ++i) {
      holders.emplace_back(
          CreateFromNode<T>(node, static_cast<int32_t>(i + start_index), ValueHolderType::kOutput, args...));
    }

    return holders;
  }

  template<typename T, typename... Args>
  static std::shared_ptr<T> CreateFromNode(ge::FastNode *node, int32_t index, ValueHolderType type, Args... args) {
    auto holder = std::shared_ptr<T>(new (std::nothrow) T(args...));
    GE_ASSERT_NOTNULL(holder);

    holder->type_ = type;
    holder->fast_node_ = node;
    holder->index_ = index;
    holder->op_desc_ = holder->fast_node_->GetOpDescPtr();
    return holder;
  }

  virtual ValueHolderPtr CreateMateFromNode(ge::FastNode *node, int32_t index, ValueHolderType type);

  static std::string GenerateNodeName(const ge::char_t *node_type, const GraphFrame &frame);

  static std::vector<ValueHolderPtr> GetLastExecNodes();

 protected:
  ValueHolder();
  
  static ge::FastNode *CreateNode(const ge::char_t *node_type, const std::vector<ValueHolderPtr> &inputs,
                                  size_t out_count);

  template<typename T, typename... Args>
  static std::vector<std::shared_ptr<T>> CreateFromNodeStart(ge::FastNode *node, size_t out_count,
                                                             Args... args) {
    return CreateFromNode<T>(node, 0U, out_count, args...);
  }

  void SetErrorMsg(const char *fmt, va_list arg);

 private:
  static std::atomic<int64_t> id_generator_;
  int64_t id_;
  ValueHolder::ValueHolderType type_;
  ge::FastNode *fast_node_; // 通过ValueHolder创建的fast_node节点如果后续在图中被删除，此处会是无效指针，不能直接使用
  ge::OpDescPtr op_desc_;
  int32_t index_;
  int32_t placement_;
  std::unique_ptr<char[]> error_msg_;
  ValueHolderPtr guarder_;
  friend class ValueHolderUtils;
};
}  // namespace bg
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_VALUE_HOLDER_H_
