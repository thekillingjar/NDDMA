/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_COMMON_GRAPH_TOP_CHECKER_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_COMMON_GRAPH_TOP_CHECKER_H_
#include <utility>
#include "exe_graph/lowering/value_holder.h"
#include "pretty_table.h"

namespace gert {
template <typename T>
class OrderedSet {
 public:
  typename std::vector<T>::iterator begin() {
    return v_.begin();
  }
  typename std::vector<T>::iterator end() {
    return v_.end();
  }
  [[nodiscard]] typename std::vector<T>::const_iterator begin() const {
    return v_.begin();
  }
  [[nodiscard]] typename std::vector<T>::const_iterator end() const {
    return v_.end();
  }
  bool insert(const T &obj) {
    if (s_.insert(obj).second) {
      v_.push_back(obj);
      return true;
    }
    return false;
  }

  [[nodiscard]] size_t count(const T &obj) const {
    return s_.count(obj);
  }

 private:
  std::set<T> s_;
  std::vector<T> v_;
};

#define MUST_OK() \
  if (!IsOk()) return *this;

template <typename T>
class BaseIoChecker {
 public:
  bool IsOk() const {
    return result_.empty();
  }

  std::string Result() const {
    if (IsOk()) {
      return "success";
    }
    return result_;
  }

 protected:
  T CheckByType(const std::string &node_type, const ge::Node::Vistor<ge::NodePtr> &nodes) {
    std::stringstream error_message;
    error_message << "Node " << node_->GetName() << '(' << node_->GetType() << "), all expect node types: ";
    for (const auto &in_data_node : nodes) {
      error_message << in_data_node->GetType() << ", ";
      if (in_data_node->GetType() == node_type) {
        return T::Success(in_data_node.get());
      }
    }
    error_message << "do not contains node type: " << node_type;
    return T::Failed(error_message.str());
  }
  T CheckByName(const std::string &node_name, const ge::Node::Vistor<ge::NodePtr> &nodes) {
    std::stringstream error_message;
    error_message << "Node " << node_->GetName() << '(' << node_->GetType() << "), all expect node name: ";
    for (const auto &in_data_node : nodes) {
      error_message << in_data_node->GetName() << ", ";
      if (in_data_node->GetName() == node_name) {
        return T::Success(in_data_node.get());
      }
    }
    error_message << "do not contains node name: " << node_name;
    return T::Failed(error_message.str());
  }

  const ge::Node *node_{nullptr};
  std::string result_;
};

class InChecker : public BaseIoChecker<InChecker> {
 public:
  static InChecker Success(const ge::Node *node) {
    InChecker checker;
    checker.node_ = node;
    checker.result_ = "";
    return checker;
  }
  static InChecker Failed(std::string message) {
    InChecker checker;
    checker.node_ = nullptr;
    checker.result_ = std::move(message);
    return checker;
  }

  InChecker DataFromByType(const std::string &node_type) {
    MUST_OK()
    return CheckByType(node_type, node_->GetInDataNodes());
  }
  InChecker CtrlFromByType(const std::string &node_type) {
    MUST_OK()
    return CheckByType(node_type, node_->GetInControlNodes());
  }
};

class OutChecker : public BaseIoChecker<OutChecker> {
 public:
  static OutChecker Success(const ge::Node *node) {
    OutChecker checker;
    checker.node_ = node;
    checker.result_ = "";
    return checker;
  }
  static OutChecker Failed(std::string message) {
    OutChecker checker;
    checker.node_ = nullptr;
    checker.result_ = std::move(message);
    return checker;
  }

  OutChecker DataToByType(const std::string &node_type) {
    MUST_OK()
    return CheckByType(node_type, node_->GetOutDataNodes());
  }
  OutChecker CtrlToByType(const std::string &node_type) {
    MUST_OK()
    return CheckByType(node_type, node_->GetOutControlNodes());
  }
};

class SrcNode {
 public:
  SrcNode(const ge::NodePtr &node)
      : check_element_(node->GetName()), index_(-1), check_name_(true), check_index_(false) {}
  SrcNode(const ge::NodePtr &node, int32_t index)
      : check_element_(node->GetName()), index_(index), check_name_(true), check_index_(true) {}
  SrcNode(std::string node_type)
      : check_element_(std::move(node_type)), index_(-1), check_name_(false), check_index_(false) {}
  SrcNode(std::string node_type, int32_t index)
      : check_element_(std::move(node_type)), index_(index), check_name_(false), check_index_(true) {}

  static SrcNode Name(std::string node_name, int32_t index) {
    SrcNode node(std::move(node_name), index);
    node.check_name_ = true;
    return node;
  }

  bool Match(const ge::NodePtr &node, int32_t index, std::stringstream &ss) const {
    if (check_name_) {
      if (node->GetName() != check_element_) {
        ss << "Expect node name " << check_element_ << ", get node " << node->GetName() << '(' << node->GetType()
           << ')';
        return false;
      }
    } else {
      if (node->GetType() != check_element_) {
        ss << "Expect node type " << check_element_ << ", get node " << node->GetName() << '(' << node->GetType()
           << ')';
        return false;
      }
    }

    if (check_index_) {
      if (index_ != index) {
        ss << "Node " << check_element_ << ", expect index " << index_ << ", get index " << index;
        return false;
      }
    }

    return true;
  }

  const std::string &GetCheckElement() const {
    return check_element_;
  }

  int32_t GetIndex() const {
    return index_;
  }

  bool IsCheckName() const {
    return check_name_;
  }

  bool IsCheckIndex() const {
    return check_index_;
  }

 private:
  std::string check_element_;
  int32_t index_;
  bool check_name_;
  bool check_index_;
};

class NodeTopoChecker {
 public:
  explicit NodeTopoChecker(const ge::NodePtr &node) {
    node_ = node.get();
  }

  /**
   * 严格模式校验输入节点，所谓严格模式，要求输入节点的数量、顺序均与`in_nodes`也必须一致
   * @param in_nodes
   * @return
   */
  [[nodiscard]] std::string StrictConnectFrom(std::vector<SrcNode> in_nodes) const {
    // 1. 先比较个数是否相等
    std::stringstream error_msg;
    if (in_nodes.size() != node_->GetInNodes().size()) {
      error_msg << "In nodes num not match " << std::endl;
      return PrintInNodesComparedTable(error_msg, in_nodes);
    }

    // 2. 按照顺序依次比较数据输入
    size_t i = 0;
    auto src_data_nodes_and_anchors = node_->GetInDataNodesAndAnchors();
    for (const auto &node_and_anchor : node_->GetInDataNodesAndAnchors()) {
      if (!in_nodes[i++].Match(node_and_anchor.first, node_and_anchor.second->GetIdx(), error_msg)) {
        error_msg << std::endl;
        return PrintInNodesComparedTable(error_msg, in_nodes);
      }
    }

    // 3. 数据输入比完后，开始比控制输入。因为控制输入是没有顺序概念的，因此通过name+type个数两次匹配进行
    std::set<std::string> expect_ctrl_node_names;
    std::map<std::string, size_t> expect_ctrl_node_types_to_num;
    for (auto iter = in_nodes.begin() + static_cast<int64_t>(i); iter != in_nodes.end(); ++iter) {
      if (iter->IsCheckIndex() && iter->GetIndex() >= 0) {
        error_msg << "The in data node " << iter->GetCheckElement() << ", index " << iter->GetIndex() << " not exists"
                  << std::endl;
        return PrintInNodesComparedTable(error_msg, in_nodes);
      }
      if (iter->IsCheckName()) {
        expect_ctrl_node_names.insert(iter->GetCheckElement());
      } else {
        expect_ctrl_node_types_to_num[iter->GetCheckElement()]++;
      }
    }

    std::map<std::string, std::string> ctrl_names_to_type;
    std::map<std::string, size_t> ctrl_types_to_num;
    for (const auto &in_ctrl_node : node_->GetInControlNodes()) {
      ctrl_names_to_type[in_ctrl_node->GetName()] = in_ctrl_node->GetType();
      ctrl_types_to_num[in_ctrl_node->GetType()]++;
    }

    for (const auto &expect_name : expect_ctrl_node_names) {
      auto iter = ctrl_names_to_type.find(expect_name);
      if (iter == ctrl_names_to_type.end()) {
        error_msg << "Expect in ctrl node " << expect_name << " does not exists" << std::endl;
        return PrintInNodesComparedTable(error_msg, in_nodes);
      }
      ctrl_types_to_num[iter->second]--;
    }
    for (const auto &expect_type_and_num : expect_ctrl_node_types_to_num) {
      auto iter = ctrl_types_to_num.find(expect_type_and_num.first);
      if (iter == ctrl_types_to_num.end()) {
        error_msg << "Expect in ctrl node type " << expect_type_and_num.first << " does not found" << std::endl;
        return PrintInNodesComparedTable(error_msg, in_nodes);
      }
      if (iter->second != expect_type_and_num.second) {
        error_msg << "In ctrl node type, expect num " << expect_type_and_num.second << ", actual num " << iter->second
                  << std::endl;
        return PrintInNodesComparedTable(error_msg, in_nodes);
      }
    }

    return "success";
  }

  [[nodiscard]] std::string StrictConnectTo(int32_t out_index, std::vector<SrcNode> out_nodes) const {
    auto dst_nodes_and_indexes = GetOutNodes(out_index);

    // 1. 先比较个数是否相等
    std::stringstream error_msg;
    if (out_nodes.size() != dst_nodes_and_indexes.size()) {
      error_msg << "Out nodes num not match " << std::endl;
      return PrintOutNodesComparedTable(error_msg, out_index, out_nodes);
    }

    // 2. 按照check优先级排序（优先级从低到高check：name+index->name->type+index->type）
    std::sort(out_nodes.rbegin(), out_nodes.rend(), [](const SrcNode &lhs, const SrcNode &rhs) {
      return std::make_pair<bool, bool>(lhs.IsCheckName(), lhs.IsCheckIndex()) <
             std::make_pair<bool, bool>(rhs.IsCheckName(), rhs.IsCheckIndex());
    });

    for (const auto &expect_out : out_nodes) {
      auto iter =
          std::find_if(dst_nodes_and_indexes.begin(), dst_nodes_and_indexes.end(),
                       [&expect_out, &error_msg](const std::pair<ge::NodePtr, int32_t> &dst_node_and_index) -> bool {
                         return expect_out.Match(dst_node_and_index.first, dst_node_and_index.second, error_msg);
                       });
      if (iter == dst_nodes_and_indexes.end()) {
        return PrintOutNodesComparedTable(error_msg, out_index, out_nodes);
      }
      dst_nodes_and_indexes.erase(iter);
    }

    return "success";
  }

  template <class T>
  bool ConnectFrom(const std::vector<T> &nodes) {
    return CheckConnect(GetInNodes<T>(), nodes);
  }

  bool ConnectFromByType(const std::vector<std::string> &node_types) {
    return CheckConnect(GetInNodesTypes(), node_types);
  }

  gert::InChecker InChecker() {
    return InChecker::Success(node_);
  }
  gert::OutChecker OutChecker() {
    return OutChecker::Success(node_);
  }

 private:
  std::string PrintOutNodesComparedTable(std::stringstream &error_msg, uint32_t out_index,
                                         const std::vector<SrcNode> &expect_nodes) const {
    std::vector<std::string> actual_nodes;
    std::vector<std::string> actual_indexes;
    GetFromCheckNodeForOutputs(out_index, actual_nodes, actual_indexes);

    PrettyTable pt;
    pt.SetHeader({"Compare Type", "Src Index", "Actual Node", "Actual Dst Index", "Expect Node", "Expect Dst Index"});
    auto row_num = std::max(actual_nodes.size(), expect_nodes.size());
    for (size_t i = 0; i < row_num; ++i) {
      std::string compare_type = "-";
      std::string expect_node = "-";
      std::string expect_dst_index = "-";
      GetFromExpectNodes(expect_nodes, i, compare_type, expect_node, expect_dst_index);

      std::string actual_node = "-";
      std::string actual_index = "-";
      if (i < actual_nodes.size()) {
        actual_node = actual_nodes[i];
        actual_index = actual_indexes[i];
      }
      pt.AddRow({compare_type, std::to_string(out_index), actual_node, actual_index, expect_node, expect_dst_index});
    }
    pt.Print(error_msg);
    return error_msg.str();
  }
  void GetFromCheckNodeForOutputs(uint32_t out_index, std::vector<std::string> &actual_nodes,
                                  std::vector<std::string> &actual_indexes) const {
    auto dst_nodes_and_indexes = GetOutNodes(out_index);
    for (auto &dst_node_and_index : dst_nodes_and_indexes) {
      actual_nodes.emplace_back(GetNodeDesc(dst_node_and_index.first));
      actual_indexes.emplace_back(std::to_string(dst_node_and_index.second));
    }
  }
  [[nodiscard]] std::vector<std::pair<ge::NodePtr, int32_t>> GetOutNodes(int32_t out_index) const {
    std::vector<std::pair<ge::NodePtr, int32_t>> dst_nodes;

    if (out_index >= 0) {
      auto out_anchor = node_->GetOutDataAnchor(static_cast<int32_t>(out_index));
      if (out_anchor == nullptr) {
        return {};
      }
      for (const auto &dst_anchor : out_anchor->GetPeerInDataAnchors()) {
        dst_nodes.emplace_back(dst_anchor->GetOwnerNode(), dst_anchor->GetIdx());
      }
    } else {
      for (const auto &dst_node : node_->GetOutControlNodes()) {
        dst_nodes.emplace_back(dst_node, -1);
      }
    }
    return dst_nodes;
  }
  std::string PrintInNodesComparedTable(std::stringstream &error_msg, const std::vector<SrcNode> &expect_nodes) const {
    PrettyTable pt;
    pt.SetHeader({"Actual Node", "Actual Index", "Compare Type", "Expect Node", "Expect Index"});
    auto row_num = std::max(node_->GetInNodes().size(), expect_nodes.size());
    for (size_t i = 0; i < row_num; ++i) {
      std::string compare_type = "-";
      std::string expect_node = "-";
      std::string expect_index = "-";
      GetFromExpectNodes(expect_nodes, i, compare_type, expect_node, expect_index);

      std::string actual_node;
      std::string actual_index;
      GetFromCheckNodeForInputs(i, actual_node, actual_index);

      pt.AddRow({actual_node, actual_index, compare_type, expect_node, expect_index});
    }
    pt.Print(error_msg);
    return error_msg.str();
  }

  void GetFromCheckNodeForInputs(size_t index, std::string &src_node_info, std::string &src_index) const {
    auto in_data_nodes_and_anchors = node_->GetInDataNodesAndAnchors();
    if (index < in_data_nodes_and_anchors.size()) {
      auto src_node_and_anchor = in_data_nodes_and_anchors.begin() + static_cast<int64_t>(index);
      auto &src_node = src_node_and_anchor->first;
      src_node_info = GetNodeDesc(src_node);
      src_index = std::to_string(src_node_and_anchor->second->GetIdx());
      return;
    }
    index -= in_data_nodes_and_anchors.size();
    auto src_ctrl_anchors = node_->GetInControlAnchor()->GetPeerOutControlAnchors();
    if (index < src_ctrl_anchors.size()) {
      auto src_anchor = src_ctrl_anchors.begin() + static_cast<int64_t>(index);
      auto src_node = (*src_anchor)->GetOwnerNode();
      src_node_info = GetNodeDesc(src_node);
      src_index = "-1";
      return;
    }
    src_node_info = "-";
    src_index = "-";
  }
  static std::string GetNodeDesc(const ge::NodePtr &node) {
    std::stringstream node_str;
    node_str << node->GetName() << "(" << node->GetType() << ")";
    return node_str.str();
  }
  static void GetFromExpectNodes(const std::vector<SrcNode> &expect_nodes, size_t index, std::string &compare_type,
                                 std::string &expect_node, std::string &expect_index) {
    compare_type = "-";
    expect_node = "-";
    expect_index = "-";
    if (index < expect_nodes.size()) {
      auto &node = expect_nodes[index];
      expect_node = node.GetCheckElement();
      if (node.IsCheckName()) {
        compare_type = "NodeName";
      } else {
        compare_type = "NodeType";
      }
      if (node.IsCheckIndex()) {
        compare_type += " And Index";
        expect_index = std::to_string(node.GetIndex());
      }
    }
  }

  template <typename T>
  bool CheckConnect(const OrderedSet<T> &prev_nodes, const std::vector<T> &expect_nodes) {
    for (const auto &node : expect_nodes) {
      if (prev_nodes.count(node) == 0) {
        return false;
      }
    }
    return true;
  }

  template <class T>
  OrderedSet<T> GetInNodes();

  OrderedSet<std::string> GetInNodesTypes() {
    OrderedSet<std::string> in_nodes;
    for (const auto &in_node : node_->GetInNodes()) {
      in_nodes.insert(in_node->GetType());
    }
    return in_nodes;
  }

 private:
  const ge::Node *node_;
};

template <>
inline OrderedSet<std::string> NodeTopoChecker::GetInNodes<std::string>() {
  OrderedSet<std::string> in_nodes;
  for (const auto &in_node : node_->GetInNodes()) {
    in_nodes.insert(in_node->GetName());
  }
  return in_nodes;
}

template <>
inline OrderedSet<ge::Node *> NodeTopoChecker::GetInNodes<ge::Node *>() {
  OrderedSet<ge::Node *> in_nodes_set;
  for (const auto &in_node : node_->GetInNodes()) {
    in_nodes_set.insert(in_node.get());
  }
  return in_nodes_set;
}

template <>
inline OrderedSet<std::pair<ge::Node *, int32_t>> NodeTopoChecker::GetInNodes<std::pair<ge::Node *, int32_t>>() {
  OrderedSet<std::pair<ge::Node *, int32_t>> in_nodes_set;
  for (const auto &in_node_and_anchor : node_->GetInDataNodesAndAnchors()) {
    in_nodes_set.insert(std::make_pair(in_node_and_anchor.first.get(), in_node_and_anchor.second->GetIdx()));
  }
  return in_nodes_set;
}

// start add for fast node
template <typename T>
class BaseFastIoChecker {
 public:
  bool IsOk() const {
    return result_.empty();
  }

  std::string Result() const {
    if (IsOk()) {
      return "success";
    }
    return result_;
  }

 protected:
  T CheckByType(const std::string &node_type, const std::vector<ge::FastNode *> &nodes) {
    std::stringstream error_message;
    error_message << "Node " << node_->GetName() << '(' << node_->GetType() << "), all expect node types: ";
    for (const auto &in_data_node : nodes) {
      error_message << in_data_node->GetType() << ", ";
      if (in_data_node->GetType() == node_type) {
        return T::Success(in_data_node);
      }
    }
    error_message << "do not contains node type: " << node_type;
    return T::Failed(error_message.str());
  }
  T CheckByName(const std::string &node_name, const std::vector<ge::FastNode *> &nodes) {
    std::stringstream error_message;
    error_message << "Node " << node_->GetName() << '(' << node_->GetType() << "), all expect node name: ";
    for (const auto &in_data_node : nodes) {
      error_message << in_data_node->GetName() << ", ";
      if (in_data_node->GetName() == node_name) {
        return T::Success(in_data_node);
      }
    }
    error_message << "do not contains node name: " << node_name;
    return T::Failed(error_message.str());
  }

  const ge::FastNode *node_{nullptr};
  std::string result_;
};

class FastInChecker : public BaseFastIoChecker<FastInChecker> {
 public:
  static FastInChecker Success(const ge::FastNode *node) {
    FastInChecker checker;
    checker.node_ = node;
    checker.result_ = "";
    return checker;
  }
  static FastInChecker Failed(std::string message) {
    FastInChecker checker;
    checker.node_ = nullptr;
    checker.result_ = std::move(message);
    return checker;
  }

  FastInChecker DataFromByType(const std::string &node_type) {
    MUST_OK()
    return CheckByType(node_type, node_->GetInDataNodes());
  }
  FastInChecker CtrlFromByType(const std::string &node_type) {
    MUST_OK()
    return CheckByType(node_type, node_->GetInControlNodes());
  }

  FastInChecker DataFrom(const bg::ValueHolderPtr &holder) {
    MUST_OK()
    return CheckByName(holder->GetFastNode()->GetName(), node_->GetInDataNodes());
  }
  FastInChecker CtrlFrom(const bg::ValueHolderPtr &holder) {
    MUST_OK()
    return CheckByName(holder->GetFastNode()->GetName(), node_->GetInControlNodes());
  }
};

class FastOutChecker : public BaseFastIoChecker<FastOutChecker> {
 public:
  static FastOutChecker Success(const ge::FastNode *node) {
    FastOutChecker checker;
    checker.node_ = node;
    checker.result_ = "";
    return checker;
  }
  static FastOutChecker Failed(std::string message) {
    FastOutChecker checker;
    checker.node_ = nullptr;
    checker.result_ = std::move(message);
    return checker;
  }

  FastOutChecker DataToByType(const std::string &node_type) {
    MUST_OK()
    return CheckByType(node_type, node_->GetOutDataNodes()); // todo：confirm
  }
  FastOutChecker CtrlToByType(const std::string &node_type) {
    MUST_OK()
    return CheckByType(node_type, node_->GetOutControlNodes());
  }

  FastOutChecker DataTo(const bg::ValueHolderPtr &holder) {
    MUST_OK()
    return CheckByName(holder->GetFastNode()->GetName(), node_->GetOutDataNodes()); // todo：confirm
  }
  FastOutChecker CtrlTo(const bg::ValueHolderPtr &holder) {
    MUST_OK()
    return CheckByName(holder->GetFastNode()->GetName(), node_->GetOutControlNodes());
  }
};

class SrcFastNode {
 public:
  SrcFastNode(const bg::ValueHolderPtr &holder)
      : check_element_(holder->GetFastNode()->GetName()),
        index_(holder->GetOutIndex()),
        check_name_(true),
        check_index_(true) {}
  SrcFastNode(const ge::FastNode *node)
      : check_element_(node->GetName()), index_(-1), check_name_(true), check_index_(false) {}
  SrcFastNode(const ge::FastNode *node, int32_t index)
      : check_element_(node->GetName()), index_(index), check_name_(true), check_index_(true) {}
  SrcFastNode(std::string node_type)
      : check_element_(std::move(node_type)), index_(-1), check_name_(false), check_index_(false) {}
  SrcFastNode(std::string node_type, int32_t index)
      : check_element_(std::move(node_type)), index_(index), check_name_(false), check_index_(true) {}

  static SrcFastNode Name(std::string node_name, int32_t index) {
    SrcFastNode node(std::move(node_name), index);
    node.check_name_ = true;
    return node;
  }

  bool Match(const ge::FastNode *node, int32_t index, std::stringstream &ss) const {
    if (check_name_) {
      if (node->GetName() != check_element_) {
        ss << "Expect node name " << check_element_ << ", get node " << node->GetName() << '(' << node->GetType()
           << ')';
        return false;
      }
    } else {
      if (node->GetType() != check_element_) {
        ss << "Expect node type " << check_element_ << ", get node " << node->GetName() << '(' << node->GetType()
           << ')';
        return false;
      }
    }

    if (check_index_) {
      if (index_ != index) {
        ss << "Node " << check_element_ << ", expect index " << index_ << ", get index " << index;
        return false;
      }
    }

    return true;
  }

  const std::string &GetCheckElement() const {
    return check_element_;
  }

  int32_t GetIndex() const {
    return index_;
  }

  bool IsCheckName() const {
    return check_name_;
  }

  bool IsCheckIndex() const {
    return check_index_;
  }

 private:
  std::string check_element_;
  int32_t index_;
  bool check_name_;
  bool check_index_;
};

class FastNodeTopoChecker {
 public:
  explicit FastNodeTopoChecker(const bg::ValueHolderPtr &holder) {
    node_ = holder->GetFastNode();
  }
  explicit FastNodeTopoChecker(const ge::FastNode *node) {
    node_ = node;
  }

  /**
   * 严格模式校验输入节点，所谓严格模式，要求输入节点的数量、顺序均与`in_nodes`也必须一致
   * @param in_nodes
   * @return
   */
  [[nodiscard]] std::string StrictConnectFrom(std::vector<SrcFastNode> in_nodes) const {
    // 1. 先比较个数是否相等
    std::stringstream error_msg;
    if (in_nodes.size() != node_->GetAllInNodes().size()) {
      error_msg << "In nodes num not match " << std::endl;
      return PrintInNodesComparedTable(error_msg, in_nodes);
    }

    // 2. 按照顺序依次比较数据输入
    size_t i = 0;
    for (const auto &edge : node_->GetAllInDataEdges()) {
      if (!in_nodes[i++].Match(edge->src, edge->src_output, error_msg)) {
        error_msg << std::endl;
        return PrintInNodesComparedTable(error_msg, in_nodes);
      }
    }

    // 3. 数据输入比完后，开始比控制输入。因为控制输入是没有顺序概念的，因此通过name+type个数两次匹配进行
    std::set<std::string> expect_ctrl_node_names;
    std::map<std::string, size_t> expect_ctrl_node_types_to_num;
    for (auto iter = in_nodes.begin() + static_cast<int64_t>(i); iter != in_nodes.end(); ++iter) {
      if (iter->IsCheckIndex() && iter->GetIndex() >= 0) {
        error_msg << "The in data node " << iter->GetCheckElement() << ", index " << iter->GetIndex() << " not exists"
                  << std::endl;
        return PrintInNodesComparedTable(error_msg, in_nodes);
      }
      if (iter->IsCheckName()) {
        expect_ctrl_node_names.insert(iter->GetCheckElement());
      } else {
        expect_ctrl_node_types_to_num[iter->GetCheckElement()]++;
      }
    }

    std::map<std::string, std::string> ctrl_names_to_type;
    std::map<std::string, size_t> ctrl_types_to_num;
    for (const auto &in_ctrl_node : node_->GetInControlNodes()) {
      ctrl_names_to_type[in_ctrl_node->GetName()] = in_ctrl_node->GetType();
      ctrl_types_to_num[in_ctrl_node->GetType()]++;
    }

    for (const auto &expect_name : expect_ctrl_node_names) {
      auto iter = ctrl_names_to_type.find(expect_name);
      if (iter == ctrl_names_to_type.end()) {
        error_msg << "Expect in ctrl node " << expect_name << " does not exists" << std::endl;
        return PrintInNodesComparedTable(error_msg, in_nodes);
      }
      ctrl_types_to_num[iter->second]--;
    }
    for (const auto &expect_type_and_num : expect_ctrl_node_types_to_num) {
      auto iter = ctrl_types_to_num.find(expect_type_and_num.first);
      if (iter == ctrl_types_to_num.end()) {
        error_msg << "Expect in ctrl node type " << expect_type_and_num.first << " does not found" << std::endl;
        return PrintInNodesComparedTable(error_msg, in_nodes);
      }
      if (iter->second != expect_type_and_num.second) {
        error_msg << "In ctrl node type, expect num " << expect_type_and_num.second << ", actual num " << iter->second
                  << std::endl;
        return PrintInNodesComparedTable(error_msg, in_nodes);
      }
    }

    return "success";
  }

  [[nodiscard]] std::string StrictConnectTo(int32_t out_index, std::vector<SrcFastNode> out_nodes) const {
    auto dst_nodes_and_indexes = GetOutNodes(out_index);

    // 1. 先比较个数是否相等
    std::stringstream error_msg;
    if (out_nodes.size() != dst_nodes_and_indexes.size()) {
      error_msg << "Out nodes num not match " << std::endl;
      return PrintOutNodesComparedTable(error_msg, out_index, out_nodes);
    }

    // 2. 按照check优先级排序（优先级从低到高check：name+index->name->type+index->type）
    std::sort(out_nodes.rbegin(), out_nodes.rend(), [](const SrcFastNode &lhs, const SrcFastNode &rhs) {
      return std::make_pair<bool, bool>(lhs.IsCheckName(), lhs.IsCheckIndex()) <
             std::make_pair<bool, bool>(rhs.IsCheckName(), rhs.IsCheckIndex());
    });

    for (const auto &expect_out : out_nodes) {
      auto iter =
          std::find_if(dst_nodes_and_indexes.begin(), dst_nodes_and_indexes.end(),
                       [&expect_out, &error_msg](const std::pair<ge::FastNode *, int32_t> &dst_node_and_index) -> bool {
                         return expect_out.Match(dst_node_and_index.first, dst_node_and_index.second, error_msg);
                       });
      if (iter == dst_nodes_and_indexes.end()) {
        return PrintOutNodesComparedTable(error_msg, out_index, out_nodes);
      }
      dst_nodes_and_indexes.erase(iter);
    }

    return "success";
  }

  template <class T>
  bool ConnectFrom(const std::vector<T> &nodes) {
    return CheckConnect(GetInNodes<T>(), nodes);
  }

  bool ConnectFromByType(const std::vector<std::string> &node_types) {
    return CheckConnect(GetInNodesTypes(), node_types);
  }

  gert::FastInChecker InChecker() {
    return FastInChecker::Success(node_);
  }
  gert::FastOutChecker OutChecker() {
    return FastOutChecker::Success(node_);
  }

 private:
  std::string PrintOutNodesComparedTable(std::stringstream &error_msg, uint32_t out_index,
                                         const std::vector<SrcFastNode> &expect_nodes) const {
    std::vector<std::string> actual_nodes;
    std::vector<std::string> actual_indexes;
    GetFromCheckNodeForOutputs(out_index, actual_nodes, actual_indexes);

    PrettyTable pt;
    pt.SetHeader({"Compare Type", "Src Index", "Actual Node", "Actual Dst Index", "Expect Node", "Expect Dst Index"});
    auto row_num = std::max(actual_nodes.size(), expect_nodes.size());
    for (size_t i = 0; i < row_num; ++i) {
      std::string compare_type = "-";
      std::string expect_node = "-";
      std::string expect_dst_index = "-";
      GetFromExpectNodes(expect_nodes, i, compare_type, expect_node, expect_dst_index);

      std::string actual_node = "-";
      std::string actual_index = "-";
      if (i < actual_nodes.size()) {
        actual_node = actual_nodes[i];
        actual_index = actual_indexes[i];
      }
      pt.AddRow({compare_type, std::to_string(out_index), actual_node, actual_index, expect_node, expect_dst_index});
    }
    pt.Print(error_msg);
    return error_msg.str();
  }
  void GetFromCheckNodeForOutputs(uint32_t out_index, std::vector<std::string> &actual_nodes,
                                  std::vector<std::string> &actual_indexes) const {
    auto dst_nodes_and_indexes = GetOutNodes(out_index);
    for (auto &dst_node_and_index : dst_nodes_and_indexes) {
      actual_nodes.emplace_back(GetNodeDesc(dst_node_and_index.first));
      actual_indexes.emplace_back(std::to_string(dst_node_and_index.second));
    }
  }
  [[nodiscard]] std::vector<std::pair<ge::FastNode *, int32_t>> GetOutNodes(int32_t out_index) const {
    std::vector<std::pair<ge::FastNode *, int32_t>> dst_nodes;

    if (out_index >= 0) {
      for (auto edge : node_->GetOutEdgesByIndex(static_cast<int32_t>(out_index))) {
        if (edge != nullptr) {
          dst_nodes.emplace_back(edge->dst, edge->dst_input);
        }
      }
    } else {
      for (const auto &dst_node : node_->GetOutControlNodes()) {
        dst_nodes.emplace_back(dst_node, -1);
      }
    }
    return dst_nodes;
  }
  std::string PrintInNodesComparedTable(std::stringstream &error_msg, const std::vector<SrcFastNode> &expect_nodes) const {
    PrettyTable pt;
    pt.SetHeader({"Actual Node", "Actual Index", "Compare Type", "Expect Node", "Expect Index"});
    auto row_num = std::max(node_->GetAllInNodes().size(), expect_nodes.size());
    for (size_t i = 0; i < row_num; ++i) {
      std::string compare_type = "-";
      std::string expect_node = "-";
      std::string expect_index = "-";
      GetFromExpectNodes(expect_nodes, i, compare_type, expect_node, expect_index);

      std::string actual_node;
      std::string actual_index;
      GetFromCheckNodeForInputs(i, actual_node, actual_index);

      pt.AddRow({actual_node, actual_index, compare_type, expect_node, expect_index});
    }
    pt.Print(error_msg);
    return error_msg.str();
  }

  void GetFromCheckNodeForInputs(size_t index, std::string &src_node_info, std::string &src_index) const {
    auto in_data_edges = node_->GetAllInDataEdges();
    if (index < in_data_edges.size()) {
      auto src_node = in_data_edges[index]->src;
      src_node_info = GetNodeDesc(src_node);
      src_index = std::to_string(in_data_edges[index]->src_output);
      return;
    }
    index -= in_data_edges.size();
    auto src_ctrl_edges = node_->GetAllInControlEdges();
    if (index < src_ctrl_edges.size()) {
      auto src_node = src_ctrl_edges[index]->src;
      src_node_info = GetNodeDesc(src_node);
      src_index = "-1";
      return;
    }
    src_node_info = "-";
    src_index = "-";
  }
  static std::string GetNodeDesc(const ge::FastNode *node) {
    std::stringstream node_str;
    node_str << node->GetName() << "(" << node->GetType() << ")";
    return node_str.str();
  }
  static void GetFromExpectNodes(const std::vector<SrcFastNode> &expect_nodes, size_t index, std::string &compare_type,
                                 std::string &expect_node, std::string &expect_index) {
    compare_type = "-";
    expect_node = "-";
    expect_index = "-";
    if (index < expect_nodes.size()) {
      auto &node = expect_nodes[index];
      expect_node = node.GetCheckElement();
      if (node.IsCheckName()) {
        compare_type = "NodeName";
      } else {
        compare_type = "NodeType";
      }
      if (node.IsCheckIndex()) {
        compare_type += " And Index";
        expect_index = std::to_string(node.GetIndex());
      }
    }
  }

  template <typename T>
  bool CheckConnect(const OrderedSet<T> &prev_nodes, const std::vector<T> &expect_nodes) {
    for (const auto &node : expect_nodes) {
      if (prev_nodes.count(node) == 0) {
        return false;
      }
    }
    return true;
  }

  template <class T>
  OrderedSet<T> GetInNodes();

  OrderedSet<std::string> GetInNodesTypes() {
    OrderedSet<std::string> in_nodes;
    for (const auto &in_node : node_->GetAllInNodes()) {
      in_nodes.insert(in_node->GetType());
    }
    return in_nodes;
  }

 private:
  const ge::FastNode *node_;
};

template <>
inline OrderedSet<std::string> FastNodeTopoChecker::GetInNodes<std::string>() {
  OrderedSet<std::string> in_nodes;
  for (const auto &in_node : node_->GetAllInNodes()) {
    in_nodes.insert(in_node->GetName());
  }
  return in_nodes;
}

template <>
inline OrderedSet<ge::FastNode *> FastNodeTopoChecker::GetInNodes<ge::FastNode *>() {
  OrderedSet<ge::FastNode *> in_nodes_set;
  for (const auto &in_node : node_->GetAllInNodes()) {
    in_nodes_set.insert(in_node);
  }
  return in_nodes_set;
}

template <>
inline OrderedSet<std::pair<ge::FastNode *, int32_t>> FastNodeTopoChecker::GetInNodes<std::pair<ge::FastNode *, int32_t>>() {
  OrderedSet<std::pair<ge::FastNode *, int32_t>> in_nodes_set;
  for (const auto edge : node_->GetAllInDataEdges()) { // todo : confirm
    in_nodes_set.insert(std::make_pair(edge->src, edge->src_output));
  }
  return in_nodes_set;
}
template <>
inline bool FastNodeTopoChecker::ConnectFrom<bg::ValueHolderPtr>(const std::vector<bg::ValueHolderPtr> &nodes) {
  std::vector<std::pair<ge::FastNode *, int32_t>> nodes_and_index;
  for (const auto &holder : nodes) {
    nodes_and_index.emplace_back(const_cast<ge::FastNode *>(holder->GetFastNode()), holder->GetOutIndex());
  }
  return CheckConnect(GetInNodes<std::pair<ge::FastNode *, int32_t>>(), nodes_and_index);
}
// end add for fast node
}  // namespace gert
#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_COMMON_GRAPH_TOP_CHECKER_H_
