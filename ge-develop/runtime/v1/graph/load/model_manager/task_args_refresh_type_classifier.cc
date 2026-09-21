/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_args_refresh_type_classifier.h"
#include "common/checker.h"
#include "graph/compute_graph.h"
#include "utils/extern_math_util.h"

namespace ge {
namespace {
class LogHelper {
 public:
  void AddRefresh(uint64_t refresh_type, TaskArgsRefreshTypeClassifier::IndexType it, size_t index) {
    if (static_cast<bool>(refresh_type & TaskArgsRefreshTypeClassifier::kRefreshByFm)) {
      fm_ << GetIndexTypeStr(it) << index << ',';
    }
    if (static_cast<bool>(refresh_type & TaskArgsRefreshTypeClassifier::kRefreshByModelIo)) {
      mio_ << GetIndexTypeStr(it) << index << ',';
    }
  }
  void AddFixedAddr(TaskArgsRefreshTypeClassifier::IndexType it, size_t index) {
    fixed_addr_ << GetIndexTypeStr(it) << index << ',';
  }

  std::string GetLog() const {
    std::stringstream ss;

    const auto fm = fm_.str();
    if (!fm.empty()) {
      ss << "refresh by fm: " << fm << ", ";
    }
    const auto mio = mio_.str();
    if (!mio.empty()) {
      ss << "refresh by model io: " << mio;
    }
    return ss.str();
  }

  static char_t GetIndexTypeStr(TaskArgsRefreshTypeClassifier::IndexType it) {
    // 内部函数，it不可能越界
    static constexpr std::array<char_t, static_cast<size_t>(TaskArgsRefreshTypeClassifier::kEnd) + 1U> strs{'i', 'o',
                                                                                                              'w', 'u'};
    return strs[static_cast<size_t>(it)];
  }

 private:
  std::stringstream fm_;
  std::stringstream mio_;
  std::stringstream fixed_addr_;
};

// 相关的临时变量太多了，没办法新增个类来处理
class OneTaskRefreshTypeClassifier {
 public:
  OneTaskRefreshTypeClassifier(const TaskArgsRefreshTypeClassifier &classifer, size_t task_index, bool need_log,
                               uint64_t &task_refresh_type,
                               std::vector<TaskArgsRefreshTypeClassifier::TaskFixedAddr> &fixed_addrs,
                               bool is_physical_memory_refreshable = false)
      : classifer_(classifer),
        task_index_(task_index),
        need_log_(need_log),
        task_refresh_type_(task_refresh_type),
        fixed_addrs_(fixed_addrs),
        is_physical_memory_refreshable_(is_physical_memory_refreshable) {}
  void ClassifyAddrs(const std::vector<AddrDesc> &addrs, TaskArgsRefreshTypeClassifier::IndexType it,
                     std::vector<uint64_t> &refresh_types) {
    refresh_types.resize(addrs.size(), 0UL);
    for (size_t i = 0UL; i < addrs.size(); ++i) {
      auto &addr_desc = addrs[i];

      /*
       * |                     | addr not need refresh(0) | addr need refresh(1/2) |
       * |---------------------|--------------------------|------------------------|
       * | support refresh     | 0                        | 1/2                    |
       * | not support refresh | 0                        | 0, fixed memory        |
       */
      auto addr_refresh_type = classifer_.GetRefreshTypeByLogicalAddr(addr_desc);
      if (((addr_refresh_type != 0UL) || is_physical_memory_refreshable_) && (!addr_desc.support_refresh)) {
        addr_refresh_type = 0UL;
        fixed_addrs_.push_back({task_index_, i, it});
        if (need_log_) {
          log_helper_.AddFixedAddr(it, i);
        }
      }
      refresh_types[i] = addr_refresh_type;
      task_refresh_type_ |= addr_refresh_type;

      if (need_log_) {
        log_helper_.AddRefresh(addr_refresh_type, it, i);
      }
    }
  }

  std::string GetLog() const {
    return log_helper_.GetLog();
  }

 private:
  const TaskArgsRefreshTypeClassifier &classifer_;
  size_t task_index_;
  bool need_log_;
  LogHelper log_helper_;

  // task级别的出参，构造时传入
  uint64_t &task_refresh_type_;
  std::vector<TaskArgsRefreshTypeClassifier::TaskFixedAddr> &fixed_addrs_;
  bool is_physical_memory_refreshable_;
};
/**
 * 根据识别出的fixed addrs在图上做扩散推导，识别出被其影响到的所有anchor并做标注，
 * 同时刷新对应task的refresh type
 */
class FixedAddrInferrer {
 public:
  explicit FixedAddrInferrer(const TaskNodeMap &task_node_map) : task_node_map_(task_node_map) {}
  /**
   *
   * @param [In/Out] fixed_addrs 推导完成后，会将新发现的fixed addrs append到fixed_addrs中
   * @return
   */
  Status Infer(const std::vector<TaskArgsRefreshTypeClassifier::TaskFixedAddr> &src_fixed_addrs,
               TaskArgsRefreshTypeClassifier::FixedAddrs &fixed_addrs,
               std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> &task_indexes_to_refresh_type) const {
    fixed_addrs.resize(src_fixed_addrs.size());
    for (size_t i = 0UL; i < src_fixed_addrs.size(); ++i) {
      const auto &fixed_addr = src_fixed_addrs[i];
      fixed_addrs[i].push_back(fixed_addr);

      auto &node_info = task_node_map_.FindNodeByTaskIndex(fixed_addr.task_index);
      GE_ASSERT_TRUE(node_info.node_id != -1, "Failed to infer fixed addr for task index %zu, invalid op index",
                     fixed_addr.task_index);

      // todo 当前仅支持基于图的查找方式，下一步支持基于编译时的符号属性来查找
      std::vector<FixedAddrPeerNodeDesc> peers;
      GE_ASSERT_SUCCESS(FindPeersByGraph(node_info.node, fixed_addr, peers));

      for (const auto &peer : peers) {
        const auto task_indexes = task_node_map_.FindTasksByNodeId(peer.node_id);
        if (task_indexes.empty()) {
          GELOGE(FAILED,
                 "The fixed addr from task %zu %s %d not satisfied with peer node %s(%s) id %lld, because the peer "
                 "node does have a task",
                 fixed_addr.task_index, TaskArgsRefreshTypeClassifier::GetIndexTypeStr(fixed_addr.iow_index_type),
                 fixed_addr.iow_index, peer.node->GetName().c_str(), peer.node->GetType().c_str(),
                 peer.node->GetOpDesc()->GetId());
          return FAILED;
        }
        for (const size_t task_index : task_indexes) {
          // todo task上的输入输出是否与node一一对应？有可能不是，例如：
          //     1. atomic集中清零，atomic算子的task虽然关联到了atomic算子，但是其地址实际在清零某个算子的输出
          //     2. atomic单独清零，atomic算子也很难遵循node的原型定义自己的输入输出
          const auto new_fa =
              TaskArgsRefreshTypeClassifier::TaskFixedAddr{task_index, peer.iow_index, peer.iow_index_type};
          GE_ASSERT_SUCCESS(UpdateRefreshType(new_fa, task_indexes_to_refresh_type[new_fa.task_index]));
          fixed_addrs[i].push_back(new_fa);
          GELOGD("Task index %zu %s %lld is inferred as fixed addr, the same with task %zu %s %lld", task_index,
                 TaskArgsRefreshTypeClassifier::GetIndexTypeStr(peer.iow_index_type), peer.iow_index,
                 fixed_addr.task_index, TaskArgsRefreshTypeClassifier::GetIndexTypeStr(fixed_addr.iow_index_type),
                 fixed_addr.iow_index);
        }
      }
    }

    return SUCCESS;
  }

 private:
  struct FixedAddrPeerNodeDesc {
    int64_t node_id;
    Node *node;
    size_t iow_index;
    TaskArgsRefreshTypeClassifier::IndexType iow_index_type;
  };

 private:
  static Status UpdateRefreshType(const TaskArgsRefreshTypeClassifier::TaskFixedAddr &new_fa,
                                  TaskArgsRefreshTypeClassifier::TaskRefreshType &task_refresh_type) {
    switch (new_fa.iow_index_type) {
      case TaskArgsRefreshTypeClassifier::kInput:
        task_refresh_type.input_refresh_types[new_fa.iow_index] = 0UL;
        break;
      case TaskArgsRefreshTypeClassifier::kOutput:
        task_refresh_type.output_refresh_types[new_fa.iow_index] = 0UL;
        break;
      default:
        // new_fa.iow_index_type不可能出现ws，因为new_fa是由对端的输入/输出推导出来的，对应到本端必然也是输入/输出
        return FAILED;
    }

    // update task refresh flag
    uint64_t task_refresh_flag = 0UL;
    for (const uint64_t rf : task_refresh_type.input_refresh_types) {
      task_refresh_flag |= rf;
    }
    for (const uint64_t rf : task_refresh_type.output_refresh_types) {
      task_refresh_flag |= rf;
    }
    for (const uint64_t rf : task_refresh_type.workspace_refresh_types) {
      task_refresh_flag |= rf;
    }
    task_refresh_type.task_refresh_type = task_refresh_flag;
    return SUCCESS;
  }

  static Status FindPeersByGraph(const NodePtr &node, const TaskArgsRefreshTypeClassifier::TaskFixedAddr &addr_desc,
                                 std::vector<FixedAddrPeerNodeDesc> &peers) {
    GELOGI("find peers by graph for node %s iow_index_type %d(%s) iow_index %zu",
           node->GetName().c_str(), static_cast<int32_t>(addr_desc.iow_index_type),
           TaskArgsRefreshTypeClassifier::GetIndexTypeStr(addr_desc.iow_index_type), addr_desc.iow_index);

    GE_ASSERT_TRUE(IntegerChecker<int32_t>::Compat(addr_desc.iow_index));
    switch (addr_desc.iow_index_type) {
      case TaskArgsRefreshTypeClassifier::kInput: {
        const auto anchor = node->GetInDataAnchor(static_cast<int32_t>(addr_desc.iow_index));
        GE_ASSERT_NOTNULL(anchor);
        const auto peer_anchor = anchor->GetPeerOutAnchor();
        GE_ASSERT_NOTNULL(peer_anchor);
        const auto peer_node = peer_anchor->GetOwnerNodeBarePtr();
        GE_ASSERT_NOTNULL(peer_node);
        GE_ASSERT_TRUE((TaskArgsRefreshTypeClassifier::IsIdentityLikeType(peer_node->GetType())),
                       "Failed to find peers by graph for node %s input index %zu, only support peer node type "
                       "%s, but get %s(%s)",
                       node->GetName().c_str(), addr_desc.iow_index, IDENTITY, peer_node->GetType().c_str(),
                       peer_node->GetName().c_str());
        peers.emplace_back(FixedAddrPeerNodeDesc{peer_node->GetOpDesc()->GetId(), peer_node,
                                                 static_cast<size_t>(peer_anchor->GetIdx()),
                                                 TaskArgsRefreshTypeClassifier::kOutput});
        break;
      }
      case TaskArgsRefreshTypeClassifier::kOutput: {
        const auto anchor = node->GetOutDataAnchor(static_cast<int32_t>(addr_desc.iow_index));
        GE_ASSERT_NOTNULL(anchor);
        const auto peer_anchors = anchor->GetPeerInDataAnchorsPtr();
        // 在历史版本中，编译时会识别fixed地址，并在fixed地址anchor对端插入一个Identity算子，
        // 此机制可以屏蔽图上多变的符号与anchor的对应关系，使得加载时一个fixed地址仅影响到对端的唯一算子(Identity)的唯一anchor，
        // 在一定中程度上降低了加载时代码复杂度。在这种场景下，通过拓扑的识别方式，也不需要重建符号表，
        // 因此，在基于拓扑的识别方式中，需要做强校验（要求对端只有一个算子，且必须是Identity）。
        // 在新版本中，编译时会将同一符号的anchor作为属性记录在op上，通过属性的查找方式，不再受此限制
        for (const auto &peer_anchor : peer_anchors) {
          const auto peer_node = peer_anchor->GetOwnerNodeBarePtr();
          GE_ASSERT_NOTNULL(peer_node);
          GE_ASSERT_TRUE((TaskArgsRefreshTypeClassifier::IsIdentityLikeType(peer_node->GetType())),
                         "Failed to find peers by graph for node %s output index %zu, only support peer node type "
                         "%s, but get %s(%s)",
                         node->GetName().c_str(), addr_desc.iow_index, IDENTITY, peer_node->GetType().c_str(),
                         peer_node->GetName().c_str());
          peers.emplace_back(FixedAddrPeerNodeDesc{peer_node->GetOpDesc()->GetId(), peer_node,
                                                   static_cast<size_t>(peer_anchor->GetIdx()),
                                                   TaskArgsRefreshTypeClassifier::kInput});
        }
        break;
      }
      case TaskArgsRefreshTypeClassifier::kWorkspace:
        // workspace不存在对端的说法，不需要考虑查找对端的事情
        break;
      default:
        GELOGE(INTERNAL_ERROR, "Unexpected iow type %d when find peers for fixed addr",
               static_cast<int32_t>(addr_desc.iow_index_type));
        return FAILED;
    }
    return SUCCESS;
  }

 private:
  const TaskNodeMap &task_node_map_;
};
}  // namespace
TaskArgsRefreshTypeClassifier::TaskArgsRefreshTypeClassifier(
    const TaskNodeMap &task_node_map,
    const map<std::pair<uint64_t, uint64_t>, MemoryAppType> &logical_addrs_to_memory_app_type,
    bool is_fm_refresh_enable)
    : task_ndoe_map_(task_node_map),
      logical_addrs_to_memory_app_type_(logical_addrs_to_memory_app_type),
      is_fm_refresh_enable_(is_fm_refresh_enable) {}

uint64_t TaskArgsRefreshTypeClassifier::GetRefreshTypeByLogicalAddr(const AddrDesc &addr_desc) const {
  auto &memory_app_type =
      logical_addrs_to_memory_app_type_.at(std::pair<uint64_t, uint64_t>{addr_desc.memory_type, addr_desc.logic_addr});
  switch (memory_app_type) {
    case MemoryAppType::kMemoryTypeModelIo:
      /*
       * model io地址只要独立存在，就总是需要被刷新的
       * 一种场景是编译时设置了alloc_byge的开关，此时模型的输入输出地址被复用到fm中，
       * 这种场景下，模型的输入输出地址是不会被刷新的。
       * 然而这种场景下，model id的逻辑地址会被一起归类到feature map地址中，
       * 在allocation表中找不到输入输出的memory app type
       */
      return kRefreshByModelIo;
    case MemoryAppType::kMemoryTypeFeatureMap:
    // todo: fmtype 是否需要刷新，不单单取决于is_fm_refresh_enable_， 还取决于placement（TS或hbm），
    // 未来logic_addrs_to_memory_app_type_不仅要带app type 还要带placement，这样才能准确判断是否需要刷新
    // 当前的处理也不会导致bug，即使因为需要刷新而被判断为fixed地址也仅仅多浪费一段fixed内存
      return is_fm_refresh_enable_ ? kRefreshByFm : 0UL;
    case MemoryAppType::kMemoryTypeFix:
      return 0UL;
    default:
      GELOGW("Unknown memory app type %s by memory type %llu addr %llu, no need refresh",
             GetMemoryAppTypeStr(memory_app_type), addr_desc.memory_type, addr_desc.logic_addr);
      return 0UL;
  }
}
Status TaskArgsRefreshTypeClassifier::ClassifyMultiTasks(const vector<TaskRunParam> &params,
                                                         std::vector<TaskRefreshType> &task_indexes_to_refresh_type,
                                                         FixedAddrs &fixed_addrs,
                                                         bool is_physical_memory_refreshable) const {
  const bool need_log = IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG);
  task_indexes_to_refresh_type.resize(params.size(), {0, {}, {}, {}});
  std::vector<TaskFixedAddr> src_fixed_addrs;
  for (size_t i = 0U; i < params.size(); ++i) {
    OneTaskRefreshTypeClassifier classifier(*this, i, need_log, task_indexes_to_refresh_type[i].task_refresh_type,
                                            src_fixed_addrs, is_physical_memory_refreshable);
    classifier.ClassifyAddrs(params[i].parsed_input_addrs, kInput, task_indexes_to_refresh_type[i].input_refresh_types);
    classifier.ClassifyAddrs(params[i].parsed_output_addrs, kOutput,
                             task_indexes_to_refresh_type[i].output_refresh_types);
    classifier.ClassifyAddrs(params[i].parsed_workspace_addrs, kWorkspace,
                             task_indexes_to_refresh_type[i].workspace_refresh_types);

    if (need_log) {
      GELOGD("The task index %zu, refresh type %llu. %s", i, task_indexes_to_refresh_type[i].task_refresh_type,
             classifier.GetLog().c_str());
    }
  }
  // todo log
  return FixedAddrInferrer(task_ndoe_map_).Infer(src_fixed_addrs, fixed_addrs, task_indexes_to_refresh_type);
}
}  // namespace ge
