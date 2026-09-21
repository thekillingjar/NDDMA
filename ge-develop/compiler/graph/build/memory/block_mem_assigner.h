/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_MEMORY_BLOCK_MEM_ASSIGNER_H_
#define GE_GRAPH_BUILD_MEMORY_BLOCK_MEM_ASSIGNER_H_

#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <list>

#include "runtime/rt.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "graph/build/memory/mem_assigner.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "common/sgt_slice_type.h"
#include "graph/build/memory/mem_reuse_strategy.h"
#include "block_type_list.h"
#include "continuous_mem.h"

namespace ge {
constexpr size_t kMaxLifeTime = 0xffffffffUL;
constexpr size_t kMinLifeTime = 1U;
const size_t kDefaultLifeTime = 0xffffffffUL - 1;
const uint64_t kSessionScopeMemory = 0x100000000UL;
const int64_t kOutputMemoryGlobalType = 2;
constexpr size_t kMaxLogLen = 512UL;
constexpr uint32_t kMaxDepthNum = 100U;

enum MemoryNoReuseScope { kReuse, kSessionNoReuse, kGraphNoReuse };

using DependStreamLife = std::map<int32_t, std::map<int64_t, std::map<int64_t, size_t>>>;

struct EdgeLife {
  size_t node_id;
  size_t peer_node_id;
};

struct CompareEdgeLife {
  bool operator() (const ge::EdgeLife &left, const ge::EdgeLife &right) const {
    return left.node_id < right.node_id;
  }
};

using DiffStreamEdgeLife = std::map<int64_t, std::map<int64_t, std::set<EdgeLife, CompareEdgeLife>>>;

enum OpMemoryType { kOutput, kWorkspace, kOutputDesc, kInput };

struct ReuseStrategy {
  explicit ReuseStrategy(bool use_range = false, bool ascending_sort = true, bool reuse_first_release = false,
                         bool memory_priority_mode = false)
      : use_range_(use_range),
        ascending_sort_(ascending_sort),
        reuse_first_release_(reuse_first_release),
        memory_priority_mode_(memory_priority_mode) {}
  bool use_range_ = false;
  bool ascending_sort_ = true;
  bool reuse_first_release_ = false;
  bool memory_priority_mode_ = false;
};

struct MemoryStat {
  size_t theory_min_memory_size_ = 0;
  size_t theory_memory_size_ = 0;
  size_t theory_no_reuse_memory_size_ = 0;
  size_t total_memory_size_ = 0;
  size_t stream_count_ = 0;
};

struct MemoryReuseInfo {
  size_t size_ = 0U;
  uint64_t mem_type_ = RT_MEMORY_HBM;
  bool pre_reuse_flag_ = true;
  bool post_reuse_flag_ = true;
  bool no_assign_mem_ = false;
  bool is_fixed_addr_prior_ = false;
  bool diff_stream_prior_ = false;
};

template<class T>
struct TAttr {
  TAttr(ge::AttrHolder *const ptr, const ge::OpDesc *const desc, int32_t index, const std::string &name, const T value)
      : ptr_(ptr), desc_(desc), name_(name), index_(index), value_(value) {}
  ge::AttrHolder *ptr_;
  const ge::OpDesc *desc_;
  const std::string &name_;
  const int32_t index_;
  const T value_;
};

struct NodeTypeIndex {
  NodeTypeIndex(const ge::Node *node, OpMemoryType mem_type, uint32_t index, bool ref_input = false, size_t begin = 0,
                int64_t stream_id = kInvalidStreamId, bool is_subgraph_workspace = false,
                size_t symbol_end_time = kDefaultLifeTime)
      : node_(node),
        mem_type_(mem_type),
        index_(index),
        ref_input_(ref_input),
        is_subgraph_workspace_(is_subgraph_workspace),
        life_time_begin_(begin),
        symbol_max_life_time_end_(symbol_end_time),
        stream_id_(stream_id) {
    if ((node_ != nullptr) && (node_->GetOpDesc() != nullptr)) {
      is_subgraph_out_ = (node_->GetOpDesc()->GetType() == ge::PARTITIONEDCALL) && (mem_type_ == kOutput);
      node_id_ = static_cast<size_t>(node_->GetOpDesc()->GetId());
      life_time_begin_is_min_ = (life_time_begin_ > 0) && (life_time_begin_ < node_id_);
    } else {
      node_id_ = life_time_begin_;
    }
  }

  const std::string GetMemType() const {
    return GetMemType(mem_type_);
  }
  static std::string GetMemType(const OpMemoryType &mem_type) {
    switch (mem_type) {
      case kOutput :
        return "output";
      case kWorkspace :
        return "workspace";
      case kOutputDesc:
        return "output_desc";
      case kInput:
        return "input";
      default :
        return "unknown";
    }
  }

  size_t GetLifeBegin(bool for_sort = false) const {
    if (life_time_begin_is_min_) {
      return life_time_begin_;
    } else {
      if (!for_sort && is_subgraph_out_) {
        return life_time_end_;
      } else {
        return node_id_;
      }
    }
  }

  std::string GetLifeBeginDesc() const {
    if (node_ == nullptr) {
      return std::to_string(life_time_begin_);
    }

    auto life_begin = GetLifeBegin();
    if (life_begin != node_id_) {
      return std::to_string(life_begin) + "--" + std::to_string(node_id_);
    } else {
      return std::to_string(life_begin);
    }
    return "";
  }

  std::vector<size_t> GetLifeEnd() const {
    std::vector<size_t> life_end_list;
    bool single_end = true;
    for (auto pair : diff_stream_life_time_) {
      if (pair.second != life_time_end_) {
        life_end_list.emplace_back(pair.second);
        single_end = false;
      }
    }
    life_end_list.emplace_back(life_time_end_);
    if (single_end && (ref_life_time_end_ != kDefaultLifeTime) && (ref_life_time_end_ != life_time_end_)) {
      life_end_list.emplace_back(ref_life_time_end_);
    }
    return life_end_list;
  }

  std::string GetLifeEndDesc() const {
    std::string end_desc;
    const auto life_end_list = GetLifeEnd();
    for (const auto life_end : life_end_list) {
      if (!end_desc.empty()) {
        end_desc.append("--");
      }
      end_desc.append(std::to_string(life_end));
    }
    return end_desc;
  }

  void SetOutStreamCount(size_t count) {
    if (count > out_stream_count_) {
      out_stream_count_ = count;
    }
  }
  void SetFirstContinuousNode(const bool flag) {
    first_continuous_node_ = flag;
  }
  void SetLastContinuousNode(const bool flag) {
    last_continuous_node_ = flag;
  }
  void SetContinuousNode(const bool flag) {
    continuous_node_ = flag;
  }
  bool GetFirstContinuousNodeFlag() const { return first_continuous_node_; }
  bool GetLastContinuousNodeFlag() const { return last_continuous_node_; }
  bool GetContinuousNodeFlag() const { return continuous_node_; }

  const ge::Node *node_ = nullptr;
  OpMemoryType mem_type_ = kOutput;
  uint32_t index_ = 0;
  bool ref_input_ = false;
  bool is_subgraph_workspace_ = false;
  bool is_subgraph_out_ = false;
  bool life_time_begin_is_min_ = false;
  bool next_is_ref_input_ = false;
  size_t life_time_begin_ = 0;
  size_t life_time_end_ = kDefaultLifeTime;
  size_t symbol_max_life_time_end_ = kDefaultLifeTime;
  int64_t stream_id_ = kInvalidStreamId;
  size_t node_id_ = 0U;
  std::map<int64_t, size_t> diff_stream_life_time_;
  std::map<int64_t, std::pair<size_t, size_t>> out_stream_life_time_;
  size_t ref_life_time_end_ = kDefaultLifeTime;
  size_t out_stream_count_ = 1U;
  size_t no_align_size_ = 0U;
  bool continuous_node_ = false;
  bool first_continuous_node_ = false;
  bool last_continuous_node_ = false;
};

class BlockMemAssigner;
class MemoryBlock {
 public:
  explicit MemoryBlock(const ReuseStrategy &reuse_strategy, size_t block_size, int64_t stream_id = 0,
                       bool reuse_mem = true, uint64_t memory_type = RT_MEMORY_HBM)
      : ref_count_(0),
        stream_id_(stream_id),
        child_block_(false),
        reuse_mem_(reuse_mem),
        same_stream_(true),
        has_sub_graph_in_out_node_(false),
        input_index_(0),
        continuous_block_(false),
        first_continuous_block_(false),
        last_continuous_block_(false),
        is_zero_copy_(false),
        is_reuse_zero_copy_(true),
        memory_type_(memory_type),
        memory_type_logic_base_(0),
        need_same_offset_in_batch_(false),
        max_real_size_(0),
        max_block_size_(0),
        window_size_(1U),
        thread_dim_(1U),
        post_reuse_flag_(true),
        is_fixed_addr_prior_(false),
        diff_stream_prior_(false),
        used_by_diff_streams_(false),
        block_size_(block_size),
        head_offset_(0U),
        tail_offset_(0U),
        child_offset_(0U),
        batch_used_size_(0U),
        reuse_strategy_(reuse_strategy) {
    switch (memory_type_) {
      case RT_MEMORY_HOST:
        memory_type_logic_base_ = kMemoryHostFeatureMapLogicBase;
        break;
      case RT_MEMORY_HOST_SVM:
        memory_type_logic_base_ = kMemoryHostSVMFeatureMapLogicBase;
        break;
      default:
        break;
    }
  }

  MemoryBlock(const MemoryBlock &) = delete;

  MemoryBlock &operator=(const MemoryBlock &) = delete;

  ~MemoryBlock() {
    node_type_index_list_.clear();
    symbol_list_.clear();
  }

  size_t Size() const { return block_size_; }

  void SetSize(size_t size) {
    if (size > block_size_) {
      block_size_ = size;
    }
  }

  size_t AlignSize();

  Status SetHeadOffset(size_t offset);

  void SetTailOffset(size_t offset);

  size_t HeadOffset() const { return head_offset_; }

  size_t TailOffset() const { return tail_offset_; }

  void AddNodeTypeIndex(const NodeTypeIndex &node_type_index, size_t real_size, size_t no_align_size,
                        int64_t stream_id) {
    if (node_type_index.ref_input_) {
      if (!node_type_index_list_.empty()) {
        node_type_index_list_.back().next_is_ref_input_ = true;
      }
    }
    node_type_index_list_.emplace_back(node_type_index);
    node_type_index_list_.back().no_align_size_ = no_align_size;
    real_size_list_.emplace_back(real_size);
    no_align_size_list_.emplace_back(no_align_size);

    if (stream_id != stream_id_) {
      same_stream_ = false;
    }

    // need recompute max real size
    max_real_size_ = 0;
    last_continuous_block_ = last_continuous_block_ || node_type_index.GetLastContinuousNodeFlag();
    first_continuous_block_ = first_continuous_block_ || node_type_index.GetFirstContinuousNodeFlag();
    continuous_block_ = continuous_block_ || node_type_index.GetContinuousNodeFlag();
    block_type_list_.WithAdded(node_type_index);
  }

  bool IsBlockTypeConflict(const MemoryBlock &other) const {
    return block_type_list_.IsConflictWithBlock(other.block_type_list_);
  }

  bool IsBlockTypeConflictWithNode(const NodeTypeIndex &node_type_index) const {
    return block_type_list_.IsConflictWithOneNode(node_type_index);
  }

  std::string BlockTypeStr() const {
    return block_type_list_.ToString();
  }

  void AddSymbol(const std::string &symbol) {
    symbol_list_.emplace_back(symbol);
  }

  void ClearOutStreamLifeInfo() {
    node_type_index_list_.back().out_stream_life_time_.clear();
  }

  void ClearDiffStreamLifeInfo() {
    node_type_index_list_.back().diff_stream_life_time_.clear();
  }

  const std::vector<NodeTypeIndex> &NodeTypeIndexList() const { return node_type_index_list_; }
  const std::vector<std::string> &SymbolList() const { return symbol_list_; }
  const std::vector<size_t> &RealSizeList() const { return real_size_list_; }
  const std::vector<MemoryBlock *> &ChildBlockList() const { return child_blocks_; }
  const std::map<std::string, std::vector<MemoryBlock *>> &BatchBlockList() const { return batch_to_blocks_; }
  const std::vector<size_t> &NoAlignSizeList() const { return no_align_size_list_; }
  const std::vector<MemoryBlock *> &ChildSubGraphBlockList() const { return sub_graph_blocks_; }
  bool IsNoAlignSizeReuseBlock() const { return continuous_block_; }
  bool IsRealSizeReuseBlock() const { return is_zero_copy_; }
  std::vector<MemoryBlock *> AllChildBlockList() const;

  inline void SetRefLifeTimeEnd() {
    for (size_t index = 0U; index < node_type_index_list_.size(); ++index) {
      auto &node_type_index = node_type_index_list_[index];
      if (!node_type_index.next_is_ref_input_ || node_type_index.ref_input_) {
        continue;
      }
      size_t ref_end_index = index;
      for (size_t i = index + 1U; i < node_type_index_list_.size(); ++i) {
        if (!node_type_index_list_[i].ref_input_) {
          break;
        }
        ref_end_index = i;
      }
      node_type_index.ref_life_time_end_ = node_type_index_list_[ref_end_index].life_time_end_;
    }
  }

  void Resize();

  std::string String() const;

  bool IsSameBatchLabel() const;

  // if the block is used by graph input, if true, return input size
  bool IsGraphInputAndGetSize(const ComputeGraphPtr &computeGraph, size_t &size) const;

  void AddContinuousLifeReuseBlock(MemoryBlock &block);

  void AddZeroCopyLifeReuseBlock(MemoryBlock &block);

  bool AddLifeReuseBlock(const BlockMemAssigner *const mem_assigner,
                         MemoryBlock *block, std::vector<MemoryBlock *> &clone_blocks, uint32_t depth,
                         DiffStreamEdgeLife &diff_stream_edge_life, bool child_reuse = false);

  void SetLifeTimeEnd(size_t time, int64_t stream_id);

  void SetOutStreamLifeTime(size_t out_time, size_t end_time, int64_t stream_id);

  size_t GetLifeBegin(bool for_sort = false) const;

  size_t GetLifeEnd(int64_t stream_id) const;

  size_t GetLifeEnd(int64_t stream_id, int64_t &end_stream_id) const;

  size_t GetSymbolLifeEnd() const;

  void SetSymbolLifeEnd(size_t symbol_life_end);

  size_t GetDependLifeBegin(int64_t stream_id, DiffStreamEdgeLife &diff_stream_edge_life) const;

  bool CrossLifeTimeNode(const std::vector<NodeTypeIndex>::const_iterator &it, const MemoryBlock &child_block) const;

  MemoryBlock *Clone() const;

  std::vector<NodeTypeIndex>::const_iterator DelNode(std::vector<NodeTypeIndex>::const_iterator &it);

  void Swap(MemoryBlock &block);

  bool AddChildBlock(MemoryBlock *block) {
    block->child_block_ = true;
    sub_graph_blocks_.emplace_back(block);
    return true;
  }

  bool AddBatchChildBlock(MemoryBlock *block) {
    if ((batch_used_size_ <= Size()) && (block->Size() <= (Size() - batch_used_size_))) {
      block->child_block_ = true;
      batch_used_size_ += block->Size();
      batch_to_blocks_[block->batch_label_].emplace_back(block);
      return true;
    }
    return false;
  }

  void Reset() {
    batch_used_size_ = 0U;
  }

  void SetOutStreamCount(size_t end_stream_count) {
    if (!node_type_index_list_.empty()) {
      node_type_index_list_.back().SetOutStreamCount(end_stream_count);
    }
  }
  void UpdateContinuousFlag();
  void SetFirstContinuousBlock() {
    first_continuous_block_ = true;
    if (!node_type_index_list_.empty()) {
      node_type_index_list_.back().SetFirstContinuousNode(true);
    }
  }
  void SetLastContinuousBlock() {
    last_continuous_block_ = true;
    if (!node_type_index_list_.empty()) {
      node_type_index_list_.back().SetLastContinuousNode(true);
    }
  }
  void SetContinuousBlock() {
    continuous_block_ = true;
    if (!node_type_index_list_.empty()) {
      node_type_index_list_.back().SetContinuousNode(true);
    }
  }
  bool GetFirstContinuousFlag() const { return first_continuous_block_; }
  bool GetLastContinuousFlag() const { return last_continuous_block_; }
  bool GetContinuousFlag() const { return continuous_block_; }

  const ReuseStrategy &GetReuseStrategy() const { return reuse_strategy_; }

  int32_t ref_count_;
  int64_t stream_id_;
  bool child_block_;
  bool reuse_mem_;
  bool same_stream_;
  bool has_sub_graph_in_out_node_;
  uint32_t input_index_;
  bool continuous_block_;
  bool first_continuous_block_;
  bool last_continuous_block_;
  bool is_zero_copy_;
  bool is_reuse_zero_copy_;
  uint64_t memory_type_;
  int64_t memory_type_logic_base_;
  std::string batch_label_;
  bool need_same_offset_in_batch_;
  size_t max_real_size_;
  size_t max_block_size_;
  uint32_t window_size_;
  uint32_t thread_dim_;
  bool post_reuse_flag_;
  bool is_fixed_addr_prior_;
  bool diff_stream_prior_;
  bool used_by_diff_streams_;
 private:
  size_t block_size_;
  std::vector<size_t> real_size_list_;
  std::vector<size_t> no_align_size_list_;
  size_t head_offset_;
  size_t tail_offset_;
  size_t child_offset_;
  size_t batch_used_size_;
  std::vector<NodeTypeIndex> node_type_index_list_;
  std::vector<std::string> symbol_list_;
  std::vector<MemoryBlock *> child_blocks_;
  std::vector<MemoryBlock *> sub_graph_blocks_;
  std::map<std::string, std::vector<MemoryBlock *>> batch_to_blocks_;
  const ReuseStrategy &reuse_strategy_;
  BlockTypeList block_type_list_;
};

using StreamIdToBlocks = std::unordered_map<int64_t, std::vector<MemoryBlock *>>;
using MemoryTypeToSubGraphIdBlocks = std::unordered_map<int64_t, StreamIdToBlocks>;

struct ApplyMemoryParam {
  size_t block_size;           // block_size applied memory block size
  size_t real_size;            // real_size actual memory size required
  size_t no_align_size;
  OpMemoryType mem_type;
  uint32_t out_index;          // out_index output node index
  const bool is_op_reuse_mem;  // is_op_reuse_mem whether the op reuses memory
  const bool continuous;       // whether the memory of op is continuous
  uint64_t memory_type;        // device memory type
  bool is_zero_copy;
};

struct MemAssistInfo {
  ComputeGraphPtr compute_graph;
  AnchorToSymbol anchor_to_symbol;
  SymbolToAnchors symbol_to_anchors;
  std::unordered_map<const Node *, std::vector<int64_t>> parent_nodes_to_stream_ids;
};

class BlockMemAssigner : public MemAssigner {
 public:
  BlockMemAssigner(const MemAssistInfo &mem_assist_info);

  BlockMemAssigner(const BlockMemAssigner &) = delete;

  BlockMemAssigner &operator=(const BlockMemAssigner &) = delete;

  ~BlockMemAssigner() override;

  Status Assign() override;

  const std::map<uint64_t, size_t> &GetMemOffsets() const { return mem_offsets_; }

  const std::map<uint64_t, MemoryStat> &GetMemoryStat() const { return memory_stat_; }

  int64_t GetAtomicAddrCleanId() const { return atomic_addr_clean_id_; }

  std::vector<MemoryBlock *> GetMemoryBlocks() const { return memory_blocks_; }

  void SetReuseStrategy(const ReuseStrategy &reuse_strategy) { reuse_strategy_ = reuse_strategy; }

  bool IsMemoryPriorityMode() const { return memory_priority_mode_; }

  /// @ingroup domi
  /// @brief   memory size fixed for reuse. get memory range
  /// @param [out] ranges return memory range
  /// @return Status result
  virtual Status GetMemoryRanges(std::vector<int64_t> &ranges) = 0;
  /// @ingroup domi
  /// @brief traverse all nodes' outputs and needed workspace mem, apply memory, consider reuse memory
  /// @param [in] ranges memory range provided
  /// @author
  Status AssignMemoryWithReuse(std::vector<int64_t> &ranges);

  std::string GetMaxBatchLabel() const { return max_batch_label_; }

  /// PreAssign and SetOpMemOffset are not thread safe, can only be called from a single thread
  /// Other function must ensure thread safety, there will be multiple thread calls
  static Status PreparationForAssign(MemAssistInfo &mem_assist_info);
  static Status SetRealStreamIdForParentNode(MemAssistInfo &mem_assist_info);

  void SetOpMemOffset(bool is_zero_copy) const;
  void SetOpMemOffset(const std::vector<MemoryBlock *> &zero_copy_blocks) const;
  void SetOffsetForContinuousMem() const;
  bool HasSameOutAnchorWithDiffStream(const Node *n, const uint32_t index) const;
 protected:
  /// @ingroup domi
  /// @brief traverse all memory size, resize, and calculate offset
  /// @param [in&out] memory_blocks memory size, resize and calculate memory address after offset
  Status ResizeMemoryBlocks();

  Status GetOutAndWorkSpaceMem(std::vector<int64_t> &all_memory_size);

  void GetNodeWorkSpaceSize(const ge::NodePtr &node, std::vector<int64_t> &workspace_memory,
                            int64_t &total_size) const;

  /// @ingroup GE
  /// @brief Determine whether it is the type of zero memory node.
  /// @param [in] node type.
  /// @return bool true: is zero memory node; false: is not zero memory node
  /// @author
  bool CheckIsZeroMemNodeType(const std::string &node_type) const;

  /// @ingroup GE
  /// @brief check if input node reuse memory
  /// @param [in] n input node
  /// @return bool
  bool GetInputNodeReuseMemFlag(const NodePtr &n);

  /// @ingroup GE
  /// @brief check if a input of net output node reuse memory
  /// @param [in] index input index of netoutput node
  /// @return bool
  bool GetOutputNodeReuseMemFlagByIndex(const int32_t index);

  /// @ingroup GE
  /// @brief Check pre_reuse flag & post_reuse glag for each symbol
  /// @return void
  void InitReuseFlag();

  /// @ingroup GE
  /// @brief get pre_reuse flag
  /// @param [in] cur_node_index_io
  /// @param [out] symbol
  /// @return bool
  bool IsPreReuse(const NodeIndexIO &cur_node_index_io, std::string &symbol) const;

  /// @ingroup GE
  /// @brief get post_reuse flag
  /// @param [in] symbol
  /// @param [out] diff_stream_prior
  /// @return bool
  bool IsPostReuse(const std::string &symbol, bool &diff_stream_prior) const;

  /// @ingroup GE
  /// @brief get post_reuse flag
  /// @param [in] mem_block
  /// @return bool
  bool IsPostReuse(const ge::MemoryBlock *const mem_block) const;

  /// @ingroup GE
  /// @brief check if symbol of cur node_index_io has block
  /// @param [in] node_index_io
  /// @param [out] symbol
  /// @return bool
  bool IsSymbolExist(const NodeIndexIO &node_index_io, std::string &symbol, MemoryBlock *&block) const;

  /// @ingroup GE
  /// @brief check if symbol of cur node_index_io has output description block
  /// @param [in] node_index_io
  /// @param [out] symbol
  /// @return bool
  bool IsSymbolDescBlockExist(const NodeIndexIO &node_index_io, std::string &symbol, MemoryBlock *&block) const;

  /// @ingroup GE
  /// @brief Print symbol
  /// @return void
  void PrintSymbolMap();

 public:
  /// @ingroup GE
  /// @brief Get the memory type corresponding to the current symbol.
  /// @param [in] node_index_io_list
  /// @param [out] memory_type
  /// @return void
  static void GetSymbolMemType(const std::list<NodeIndexIO> &node_index_io_list, int64_t &memory_type);

  /// @ingroup GE
  /// @brief add the memory type with symbol.
  /// @param [in] symbol
  /// @param [in] memory_type
  /// @return void
  void AddSymbolMemType(const std::string &symbol, int64_t memory_type);

  /// @ingroup GE
  /// @brief Update input tensor or output tensor of op to new memory type attr.
  /// @param [in] node_index_io_list
  /// @param [in] memory_type
  /// @return void
  void UpdateOpTensorMemType(const std::list<NodeIndexIO> &node_index_io_list, int64_t memory_type);

  /// @ingroup GE
  /// @brief Print memory block info
  /// @return void
  void PrintMemBlock();

  /// @ingroup GE
  /// @brief Nano Determine whether it is the type of zero memory output node.
  /// @param [in] node type.
  /// @return bool true: is zero memory node; false: is not zero memory node
  /// @author
  bool CheckIsZeroMemNodeOutputIndex(const NodePtr &n, uint32_t index) const;

  virtual bool NeedLevel2Reuse() { return true; };

  Status GetRealStreamIdForParentNode(const NodePtr &node, const uint32_t out_index, int64_t &stream_id,
                                      bool &is_reuse) const;

  std::map<uint64_t, size_t> mem_offsets_;
  ge::ComputeGraphPtr compute_graph_;
  std::vector<MemoryBlock *> memory_blocks_;
  std::vector<MemoryBlock *> blocks_store_;
  std::vector<NodeTypeIndex> zero_memory_list_;

  // ref mapping
  const SymbolToAnchors &symbol_to_anchors_;
  const AnchorToSymbol &anchor_to_symbol_;
  std::map<std::string, MemoryReuseInfo> symbol_mem_reuse_info_;

 private:
  Status GetOutputTotalSizeAndOutCount(const NodePtr &n, uint32_t output_index, size_t &max_size, size_t &no_align_size,
                                       int32_t &out_count, bool is_separate_clean_continuous_inputs) const;
  /// @ingroup GE
  /// @brief Traversing the compute_graph_ to apply for output memory while considering reuse
  /// @param [in] n: node in compute_graph_
  /// @param [in] index: output node index
  /// @param [in] ranges: available memory specifications
  /// @param [in] is_op_reuse_mem: Whether the op reuses the memory, true: reuse; false: not reuse
  /// @param [in] out_node_need_continuous_input: Whether the downstream node's need continuous input memory
  /// @return MemoryBlock*
  /// @author
  MemoryBlock *ApplyOutMemory(const ge::NodePtr &n, uint32_t index, const std::vector<int64_t> &ranges,
                              const bool is_op_reuse_mem, const bool out_node_need_continuous_input);

  Status AssignOutputMemoryWithReuse(const NodePtr &node, std::vector<int64_t> &ranges);

  Status AssignWorkSpaceMemoryWithReuse(const NodePtr &node, std::vector<int64_t> &ranges);

  /// @ingroup GE
  /// @brief Traversing the compute_graph_ to apply for output description memory
  /// @param [in] n: node in compute_graph_
  /// @param [in] index: output node index
  /// @param [in] ranges: available memory specifications
  /// @return MemoryBlock*
  /// @author
  MemoryBlock *ApplyOutDescMemory(const NodePtr &n, uint32_t index, const std::vector<int64_t> &ranges);

  /// @ingroup GE
  /// @brief Traversing the compute_graph_ to apply for memory while considering reuse
  /// @param [in] n node in compute_graph_
  /// @param [in] workspace_reuse_flag reuse flag for workspace
  /// @param [in] ApplyMemoryParam apply memory param
  /// @return MemoryBlock*
  /// @author
  MemoryBlock *ApplyMemory(const NodePtr &n,  const std::vector<bool> &workspace_reuse_flag,
                           const ApplyMemoryParam &param);
  bool IsNodeOutputUseSameMemWithNetOutput(const ge::NodePtr &node, uint32_t out_index) const;
  /// @ingroup GE
  /// @brief Get the block: release first, reuse first
  /// @param [in] block_size applied memory block size
  /// @param [in] batch_label batch label
  /// @param [in] reusable_blocks all reusable blocks
  /// @param [in] node_type_index node to assign memory
  /// @return MemoryBlock*
  /// @author
  MemoryBlock *GetFirstReleaseBlock(const size_t block_size, const std::string &batch_label,
                                    std::vector<MemoryBlock *> &reusable_blocks,
                                    const NodeTypeIndex &node_type_index) const;

  /// @ingroup GE
  /// @brief Get the block: release first, reuse first
  /// @param [in] block_size applied memory block size
  /// @param [in] batch_label batch label
  /// @param [in] reusable_blocks all reusable blocks
  /// @param [in] node_type_index node to assign memory
  /// @return MemoryBlock*
  /// @return MemoryBlock*
  /// @author
  MemoryBlock *GetLastReleaseBlock(const size_t block_size, const std::string &batch_label,
                                   std::vector<MemoryBlock *> &reusable_blocks,
                                   const NodeTypeIndex &node_type_index) const;
  /// @ingroup GE
  /// @brief check workspace_reuse_flag to judge if add workspace block wait reuse
  /// @param [in] workspace_reuse_flag mark out index if support resue
  /// @param [in] index out index
  /// @param [in] stream_id which stream op in
  /// @param [in] mem_block node workspace mem_block
  /// @param [in] memory_type workspace memory type
  /// @return void
  /// @author
  void CheckWorkspaceReuse(const std::vector<bool> &workspace_reuse_flag, uint32_t index, int64_t stream_id,
                           MemoryBlock *const mem_block, uint64_t memory_type);

  /// @ingroup GE
  /// @brief Release memory block to reusable list
  /// @param [in] to_release memory block to be released
  /// @param [in] reusable_memory reusable list
  /// @return void
  /// @author
  void ReleaseMemory(MemoryBlock *const to_release, std::vector<MemoryBlock *> &reusable_memory, int64_t stream_id,
                     const std::string &symbol, bool no_release = false);

  /// @ingroup GE
  /// @brief Release memory blocks to reusable list
  /// @param [in] to_releases memory blocks to be released
  /// @param [in] reusable_memory reusable list
  /// @return void
  /// @author
  void ReleaseMemorys(StreamIdToBlocks &to_releases,
                      StreamIdToBlocks &reusable_memory);

  /// @ingroup GE
  /// @brief Release memory block to reusable list
  /// @param [in] n node in compute_graph_
  /// @param [in] node_out_blocks output memory blocks for ops
  /// @param [in] reusable_memory reusable list
  /// @return void
  /// @author
  void ReleaseInputNodeOutMemory(const NodePtr &node);

  void AssignContinuousBlocks();

  bool IsNodeAndPeerNodeTaskSupportZeroCopy(const ge::NodePtr &node, uint32_t output_index) const;

  bool IsZeroCopyBlock(const NodePtr &node, uint32_t output_index, bool continuous, size_t output_size = 0) const;
  bool IsAtomicOutputMemory(const ge::NodePtr &node, uint32_t output_index, bool is_atomic,
                            bool out_node_set_continuous_input) const;
  bool IsOutNodeSetContinuousInput(const NodePtr &n, uint32_t out_index,
                                   InDataAnchor *&continuous_in_anchor, bool &is_reuse_zero_copy,
                                   std::set<int64_t> &streams);
  Status GetNoNeedAssignMemoryFlag(const NodePtr &n, uint32_t out_index, bool &no_need_assign_memory) const;

  bool IsContinuousMemoryReuse(const Node *const n, uint32_t out_index, const Node *const continuous_node,
                               std::set<int64_t> &streams);

  Status CalNodeAsContinuousInputMaxLife(const Node *const n, uint32_t out_index, const Node *const continuous_node,
                                         int64_t &first_node_max_life, std::set<int64_t> &streams);

  void CalExitSymbolNodeLifeTime(const Node *const n, uint32_t out_index, size_t &max_life_time);
  /// @ingroup GE
  /// @|+++++++++block1++++++++|                               |+++++++++block1++++++++|
  /// @|+++++++++block1++++++++||++block2++|                   |+++++++++block1++++++++||++block2++|
  /// @                         |++block2++||++block3++|  ==>  |++block3++|             |++block2++|
  /// @                                     |++block3++|       |++block3++|
  /// @return void
  /// @author
  void ReuseBlocksByLifeTime();

  uint64_t GetWorkSpaceMemoryType(const size_t no_reuse_scope_size, const size_t index, const bool is_p2p_memory,
                                  const bool session_scope_memory, std::vector<bool> &workspace_reuse_flag) const;

  void ContinuousOutRefCheck(bool &is_all_output_ref, bool &is_output_has_ref, const NodePtr &n);

  Status ApplyContinuousMemory(const NodePtr &n, const std::vector<int64_t> &ranges, const bool is_op_reuse_mem);
  Status ApplyContinuousMemWithMng(const NodePtr &n, int32_t idx, const std::vector<int64_t> &ranges);
  Status GetContinuousMemType(const ContinuousMem &continuous_mem, uint64_t &memory_type) const;
  void CheckAndReleaseSuspendedBlock(const NodePtr &node, uint32_t idx, MemoryBlock *block);

  int32_t GetAllRefCount(const NodeIndexIO &out_node_index_io, bool &is_reuse_zero_copy) const;
  Status GetContinuousMemLifeTime(const ContinuousMem &continuous_mem, int64_t &begin_time,
                                  int64_t &out_time, int64_t &end_time, size_t &out_streams_cnt) const;
  static void OptimizeStreamIdForMemoryReuse(const NodePtr &node);
  static void SetRealStreamIdForDataNode(const Node *const node);

  void GetDiffStreamEdgeLife(const NodePtr &node, const std::set<int64_t> &exclude_merge_streams);
  void AddInStreamEdge(const ge::OpDesc *const node_desc, const ge::OpDesc *const in_node_desc);
  void InsertStreamOutEdge();
  void InsertStreamInEdge(const EdgeLife &new_in_edge, const int64_t src_stream_id, const int64_t dst_stream_id,
                          const char *src_name = nullptr, const char *dst_name = nullptr);
      /// @ingroup GE
  /// @brief Cascade memory scenarios to obtain the actual life time begin of continuous input memory
  /// @return void
  /// @author
  void GetContinuousNodeLifeTimeBegin(const Node *const org_node, const Node *const node, const int32_t index,
                                      uint32_t depth);

  void SetContinuousNodeLifeTimeBegin(const Node *const org_node, const Node *const node, uint32_t depth);

  void GetRefContinuousInputNodeAndFixedAddrPriorFlag(const std::string &symbol, const std::list<NodeIndexIO> &anchors);

  bool IsNoNeedAssignMemory(const NodePtr &n, const NodeIndexIO &out_node_index_io, const uint32_t index) const;

  void SetOffsetSize(const NodeTypeIndex &node_type, const MemoryBlock &block,
                     size_t real_size, size_t no_align_size, int32_t child_block_level) const;

  void SetBlockOpMemOffset(const MemoryBlock *const block, int32_t child_block_level, bool &is_fixed_addr_prior) const;

  void ParseGraphIoAllocMode();
  void ParseIoReuseMemOption();
  Status InitIoReuseFlag();
  void AddMemoryStat(uint64_t memory_type, size_t real_size, bool is_reuse_memory);
  void MarkReuseZeroCopyBlockFlag(const NodePtr &n, MemoryBlock *const block, const uint32_t index) const;

  void InitDiffStreamSameOutTable();
  // [memory type][sub graph id][stream id]
  MemoryTypeToSubGraphIdBlocks reusable_blocks_;

  MemoryTypeToSubGraphIdBlocks stream_workspace_blocks_;

  std::unordered_map<std::string, MemoryBlock *> symbol_blocks_;

  std::unordered_map<std::string, MemoryBlock *> symbol_desc_blocks_;

  // 用于给带Padding连续输入节点分配连续的内存，记录所有输入的block
  // <int64_t, <input_index, blockPtr>>
  std::unordered_map<int64_t, std::unordered_map<uint32_t, MemoryBlock *>> node_continuous_input_blocks_;

  // 记录带Padding连续输入节点
  std::unordered_map<int64_t, std::pair<std::string, uint32_t>> node_continuous_input_counts_;

  std::unordered_map<std::string, size_t> cascade_min_life_time_;

  // reuse memory
  std::vector<std::string> op_no_reuse_mem_vec_;  // names and types of Op which is not reuse memory

  bool op_reuse_env_valid_ = false;  // init flag for op_no_reuse_mem_vec_

  bool is_ge_reuse_mem_ = true; // global, controlled by ge option ge.exec.disableReuseMemory

  bool is_op_reuse_mem_ = true; // op-level, changed and shared in the process of an op

  bool is_separate_clean_continuous_inputs_ = false;  // op-output-level, changed and shared in the process of an output

  size_t life_time_;

  int64_t atomic_addr_clean_id_ = 0;

  std::map<uint64_t, MemoryStat> memory_stat_;  // key: device memory type

  std::string max_batch_label_;

  size_t life_begin_ = 0U;

  size_t life_end_ = 0U;

  bool root_unknown_shape_flag_ = false;

  // [sub graph id] streamid, out streamid, nodeid, outnodeid
  DiffStreamEdgeLife out_stream_edges_;

  // [sub graph id] streamid, in streamid, nodeid, innodeid
  DiffStreamEdgeLife in_stream_edges_;

  bool memory_priority_mode_ = false;

  bool is_io_alloc_by_ge_in_run_graph_ = false;

  bool strict_reuse_zero_memory_mode_ = false;

  bool is_feature_map_refreshable_ = false;

  uint64_t input_fusion_size_ = 0U;

  ReuseStrategy reuse_strategy_{};

  // Saved and finally modified by a single thread
  std::vector<TAttr<bool>> bool_attr_;
  std::vector<TAttr<int64_t>> int_attr_;
  std::vector<bool> input_index_to_reuse_mem_flag_;
  std::vector<bool> output_index_to_reuse_mem_flag_;

  std::unordered_map<OutDataAnchor *, std::list<OutDataAnchor *> *> same_out_group_;
  std::list<std::list<OutDataAnchor *>> same_out_group_holder_;
  bool is_static_model_addr_fixed_ = false;
  const std::unordered_map<const Node *, std::vector<int64_t>> &parent_nodes_to_stream_ids_;
  ContinuousMemMng continuous_mem_mng_;
};
}  // namespace ge
#endif  // GE_GRAPH_BUILD_MEMORY_BLOCK_MEM_ASSIGNER_H_
