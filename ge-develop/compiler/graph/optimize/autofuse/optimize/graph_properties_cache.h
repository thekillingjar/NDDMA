/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#ifndef OPTIMIZE_AUTOSCHEDULE_GRAPH_PROPERTIES_CACHE_H_
#define OPTIMIZE_AUTOSCHEDULE_GRAPH_PROPERTIES_CACHE_H_

#include <bitset>

#include "ascendc_ir/ascendc_ir_core/ascendc_ir_def.h"
#include "ascir.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace optimize {
  class GraphPropertiesCache {
  public:
    enum class Property : size_t {
      kHasReduce = 0,
      kHasGather,
      kHasCube,
      kHasTranspose,
      kHasConcat,
      kHasSplit,
      kHasLoad,
      kHasStore,
      kHasBuffer,
      kLastAxisReduce,
      kEnd
    };

    explicit GraphPropertiesCache(const ::ascir::ImplGraph &graph)
      : graph_(graph), cached_(false) {
    }

    GraphPropertiesCache(const GraphPropertiesCache &) = delete;

    GraphPropertiesCache &operator=(const GraphPropertiesCache &) = delete;

    GraphPropertiesCache(GraphPropertiesCache &&) = default;

    GraphPropertiesCache &operator=(GraphPropertiesCache &&) = delete;

    bool HasComputeType(ge::ComputeType compute_type) {
      EnsureCached();
      switch (compute_type) {
        case ge::ComputeType::kComputeReduce:
          return properties_[static_cast<size_t>(Property::kHasReduce)];
        case ge::ComputeType::kComputeGather:
          return properties_[static_cast<size_t>(Property::kHasGather)];
        case ge::ComputeType::kComputeCube:
          return properties_[static_cast<size_t>(Property::kHasCube)];
        case ge::ComputeType::kComputeTranspose:
          return properties_[static_cast<size_t>(Property::kHasTranspose)];
        case ge::ComputeType::kComputeConcat:
          return properties_[static_cast<size_t>(Property::kHasConcat)];
        case ge::ComputeType::kComputeSplit:
          return properties_[static_cast<size_t>(Property::kHasSplit)];
        case ge::ComputeType::kComputeLoad:
          return properties_[static_cast<size_t>(Property::kHasLoad)];
        case ge::ComputeType::kComputeStore:
          return properties_[static_cast<size_t>(Property::kHasStore)];
        default:
          return false;
      }
    }

    bool HasReduce() {
      EnsureCached();
      return properties_[static_cast<size_t>(Property::kHasReduce)];
    }

    bool HasGather() {
      EnsureCached();
      return properties_[static_cast<size_t>(Property::kHasGather)];
    }

    bool HasCube() {
      EnsureCached();
      return properties_[static_cast<size_t>(Property::kHasCube)];
    }

    bool HasTranspose() {
      EnsureCached();
      return properties_[static_cast<size_t>(Property::kHasTranspose)];
    }

    bool HasConcat() {
      EnsureCached();
      return properties_[static_cast<size_t>(Property::kHasConcat)];
    }

    bool HasSplit() {
      EnsureCached();
      return properties_[static_cast<size_t>(Property::kHasSplit)];
    }

    bool HasLoad() {
      EnsureCached();
      return properties_[static_cast<size_t>(Property::kHasLoad)];
    }

    bool HasStore() {
      EnsureCached();
      return properties_[static_cast<size_t>(Property::kHasStore)];
    }

    bool HasBuffer() {
      EnsureCached();
      return properties_[static_cast<size_t>(Property::kHasBuffer)];
    }

    bool IsLastAxisReduce() {
      EnsureCached();
      return properties_[static_cast<size_t>(Property::kLastAxisReduce)];
    }

    /**
     * @brief 使缓存失效，当图结构发生变化时应调用此方法
     */
    void Invalidate() {
      cached_ = false;
      properties_.reset();
    }

  private:
    void EnsureCached() {
      if (cached_) {
        return;
      }
      BuildCache();
      cached_ = true;
    }

    static bool IsLastAxisReduceNode(const ::ascir::NodeView &node) {
      const std::vector<ascir::SizeExpr> &src_strides = node->inputs[0].attr.strides;
      const std::vector<ascir::SizeExpr> &dst_strides = node->outputs[0].attr.strides;
      if (src_strides.empty() || dst_strides.size() <= src_strides.size() - 1) {
        return false;
      }
      auto last_index = src_strides.size() - 1;
      return (ge::SymbolicUtils::StaticCheckEq(src_strides[last_index],
                                               dst_strides[last_index]) != ge::TriBool::kTrue) &&
             (ge::SymbolicUtils::StaticCheckEq(dst_strides[last_index],
                                               ge::sym::kSymbolZero) == ge::TriBool::kTrue);
    }

    void BuildCache() {
      auto all_nodes = graph_.GetAllNodes();

      for (const auto &node: all_nodes) {
        if (node->attr.api.type == ge::ApiType::kAPITypeBuffer) {
          properties_[static_cast<size_t>(Property::kHasBuffer)] = true;
          continue;
        }

        switch (node->attr.api.compute_type) {
          case ge::ComputeType::kComputeReduce:
            properties_[static_cast<size_t>(Property::kHasReduce)] = true;
            properties_[static_cast<size_t>(Property::kLastAxisReduce)] = IsLastAxisReduceNode(node);
            break;
          case ge::ComputeType::kComputeGather:
            properties_[static_cast<size_t>(Property::kHasGather)] = true;
            break;
          case ge::ComputeType::kComputeCube:
            properties_[static_cast<size_t>(Property::kHasCube)] = true;
            break;
          case ge::ComputeType::kComputeTranspose:
            properties_[static_cast<size_t>(Property::kHasTranspose)] = true;
            break;
          case ge::ComputeType::kComputeConcat:
            properties_[static_cast<size_t>(Property::kHasConcat)] = true;
            break;
          case ge::ComputeType::kComputeSplit:
            properties_[static_cast<size_t>(Property::kHasSplit)] = true;
            break;
          case ge::ComputeType::kComputeLoad:
            properties_[static_cast<size_t>(Property::kHasLoad)] = true;
            break;
          case ge::ComputeType::kComputeStore:
            properties_[static_cast<size_t>(Property::kHasStore)] = true;
            break;
          default:
            break;
        }
      }
    }

    const ::ascir::ImplGraph &graph_;
    bool cached_;
    std::bitset<static_cast<size_t>(Property::kEnd)> properties_;
  };
} // namespace optimize

#endif  // OPTIMIZE_AUTOSCHEDULE_GRAPH_PROPERTIES_CACHE_H_
