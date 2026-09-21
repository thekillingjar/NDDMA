/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_TRANS_OP_CREATOR_H_
#define GE_GRAPH_TRANS_OP_CREATOR_H_

#include <string>

#include "graph/op_desc.h"
#include "graph/node.h"
#include "graph/compute_graph.h"
#include "framework/common/debug/ge_log.h"
#include "base/err_msg.h"

namespace ge {
struct GeShapeHasher {
  uint64_t operator()(const GeShape &shape) const;
};
class TransOpCreator {
 public:
  static OpDescPtr CreateTransDataOp(const std::string &op_name, const GeTensorDesc &input_desc,
                                     const GeTensorDesc &output_desc, bool enable_check_acc_support = true);

  static OpDescPtr CreateTransPoseDOp(const std::string &op_name, const GeTensorDesc &input_desc,
                                      const GeTensorDesc &output_desc);

  static OpDescPtr CreateCastOp(const std::string &op_name, const GeTensorDesc &input_desc,
                                const GeTensorDesc &output_desc, bool enable_check_acc_support = true);

  /**
   * 往compute_graph中添加Reshape节点。该接口创建Reshape节点时会同时创建一个const节点，并作为Reshape的shape输入。
   * @param compute_graph Reshape节点所在的计算图
   * @param op_name Reshape算子名
   * @param input_desc_x Reshape节点的X输入描述
   * @param output_desc  Reshape节点的输出描述
   * @return Reshape节点的SharePtr
   */
  static NodePtr CreateReshapeNodeToGraph(const ComputeGraphPtr &compute_graph, const std::string &op_name,
                                          const GeTensorDesc &input_desc_x, const GeTensorDesc &output_desc);

  /**
   * 往compute_graph中添加Reshape节点。该接口创建Reshape节点时会优先从reshape_shape_input_2_const_nodes中根据
   * 目标shape查找是否存在对应的Const节点作为Shape输入，避免Const节点重复创建。若没找到，才会创建对应的Const节点。
   * @param compute_graph Reshape节点所在的计算图
   * @param op_name Reshape算子名
   * @param input_desc_x Reshape节点的X输入描述
   * @param output_desc  Reshape节点的输出描述
   * @param reshape_target_shape_2_const_nodes 用于缓存Reshape目标shape对应的Const节点的哈希表
   * @return Reshape节点的SharePtr
   */
  static NodePtr CreateReshapeNodeToGraph(
      const ComputeGraphPtr &compute_graph, const std::string &op_name,
      const GeTensorDesc &input_desc_x, const GeTensorDesc &output_desc,
      std::unordered_map<GeShape, NodePtr, GeShapeHasher> &reshape_target_shape_2_const_nodes);

  static OpDescPtr CreateOtherTransOp(const std::string &op_name, const std::string &op_type,
                                      const GeTensorDesc &input_desc, const GeTensorDesc &output_desc);

  static graphStatus CheckAccuracySupported(const OpDescPtr &op_desc, bool &is_supported);

  /**
   *
   * @param op_desc
   * @param engine_name 若传入空字符串，意味着任何一个引擎中支持即认为支持
   * @param is_supported
   * @param unsupported_reason
   * @return
   */
  static graphStatus CheckAccuracySupported(const OpDescPtr &op_desc, const std::string &engine_name,
                                            bool &is_supported, std::string &unsupported_reason);

  TransOpCreator() = default;
  TransOpCreator(TransOpCreator &&) = delete;
  TransOpCreator(const TransOpCreator &) = delete;
  TransOpCreator &operator=(const TransOpCreator &) = delete;

 private:
  inline static graphStatus AddInputOutputDesc(const OpDescPtr &op_desc, const std::vector<GeTensorDesc> &input_descs,
                                               const GeTensorDesc &output_desc) {
    for (const auto &input_desc : input_descs) {
      auto ret = op_desc->AddInputDesc(input_desc);
      if (ret != GRAPH_SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Add input desc into op:%s(%s) failed", op_desc->GetName().c_str(),
                          op_desc->GetType().c_str());
        GELOGE(INTERNAL_ERROR, "[Add][InputDesc] into op:%s(%s) failed", op_desc->GetName().c_str(),
               op_desc->GetType().c_str());
        return ret;
      }
    }

    auto ret = op_desc->AddOutputDesc(output_desc);
    if (ret != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Add output desc into op:%s(%s) failed", op_desc->GetName().c_str(),
                        op_desc->GetType().c_str());
      GELOGE(INTERNAL_ERROR, "[Add][OutputDesc] into op:%s(%s) failed", op_desc->GetName().c_str(),
             op_desc->GetType().c_str());
      return ret;
    }
    return GRAPH_SUCCESS;
  };
};
}  // namespace ge

#endif  // GE_GRAPH_TRANS_OP_CREATOR_H_
