/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef GE_GRAPH_PASSES_TENSOR_MOVE_DELETE_PASS_H_
#define GE_GRAPH_PASSES_TENSOR_MOVE_DELETE_PASS_H_

#include "graph/node.h"
#include "graph/passes/base_pass.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
/**
 * @brief 冗余 TensorMove 节点消除优化
 *
 * 识别并删除计算图中冗余的TensorMove算子，以减少不必要的内存拷贝
 *
 * 核心逻辑流程
 * 1. 源头回溯:
 * 从 TensorMove 的输入端口向上回溯，寻找真正产生数据的“源头节点”。
 * 回溯过程具备穿透能力：
 * - 跨子图边界：从子图 Data 跳出到父图，或钻入 PartitionedCall 内部。
 * - RefOp 透传：穿过 Reshape/Cast 等具有引用关系的节点。
 * - 多TensorMove：连续的拷贝节点。
 * 2. 安全校验:
 * 仅当满足以下所有规则时，才执行删除：
 * - Rule 1: 源头节点不能是特殊变量/常量（除非特定场景）
 * - Rule 2: 如果源头是图的输入 (Data)，仅当以下条件同时满足时允许删除：
 *           a) 数据流最终直接输出到全图输出 (NetOutput)，中间没有计算节点消费
 *           b) 用户显式配置了内存复用选项 (ge.exec.outputReuseInputMemIndexes)，承诺输入输出内存复用。
 * - Rule 3: 单通路校验:
 *           从源头到 TensorMove 的路径上，所有节点（包括 RefOp 的隐式分叉）必须是单输出。
 *           如果有其他分支使用了内存，则不能移除该拷贝节点。
 * 3. 拓扑重连: 满足条件后，将TensorMove消除，直接连接其上游与下游。
 *
 * 【典型优化场景】
 * 1. 计算节点直连: Relu -> TensorMove -> NetOutput  ==>  Relu -> NetOutput
 * 2. 零拷贝输出: Data -> TensorMove -> NetOutput (需配置内存复用)  ==>  Data -> NetOutput
 * 3. 跨子图优化: Root_Data -> SubGraph(Data->TensorMove->NetOutput)(需配置内存复用) -> Root_NetOutput ==> 删除TensorMove
 */
class TensorMoveDeletePass : public BaseNodePass  {
  public:
  TensorMoveDeletePass() = default;
  ~TensorMoveDeletePass() override = default;

  Status Run(NodePtr &node) override;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_TENSOR_MOVE_DELETE_PASS_H_

