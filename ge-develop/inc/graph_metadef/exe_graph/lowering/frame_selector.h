/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_LOWERING_FRAME_SELECTOR_H_
#define METADEF_CXX_INC_EXE_GRAPH_LOWERING_FRAME_SELECTOR_H_
#include <vector>
#include "value_holder.h"
#include "common/checker.h"
namespace gert {
namespace bg {
/**
 * frame选择器，通过frame选择器，可以将执行图逻辑生成到执行的frame上
 *
 * 当前frame选择器还无法提供OnInitFrame/OnDeInitFrame功能，因为ValueHolder跨图连接的能力仅支持从父图向子图连接，
 * 若出现了Init向Main图中节点连边的场景，当前无法处理。对于Init/DeInit需求，按需开发即可
 */
class FrameSelector {
 public:
  /**
   * 选择Init图，将builder中的逻辑生成到Init图上，并返回**Init节点**的输出ValueHolder。
   * 注意：如果builder中创建的输出ValueHolder具有guarder，那么此guarder节点会被自动移动到DeInit图中
   * @param builder 执行图构建函数
   * @return 成功时，将builder返回的ValueHolderPtr作为本函数的返回值；失败时，本函数返回空vector
   */
  static std::vector<ValueHolderPtr> OnInitRoot(const std::function<std::vector<ValueHolderPtr>()> &builder);

  static ge::graphStatus OnInitRoot(const std::function<std::vector<ValueHolderPtr>()> &builder,
                                    std::vector<ValueHolderPtr> &init_graph_outputs,
                                    std::vector<ValueHolderPtr> &init_node_outputs);
  /**
   * 选择DeInit图，将builder中的逻辑生成到DeInit图上，并返回builder返回的ValueHolder。
   * 注意：builder中创建无输出的ValueHolder，即CreateVoid，并将其返回。
   * @param builder 执行图构建函数
   * @return 成功时，将DeInit图中builder返回的ValueHolder节点作为本函数的返回值；失败时，本函数返回空vector
   */
  static std::vector<ValueHolderPtr> OnDeInitRoot(const std::function<std::vector<ValueHolderPtr>()> &builder);

  /**
   * 选择Main图，将builder中的逻辑生成到Main图上。
   *
   * 需要注意的是，本函数仅保证当前将builder中的逻辑生成到Main图上，但不保证其始终在Main图上。
   * 在lowering构图完成后，在图优化阶段，CEM等优化可能将Main图上的Node移动到Init图中。
   *
   * @param builder 执行图构建函数
   * @return 成功时，将builder返回的ValueHolderPtr作为本函数的返回值；失败时，本函数返回空vector
   */
  static std::vector<ValueHolderPtr> OnMainRoot(const std::function<std::vector<ValueHolderPtr>()> &builder);

  static ge::graphStatus OnMainRoot(const std::function<std::vector<ValueHolderPtr>()> &builder,
                                    std::vector<ValueHolderPtr> &outputs);
  /**
 * 选择Main图，将builder中的逻辑生成到Main图上, 并且保证builder生成的节点在main图最开始执行
 * 当前已有阶段，请参考bg::OnMainRootFirstExecStage的枚举值
 *
 * @param builder 执行图构建函数
 * @return 成功时，将builder返回的ValueHolderPtrs作为本函数的返回值；失败时，本函数返回空vector
 */
  static std::vector<ValueHolderPtr> OnMainRootFirst(const std::function<std::vector<bg::ValueHolderPtr>()> &builder);

  static ValueHolderPtr OnMainRootLast(const std::function<bg::ValueHolderPtr()> &builder);

  /**
 * 选择Main图，将builder中的逻辑生成到Main图上, builder生成的节点在LastEventSync阶段执行.
 * 当前已有阶段，请参考bg::OnMainRootLastExecStage的枚举值
 *
 * @param builder 执行图构建函数
 * @return 成功时，将builder返回的ValueHolderPtrs作为本函数的返回值；失败时，本函数返回空vector
 */
  static std::vector<ValueHolderPtr> OnMainRootLastEventSync(
      const std::function<std::vector<bg::ValueHolderPtr>()> &builder);

  /**
   * 选择Main图，将builder中的逻辑生成到Main图上, builder生成的节点在LastResourceClean阶段执行.
   * 当前已有阶段，请参考bg::OnMainRootLastExecStage的枚举值
   *
   * @param builder 执行图构建函数
   * @return 成功时，将builder返回的ValueHolderPtrs作为本函数的返回值；失败时，本函数返回空vector
   */
  static std::vector<ValueHolderPtr> OnMainRootLastResourceClean(
      const std::function<std::vector<bg::ValueHolderPtr>()> &builder);
};

ValueHolderPtr HolderOnInit(const ValueHolderPtr &holder);
}  // namespace bg
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_LOWERING_FRAME_SELECTOR_H_
