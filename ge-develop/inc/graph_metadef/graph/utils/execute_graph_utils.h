/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_UTILS_EXECUTE_GRAPH_UTILS_H
#define INC_GRAPH_UTILS_EXECUTE_GRAPH_UTILS_H
#include "graph/fast_graph/fast_node.h"
#include "graph/fast_graph/execute_graph.h"

namespace ge {
class ExecuteGraphUtils {
 public:
  /**
   * 查找`exe_graph`的根图，如果当前图就是根图或者当前图没有父图，则返回当前图
   * @param exe_graph
   * @return
   */
  static ExecuteGraph *FindRootGraph(ExecuteGraph *exe_graph);

  /**
   * 查找`exe_graph`中节点名为`name`的节点，包含子图
   * @param exe_graph
   * @return
   */
  static FastNode *FindNodeFromAllNodes(ExecuteGraph *exe_graph, const char_t *const name);

  /**
   * 查找`exe_graph`中节点类型为`type`的节点，包含子图
   * @param exe_graph
   * @return
   */
  static std::vector<FastNode *> FindNodesByTypeFromAllNodes(ExecuteGraph *exe_graph, const char_t *const type);

  /**
   * 查找`exe_graph`中首个节点类型为`type`的节点，不包含子图
   * @param exe_graph
   * @param type
   * @return
   */
  static FastNode *FindFirstNodeMatchType(ExecuteGraph *exe_graph, const char_t *const type);

  /**
   * 接口行为是在'src'的源节点输出端和'dst'目的节点输入端们之间插入一个`insert_node`节点,
   * 默认是`insert_node`的`0`号数据输入端和`0`号输出端参与连边，`insert_node`插入之后, `src_node`和`insert_node`
   * 作为一个整体与原来的`src_node`具备等价的控制和数据关系
   * @param src 源数据输出端
   * @param dsts 源数据输出端连接的目的数据输入端，使用vector的原因是存在一个源节点输出端给到多个目的节点输入端的情况
   * @param insert_node 表示要插入的节点
   * @param input_index 表示插入节点的哪个数据输入端要跟src相连，如果不传递，默认取0
   * @param output_index 表示插入节点的哪个数据输出端要跟dsts依次相连，如果不传递，默认取0
   * @return 如果插入成功返回GRAPH_SUCCESS，失败返回GRAPH_FAILED
   */
  static graphStatus InsertNodeAfter(const EdgeSrcEndpoint &src, const std::vector<EdgeSrcEndpoint> &dsts,
                                     FastNode *insert_node, const uint32_t input_index = 0U,
                                     const uint32_t output_index = 0U);

  /**
   * 接口行为是在数据`dst`目的节点输入端和其对端源节点输出端之间插入一个`insert_node`节点,
   * 默认是`insert_node`的`0`号数据输入端和`0`号数据输出数据端参与连边，`insert_node`插入之后,
   * `dst_node`和`insert_node`作为一个整体与原来的`dst_node`具备等价的控制和数据关系
   * @param dst 目的数据输入端
   * @param insert_node 表示要插入的节点
   * @param input_index 表示插入节点的哪个数据输入端要跟dst的对端src输出端相连，如果不传递，默认取0
   * @param output_index 表示插入节点的哪个数据输出端要跟dst相连，如果不传递，默认取0
   * @return 如果插入成功返回GRAPH_SUCCESS，失败返回GRAPH_FAILED
   */
  static graphStatus InsertNodeBefore(const EdgeSrcEndpoint &dst, FastNode *insert_node,
                                      const uint32_t input_index = 0U, const uint32_t output_index = 0U);

  /**
   * 移动`src_node`的控制输入边到`dst_node`上
   * @param src_node
   * @param dst_node
   * @return
   */
  static graphStatus MoveInCtrlEdges(const FastNode *src_node, FastNode *dst_node);

  /**
   * 移动`src_node`的控制输出边到`dst_node`上
   * @param src_node
   * @param dst_node
   * @return
   */
  static graphStatus MoveOutCtrlEdges(const FastNode *src_node, FastNode *dst_node);

  /**
   * 将node所有的输入、输出边断开，并移动到dst_graph
   * @param dst_graph 目的Graph，
   * @param node 需要移动的Node
   * @return 成功时，返回ge::GRAPH_SUCCESS
   */
  static graphStatus MoveNodeToGraph(FastNode *node, ExecuteGraph *dst_graph);

  /**
   * 接口行为是根据`inputs_map`和`outputs_map`把`old_node`上的数据关系`移动`到`new_node`上；具体操作是
   * 把`old_node`的第`inputs_map[i]`/`outputs_map[i]`个数据输入、输出端点的数据关系替换到`new_node`的第`i`个
   * 输入、输出端点上, `i`的取值范围是[0, `inputs_map`/`outputs_map`的元素个数）; 如果`inputs_map[i]`/`outputs_map[i]`
   * 的值小于0或者不在`old_node`的输入、输出端点范围之内，那么`new_node`的第`i`个数据输入、输出端点的数据关系保持原样
   * @param new_node
   * @param old_node
   * @param inputs_map 用于指导输入数据端点的替换，注意元素个数不应该超过`new_node`的输入端点总个数
   * @param outputs_map 用于指导输出端点的替换，注意元素个数不应该超过`new_node`的输出端点总个数
   * @param graph 表示`new_node`和`old_node`所在的graph，如果不传会通过`new_node`获取
   * @return
   */
  static graphStatus ReplaceNodeDataEdges(FastNode *new_node, FastNode *old_node,
                                          const std::initializer_list<int32_t> inputs_map,
                                          const std::initializer_list<int32_t> outputs_map,
                                          ExecuteGraph *graph = nullptr);
  static graphStatus ReplaceNodeDataEdges(FastNode *new_node, FastNode *old_node,
                                          const std::vector<int32_t> &inputs_map,
                                          const std::vector<int32_t> &outputs_map, ExecuteGraph *graph = nullptr);

  /**
   * 此接口对数据关系的处理与`ReplaceNodeDataEdges`的处理行为一致， 在此基础上，
   * 复制了`old_node`的所有控制关系到`new_node`上，这也是要注意的一点：
   * `数据`关系是`移动`操作，`控制`关系是`复制`操作
   * @param new_node
   * @param old_node
   * @param inputs_map 用于指导输入数据锚点的替换，注意元素个数不应该超过`new_node`的输入短点总个数
   * @param outputs_map 用于指导输出锚点的替换，注意元素个数不应该超过`new_node`的输出短点总个数
   * @return 如果替换成功返回GRAPH_SUCCESS，失败返回GRAPH_FAILED
   */
  static graphStatus ReplaceNodeEdges(FastNode *new_node, FastNode *old_node,
                                      const std::initializer_list<int32_t> inputs_map,
                                      const std::initializer_list<int32_t> outputs_map);
  static graphStatus ReplaceNodeEdges(FastNode *new_node, FastNode *old_node, const std::vector<int32_t> &inputs_map,
                                      const std::vector<int32_t> &outputs_map);

  /**
   * 孤立`node`, 根据`io_map`完成node的输入输出数据边的重连；同时会添加必要的控制边保证`node`的所有输入节点
   * 均在`node`的输出节点之前执行
   * @param node
   * @param io_map 把第`io_map[i]`个输入的对端输出，连接到第`i`个输出的对端输入。因此`io_map`的元素个数应该与
   * `node`的输出端点的个数相等，如果`io_map[i]`小于0，则仅断开第`i`个输出端点到对端的所有连边
   * @return
   */
  static graphStatus IsolateNode(FastNode *node, const std::initializer_list<int32_t> &io_map);
  static graphStatus IsolateNode(FastNode *node, const std::vector<int32_t> &io_map);

  /**
   * 替换`old_edge`的``src`为`new_src`指示的`node`和`index`
   * @param old_edge
   * @param new_src
   * @return 替换成功返回GRAPH_SUCCESS, 替换失败返回GRAPH_FAILED
   */
  static graphStatus ReplaceEdgeSrc(FastEdge *old_edge, const EdgeSrcEndpoint &new_src);

  /**
   * 从`execute_graph`上删除`直接`或者`间接`父节点为remove_node的所有子图对象
   * @param execute_graph
   * @param remove_node
   * @return 成功返回GRAPH_SUCCESS, 失败返回GRAPH_FAILED
   */
  static graphStatus RemoveSubgraphRecursively(ExecuteGraph *execute_graph, FastNode *remove_node);

  /**
   * 从`execute_graph`中删除`node`对象的所有关系，包括子图关系，从属关系，作为`execute_graph`的输入，输出的关系；
   * 仅删除，不进行断边连边，不保证删除后节点前后的控制关系传递
   * @param execute_graph
   * @param node
   * @return 成功返回GRAPH_SUCCESS, 失败返回GRAPH_FAILED
   */
  static graphStatus RemoveNodeWithoutRelink(ExecuteGraph *execute_graph, FastNode *node);

  /**
   * 拷贝`src_node`的输入控制边到`dst_node`上
   * @param src_node
   * @param dst_node
   * @return 成功返回GRAPH_SUCCESS, 失败返回GRAPH_FAILED
   */
  static graphStatus CopyInCtrlEdges(const FastNode *src_node, FastNode *dst_node);

  /**
   * 拷贝`src_node`的输出控制边到`dst_node`上
   * @param src_node
   * @param dst_node
   * @return 成功返回GRAPH_SUCCESS, 失败返回GRAPH_FAILED
   */
  static graphStatus CopyOutCtrlEdges(const FastNode *src_node, FastNode *dst_node);

  /**
 * 构建并返回根图中所有节点名到节点的映射，包含子图中的节点
 * @param exe_graph
 * @return 节点名到节点的映射
 */
  static std::unordered_map<std::string, FastNode *> GetNodeMapFromAllNodes(ExecuteGraph *exe_graph);
};
}  // namespace ge
#endif  // INC_GRAPH_UTILS_EXECUTE_GRAPH_UTILS_H
