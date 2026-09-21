/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "assemble_json/te_op_custom_utils.h"
#include "inc/te_fusion_log.h"
#include "inc/te_fusion_check.h"
#include "inc/te_fusion_util_constants.h"
#include "common/tbe_op_info_cache.h"

namespace te {
namespace fusion {
using nlohmann::json;
namespace {
constexpr size_t SHAPE_4D = 4;
constexpr size_t SHAPE_5D = 5;
constexpr size_t SHAPE_6D = 6;
constexpr int DIM_INDEX_2 = 2;
constexpr int DIM_INDEX_3 = 3;
constexpr int DIM_INDEX_4 = 4;
constexpr int DIM_INDEX_5 = 5;
/*
 * @brief: get multi input elementwise node for convolution fusion
 * @param [in] teGraphNode: sub fusion graph
 * @param [out] elewiseNode: multi input elementwise nodes for convolution fusion
 * @return bool: get elewise Node ok or not
 */
bool GetConvElewise(const std::vector<ge::Node *> &teGraphNode, std::vector<ge::Node *> &elewiseNode,
                    uint32_t inputGroupIdx, bool &hasFixpipe)
{
    // func start log
    TE_DBGLOG("Begin obtaining multiple input element-wise nodes for the purpose of convolution fusion.");

    std::string fixpipePattern = "fixpipe";
    std::set<std::string> convPatterns = {"QuantConvolution", "Convolution", "Conv2d_backprop_input",
                                          "Conv3d", "Conv3d_backprop_input"};
    std::set<std::string> vectorPatterns = {"ElemWise", "Broadcast"};
    bool convFusionFlag = false;
    for (auto &node : teGraphNode) {
        TE_FUSION_CHECK(node == nullptr, continue);
        std::string opTypePattern;
        std::string name = node->GetName();
        bool bres = ge::AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_OP_PATTERN, opTypePattern);
        if (!bres) {
            TE_ERRLOG("Node[%s] does not have the attribute '_pattern'", name.c_str());
            return false;
        };
        convFusionFlag = convFusionFlag || (convPatterns.count(opTypePattern) != 0);
    }
    for (auto &node : teGraphNode) {
        TE_FUSION_CHECK(node == nullptr, continue);
        std::string opTypePattern;
        std::string name = node->GetName();
        // get op type
        bool bres = ge::AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_OP_PATTERN, opTypePattern);
        if (!bres) {
            TE_ERRLOG("current Node[%s] does not have the attribute '_pattern'.", name.c_str());
            return false;
        };
        if (opTypePattern == fixpipePattern) {
            hasFixpipe = true;
        }
        if (convFusionFlag && ((vectorPatterns.count(opTypePattern) != 0) ||
                               (fixpipePattern == opTypePattern && inputGroupIdx == 1))) {
            if (node->GetInDataNodes().size() > 1) {
                elewiseNode.push_back(node);
            };
        }
    }

    // func end log
    TE_DBGLOG("Successfully obtained convolution multi-input elewise node.");
    return true;
}


/*
 * @brief: get dx and drelu node for fusion
 * @param [in] teGraphNode: sub fusion graph
 * @param [out] dReluNode: drelu nodes for dx and drelu fusion
 * @return bool: get drelu Node ok or not
 */
bool GetDxDreluNode(const std::vector<ge::Node *> &teGraphNode, std::vector<ge::Node *> &dReluNode)
{
    // func start log
    TE_DBGLOG("Start getting dx and drelu nodes for fusion.");

    std::string dxPattern = "Conv2d_backprop_input";
    bool dxFusionFlag = false;
    for (auto &node : teGraphNode) {
        TE_FUSION_CHECK(node == nullptr, continue);
        std::string opTypePattern;
        std::string name = node->GetName();
        bool bres = ge::AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_OP_PATTERN, opTypePattern);
        TE_FUSION_CHECK(!bres, {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Node[%s] has no attr _pattern.", name.c_str());
            return false;
        });
        if (dxPattern == opTypePattern) {
            dxFusionFlag = true;
        }
        if (dxFusionFlag) {
            std::string funcName;
            if (!TbeOpInfoCache::Instance().GetOpFuncNameByNode(node, funcName)) {
                TE_ERRLOG("Failed to retrieve func name for node [%s].", node->GetName().c_str());
                return false;
            }

            TE_FUSION_CHECK(funcName == "relu_grad_v2", {
                dReluNode.push_back(node);
            });
        }
    }

    // func end log
    TE_DBGLOG("Successfully obtained dx and drelu nodes.");

    return true;
}

void CheckCurrNode(const std::vector<ge::Node *> &elewiseNode, const std::vector<ge::Node *> &dReluNode,
                   const ge::Node *currNode, bool &transShape4hd, bool &isDRelu)
{
    if (std::find(elewiseNode.begin(), elewiseNode.end(), currNode) !=
        elewiseNode.end()) {
        transShape4hd = true;
    }
    if (std::find(dReluNode.begin(), dReluNode.end(), currNode) != dReluNode.end()) {
        isDRelu = true;
    }
}

void ChangeShape(const std::vector<int64_t> &currShape, const bool &transShape,
                 const bool &hasFixpipe, const std::vector<std::pair<int64_t, int64_t>> &currShapeRange,
                 std::vector<int64_t> &shape, json &inputDescJson)
{
    std::vector<std::pair<int64_t, int64_t>> shapeRange;
    if (!transShape) {
        shape = currShape;
        return;
    }
    size_t shapeDims = currShape.size();
    if (hasFixpipe && shapeDims == SHAPE_4D) {
        // fixpipe only support NHWC transdata, fusion axis H and W
        shape.push_back(currShape[0]);
        if (currShape[1] == -1 || currShape[DIM_INDEX_2] == -1) {
            shape.push_back(-1);
            shapeRange.push_back(currShapeRange[0]);
            std::pair<int64_t, int64_t> secondRange;
            secondRange.first = currShapeRange[1].first * currShapeRange[DIM_INDEX_2].first;
            secondRange.second = currShapeRange[1].second * currShapeRange[DIM_INDEX_2].second;
            shapeRange.push_back(secondRange);
            shapeRange.push_back(currShapeRange[DIM_INDEX_3]);
            inputDescJson["range"] = shapeRange;
        } else {
            shape.push_back(currShape[1]*currShape[DIM_INDEX_2]);
        }
        shape.push_back(currShape[DIM_INDEX_3]);
        inputDescJson["shape"] = shape;
    } else if (shapeDims == SHAPE_5D) {
        shape.push_back(currShape[0]);
        shape.push_back(currShape[1]);
        if (currShape[DIM_INDEX_2] == -1 || currShape[DIM_INDEX_3] == -1) {
            shape.push_back(-1);
            shapeRange.push_back(currShapeRange[0]);
            shapeRange.push_back(currShapeRange[1]);
            std::pair<int64_t, int64_t> secondRange;
            secondRange.first = currShapeRange[DIM_INDEX_2].first * currShapeRange[DIM_INDEX_3].first;
            secondRange.second = currShapeRange[DIM_INDEX_2].second * currShapeRange[DIM_INDEX_3].second;
            shapeRange.push_back(secondRange);
            shapeRange.push_back(currShapeRange[DIM_INDEX_4]);
            inputDescJson["range"] = shapeRange;
        } else {
            shape.push_back(currShape[DIM_INDEX_2]*currShape[DIM_INDEX_3]);
        }
        shape.push_back(currShape[DIM_INDEX_4]);
        inputDescJson["shape"] = shape;
    } else if (shapeDims == SHAPE_6D) {
        shape.push_back(currShape[0] * currShape[1]);
        shape.push_back(currShape[DIM_INDEX_2]);
        shape.push_back(currShape[DIM_INDEX_3] * currShape[DIM_INDEX_4]);
        shape.push_back(currShape[DIM_INDEX_5]);
        for (size_t idx = 0; idx < currShape.size(); idx++) {
            if (currShape[idx] == -1) {
                std::pair<int64_t, int64_t> oneRange;
                oneRange.first = currShapeRange[0].first * currShapeRange[1].first;
                oneRange.second = currShapeRange[0].second * currShapeRange[1].second;
                shapeRange.push_back(oneRange);
                shapeRange.push_back(currShapeRange[DIM_INDEX_2]);
                oneRange.first = currShapeRange[DIM_INDEX_3].first * currShapeRange[DIM_INDEX_4].first;
                oneRange.second = currShapeRange[DIM_INDEX_3].second * currShapeRange[DIM_INDEX_4].second;
                shapeRange.push_back(oneRange);
                shapeRange.push_back(currShapeRange[DIM_INDEX_5]);
                inputDescJson["range"] = shapeRange;
                break;
            }
        }
        inputDescJson["shape"] = shape;
    } else {
        shape = currShape;
    }
}
}

bool ChangeInputDescShape(const std::vector<ge::Node *> &teGraphNode, const ge::Node *currNode,
                          const std::vector <int64_t> &currShape, uint32_t inputGroupIdx, json &inputDescJson)
{
    std::vector <ge::Node *> elewiseNode;
    bool hasFixpipe = false;
    bool bres = GetConvElewise(teGraphNode, elewiseNode, inputGroupIdx, hasFixpipe);
    TE_FUSION_CHECK(!bres, {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                           "Failed to get multi input elewise for conv fusion.");
        return false;
    });
    std::vector <ge::Node *> dReluNode;
    bres = GetDxDreluNode(teGraphNode, dReluNode);
    TE_FUSION_CHECK(!bres, {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
                           "Failed to get drelu for dx fusion.");
        return false;
    });

    bool transShape4hd = false;
    bool isDRelu = false;
    CheckCurrNode(elewiseNode, dReluNode, currNode, transShape4hd, isDRelu);
    std::vector <int64_t> shape4Hd;
    std::vector <std::pair<int64_t, int64_t>> currShapeRange = inputDescJson["range"];
    ChangeShape(currShape, transShape4hd, hasFixpipe, currShapeRange, shape4Hd, inputDescJson);

    // need to transform DRelu input parameter[1] from int8 to bool
    if (isDRelu && inputGroupIdx == 1) {
        std::string dType = "bool";
        std::string oldDType = inputDescJson["data_type"];
        auto oldDTypeIter = kDataTypeStrToLength.find(oldDType);
        auto newDTypeIter = kDataTypeStrToLength.find(dType);
        inputDescJson["data_type"] = dType;
        std::vector <int64_t> shapeBool;
        for (size_t idx = 0; idx < shape4Hd.size(); ++idx) {
            if (idx < shape4Hd.size() - 1) {
                shapeBool.push_back(shape4Hd[idx]);
            }
        }
        if (shape4Hd.size() > 0 && oldDTypeIter != kDataTypeStrToLength.end() &&
            newDTypeIter != kDataTypeStrToLength.end()) {
            // change shape size from int8 to bool by mul 8
            shapeBool.push_back(shape4Hd[shape4Hd.size() - 1] * oldDTypeIter->second / newDTypeIter->second);
            TE_DBGLOG("shape last dim change from [%ld] to [%ld], type from [%s] to [%s]",
                      shape4Hd[shape4Hd.size() - 1], shapeBool[shapeBool.size() - 1],
                      oldDType.c_str(), dType.c_str());
            inputDescJson["shape"] = shapeBool;
        }
    }
    return true;
}
}
}