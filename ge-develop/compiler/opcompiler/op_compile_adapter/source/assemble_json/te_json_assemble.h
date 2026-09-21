/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_TE_FUSION_SOURCE_ASSEMBLE_JSON_TE_JSON_ASSEMBLE_H_
#define ATC_OPCOMPILER_TE_FUSION_SOURCE_ASSEMBLE_JSON_TE_JSON_ASSEMBLE_H_

#include <string>
#include <vector>
#include <map>
#include <unordered_set>
#include <nlohmann/json.hpp>
#include "inc/te_fusion_types.h"
#include "tensor_engine/tbe_op_info.h"
#include "graph/node.h"

namespace te {
namespace fusion {
struct InOutToJsonParam {
    std::string keyName;
    size_t nodeListIdx;
    std::vector<std::string> verifyOpTypeList;
    ConstTbeOpInfoPtr tbeOpInfo;
    ge::Node *node = nullptr;
    InOutToJsonParam(const string &keyNamePara, const size_t &nodeListIdxPara,
                     std::vector<std::string> &verifyOpTypeListPara,
                     const ConstTbeOpInfoPtr &tbeOpInfoPara, ge::Node *nodePara)
        : keyName(std::move(keyNamePara)),
          nodeListIdx(nodeListIdxPara),
          verifyOpTypeList(verifyOpTypeListPara),
          tbeOpInfo(tbeOpInfoPara),
          node(nodePara) {}
};

struct InputDescJsonParam {
    std::vector<TbeOpParam> inputs;
    uint32_t inputSize = 0;
    std::string keyName;
    ge::Node *node;
    std::vector<std::string> inputNameList;
    std::vector<std::string> inputsLinkKeyList;
    std::map<int, int> inputOrder;
    InputDescJsonParam(const string &keyNamePara, ge::Node *nodePara)
        : keyName(std::move(keyNamePara)),
          node(nodePara) {}
};

class TeJsonAssemble {
public:
    static bool GenerateJsonAndKernelName(const std::vector<ge::Node *> &nodes, const bool isAllowRepeated,
                                          std::string &jsonStr, std::string &kernelName);

    static bool GeneratePrebuildJsonAndKernelName(const std::vector<ge::Node *> &nodes, std::string &jsonStr,
                                                  std::string &kernelName);

    static bool GenerateOpJson(const ge::NodePtr &nodePtr, const TbeOpInfo &tbeOpInfo, std::string &jsonStr);

    /**
     * @brief Generate node data json
     * @param [in] teGraphNode             fusion subgraph node ptr vector
     * @param [in] currNode                current node
     * @param [in] inputTensorIdx          input tensor index in graph
     * @param [in] inputGroupIdx           input index in op
     * @param [out] jsonStr                data desc json
     * @return [out] bool                  get data desc json succ or fail
     */
    static bool GenNodeDataJson(const std::vector<ge::Node *> &teGraphNode, ge::Node *currNode, uint32_t inputTensorIdx,
                                uint32_t inputGroupIdx, nlohmann::json &inputDescJson, bool needConvertBoolToInt8);

    static bool GenerateOptionsMap(const std::vector<ConstTbeOpInfoPtr> &tbeOpInfoVec,
                                   std::map<std::string, std::string> &options);

    static void GetOutputSuffix(ge::InDataAnchorPtr anchor, const std::unordered_set<ge::Node *> &allNodes,
                                std::string &suffix, size_t &refCount);

private:
    static bool GenerateJsonByNodes(const std::vector<ge::Node *> &teGraphNode, nlohmann::json &jsonData);
    static bool GenerateKernelName(const std::string &prefix, const std::string &uniqueJsonStr,
                                   const bool isAllowRepeatedHash, std::string &kernelName);
    static bool GeneratePreBuildKernelName(const std::string &prefix, const std::string &uniqueJsonStr,
                                           std::string &kernelName);
    static void GetUnrepeatedHash(std::string &kernelHash);
    static void GenerateSocInfoJson(const std::vector<ConstTbeOpInfoPtr> &tbeOpInfoVec, nlohmann::json &socInfoJson);
    static bool GenBuildinIndescJson(const InOutToJsonParam &inputPara, const std::unordered_set<ge::Node *> &allNodes,
                                     nlohmann::json &opJson, std::map<string, int> &allInputNames,
                                     const std::vector<ge::Node *> &teGraphNode);
    static void GenBuildinOutdescJson(const InOutToJsonParam &outputPara,
                                      const std::unordered_set<ge::Node *> &allNodes, nlohmann::json &jsonStr,
                                      const std::vector<ge::Node *> &teGraphNode);
    /**
     * @brief Generate plug-in node input desc Json
     * @param [in] opNode                  graph node ptr vector
     * @param [in] transShape4hd           flag whether need to transform 5hd to 4hd
     * @param [out] jsonStr                input desc json string
     * @return [out] bool                  create json succ or fail
     */
    static void GenBuildinAttrsAndModuleJson(const ConstTbeOpInfoPtr &tbeOpInfo, nlohmann::json &jsonStr);

    /**
     * @brief Generate plug-in node input desc Json
     * @param [in] opNode                  graph node ptr vector
     * @param [in] transShape4hd           flag whether need to transform 5hd to 4hd
     * @param [out] jsonStr                input desc json string
     * @return [out] bool                  create json succ or fail
     */
    static bool GenDatadescJson(const std::vector<ge::Node *> &teGraphNode, nlohmann::json &jsonStr);

    static bool InputsDescToJsonProcess(const std::unordered_set<ge::Node *> &allNodes,
                                        const InputDescJsonParam &const_para, nlohmann::json &jsonStr,
                                        std::map<string, int> &allInputNames);

    static void InputsTensorDescToJson(const TbeOpTensor &tensor, nlohmann::json &inputDescJson);

    static void GenerateOutputLinkKey(const std::unordered_set<ge::Node *> &allNodes, const ge::OutDataAnchorPtr &anchor,
                                      size_t i, const InOutToJsonParam &outputParam, const std::vector<ge::Node *> &teGraphNode,
                                      nlohmann::json &outputDescJson);

    static void OutputsDescToJsonProcess(nlohmann::json &json_str, const std::unordered_set<ge::Node *> &allNodes,
                                         const InOutToJsonParam &outputParam, uint32_t idx,
                                         const ge::OutDataAnchorPtr &anchor, size_t peerDataNodesSize,
                                         const std::vector<ge::Node *> &teGraphNode);

    static void GenBuildinAttrsJson(const TbeOpInfo &tbeOpInfo, nlohmann::json &jsonStr);

    static void GenBuildinPrivateAttrsJson(const TbeOpInfo &tbeOpInfo, nlohmann::json &jsonStr,
                                           const std::vector<std::string> &variableAttrs);

    static bool CheckInputsSize(InputDescJsonParam &inputDescJsonPara);

    static bool GenInputsLinkKey(const ge::Node *node, size_t nodeListIdx,
                                 std::vector<std::string> verifyOpTypeList,
                                 std::vector<std::string> &inputsLinkKeyList,
                                 const std::unordered_set<ge::Node *> &allNodes,
                                 const std::vector<ge::Node *> &teGraphNode);

    static bool CheckInputsNullTensor(const TbeOpParam &input, bool &isNull);

    static void GetPeerInputsOrder(const ge::OutDataAnchorPtr &anchor, uint32_t idx,
                                   const std::unordered_set<ge::Node *> &allNodes,
                                   std::vector<string> &peerInputsOrder);

    /* When the output node has two output and both output is the
     * input of current node, we need a output id differencient two output.
     * output id is the output sequence in the output node.
     * For example:
     * A ----> B
     *   \---/
     * A has two different output and they are both linked to B. We
     * need to record the output index of A because if A's two output have
     * same format\attr\dtype and shape, compiling cache will consider them as
     * the same. So we need the output index to differenciate the output.
     * @parmater allInputsId: {input index, input sequence from 0 to the size of input
     * sorted by the peer output's topo id.} */
    static void GetPeerOutputsOrder(const ge::Node *node, const std::unordered_set<ge::Node *> &allNodes,
                                    std::map<size_t, int64_t> &peerOutputsOrder);

    static void GetAllInnerSuccessorTypes(const std::unordered_set<ge::Node *> &allNodes,
                                          const std::vector<ge::NodePtr> &peerInNodes,
                                          std::vector<string> &successorTypes);

    template<typename Container>
    static bool ContainsDuplicatedType(const Container &allNodes)
    {
        std::set<string> nodeSet;
        for (auto &node : allNodes) {
            if (nodeSet.count(node->GetType()) != 0) {
                return true;
            }
            nodeSet.emplace(node->GetType());
        }
        return false;
    }

    static void SetJsonRange(nlohmann::json &desc, const TbeOpTensor &tensor, const std::vector<int64_t> &currShape,
                             const std::vector<int64_t> &oriShape);
    static void GetRealShapeRange(const bool &hasSet, const std::vector<int64_t> &currShape,
                                  std::vector<std::pair<int64_t, int64_t>> &shapeRange);
    static void SetConstValueToJson(const TbeOpTensor &tensor, nlohmann::json &inputDescJson);
    static void AssembleComipleParams(const std::vector<ge::Node *> &fusionNodes, nlohmann::json &jsonData);
    static void FillOptionalOutputWithNull(const std::vector<ge::Node *> &teGraphNode, nlohmann::json &jsonData);
    static void GetPrebuildOutput(const std::string &nodeName, nlohmann::json &jsonStr);
    static void RefreshSgtSliceShape(nlohmann::json &outputDesc, nlohmann::json &outputDataDesc);
    static void FilterOutputMultipleReference(nlohmann::json &jsonData);
    static void GetAllInputNameVec(nlohmann::json &jsonData, std::vector<std::string> &inputNameVec);
    static void RemoveMultipleReferOutput(const std::vector<std::string> &inputNameVec, nlohmann::json &outputDescVec);
    static void GetUniqueJsonStr(const nlohmann::json &oriJson, std::string &uniqueJsonStr);
    static bool GetJsonDataWithOpList(const nlohmann::json &oriJson, nlohmann::json &jsonData);
};
}
}
#endif  // ATC_OPCOMPILER_TE_FUSION_SOURCE_ASSEMBLE_JSON_TE_JSON_ASSEMBLE_H_
