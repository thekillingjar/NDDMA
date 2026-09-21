/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_TE_FUSION_SOURCE_BINARY_GENERATE_SIMPLE_KEY_H_
#define ATC_OPCOMPILER_TE_FUSION_SOURCE_BINARY_GENERATE_SIMPLE_KEY_H_

#include <string>
#include "tbe_op_info.h"
#include "binary/fusion_binary_info.h"
#include "graph/operator.h"

namespace te {
namespace fusion {

class GenerateSimpleKey {
public:
    GenerateSimpleKey(std::string opType, SimpleKeyModeType simpleKeyMode,
        std::string implMode, std::string deterministic, bool isUnknowShape)
        : opType_(opType), simpleKeyMode_(simpleKeyMode), implMode_(implMode), deterministic_(deterministic),
        isUnknowShape_(isUnknowShape) {}
    GenerateSimpleKey(const GenerateSimpleKey&) = delete;
    GenerateSimpleKey& operator=(const GenerateSimpleKey&) = delete;
    ~GenerateSimpleKey() {}

    bool GenerateSimpleKeyStr(std::string &simpleKeystr);
    void SetInputs(const std::vector<TbeOpParam> &inputs);
    void SetOutputs(const std::vector<TbeOpParam> &outputs);
    void SetAttrs(const std::vector<TbeAttrValue> &attrs);
    void SetBinaryInfoPtr(const BinaryInfoBasePtr binaryInfo);
    void SetOpInfoPtr(const TbeOpInfoPtr &opInfo);

private:
    std::string GenerateImplMode(const std::string implMode) const;
    bool GenerateInOutPut(const std::vector<TbeOpParam> &inOutPuts, const std::string &putsType,
                          std::string &inOutPutStr) const;
    bool FormatNormalize(const int index, const DtypeFormatMode &mode, std::string &format) const;
    bool DtypeNormalize(const int index, const DtypeFormatMode &mode, std::string &dtype) const;
    bool Normalization(const int index, std::string putsType, std::string &format, std::string &dtype) const;
    bool MatchInOutPutCnt(const std::vector<TbeOpParam> &inOutPuts, const std::string &putsType) const;
    std::string GenerateDetermin(const std::string deterministic) const;
    std::string GenerateAttrs(const std::vector<TbeAttrValue> &attrValues) const;
    bool CheckParamSetDefaultVal();
    bool GenerateCustomizeSimplifiedKey(std::string &simpleKeystr) const;

    const std::string opType_;
    const SimpleKeyModeType simpleKeyMode_;
    std::string implMode_;
    std::string deterministic_;
    bool isUnknowShape_;
    std::vector<TbeOpParam> inputs_;
    std::vector<TbeOpParam> outputs_;
    std::vector<TbeAttrValue> attrs_;
    BinaryInfoBasePtr binaryInfo_{nullptr};
    TbeOpInfoPtr opInfo_{nullptr};
};
} // namespace fusion
} // namespace te
#endif

