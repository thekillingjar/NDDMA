/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACL_OP_EXEC_OP_MODEL_PARSER_H_
#define ACL_OP_EXEC_OP_MODEL_PARSER_H_

#include "single_op/acl_op_resource_manager.h"

#include "graph/model.h"

namespace acl {
class OpModelParser {
public:
    static aclError ParseOpModel(OpModel &opModel, OpModelDef &modelDef);

private:
    static aclError DeserializeModel(const OpModel &opModel, ge::Model &model);

    static aclError ToModelConfig(ge::Model &model, OpModelDef &modelDef);

    static aclError ParseModelContent(const OpModel &opModel, uint64_t &modelSize, uint8_t *&modelData);

    static aclError ParseGeTensorDesc(std::vector<ge::GeTensorDesc> &geTensorDescList,
                                      std::vector<aclTensorDesc> &output, const std::string &opType);

    static void ParseOpAttrs(const ge::Model &model, aclopAttr &attrs);
};
} // namespace acl

#endif // ACL_OP_EXEC_OP_MODEL_PARSER_H_
