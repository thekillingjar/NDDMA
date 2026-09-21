/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_ADAPTER_TBE_ADAPTER_TBE_INFO_TBE_SINGLE_OP_INFO_ASSEMBLER_H_
#define FUSION_ENGINE_OPTIMIZER_ADAPTER_TBE_ADAPTER_TBE_INFO_TBE_SINGLE_OP_INFO_ASSEMBLER_H_

#include "adapter/tbe_adapter/tbe_info/tbe_info_assembler.h"
#include "tensor_engine/tbe_op_info.h"

namespace fe {
using TbeInfoAssemblerPtr = std::shared_ptr<fe::TbeInfoAssembler>;
struct OpTensorStruct {
  uint32_t index;
  OpParamType op_param_type;
  bool is_last_dynamic_tensor;
};

const std::map<std::string, ge::GeAttrValue::ValueType> ATTR_STRING_TO_VALUETYPE_MAP{
    {"str", ge::GeAttrValue::VT_STRING},
    {"int", ge::GeAttrValue::VT_INT},
    {"float", ge::GeAttrValue::VT_FLOAT},
    {"bool", ge::GeAttrValue::VT_BOOL},
    {"listStr", ge::GeAttrValue::VT_LIST_STRING},
    {"listInt", ge::GeAttrValue::VT_LIST_INT},
    {"listFloat", ge::GeAttrValue::VT_LIST_FLOAT},
    {"listBool", ge::GeAttrValue::VT_LIST_BOOL},
    {"listListInt", ge::GeAttrValue::VT_LIST_LIST_INT},
    {"tensor", ge::GeAttrValue::VT_TENSOR},
    {"listTensor", ge::GeAttrValue::VT_LIST_TENSOR}};

class TbeSingleOpInfoAssembler {
 public:
  TbeSingleOpInfoAssembler();

  ~TbeSingleOpInfoAssembler();

  Status AssembleSingleTbeInfo(ge::Node *node, te::TbeOpInfo &tbe_op_info, const string &engine_name);
  Status Initialize();
 private:
  TbeInfoAssemblerPtr tbe_info_assembler_ptr_;

  Status FeedInputInfoToSingleTbeInfo(const ge::OpDescPtr &op_desc_ptr,
                                      const vector<OpTensorStruct> op_input_tensor_struct,
                                      te::TbeOpInfo &tbe_op_info);

  Status FeedOutputInfoToSingleTbeInfo(const ge::OpDescPtr &op_desc_ptr,
                                       const vector<OpTensorStruct> op_output_tensor_struct,
                                       te::TbeOpInfo &tbe_op_info);

  Status JudgeShapeToSetFlag(const ge::OpDescPtr &op_desc, const bool &is_input,
                             te::TbeOpInfo &op_info, bool &flag) const;

  /*
   *  @ingroup fe
   *  @brief   set Attrs to single_tbe_op_info
   *  @param   [in]  op              op desc ptr
   *  @param   [in/out]  op_info      tbe data item
   *  @return  SUCCESS or FAILED
   */
  Status FeedAttrsToSingleTbeOpInfo(const ge::OpDescPtr &op_desc_ptr, te::TbeOpInfo &tbe_op_info) const;
  /*
    *  @ingroup fe
    *  @brief   set Attrs:flagint64 to tbe_op_info
    *  @param   [in]  node            input node pointer
    *  @param   [in/out]  op_info      tbe data item
    *  @return  SUCCESS or FAILED
    */
  Status FeedFlagInt64ToTbeOpInfo(const ge::Node *node, te::TbeOpInfo &op_info) const;
};
}  // namespace fe

#endif  // FUSION_ENGINE_OPTIMIZER_ADAPTER_TBE_ADAPTER_TBE_INFO_TBE_SINGLE_OP_INFO_ASSEMBLER_H_
