/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_INC_OPS_STORE_OP_KERNEL_INFO_CONSTRUCTOR_H_
#define FUSION_ENGINE_INC_OPS_STORE_OP_KERNEL_INFO_CONSTRUCTOR_H_
#include <map>
#include <memory>
#include <string>
#include "graph_optimizer/graph_optimize_register_error_codes.h"
#include "ops_store/op_kernel_info.h"

namespace fe {
struct OpContent {
  // optype
  std::string op_type_;
  std::string ops_path_name_prefix_;

  // opType content map
  std::map<std::string, std::map<std::string, std::string>> map_kernel_info_;
};

void GetIntputAndOutputFlag(const OpContent &op_content, vector<string> &input_vec, vector<string> &output_vec,
                            bool &input_found, bool &output_found);
bool RemoveHeavyFormatSupportWhenC0Change(vector<vector<ge::DataType> *>& all_support_dtypes,
                                          vector<vector<ge::Format> *>& all_support_format);
Status ConvertStr2VecNumber(const string &attr_str, vector<float> &temp_plate_vec);
Status ConvertStr2VecNumber(const string &attr_str, vector<string> &temp_plate_vec);

class OpKernelInfoConstructor {
 public:
  OpKernelInfoConstructor();
  ~OpKernelInfoConstructor();
  Status InitializeOpKernelInfo(const std::string &engine_name, const OpContent &op_content,
                                OpKernelInfoPtr op_kernel_info);
  Status FinalizeOpKernelInfo(OpKernelInfoPtr op_kernel_info) const;

  Status GetStrFromOpContent(const OpContent &op_content, const std::string &key1,
                             const std::string &key2, std::string &value) const;

 private:
  void ParseOpOutputInplaceAbility(const OpContent &op_content,
                                   OpKernelInfoPtr &op_kernel_info) const;

  Status ParseBasicParameter(const OpContent &op_content, OpKernelInfoPtr op_kernel_info) const;

  Status ParseInputAndOutputFromOpContent(const OpContent &op_content, OpKernelInfoPtr op_kernel_info);

  bool RemoveUnknownShapeAndShapeHeavyFormatSupportWhenC0Change(vector<InputOrOutputInfoPtr> &inputs,
                                                                vector<InputOrOutputInfoPtr> &outputs) const;
  void LogSupportFormatAndType(const char *message, OpKernelInfoPtr op_kernel_info);

  Status InitFormatAndDtypeForSingleInputAndOutput(OpPattern op_pattren, const std::map<string, string> &map_info,
                                                   const InputOrOutputInfoPtr &input_or_output_info,
                                                   OpKernelInfoPtr op_kernel_info,
                                                   uint32_t &dtype_and_format_size_of_first_input);

  Status InitializeInputAndOutput(OpPattern op_pattren, const std::string &op_type,
                                  const std::map<std::string, std::string> &map_info, uint32_t index,
                                  InputOrOutputInfoPtr input_or_output_info,
                                  uint32_t &dtype_and_format_size_of_first_input, OpKernelInfoPtr op_kernel_info);
  Status FinalizeInputAndOutput(InputOrOutputInfoPtr input_or_output_info) const;

  Status InitDtypeAndFormat(const std::map<string, string> &map_info, InputOrOutputInfoPtr input_or_output_info,
                            uint32_t &dtype_and_format_size_of_first_input);

  Status ParseFallbackFlags(const OpContent &op_content, const string &fallback_key,
                            std::vector<bool> &cfg_fallbacks) const;

  Status InitFallback(const OpContent &op_content, const OpKernelInfoPtr &op_kernel_info) const;

  Status InitUnknownFormatAndDtype(const std::map<string, string> &map_info, InputOrOutputInfoPtr input_or_output_info,
                                   uint32_t &dtype_and_format_size_of_first_input);

  Status InitDtypeAndAllFormat(const std::map<string, string> &map_info, InputOrOutputInfoPtr input_or_output_info,
                               uint32_t &dtype_of_first_input, OpKernelInfoPtr op_kernel_info);

  Status InitDtype(const std::map<string, string> &map_info, InputOrOutputInfoPtr input_or_output_info,
                   uint32_t &dtype_and_format_size_of_first_input);

  Status GetStrFromMap(const std::map<std::string, std::string> &map_info, std::string key, std::string &value) const;

  /* Convert listed attribute value from a long string to a 2D-Vector */
  template <typename T>
  Status ConvertListAttrValue(const OpContent &op_content, const std::string &attr_name, const std::string &key_name,
                              std::vector<std::vector<T>> &list_attr_vec) const;

  template <typename T>
  Status FillUpListAttrValue(const OpContent &op_content, const std::string &attr_name,
                             std::vector<std::vector<T>> &list_attr_vec, const AttrInfoPtr &attr_info) const;

  template <typename T>
  Status ConvertListListAttrValue(const OpContent &op_content, const std::string &attr_name,
                                  const std::string &key_name, vector<vector<vector<T>>> &list_list_attr_vec) const;

  template <typename T>
  Status FillUpListListAttrValue(const OpContent &op_content, const std::string &attr_name,
                                 vector<vector<vector<T>>> &list_list_attr_vec, const AttrInfoPtr &attr_info) const;

  template <typename T>
  Status InitAttrTemplate(const OpContent &op_content, const std::string &attr, const AttrInfoPtr &attr_info) const;

  template <typename T>
  Status InitAttrListTemplate(const OpContent &op_content, const std::string &attr, const AttrInfoPtr &attr_info) const;

  template <typename T>
  Status InitAttrListListTemplate(const OpContent &op_content, const std::string &attr,
                                  const AttrInfoPtr &attr_info) const;

  Status InitAttrValue(const OpContent &op_content, OpKernelInfoPtr op_kernel_info);
  Status InitAttrValueSub(const OpContent &op_content, OpKernelInfoPtr op_kernel_info);

  Status InitOpStrItem(const std::string &item_name, const std::string &item_key, const OpContent &op_content,
                       const std::set<std::string> &value_range, std::string &value) const;

  Status InitOpStrItem(const std::string &item_name, const std::string &item_key, const OpContent &op_content,
                       std::string &value) const;

  template <typename T>
  Status InitOpItemValueByMaping(const std::string &item_name, const std::string &item_key, const OpContent &op_content,
                                 const std::map<std::string, T> &str_item_map, T &value) const;

  Status InitOpInfo(const std::string &engine_name, const OpContent &op_content,
                    OpKernelInfoPtr op_kernel_info) const;

  Status InitShape(const string &op_type, const std::map<string, string> &map_info,
                   InputOrOutputInfoPtr input_or_output_info);

  Status CheckFormatAgnosticOp(OpKernelInfoPtr op_kernel_info) const;

  void SetUniqueName(InputOrOutputInfoPtr input_or_output_info_ptr) const;

  Status InitDtypeByPattern(const std::map<string, string> &map_info, InputOrOutputInfoPtr input_or_output_info,
                            uint32_t &dtype_and_format_size_of_first_input, const OpPattern &op_pattern);

  void ParserReferTensorNameVec(OpKernelInfoPtr op_kernel_info) const;

  Status InitTuneFormatSwitch(const map<string, string> &map_info, InputOrOutputInfoPtr &input_or_output_info,
                              const string &op_type);

  void ParsePromoteStr(OpKernelInfoPtr op_kernel_info) const;
};

}  // namespace fe

#endif  // FUSION_ENGINE_INC_OPS_STORE_OP_KERNEL_INFO_CONSTRUCTOR_H_
