/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_IR2TF_BASE_PARSER_H_
#define AICPU_IR2TF_BASE_PARSER_H_

#include <mutex>
#include <unordered_map>
#include <vector>
#include "proto/tensorflow/graph.pb.h"
#include "graph/op_desc.h"
#include "graph/compute_graph.h"
#include "graph/detail/attributes_holder.h"
#include "framework/common/ge_inner_error_codes.h"
#include "ir2tf_struct.h"

namespace aicpu {
using NameRangeMap = std::unordered_map<std::string, std::pair<int, int>>;

enum class DefaultType {
  DEFAULT_STRING,
  DEFAULT_INT,
  DEFAULT_FLOAT,
  DEFAULT_BOOL,
  DEFAULT_TYPE,
  DEFAULT_LIST,
  DEFAULT_UNDEFIN
};

class Ir2tfBaseParser {
 public:
  /**
   * @return Single instance of Ir2tfBaseParser
   */
  static std::shared_ptr<Ir2tfBaseParser> Instance();

  /**
   * Convert IR from GE node to TF node_def.
   * @param node
   * @param node_def
   * @return result is success or failed.
   */
  ge::Status ParseNodeDef(const ge::Node &node, domi::tensorflow::NodeDef *node_def);

  /**
   * Convert IR from GE node to TF opDef. Now opDef only contains output_arg
   * @param node Ge node
   * @param op_name Ge op name
   * @param outputs Output index range
   * @return result is success or failed.
   */
  __attribute__((visibility("hidden")))
  ge::Status ParseOutputArgs(const ge::NodePtr &node, const std::string &op_type, NameRangeMap &outputs);

  /**
   * Get ref input's name of an op
   * @param op_type Op type
   * @param ref_set ref input name set
   * @return void
   */
  void GetRefInputSet(const std::string &op_type, std::set<std::string> &ref_set);

  /**
   * Get ref output's name of an op
   * @param op_type Op type
   * @param ref_set ref output name set
   * @return void
   */
  void GetRefOutputSet(const std::string &op_type, std::set<std::string> &ref_set);

  /**
   * Judge whether an op has input with ref data type
   * @param op_type Op type
   * @return whether an op has input with ref data type
   */
  bool InputHaveRef(const std::string &op_type);

  /**
   * @return Single instance of Ir2tfBaseParser
   */
  ge::Status LoadMappingConfig();

  /**
   * Destructor
   */
  virtual ~Ir2tfBaseParser() = default;
  // Copy prohibit
  Ir2tfBaseParser(const Ir2tfBaseParser &ir2tf_base_parser) = delete;
  // Copy prohibit
  Ir2tfBaseParser &operator=(const Ir2tfBaseParser &ir2tf_base_parser) = delete;
  // Move prohibit
  Ir2tfBaseParser(Ir2tfBaseParser &&ir2tf_base_parser) = delete;
  // Move prohibit
  Ir2tfBaseParser &operator=(Ir2tfBaseParser &&ir2tf_base_parser) = delete;

 protected:
  /**
   * Convert attrs from node to node_def. Children class can override this function.
   * @param node
   * @param node_def
   * @return result is success or failed.
   */
  virtual ge::Status ParseAttr(const ge::Node &node, domi::tensorflow::NodeDef *node_def);

 private:
  /**
   * Contructor
   */
  Ir2tfBaseParser() = default;

  /**
   * Get Config file real path.
   * @return config relative path.
   */
  std::string GetConfigFilePath() const;

  /**
   * Convert Inputs from Node to NodeDef.
   * @param node
   * @param node_def
   * @return result is success or failed.
   */
  ge::Status ParseInput(const ge::Node &node, domi::tensorflow::NodeDef *node_def) const;

  /**
  * Parse attrs for IR definition that every field can match.
  * @param node
  * @param node_def
  * @return result is success or failed.
  */
  ge::Status ParseBaseAttr(const ge::Node &node, domi::tensorflow::NodeDef *node_def);

  /**
   * Parse attrs according to some rules defined in configuration file.
   * Take ir2tf_op_mapping_info.proto as reference.
   * @param node
   * @param node_def
   * @param ir2tf Mapping info
   * @return result is success or failed.
   */
  ge::Status ParseConfigAttr(const ge::Node &node, domi::tensorflow::NodeDef *node_def,
                             const OpMapInfo *ir2tf);

  /**
   * Add tf type value and set in.
   * @param node_def Tf node
   * @param ir2tf Mapping info
   * @return result is success or failed.
   */
  ge::Status ParseExtendAttr(const ge::Node &node, domi::tensorflow::NodeDef *node_def,
                             const OpMapInfo *ir2tf) const;

  /**
   * Validate configuration file is valid or not.
   * @param op_map_info Mapping info
   * @return result is success or failed.
   */
  ge::Status ValidateOpMappingConfig(const std::vector<OpMapInfo> &op_map_info_list) const;

  /**
   * Validate configuration attrsMapDesc is valid or not.
   * @param op_map_info Mapping info
   * @return result is success or failed.
   */
  ge::Status ValidateAttrMapConfig(const OpMapInfo &op_map_info) const;

  /**
   * Validate configuration InputAttrMapDesc is valid or not.
   * @param op_map_info Mapping info
   * @return result is success or failed.
   */
  ge::Status ValidateInputAttrMapConfig(const OpMapInfo &op_map_info) const;

  /**
   * Validate configuration outputAttrMapDesc is valid or not.
   * @param op_map_info Mapping info
   * @return result is success or failed.
   */
  ge::Status ValidateOutputAttrMapConfig(const OpMapInfo &op_map_info) const;

  /**
   * Validate configuration outputRefDesc is valid or not.
   * @param op_map_info Op mapping info
   * @return result is success or failed.
   */
  ge::Status ValidateRefTransDesc(const OpMapInfo &op_map_info) const;

  /**
   * Handle attrsMapDesc rules defined in configuration file.
   * @param node
   * @param node_def
   * @param attr_mapping
   * @return result is success or failed.
   */
  ge::Status HandleAttrMapping(const ge::Node &node, domi::tensorflow::NodeDef *node_def,
                               const ParserExpDesc &attr_mapping);

  /**
   * Handle default string attr defined in configuration file.
   * @param attr_value attr value
   * @param default_value_str default string data
   * @param node_name node name
   */
  void SetDefaultString(domi::tensorflow::AttrValue &attr_value,
                        const std::string &default_value_str,
                        const std::string &node_name) const;

  /**
   * Add tf type value and set in.
   * @param node_def Tf node
   * @param attr_default_value attrExtDesc config
   * @param node_name node name
   * @return result is success or failed.
   */
  ge::Status HandleDefaultAttrType(domi::tensorflow::NodeDef *node_def,
                                   const ExtendFieldDesc &attr_default_value,
                                   const std::string &node_name) const;

  /**
   * Handle inputAttrMapDesc rules defined in configuration file.
   * @param node Ge node
   * @param node_def Tensorflow node
   * @param ir2tf Json config
   * @return result is success or failed.
   */
  ge::Status HandleInputAttrMapping(const ge::Node &node, domi::tensorflow::NodeDef *node_def,
                                    const OpMapInfo *ir2tf) const;

  /**
   * Handle outputAttrMapDesc rules defined in configuration file.
   * @param node Ge node
   * @param node_def Tensorflow node
   * @param ir2tf Json config
   * @return result is success or failed.
   */
  ge::Status HandleOutputAttrMapping(const ge::Node &node, domi::tensorflow::NodeDef *node_def,
                                     const OpMapInfo *ir2tf) const;

  /**
    * Get AttrValue from ge value type
    * @param node Ge node
    * @param attr_name  Set attr_value for this attrname
    * @param ge_value_type Value type from ge
    * @param attr_value This func going to fill this param
    * @return result is success or not
    */
  ge::Status GetAttrValueFromGe(const ge::Node &node, const string &attr_name,
                                const ge::GeAttrValue::ValueType ge_value_type,
                                domi::tensorflow::AttrValue &attr_value);

  /**
   * Calc tf tensor value and set in.
   * @param ge_tensor Calc tf tensor from ge tensor
   * @param tf_tensor Calc tf tesor and fill this param
   * @param data_count
   * @return result is success or not
   */
  ge::Status SetTfTensorValue(const ge::ConstGeTensorPtr &ge_tensor,
                              domi::tensorflow::TensorProto *tf_tensor,
                              int32_t data_count) const;

  /**
   * Set input/output dataType list
   * @param node_def Tensorflow node
   * @param data_type_map Input/output datatype list
   * @return result is success or failed.
   */
  __attribute__((visibility("hidden")))
  ge::Status SetAttrTypeList(domi::tensorflow::NodeDef *node_def,
                             std::unordered_map<std::string,
                             std::vector<domi::tensorflow::DataType>> &data_type_map) const;

  /**
   * Add tf list(string) value and set in.
   * @param op_desc_ptr Op desc pointer
   * @param attr_name Attr name
   * @param attr_value Attr value
   * @return result is success or not
   */
  ge::Status SetTfListString(const ge::OpDescPtr &op_desc_ptr,
                             const std::string &attr_name,
                             domi::tensorflow::AttrValue &attr_value) const;

  /**
   * Get srcFieldName and dstFieldName from input/output AttrMapDesc config
   * @param exp_desc_list InputAttrMapDesc or inputAttrMapDesc list
   * @param src_dst_name_map SrcFieldName and dstFieldName map
   */
  __attribute__((visibility("hidden")))
  void GetSrcAndDstFieldName(const std::vector<ParserExpDesc> &exp_desc_list,
                             std::unordered_map<std::string, std::string> &src_dst_name_map) const;

  /**
   * Get output list size
   * @param node Ge node
   * @param dst_name InputAttrMapDesc/outputAttrMapDesc dstFieldName
   * @return output list size
   */
  int GetOutputListSize(const ge::NodePtr &node, const std::string &dest_name) const;

  /**
   * Judge the input/output is list type
   * @param src_dst_field_name_map Josn cofig
   * @param name_idx Input_name/output_name:index
   * @param dst_field_name Input/output dstFieldName
   * @return result is list type or not
   */
  __attribute__((visibility("hidden")))
  bool IsListType(const std::unordered_map<std::string, std::string> &src_dst_field_name_map,
                  const std::string &name_idx,
                  std::string &dst_field_name) const;

  /**
   * Judge the input is list type or not
   * @param name_idx Input_name/output_name:index
   * @param config_name Input/output dstFieldName
   * @return result is list type or not
   */
  bool IsListType(const std::string &name_idx, const std::string &config_name) const;

  /**
   * Get input/output list index
   * @param name_idx Input name and index
   * @param start Index begin
   * @param end Index end
   * @return result is input/output list index
   */
  int GetListIndex(const std::string &name_idx, int start, int end) const;

  /**
   * Get input/output data type
   * @param ge_data_type Ge input/output data type
   * @param dst_attr_name Tf attr name
   * @param data_type_map Input/output data type map
   * @return result is success or failed.
   */
  __attribute__((visibility("hidden")))
  ge::Status GetDataTypeList(const ge::DataType ge_data_type,
                             const std::string &dst_attr_name,
                             std::unordered_map<string,
                             vector<domi::tensorflow::DataType>> &data_type_map) const;

  /**
   * Get single output name range
   * @param ref_trans_list Json config outputRefDesc
   * @param outputs Output name range
   * @param op_map_info op info
   * @param node Ge node
   * @return result is success or failed.
   */
  __attribute__((visibility("hidden")))
  ge::Status GetSingleOutputNameRange(const std::vector<RefTransDesc> &ref_trans_list,
                                      NameRangeMap &outputs,
                                      const aicpu::OpMapInfo &op_map_info,
                                      const ge::NodePtr &node) const;

  /**
   * Get output list name range
   * @param output_attr_list Json config outputAttrMapDesc
   * @param ref_trans_list Json config outputRefDesc
   * @param node Ge node
   * @param outputs Output name range
   * @return result is success or failed.
   */
  __attribute__((visibility("hidden")))
  ge::Status GetListOutputNameRange(const std::vector<ParserExpDesc> &output_attr_list,
                                    const std::vector<RefTransDesc> &ref_trans_list,
                                    const ge::NodePtr &node, NameRangeMap &outputs) const;

  /**
   * Add tf tensor value and set in.
   * @param op_desc_ptr Ge op desc
   * @param attr_name Tf attr name
   * @param attr_value Tf tesor value
   * @param result is success or failed.
   */
  ge::Status SetTfTensorValue(const ge::OpDescPtr &op_desc_ptr,
                              const std::string &attr_name,
                              domi::tensorflow::AttrValue &attr_value) const;

  /**
   * Add tf list(shape) value and set in.
   * @param op_desc_ptr Ge op desc
   * @param attr_name Tf attr name
   * @param attr_value Tf list(shape) value
   * @param result is success or failed.
   */
  ge::Status SetTfListShape(const ge::OpDescPtr &op_desc_ptr,
                            const std::string &attr_name,
                            domi::tensorflow::AttrValue &attr_value) const;

  /**
   * Add tf list(type) value and set in.
   * @param op_desc_ptr Ge op desc
   * @param attr_name Tf attr name
   * @param attr_value Tf list(type) value
   * @param result is success or failed.
   */
  ge::Status SetTfListType(const ge::OpDescPtr &op_desc_ptr,
                           const std::string &attr_name,
                           domi::tensorflow::AttrValue &attr_value) const;

  /**
   * Add tf type value and set in.
   * @param op_desc_ptr Ge op desc
   * @param node_def Tf node
   * @param attr_mapping AttrsMapDesc config
   * @param result is success or failed.
   */
  ge::Status SetTfAttrType(const ge::OpDescPtr &op_desc_ptr,
                           domi::tensorflow::NodeDef *node_def,
                           const ParserExpDesc &attr_mapping);

  /**
   * Add tf shape value and set in.
   * @param op_desc_ptr Ge op desc
   * @param node_def Tf node
   * @param attr_mapping AttrsMapDesc config
   * @param result is success or failed.
   */
  ge::Status SetTfAttrShape(const ge::OpDescPtr &op_desc_ptr,
                            domi::tensorflow::NodeDef *node_def,
                            const ParserExpDesc &attr_mapping) const;

  ge::Status SetTfAttrEnum(const ge::OpDescPtr &op_desc_ptr,
                            domi::tensorflow::NodeDef *node_def,
                            const ParserExpDesc &attr_mapping) const;

  /**
   * Add tf list(int) value and set in.
   * @param node Ge node
   * @param attr_name Tf attr name
   * @param attr_value Tf list(int) value
   * @param result is success or failed.
   */
  ge::Status SetTfListInt(const ge::Node &node,
                          const string &attr_name,
                          domi::tensorflow::AttrValue &attr_value);

  /**
   * Add tf type value and set in.
   * @param op_desc_ptr Ge op desc
   * @param attr_name Tf attr name
   * @param attr_value Tf type value
   * @param result is success or failed.
   */
  ge::Status SetTfType(const ge::OpDescPtr &op_desc_ptr,
                       const string &attr_name,
                       domi::tensorflow::AttrValue &attr_value) const;

  /**
   * Get op blacklist from json config.
   * @param op_type Op type
   * @param attr_blacklist Json config attrs blacklist
   */
  void GetOpBlacklist(std::string op_type, std::set<std::string> &attr_blacklist);
 private:
  // map to store ir2tf config
  std::map<std::string, OpMapInfo> ir2tf_map_;
  std::unordered_map<std::string, std::set<std::string>> ref_input_map_;
  std::unordered_map<std::string, std::set<std::string>> ref_output_map_;
  bool is_loaded_ = false;
};
} // namespace aicpu
#endif // AICPU_IR2TF_BASE_PARSER_H_