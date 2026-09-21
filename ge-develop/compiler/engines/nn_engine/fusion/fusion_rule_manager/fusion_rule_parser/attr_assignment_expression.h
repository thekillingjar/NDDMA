/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_FUSION_RULE_MANAGER_FUSION_RULE_PARSER_ATTR_ASSIGNMENT_EXPRESSION_H_
#define FUSION_ENGINE_FUSION_FUSION_RULE_MANAGER_FUSION_RULE_PARSER_ATTR_ASSIGNMENT_EXPRESSION_H_

#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include "common/fe_utils.h"
#include "fusion_rule_manager/fusion_rule_data/fusion_rule_pattern.h"

namespace fe {

using FusionRuleAttrValuePtr = std::shared_ptr<FusionRuleAttrValue>;

/** @brief provide attribute assginment expression parse methods, and get
*        attribute's info from OpKernleInfoStore */
class AttrAssignmentExpression {
 public:
  AttrAssignmentExpression()
      : VALUE("value"),
        EXPR("expr"),
        OPERATOR_ASSIGN("="),
        BOOL_TRUE("true"),
        BOOL_FALSE("false"),
        parse_to_attr_success_(false) {}

  ~AttrAssignmentExpression() {}

  AttrAssignmentExpression(const AttrAssignmentExpression &) = delete;

  AttrAssignmentExpression &operator=(const AttrAssignmentExpression &) = delete;
  /*
   * @brief: Step 1: parse basic info ("attr", "expr") from json
   *         attr assignment expression json format:
   *                      {
   *                          "attr":"node_name.attr_name",
   *                          "expr":"=",
   *                          "value":"node_name.attr_name"/ "18.12"
   *                      }
   */
  Status ParseToJsonAttr(const nlohmann::json &json_object);

  Status ParseJson(const nlohmann::json &json_object);
  /*
   * @brief: Step 2: parse r_value of assignment expression from json, according
   * to whole rule info
   */
  Status ParseToAttrValue(const std::map<string, std::vector<string>> &node_map);

  const FusionRuleAttr &GetAttr() const { return attr_; }

  FusionRuleAttrValuePtr GetValue() const { return value_; }

 private:
  /*
   * @brief: Get ge::GeAttrValue::ValueType of "attr" from OpKernelInfoStore
   */
  Status GetValueType(const std::map<string, std::vector<string>> &node_map, const FusionRuleAttr &attr,
                      ge::GeAttrValue::ValueType &value_type) const;
  /*
   * @brief: Convert fixed value or relected value of r_value from json
   */
  Status ConvertToAttrValue(const nlohmann::json &json_object, ge::GeAttrValue::ValueType value_type,
                            FusionRuleAttrValuePtr value);
  /*
   * @brief: Get fixed list int64_t value from json
   */
  Status GetStrAndConvert(const nlohmann::json &json_object, std::vector<int64_t> &value,
                          FusionRuleAttrValuePtr attr_value) const;
  /*
   * @brief: Get fixed list float value from json
   */
  Status GetStrAndConvert(const nlohmann::json &json_object, std::vector<float> &value,
                          FusionRuleAttrValuePtr attr_value) const;
  /*
   * @brief: Get fixed list bool value from json
   */
  Status GetStrAndConvert(const nlohmann::json &json_object, std::vector<bool> &value,
                          FusionRuleAttrValuePtr attr_value);
  /*
   * @brief: Get fixed list string value from json
   */
  Status GetStrAndConvert(const nlohmann::json &json_object, std::vector<std::string> &value,
                          const FusionRuleAttrValuePtr &attr_value) const;
  /*
   * @brief: Get fixed int64_t value from json
   */
  Status GetStrAndConvert(const nlohmann::json &json_object, int64_t &value, FusionRuleAttrValuePtr attr_value) const;
  /*
   * @brief: Get fixed list float value from json
   */
  Status GetStrAndConvert(const nlohmann::json &json_object, float &value, FusionRuleAttrValuePtr attr_value) const;
  /*
   * @brief: Get fixed bool value from json
   */
  Status GetStrAndConvert(const nlohmann::json &json_object, bool &value, FusionRuleAttrValuePtr attr_value);
  /*
   * @brief: Get fixed string value from json
   */
  Status GetStrAndConvert(const nlohmann::json &json_object, std::string &value, FusionRuleAttrValuePtr attr_value) const;

  const std::string ATTR = "attr";
  const std::string VALUE;
  const std::string EXPR;
  const std::string OPERATOR_ASSIGN;
  const std::string BOOL_TRUE;
  const std::string BOOL_FALSE;

  bool parse_to_attr_success_;

  FusionRuleAttr attr_;

  FusionRuleAttrValuePtr value_;

  nlohmann::json tmp_value_;
};

}  // namespace fe
#endif  // FUSION_ENGINE_FUSION_FUSION_RULE_MANAGER_FUSION_RULE_PARSER_ATTR_ASSIGNMENT_EXPRESSION_H_
