/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "parser/common/op_registration_tbe.h"
#include <map>
#include <memory>
#include <string>
#include "parser/common/acl_graph_parser_util.h"
#include "common/op_map.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/type_utils_inner.h"
#include "parser/common/op_parser_factory.h"
#include "parser/tensorflow/tensorflow_custom_parser_adapter.h"
#include "parser/tensorflow/tensorflow_fusion_custom_parser_adapter.h"
#include "base/err_msg.h"

namespace ge {
namespace {
std::string GetOmOptype(const OpRegistrationData &reg_data) {
  AscendString om_op_type;
  (void)reg_data.GetOmOptype(om_op_type);
  return om_op_type.GetString() == nullptr ? "" : std::string(om_op_type.GetString());
}
}
using PARSER_CREATOR_FN = std::function<std::shared_ptr<OpParser>(void)>;

FMK_FUNC_HOST_VISIBILITY OpRegistrationTbe *OpRegistrationTbe::Instance() {
  static OpRegistrationTbe instance;
  return &instance;
}

bool OpRegistrationTbe::Finalize(const OpRegistrationData &reg_data, bool is_train, bool is_custom_op) {
  static std::map<domi::FrameworkType, std::map<std::string, std::string> *> op_map = {{domi::CAFFE, &caffe_op_map}};
  if (is_train) {
      op_map[domi::TENSORFLOW] = &tensorflow_train_op_map;
  } else {
      op_map[domi::TENSORFLOW] = &tensorflow_op_map;
  }

  if (op_map.find(reg_data.GetFrameworkType()) != op_map.end()) {
    std::map<std::string, std::string> *fmk_op_map = op_map[reg_data.GetFrameworkType()];
    std::set<AscendString> ori_optype_set;
    (void)reg_data.GetOriginOpTypeSet(ori_optype_set);
    for (auto &tmp : ori_optype_set) {
      if (tmp.GetString() == nullptr) {
        continue;
      }
      if (((*fmk_op_map).find(tmp.GetString()) != (*fmk_op_map).end()) && !is_custom_op) {
        GELOGW("Op type does not need to be changed, om_optype:%s, orignal type:%s.", (*fmk_op_map)[tmp.GetString()].c_str(), tmp.GetString());
        continue;
      } else {
        (*fmk_op_map)[tmp.GetString()] = GetOmOptype(reg_data);
        GELOGD("First register in parser initialize, original type: %s, om_optype: %s, imply type: %s.",
        tmp.GetString(), GetOmOptype(reg_data).c_str(),
        TypeUtilsInner::ImplyTypeToSerialString(reg_data.GetImplyType()).c_str());
      }
    }
  }
  bool ret = RegisterParser(reg_data, is_custom_op);
  return ret;
}

bool OpRegistrationTbe::RegisterParser(const OpRegistrationData &reg_data, bool is_custom_op) const {
  if (reg_data.GetFrameworkType() == domi::TENSORFLOW) {
    std::shared_ptr<OpParserFactory> factory = OpParserFactory::Instance(domi::TENSORFLOW);
    if (factory == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Get OpParserFactory failed.");
      GELOGE(INTERNAL_ERROR, "[Get][OpParserFactory] for tf failed.");
      return false;
    }
    if (reg_data.GetParseParamFn() != nullptr || reg_data.GetParseParamByOperatorFn() != nullptr) {
      bool is_registed = factory->OpParserIsRegistered(GetOmOptype(reg_data));
      if (is_registed && !is_custom_op) {
        GELOGW("Parse param func has already register for op:%s.", GetOmOptype(reg_data).c_str());
        return false;
      }
      std::shared_ptr<TensorFlowCustomParserAdapter> tf_parser_adapter =
          ge::parser::MakeShared<TensorFlowCustomParserAdapter>();
      if (tf_parser_adapter == nullptr) {
        REPORT_INNER_ERR_MSG("E19999", "Create TensorFlowCustomParserAdapter failed.");
        GELOGE(PARAM_INVALID, "[Create][TensorFlowCustomParserAdapter] failed.");
        return false;
      }
      OpParserRegisterar registerar __attribute__((unused)) = OpParserRegisterar(
          domi::TENSORFLOW, GetOmOptype(reg_data), [tf_parser_adapter]() -> std::shared_ptr<OpParser>
          { return tf_parser_adapter; });
    }
    if (reg_data.GetFusionParseParamFn() != nullptr || reg_data.GetFusionParseParamByOpFn() != nullptr) {
      bool is_registed = factory->OpParserIsRegistered(GetOmOptype(reg_data), true);
      if (is_registed) {
        GELOGW("Parse param func has already register for fusion op:%s.", GetOmOptype(reg_data).c_str());
        return false;
      }
      GELOGI("Register fusion custom op parser: %s", GetOmOptype(reg_data).c_str());
      std::shared_ptr<TensorFlowFusionCustomParserAdapter> tf_fusion_parser_adapter =
          ge::parser::MakeShared<TensorFlowFusionCustomParserAdapter>();
      if (tf_fusion_parser_adapter == nullptr) {
        REPORT_INNER_ERR_MSG("E19999", "Create TensorFlowFusionCustomParserAdapter failed.");
        GELOGE(PARAM_INVALID, "[Create][TensorFlowFusionCustomParserAdapter] failed.");
        return false;
      }
      OpParserRegisterar registerar __attribute__((unused)) = OpParserRegisterar(
          domi::TENSORFLOW, GetOmOptype(reg_data),
          [tf_fusion_parser_adapter]() -> std::shared_ptr<OpParser> { return tf_fusion_parser_adapter; }, true);
    }
  } else {
    std::shared_ptr<OpParserFactory> factory = OpParserFactory::Instance(reg_data.GetFrameworkType());
    if (factory == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Get OpParserFactory for %s failed.",
                        TypeUtilsInner::FmkTypeToSerialString(reg_data.GetFrameworkType()).c_str());
      GELOGE(INTERNAL_ERROR, "[Get][OpParserFactory] for %s failed.",
             TypeUtilsInner::FmkTypeToSerialString(reg_data.GetFrameworkType()).c_str());
      return false;
    }
    bool is_registed = factory->OpParserIsRegistered(GetOmOptype(reg_data));
    if (is_registed) {
      GELOGW("Parse param func has already register for op:%s.", GetOmOptype(reg_data).c_str());
      return false;
    }

    PARSER_CREATOR_FN func = CustomParserAdapterRegistry::Instance()->GetCreateFunc(reg_data.GetFrameworkType());
    if (func == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Get custom parser adapter failed for fmk type %s.",
                        TypeUtilsInner::FmkTypeToSerialString(reg_data.GetFrameworkType()).c_str());
      GELOGE(INTERNAL_ERROR, "[Get][CustomParserAdapter] failed for fmk type %s.",
             TypeUtilsInner::FmkTypeToSerialString(reg_data.GetFrameworkType()).c_str());
      return false;
    }
    OpParserFactory::Instance(reg_data.GetFrameworkType())->RegisterCreator(GetOmOptype(reg_data), func);
    GELOGD("Register custom parser adapter for op %s of fmk type %s success.", GetOmOptype(reg_data).c_str(),
           TypeUtilsInner::FmkTypeToSerialString(reg_data.GetFrameworkType()).c_str());
  }
  return true;
}
}  // namespace ge
