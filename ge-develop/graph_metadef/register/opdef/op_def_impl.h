/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_DEF_IMPL_H
#define OP_DEF_IMPL_H

#include "register/op_def.h"
#include "register/op_impl_registry.h"
#include "register/op_check_register.h"
#include "graph/operator_reg.h"

namespace ops {
enum ListParamStatus : int32_t {
  UNSET = 0,
  LIST = 1,
  NON_LIST = 2,
};
class OpParamDefImpl {
public:
  ge::AscendString name;
  Option param_type = Option::REQUIRED;
  std::vector<ge::DataType> types;
  std::vector<ge::DataType> origin_types;
  std::vector<ge::DataType> types_list;
  std::vector<ge::Format> formats;
  std::vector<ge::Format> formats_list;
  std::vector<ge::DataType> types_for_bin;
  std::vector<ge::Format> formats_for_bin;
  ListParamStatus types_status = UNSET;
  ListParamStatus formats_status = UNSET;
  ge::AscendString need_compile = "";
  ge::AscendString reshape_type = "";
  ge::AscendString value_depend = "";
  DependScope depend_scope = DependScope::ALL;
  std::vector<ge::Format> unknown_shape_formats;
  bool ignore_contiguous = false;
  bool auto_contiguous = false;
  bool is_scalar = false;
  bool is_scalar_list = false;
  bool set_type_for_bin = false;
  bool set_format_for_bin = false;
  ge::AscendString scalar_name = "";
  ge::DataType scalar_type = ge::DT_UNDEFINED;
  uint32_t version{0};
  InitValueType init_value_type = InitValueType::INIT_VALUE_DEFAULT;
  InitValueNum init_value;
  std::vector<ScalarVar> init_value_list;
  bool is_output_shape_depend_on_compute = false;
  ge::AscendString follow_port_name = "";
  FollowType follow_type = FollowType::INVALID_TYPE;
  ge::AscendString comment = "";
};

class OpParamTrunk {
public:
  OpParamDef &Input(const char *name);
  OpParamDef &Output(const char *name);
  std::vector<OpParamDef> &GetInputs(void);
  std::vector<OpParamDef> &GetOutputs(void);

private:
  friend class OpDef;
  friend class OpProtoGenerator;

  ItemFindStatus ParamFind(const char *name, bool is_output, OpParamDef **param);
  OpParamDef &ParamAdd(OpParamDef &param, bool is_output);
  OpParamDef &ParamGetOrCreate(const char *name, bool is_output);
  OpParamDef &GetParamDef(const ge::AscendString& name, OpDef::PortStat stat);
  void FollowMapUpdate(OpParamDef &param, bool is_output);
  void FollowDataImpl(void);
  void DfsFollow(OpParamDef& op_param_def, OpDef::PortStat stat);
  void ParamFollow(OpParamDef &op_param_def, OpParamDef &target_param, OpDef::PortStat stat);
  void FollowListDataImpl(const OpDef::DfsParam &dfs_param, std::vector<OpParamDef> &input,
                          std::vector<OpParamDef> &output);
  std::map<ge::AscendString, OpDef::PortFollowInfo> GetFollowMap(void);
  std::map<ge::AscendString, std::vector<std::pair<ge::AscendString, OpDef::PortStat>>> GetShapeMap(void);
  std::map<ge::AscendString, std::vector<std::pair<ge::AscendString, OpDef::PortStat>>> GetDtypeMap(void);
  bool follow_isimpl = false;
  std::vector<OpParamDef> inputs_;
  std::vector<OpParamDef> outputs_;
  std::map<ge::AscendString, OpDef::PortFollowInfo> follow_map;
  std::vector<std::pair<ge::AscendString, OpDef::PortStat>> follow_dtypelist;
  std::vector<std::pair<ge::AscendString, OpDef::PortStat>> follow_formatlist;
  std::map<ge::AscendString, std::vector<std::pair<ge::AscendString, OpDef::PortStat>>> follow_shape_map;
  std::map<ge::AscendString, std::vector<std::pair<ge::AscendString, OpDef::PortStat>>> follow_dtype_map;
};

class OpAttrDefImpl {
public:
  ge::AscendString name;
  AttrDataType data_type = AttrDataType::ATTR_DT_BOOL;
  bool required = true;
  bool bool_value = false;
  float float_value = 0;
  int64_t int_value = 0;
  ge::AscendString str_value = "";
  std::vector<bool> list_bool = {};
  std::vector<float> list_float = {};
  std::vector<int64_t> list_int = {};
  std::vector<std::vector<int64_t>> list_list_int = {};
  ge::AscendString value = "";
  uint32_t version = 0;
  ge::AscendString comment = "";
};

class OpAICoreConfigImpl {
public:
  OpParamTrunk op_params;
  std::vector<ge::AscendString> cfg_keys;
  std::map<ge::AscendString, ge::AscendString> cfg_info;
};

class OpAICoreDefImpl {
public:
  gert::OpImplRegisterV2::TilingKernelFunc tiling_func = nullptr;
  gert::OpImplRegisterV2::TilingParseFunc tiling_parse = nullptr;
  gert::OpImplRegisterV2::CompileInfoCreatorFunc ci_creator = nullptr;
  gert::OpImplRegisterV2::CompileInfoDeleterFunc ci_deleter = nullptr;
  optiling::OP_CHECK_FUNC op_chk_support = nullptr;
  optiling::OP_CHECK_FUNC op_sel_format = nullptr;
  optiling::OP_CHECK_FUNC op_get_support = nullptr;
  optiling::OP_CHECK_FUNC op_get_spec = nullptr;
  optiling::PARAM_GENERALIZE_FUNC op_generlize_func = nullptr;
  bool zero_ele_output_launch_flag = false;
  std::map<ge::AscendString, OpAICoreConfig> aicore_configs = {};
};

class OpAICPUDefImpl {
public:
  std::vector<ge::AscendString> cfg_keys;
  std::map<ge::AscendString, ge::AscendString> cfg_info;
};

class OpHostCPUDefImpl {
public:
  std::vector<ge::AscendString> cfg_keys;
  std::map<ge::AscendString, ge::AscendString> cfg_info;
};

class OpMC2DefImpl {
public:
  std::vector<ge::AscendString> group_list = {};
  std::map<ge::AscendString, HcclServerType> server_type_ = {};
};

class OpDefImpl {
public:
  gert::OpImplRegisterV2::InferShapeKernelFunc infer_shape = nullptr;
  gert::OpImplRegisterV2::InferShapeRangeKernelFunc infer_shape_range = nullptr;
  gert::OpImplRegisterV2::InferDataTypeKernelFunc infer_data_type = nullptr;
  OpParamTrunk op_params;
  std::vector<OpAttrDef> attrs;
  OpAICoreDef op_aicore;
  OpAICPUDef op_aicpu;
  OpHostCPUDef op_hostcpu;
  ge::AscendString op_type;
  ge::AscendString category = "op_proto";
  std::map<ops::CommentSection, std::vector<ge::AscendString>> comment_map = {
    {ops::CommentSection::BRIEF, {}},
    {ops::CommentSection::CONSTRAINTS, {}},
    {ops::CommentSection::RESTRICTIONS, {}},
    {ops::CommentSection::SEE, {}},
    {ops::CommentSection::THIRDPARTYFWKCOMPAT, {}}
  };
  bool has_workspace = true;
  uint32_t non_list_len = 0;
  OpMC2Def op_mc2;
  FormatCheckOption format_mode = FormatCheckOption::MAX;
  bool enable_fall_back = false;
};
}  // namespace ops

#endif
