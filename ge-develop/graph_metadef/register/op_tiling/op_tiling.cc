/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_tiling.h"

#include <nlohmann/json.hpp>
#include "graph/operator.h"
#include "framework/common/debug/ge_log.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/anchor_utils.h"
#include "op_tiling/op_tiling_constants.h"
#include "op_tiling/op_tiling_utils.h"
#include "op_tiling/op_compile_info_manager.h"
#include "common/sgt_slice_type.h"
#include "graph/def_types.h"
#include "graph/utils/node_utils_ex.h"

namespace optiling {
using Status = domi::Status;
using DataBuf = std::tuple<const uint8_t *, size_t>;

class AnyValueBase {
public:
  virtual ~AnyValueBase() = default;
  virtual DataBuf GetDataBuf() const = 0;
};

template<typename VT>
class AnyValue : public AnyValueBase {
public:
  explicit AnyValue(const VT &value) : value_(value) {}
  virtual ~AnyValue() override = default;
  virtual DataBuf GetDataBuf() const override {
    return DataBuf(reinterpret_cast<const uint8_t *>(&value_), sizeof(value_));
  }

private:
  VT value_;
};

template<typename VT>
class AnyVecValue : public AnyValueBase {
public:
  explicit AnyVecValue(const std::vector<VT> &value) : value_(std::move(value)) {}
  virtual ~AnyVecValue() override = default;
  virtual DataBuf GetDataBuf() const override {
    return DataBuf(reinterpret_cast<const uint8_t *>(value_.data()), sizeof(VT) * value_.size());
  }

private:
  std::vector<VT> value_;
};

template<typename T, typename Enabled = void>
struct Getter;

template<typename T>
struct Getter<T, typename std::enable_if<std::is_integral<T>::value>::type> {
  using ST = int64_t;
  static constexpr bool (*func)(ge::AttrUtils::ConstAttrHolderAdapter &&, const string &,
                                int64_t &) = ge::AttrUtils::GetInt;
  static constexpr bool (*list_func)(ge::AttrUtils::ConstAttrHolderAdapter &&, const string &,
                                     vector<int64_t> &) = ge::AttrUtils::GetListInt;
};
template<typename T>
struct Getter<T, typename std::enable_if<std::is_floating_point<T>::value>::type> {
  using ST = float;
  static constexpr bool (*func)(ge::AttrUtils::ConstAttrHolderAdapter &&, const string &,
                                float &) = ge::AttrUtils::GetFloat;
  static constexpr bool (*list_func)(ge::AttrUtils::ConstAttrHolderAdapter &&, const string &,
                                     vector<float> &) = ge::AttrUtils::GetListFloat;
};

class TeOpVarAttrArgsImpl {
public:
  explicit TeOpVarAttrArgsImpl(const ge::OpDescPtr &op_desc) : op_desc_(op_desc){};
  ~TeOpVarAttrArgsImpl() = default;

  Status GetDataByName(const string &name, const string &dtype, DataBuf &data);

private:
  template<typename T>
  Status GetNodeAttrDataIntListList(const std::string &name, DataBuf &data) {
    std::vector<std::vector<int64_t>> value;
    const bool res = ge::AttrUtils::GetListListInt(op_desc_, name, value);
    if (!res) {
      GE_LOGE("Attribute not found: %s", name.c_str());
      return domi::FAILED;
    }

    std::vector<T> dest;
    for (const auto &vec : value) {
      for (const auto elem : vec) {
        dest.emplace_back(static_cast<T>(elem));
      }
    }
    const auto dest_ptr = std::make_shared<AnyVecValue<T>>(dest);
    data_map_.emplace(name + '_' + typeid(T).name(), dest_ptr);
    data = dest_ptr->GetDataBuf();
    GELOGI("IntListList Attribute found: %s", name.c_str());
    return domi::SUCCESS;
  }

  template<typename T, bool IsList = false, typename std::enable_if<!IsList, bool>::type = true>
  Status GetNodeAttrDataTmpl(const std::string &name, DataBuf &data) {
    const auto func = Getter<T>::func;
    typename Getter<T>::ST value;
    const bool res = func(op_desc_, name, value);
    if (!res) {
      GE_LOGE("Attribute not found: %s", name.c_str());
      return domi::FAILED;
    }

    const auto dest_ptr = std::make_shared<AnyValue<T>>(static_cast<T>(value));
    (void)data_map_.emplace(name + '_' + typeid(T).name(), dest_ptr);
    data = dest_ptr->GetDataBuf();
    GELOGI("Single Attribute found: %s", name.c_str());
    return domi::SUCCESS;
  }

  template<typename T, bool IsList = false, typename std::enable_if<IsList, bool>::type = true>
  Status GetNodeAttrDataTmpl(const std::string &name, DataBuf &data) {
    const auto func = Getter<T>::list_func;
    std::vector<typename Getter<T>::ST> value;
    const bool res = func(op_desc_, name, value);
    if (!res) {
      GE_LOGE("List Attribute missing: %s", name.c_str());
      return domi::FAILED;
    }

    std::vector<T> dest;
    for (const auto elem : value) {
      (void) dest.emplace_back(static_cast<T>(elem));
    }
    const auto dest_ptr = std::make_shared<AnyVecValue<T>>(dest);
    (void)data_map_.emplace(name + '_' + typeid(T).name(), dest_ptr);
    data = dest_ptr->GetDataBuf();
    GELOGI("Attribute found: %s", name.c_str());
    return domi::SUCCESS;
  }

private:
  static std::map<std::string, std::function<Status(TeOpVarAttrArgsImpl *, const std::string &, DataBuf &)>>
          data_getter_;
  ge::OpDescPtr op_desc_;
  std::map<std::string, std::shared_ptr<AnyValueBase>> data_map_;
};

std::map<std::string, std::function<Status(TeOpVarAttrArgsImpl *, const std::string &, DataBuf &)>>
        TeOpVarAttrArgsImpl::data_getter_ = {{"Int8", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<int8_t>},
                                             {"Int16", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<int16_t>},
                                             {"Int32", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<int32_t>},
                                             {"Int64", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<int64_t>},
                                             {"UInt8", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<uint8_t>},
                                             {"UInt16", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<uint16_t>},
                                             {"UInt32", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<uint32_t>},
                                             {"UInt64", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<uint64_t>},
                                             {"Float", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<float>},
                                             {"ListInt8", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<int8_t, true>},
                                             {"ListInt16", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<int16_t, true>},
                                             {"ListInt32", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<int32_t, true>},
                                             {"ListInt64", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<int64_t, true>},
                                             {"ListUInt8", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<uint8_t, true>},
                                             {"ListUInt16", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<uint16_t, true>},
                                             {"ListUInt32", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<uint32_t, true>},
                                             {"ListUInt64", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<uint64_t, true>},
                                             {"ListFloat", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<float, true>}};

Status TeOpVarAttrArgsImpl::GetDataByName(const std::string &name, const std::string &dtype, DataBuf &data) {
  const auto iter = data_getter_.find(dtype);
  if (iter == data_getter_.end()) {
    GE_LOGE("wrong dtype: %s", dtype.c_str());
    return domi::FAILED;
  } else {
    return iter->second(this, name, data);
  }
}

const uint8_t *TeOpVarAttrArgs::GetData(const std::string &name, const std::string &dtype, size_t &size) const {
  DataBuf data(nullptr, 0);
  const auto rc = impl_->GetDataByName(name, dtype, data);
  if (rc == domi::SUCCESS) {
    GELOGI("Attribute found: %s, %s, %p, %ld", name.c_str(), dtype.c_str(), std::get<0>(data), std::get<1>(data));
  }
  size = std::get<1>(data);
  return std::get<0>(data);
}

class VarAttrHelper {
public:
  static bool InitTeOpVarAttr(const ge::OpDescPtr &op_desc_ptr, TeOpVarAttrArgs &attr) {
    OP_TILING_MAKE_SHARED(attr.impl_ = std::make_shared<TeOpVarAttrArgsImpl>(op_desc_ptr), return false);
    return true;
  }
};

bool FeedTeOpTensorArg(ge::OpDesc::Vistor<ge::GeTensorDescPtr> &tensor_desc_vec,
                       std::vector<TeOpTensorArg> &tensor_arg, const ge::OpDescPtr &op_desc) {
  size_t index = 0U;
  for (ge::GeTensorDescPtr &tensor_desc_ptr : tensor_desc_vec) {
    TeOpTensorArg arg_tensor;
    TeOpTensor tensor;
    arg_tensor.arg_type = TensorArgType::TA_SINGLE;
    tensor.shape = tensor_desc_ptr->MutableShape().GetDims();
    if (tensor.shape.empty()) {
      tensor.shape = {1};
    }
    tensor.ori_shape = tensor_desc_ptr->GetOriginShape().GetDims();
    tensor.name = op_desc->GetInputNameByIndex(static_cast<uint32_t>(index));

    const ge::Format primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(tensor_desc_ptr->GetFormat()));
    tensor.format = ge::TypeUtils::FormatToSerialString(primary_format);
    tensor.ori_format = ge::TypeUtils::FormatToSerialString(tensor_desc_ptr->GetOriginFormat());

    const ge::DataType dtype = tensor_desc_ptr->GetDataType();
    const auto dataTypeIter = DATATYPE_STRING_MAP.find(dtype);
    if (dataTypeIter == DATATYPE_STRING_MAP.end()) {
      GE_LOGE("datatype error %d", static_cast<int32_t>(dtype));
      return false;
    }
    tensor.dtype = dataTypeIter->second;
    if (IsLogEnable(GE_MODULE_NAME, DLOG_INFO)) {
      std::stringstream shapestr;
      shapestr << "shape:[";
      for (auto &i : tensor.shape) {
        shapestr << i << ",";
      }
      shapestr << "], ori_shape:[";
      for (auto &i : tensor.ori_shape) {
        shapestr << i << ",";
      }
      shapestr << "], format:" << tensor.format;
      shapestr << ", ori_format:" << tensor.ori_format;
      shapestr << ", dtype: " << tensor.dtype;
      GELOGI("calling optiling shape info: %s", shapestr.str().c_str());
    }

    (void) arg_tensor.tensor.emplace_back(tensor);
    (void) tensor_arg.emplace_back(arg_tensor);
    index++;
  }
  return true;
}

void FeedTeOpConstTensor(const ge::Operator &op, const ge::OpDescPtr &op_desc,
                         std::map<std::string, TeConstTensorData> &const_inputs) {
  std::vector<std::string> depend_names;
  (void)ge::AttrUtils::GetListStr(op_desc, ATTR_NAME_OP_INFER_DEPENDS, depend_names);
  for (const std::string &depend : depend_names) {
    ge::Tensor data;
    const ge::graphStatus rc = op.GetInputConstData(depend.c_str(), data);
    GELOGI("GetInputConstData: %s, %d", depend.c_str(), rc);
    if (rc != ge::GRAPH_SUCCESS) {
      continue;
    }

    const uint8_t * const pbuf = data.GetData();
    const size_t buflen = data.GetSize();

    GELOGI("Const input tensor data: %s, %p %zu", depend.c_str(), pbuf, buflen);
    (void)const_inputs.emplace(depend, TeConstTensorData{pbuf, buflen, data});
  }
}

ge::graphStatus OpParaCalculate(const ge::Operator &op, OpRunInfo &run_info, const OpTilingFunc &tiling_func) {
  ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GELOGI("Do optiling, op_type: %s, op_name: %s", op_desc->GetType().c_str(), op_desc->GetName().c_str());
  TeOpParas op_param;
  op_param.op_type = op_desc->GetType();
  (void)VarAttrHelper::InitTeOpVarAttr(op_desc, op_param.var_attrs);

  ge::OpDesc::Vistor<ge::GeTensorDescPtr> inputs = op_desc->GetAllInputsDescPtr();
  if (!FeedTeOpTensorArg(inputs, op_param.inputs, op_desc)) {
    GE_LOGE("Do optiling, op_type: %s, op_name: %s", op_desc->GetType().c_str(), op_desc->GetName().c_str());
    return ge::GRAPH_FAILED;
  }
  ge::OpDesc::Vistor<ge::GeTensorDescPtr> outputs = op_desc->GetAllOutputsDescPtr();
  if (!FeedTeOpTensorArg(outputs, op_param.outputs, op_desc)) {
    return ge::GRAPH_FAILED;
  }
  FeedTeOpConstTensor(op, op_desc, op_param.const_inputs);

  OpCompileInfo op_compile_info;
  if (!ge::AttrUtils::GetStr(op_desc, COMPILE_INFO_KEY, op_compile_info.key)) {
    GE_LOGE("Op [%s] does not have attribute [%s].", op_desc->GetName().c_str(), COMPILE_INFO_KEY.c_str());
    return ge::GRAPH_FAILED;
  }
  if (!ge::AttrUtils::GetStr(op_desc, COMPILE_INFO_JSON, op_compile_info.str)) {
    GE_LOGE("Op [%s] does not have attribute [%s].", op_desc->GetName().c_str(), COMPILE_INFO_JSON.c_str());
    return ge::GRAPH_FAILED;
  }

  const bool ret = (tiling_func)(op_param, op_compile_info, run_info);
  if (ret) {
    GELOGI("Do optiling succeed. op_type:%s, op_name:%s",
           op_desc->GetType().c_str(), op_desc->GetName().c_str());
  } else {
    GELOGW("Failed to call tiling function v1 of op [%s, %s].",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
  }
  return ret ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

ge::graphStatus TurnToOpParaCalculateV1(const ge::Operator &op, OpRunInfoV2 &run_info,
                                        const OpTilingFunc &tiling_func) {
  OpRunInfo run_info_struct;
  run_info_struct.block_dim = run_info.GetBlockDim();
  run_info_struct.clear_atomic = run_info.GetClearAtomic();
  run_info_struct.tiling_key = run_info.GetTilingKey();
  if (OpParaCalculate(op, run_info_struct, tiling_func) != ge::GRAPH_SUCCESS) {
    ge::AscendString op_type;
    (void)op.GetOpType(op_type);
    ge::AscendString op_name;
    (void)op.GetName(op_name);
    REPORT_INNER_ERR_MSG("E19999", "OpParaCalculate failed, op_type[%s], op_name[%s]", op_type.GetString(),
                         op_name.GetString());
    return ge::GRAPH_FAILED;
  }

  run_info.SetBlockDim(run_info_struct.block_dim);
  run_info.SetClearAtomic(run_info_struct.clear_atomic);
  run_info.SetTilingKey(run_info_struct.tiling_key);
  run_info.InternelSetTiling(run_info_struct.tiling_data);
  if (!run_info_struct.workspaces.empty()) {
    for (const int64_t &workspace : run_info_struct.workspaces) {
      run_info.AddWorkspace(workspace);
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TurnToOpParaCalculateV2(const ge::Operator &op_param, OpRunInfoV2 &run_info,
                                        const OpTilingFuncV2 &tiling_func) {
  const ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_param);
  GELOGI("Do optiling, op_type: %s, op_name: %s", op_desc->GetType().c_str(), op_desc->GetName().c_str());
  const std::string *op_compile_info_key = ge::AttrUtils::GetStr(op_desc, COMPILE_INFO_KEY);
  if (op_compile_info_key == nullptr) {
    GE_LOGE("Op [%s] does not have attribute [%s].", op_desc->GetName().c_str(), COMPILE_INFO_KEY.c_str());
    return ge::GRAPH_FAILED;
  }
  const std::string *op_compile_info_json = ge::AttrUtils::GetStr(op_desc, COMPILE_INFO_JSON);
  if (op_compile_info_json == nullptr) {
    GE_LOGE("Op [%s] does not have attribute [%s].", op_desc->GetName().c_str(), COMPILE_INFO_JSON.c_str());
    return ge::GRAPH_FAILED;
  }
  const OpCompileInfoV2 op_compile_info(*op_compile_info_key, *op_compile_info_json);

  std::vector<int32_t> indexes;
  ReplaceEmptyShapeOfTensorDesc(op_desc, indexes);

  const bool ret = (tiling_func)(op_param, op_compile_info, run_info);
  if (ret) {
    GELOGI("Do optiling v2 succeed. op_type: %s, op_name: %s",
           op_desc->GetType().c_str(), op_desc->GetName().c_str());
  } else {
    GELOGW("Failed to call the tiling function v2 for op [%s, %s].",
           op_desc->GetType().c_str(), op_desc->GetName().c_str());
  }
  RecoveryEmptyShapeOfTensorDesc(op_desc, indexes);
  return ret ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

ge::graphStatus TurnToOpParaCalculateV3(const ge::Operator &op_param, OpRunInfoV2 &run_info,
                                        const OpTilingFuncV3 &tiling_func, const OpParseFuncV3 &parse_func) {
  const ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_param);
  GELOGI("Do optiling, op_type: %s, op_name: %s", op_desc->GetType().c_str(), op_desc->GetName().c_str());
  const std::string *op_compile_info_key = ge::AttrUtils::GetStr(op_desc, COMPILE_INFO_KEY);
  if (op_compile_info_key == nullptr) {
    GE_LOGE("Op [%s] does not have attribute [%s].", op_desc->GetName().c_str(), COMPILE_INFO_KEY.c_str());
    return ge::GRAPH_FAILED;
  }
  void* op_compile_json_ptr = CompileInfoCache::Instance().GetCompileInfo(*op_compile_info_key);
  if (op_compile_json_ptr == nullptr) {
    const std::string *op_compile_info_json = ge::AttrUtils::GetStr(op_desc, COMPILE_INFO_JSON);
    if (op_compile_info_json == nullptr) {
      GE_LOGE("Op [%s] does not have attribute [%s].", op_desc->GetName().c_str(), COMPILE_INFO_JSON.c_str());
      return ge::GRAPH_FAILED;
    }
    const ge::AscendString compile_info_json_str = op_compile_info_json->c_str();
    op_compile_json_ptr = (parse_func)(op_param, compile_info_json_str);
    if (op_compile_json_ptr == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Failed to parse compile json[%s] for op [%s, %s].", op_compile_info_json->c_str(),
                           op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GE_LOGE("Failed to parse compile json[%s] for op [%s, %s].", op_compile_info_json->c_str(),
              op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return ge::GRAPH_FAILED;
    }
    CompileInfoCache::Instance().SetCompileInfo(*op_compile_info_key, op_compile_json_ptr);
  }

  std::vector<int32_t> indexes;
  ReplaceEmptyShapeOfTensorDesc(op_desc, indexes);

  const bool ret = (tiling_func)(op_param, op_compile_json_ptr, run_info);
  if (ret) {
    GELOGI("Do optiling v3 succeed. op_type: %s, op_name: %s",
           op_desc->GetType().c_str(), op_desc->GetName().c_str());
  } else {
    GELOGW("Failed to call the tiling function v3 for op [%s, %s].",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
  }
  RecoveryEmptyShapeOfTensorDesc(op_desc, indexes);
  return ret ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

ge::graphStatus TurnToOpParaCalculateV4(const ge::Operator &op_param, OpRunInfoV2 &run_info,
                                        const OpTilingFuncV4 &tiling_func, const OpParseFuncV4 &parse_func) {
  const ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_param);
  GELOGI("Do optiling, op_type: %s, op_name: %s", op_desc->GetType().c_str(), op_desc->GetName().c_str());
  const std::string *op_compile_info_key = ge::AttrUtils::GetStr(op_desc, COMPILE_INFO_KEY);
  if (op_compile_info_key == nullptr) {
    GE_LOGE("Op [%s] does not have attribute [%s].", op_desc->GetName().c_str(), COMPILE_INFO_KEY.c_str());
    return ge::GRAPH_FAILED;
  }
  CompileInfoPtr op_compile_info_ptr = CompileInfoManager::Instance().GetCompileInfo(*op_compile_info_key);
  if (op_compile_info_ptr == nullptr) {
    const std::string *op_compile_info_json = ge::AttrUtils::GetStr(op_desc, COMPILE_INFO_JSON);
    if (op_compile_info_json == nullptr) {
      GE_LOGE("Op [%s] does not have attribute [%s].", op_desc->GetName().c_str(), COMPILE_INFO_JSON.c_str());
      return ge::GRAPH_FAILED;
    }
    const ge::AscendString compile_info_json_str = op_compile_info_json->c_str();
    op_compile_info_ptr = (parse_func)(op_param, compile_info_json_str);
    if (op_compile_info_ptr == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Failed to parse compile json[%s] for op [%s, %s].", op_compile_info_json->c_str(),
                           op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GE_LOGE("Failed to parse compile json[%s] for op [%s, %s].", op_compile_info_json->c_str(),
              op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return ge::GRAPH_FAILED;
    }
    CompileInfoManager::Instance().SetCompileInfo(*op_compile_info_key, op_compile_info_ptr);
  }

  std::vector<int32_t> indexes;
  ReplaceEmptyShapeOfTensorDesc(op_desc, indexes);

  const bool ret = (tiling_func)(op_param, op_compile_info_ptr, run_info);
  if (ret) {
    GELOGI("Do optiling v4 succeed. op_type:%s, op_name:%s",
           op_desc->GetType().c_str(), op_desc->GetName().c_str());
  } else {
    GELOGW("Failed to call the tiling function v4 of op [%s, %s].",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
  }
  RecoveryEmptyShapeOfTensorDesc(op_desc, indexes);
  return ret ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

ge::graphStatus PostProcCalculateV2(const ge::Operator &op, OpRunInfoV2 &run_info)
{
  const ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GE_CHECK_NOTNULL(op_desc);
  const std::vector<int64_t> all_workspaces = op_desc->GetWorkspaceBytes();
  std::vector<int64_t> op_workspaces;
  run_info.GetAllWorkspaces(op_workspaces);
  const size_t op_work_size = op_workspaces.size();
  if (op_work_size > all_workspaces.size()) {
    GELOGW("Op name:%s tiling return workspace number(%zu) large than all workspace num(%zu).",
           op_desc->GetName().c_str(), op_work_size, all_workspaces.size());
    return ge::GRAPH_SUCCESS;
  }

  if (op_work_size == all_workspaces.size()) {
    return ge::GRAPH_SUCCESS;
  }

  GELOGD("Op name: %s post proc, op work num: %zu, all work num: %zu.", op_desc->GetName().c_str(), op_work_size,
         all_workspaces.size());

  // mixl2--pass will add additional works after op_workspaces
  for (size_t i = op_work_size; i < all_workspaces.size(); ++i) {
    (void) op_workspaces.emplace_back(all_workspaces[i]);
  }
  for (size_t i = 0; i < op_workspaces.size(); ++i) {
    GELOGD("Op's workspace: %zu, value: %ld.", i, op_workspaces[i]);
  }
  run_info.SetWorkspaces(op_workspaces);
  return ge::GRAPH_SUCCESS;
}

OpTilingFuncInfo *GetOpTilingInfo(const ge::OpDescPtr &op_desc) {
  if (op_desc == nullptr) {
    GE_LOGE("[Get][OpTilingInfo] failed, op_desc is nullptr.");
    REPORT_INNER_ERR_MSG("EZ9999", "[Get][OpTilingInfo] failed, op_desc is nullptr.");
    return nullptr;
  }
  if (op_desc->GetTilingFuncInfo() == nullptr) {
    const std::string op_type = op_desc->GetType();
    auto &op_func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
    auto iter = op_func_map.find(op_type);
    if (iter == op_func_map.end()) {
      GELOGI("The optiling function is not found by op type [%s].", op_type.c_str());
      iter = op_func_map.find(OP_TYPE_AUTO_TILING);
      if (iter == op_func_map.end()) {
        GE_LOGE("Optiling function of op type[%s] is not found by Autotiling.", op_type.c_str());
        REPORT_INNER_ERR_MSG("EZ9999", "Optiling function not found. op_type[%s].", op_type.c_str());
        return nullptr;
      }
    }
    op_desc->SetTilingFuncInfo(::ge::PtrToPtr<OpTilingFuncInfo, void>(&(iter->second)));
    return &(iter->second);
  }
  return ::ge::PtrToPtr<void, OpTilingFuncInfo>(op_desc->GetTilingFuncInfo());
}

void parse_tiling_data(const void* base, const size_t max_size) {
  std::stringstream result;
  int32_t tmp = 0;
  const char* base_addr = static_cast<const char*>(base);
  for (size_t i = 0U; i < max_size; i += sizeof(int32_t)) {
    if ((max_size - i) < sizeof(tmp)) {
      return;
    }
    if (memcpy_s(&tmp, sizeof(tmp), base_addr + i, sizeof(tmp)) != EOK) {
      return;
    }
    result << std::to_string(tmp);
    result << " ";
  }
  GELOGD("Parse tiling data %s.", result.str().c_str());
  return;
}

ge::graphStatus PostProcMemoryCheck(const ge::Operator &op, OpRunInfoV2 &run_info)
{
  const ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  bool value = false;
  if (!ge::AttrUtils::GetBool(op_desc, kMemoryCheck, value) || !value) {
    return ge::GRAPH_SUCCESS;
  }
  uint64_t ori_op_para_size = 0;
  if (ge::AttrUtils::GetInt(op_desc, kOriOpParaSize, ori_op_para_size)) {
    GELOGD("The ori_op_para_size of node [%s] is %lu.", op_desc->GetName().c_str(), ori_op_para_size);
    if (!run_info.SetMemCheckBaseOffset(ori_op_para_size)) {
      REPORT_INNER_ERR_MSG("E19999",
                           "[register][op_tiling][PostProcMemoryCheck]Node:%s set mem check offset:%lu failed.",
                           op_desc->GetName().c_str(), ori_op_para_size);
      return ge::GRAPH_FAILED;
    }
  } else {
    run_info.AlignOffsetWith64();
  }
  for (size_t i = 0U; i < op_desc->GetAllInputsSize(); ++i) {
    const ge::GeTensorDescPtr tensor = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    if (tensor == nullptr) {
      continue;
    }
    int64_t clean_size = 0;
    if (ge::TensorUtils::GetSize(*tensor, clean_size) != ge::GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "[register][op_tiling][PostProcMemoryCheck]Get op:%s tensor:%zu size failed.",
                           op_desc->GetName().c_str(), i);
      return ge::GRAPH_FAILED;
    }
    GELOGD("Op input tensor: %zu has a size of %ld.", i, clean_size);
    run_info.AddTilingData(clean_size);
  }
  for (size_t j = 0U; j < op_desc->GetOutputsSize(); ++j) {
    const ge::GeTensorDescPtr tensor = op_desc->MutableOutputDesc(static_cast<uint32_t>(j));
    if (tensor == nullptr) {
      continue;
    }
    int64_t clean_size = 0;
    if (ge::TensorUtils::GetSize(*tensor, clean_size) != ge::GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "[register][op_tiling][PostProcMemoryCheck]Get op:%s tensor:%zu size failed.",
                           op_desc->GetName().c_str(), j);
      return ge::GRAPH_FAILED;
    }
    GELOGD("Op output tensor: %zu with size %ld.", j, clean_size);
    run_info.AddTilingData(clean_size);
  }
  for (size_t k = 0U; k < run_info.GetWorkspaceNum(); ++k) {
    int64_t workspace = 0;
    (void)run_info.GetWorkspace(k, workspace);
    GELOGD("Op workspace: %zu size is %ld bytes.", k, workspace);
    run_info.AddTilingData(workspace);
  }
  const uint64_t cur_size = run_info.GetTilingDataSize();
  GELOGD("Adding tiling data; current size: %lu.", cur_size);
  run_info.AddTilingData(cur_size);

  uint64_t max_size = 0U;
  const void* base = run_info.GetAddrBase(max_size);
  parse_tiling_data(base, static_cast<size_t>(max_size));
  return ge::GRAPH_SUCCESS;
}

extern "C" ge::graphStatus OpParaCalculateV2(const ge::Operator &op, OpRunInfoV2 &run_info) {
  const ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  OpTilingFuncInfo *op_func_info = GetOpTilingInfo(op_desc);
  if (op_func_info == nullptr) {
    GE_LOGE("Optiling function not found.");
    REPORT_INNER_ERR_MSG("EZ9999", "Optiling function not found.");
    return ge::GRAPH_FAILED;
  }
  ge::graphStatus ret = ge::GRAPH_FAILED;
  if (op_func_info->IsFunctionV4()) {
    const OpTilingFuncV4 &tiling_func = op_func_info->GetOpTilingFuncV4();
    const OpParseFuncV4 &parse_func = op_func_info->GetOpParseFuncV4();
    ret = TurnToOpParaCalculateV4(op, run_info, tiling_func, parse_func);
  } else if (op_func_info->IsFunctionV3()) {
    const OpTilingFuncV3 &tiling_func = op_func_info->GetOpTilingFuncV3();
    const OpParseFuncV3 &parse_func = op_func_info->GetOpParseFuncV3();
    ret = TurnToOpParaCalculateV3(op, run_info, tiling_func, parse_func);
  } else if (op_func_info->IsFunctionV2()) {
    const OpTilingFuncV2  &tiling_func = op_func_info->GetOpTilingFuncV2();
    ret = TurnToOpParaCalculateV2(op, run_info, tiling_func);
  } else if (op_func_info->IsFunctionV1()) {
    const OpTilingFunc  &tiling_func = op_func_info->GetOpTilingFunc();
    ret = TurnToOpParaCalculateV1(op, run_info, tiling_func);
  } else {
    GE_LOGE("Optiling function for op type [%s] is entirely empty.", op_desc->GetType().c_str());
  }
  if (ret != ge::GRAPH_SUCCESS) {
    return ret;
  }
  ret = PostProcCalculateV2(op, run_info);
  if (ret == ge::GRAPH_SUCCESS) {
    return PostProcMemoryCheck(op, run_info);
  }
  return ret;
}

void GenerateCompileInfoKey(const std::vector<int64_t> &workspace_size_list, std::string &op_compile_info_key) {
  for (const int64_t &workspace_size : workspace_size_list) {
    (void)op_compile_info_key.append(",").append(std::to_string(workspace_size));
  }
}

ge::graphStatus AssembleCompileInfoJson(const ge::OpDescPtr &op_desc_ptr,
                                        const std::vector<int64_t> &workspace_size_list,
                                        std::string &op_compile_info_json) {
  nlohmann::json compile_info_json;
  try {
    compile_info_json = nlohmann::json::parse(op_compile_info_json);
  } catch (nlohmann::json::parse_error& ex) {
    REPORT_INNER_ERR_MSG("E19999",
                         "Failed to set compile_info_value to the json format for op[%s]. op_compile_info_json: %s",
                         op_desc_ptr->GetName().c_str(), op_compile_info_json.c_str());
    GE_LOGE("Failed to set compile_info_value to the json format for op[%s]. op_compile_info_json: %s",
            op_desc_ptr->GetName().c_str(), op_compile_info_json.c_str());
    return ge::GRAPH_FAILED;
  }
  for (const int64_t &workspace_size : workspace_size_list) {
    compile_info_json[COMPILE_INFO_WORKSPACE_SIZE_LIST].push_back(workspace_size);
  }
  op_compile_info_json = compile_info_json.dump();
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AssembleWorkspaceList(const ge::OpDescPtr &op_desc_ptr,
                                      int64_t &first_clean_size,
                                      std::vector<int64_t> &workspace_size_list) {
  std::vector<int64_t> atomic_output_indices;
  (void) ge::AttrUtils::GetListInt(op_desc_ptr, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indices);
  std::map<string, std::map<int64_t, int64_t>> atomic_workspace_info;
  atomic_workspace_info = op_desc_ptr->TryGetExtAttr(ge::EXT_ATTR_ATOMIC_WORKSPACE_INFO, atomic_workspace_info);
  const bool atomic_flag = atomic_output_indices.empty() && atomic_workspace_info.empty();
  if (atomic_flag) {
    GE_LOGE("Do not find ATOMIC_ATTR_OUTPUT_INDEX and EXT_ATTR_ATOMIC_WORKSPACE_INFO, op_type:%s, op_name:%s",
            OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN.c_str(), op_desc_ptr->GetName().c_str());
    return ge::GRAPH_FAILED;
  }

  if (!atomic_output_indices.empty()) {
    bool is_first_index = true;
    for (const int64_t &atomic_output_indice : atomic_output_indices) {
      const ge::ConstGeTensorDescPtr tensor =
          op_desc_ptr->GetOutputDescPtr(static_cast<uint32_t>(atomic_output_indice));
      if (tensor == nullptr) {
        GE_LOGE("Failed to get atomic_output_indice. op_type: %s, op_name: %s",
                OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN.c_str(), op_desc_ptr->GetName().c_str());
        return ge::GRAPH_FAILED;
      }

      int64_t clean_size = 0;
      if (ge::TensorUtils::GetSize(*tensor, clean_size) != ge::GRAPH_SUCCESS) {
        GE_LOGE("Failed to get size of tensor desc. op_type: %s, op_name: %s",
                OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN.c_str(), op_desc_ptr->GetName().c_str());
        return ge::GRAPH_FAILED;
      }
      workspace_size_list.push_back(clean_size);
      if (is_first_index) {
        first_clean_size = clean_size;
        is_first_index = false;
      }
    }
  }
  GELOGI("Atomic clean size: %ld, op_name:%s", first_clean_size, op_desc_ptr->GetName().c_str());

  if (!atomic_workspace_info.empty()) {
    const std::vector<int64_t> workspace_bytes = op_desc_ptr->GetWorkspaceBytes();
    const std::map<int64_t, int64_t> workspace_bytes_map = atomic_workspace_info[op_desc_ptr->GetName()];
    for (auto &workspace_idxs : workspace_bytes_map) {
      if (workspace_idxs.first < static_cast<int64_t>(workspace_bytes.size())) {
        workspace_size_list.push_back(static_cast<int64_t>(
            workspace_bytes[static_cast<size_t>(workspace_idxs.first)]));
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus OpAtomicCalculateV1(const ge::OpDescPtr &op_desc_ptr, OpRunInfo &run_info,
                                    const OpTilingFunc &tiling_func) {
  GELOGI("Begin to perform atomic optiling. op_type: %s, op_name: %s",
         OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN.c_str(), op_desc_ptr->GetName().c_str());
  OpCompileInfo op_compile_info;
  if (!ge::AttrUtils::GetStr(op_desc_ptr, ATOMIC_COMPILE_INFO_KEY, op_compile_info.key)) {
    GE_LOGE("Op [%s] does not have attribute [%s].", op_desc_ptr->GetName().c_str(), ATOMIC_COMPILE_INFO_KEY.c_str());
    return ge::GRAPH_FAILED;
  }
  if (!ge::AttrUtils::GetStr(op_desc_ptr, ATOMIC_COMPILE_INFO_JSON, op_compile_info.str)) {
    GE_LOGE("Op [%s] does not have attribute [%s].", op_desc_ptr->GetName().c_str(), ATOMIC_COMPILE_INFO_JSON.c_str());
    return ge::GRAPH_FAILED;
  }

  int64_t first_clean_size = 0;
  std::vector<int64_t> workspace_size_list;
  if (AssembleWorkspaceList(op_desc_ptr, first_clean_size, workspace_size_list) != ge::GRAPH_SUCCESS) {
    GE_LOGE("Failed to retrieve the workspace size list from op[%s, %s].",
            op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
    return ge::GRAPH_FAILED;
  }

  TeOpParas op_param;
  op_param.op_type = OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN;
  (void)op_param.const_inputs.emplace("workspace_size",
                                      TeConstTensorData(nullptr, static_cast<size_t>(first_clean_size), ge::Tensor()));

  GenerateCompileInfoKey(workspace_size_list, op_compile_info.key);
  if (AssembleCompileInfoJson(op_desc_ptr, workspace_size_list, op_compile_info.str) != ge::GRAPH_SUCCESS) {
    GE_LOGE("Failed to assemble compile info json for op [%s, %s].",
            op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
    return ge::GRAPH_FAILED;
  }

  const bool ret = (tiling_func)(op_param, op_compile_info, run_info);
  if (ret) {
    GELOGI("Atomic optiling v1 operation succeeded. op_type: %s, op_name: %s.",
           op_desc_ptr->GetType().c_str(), op_desc_ptr->GetName().c_str());
  } else {
    GELOGW("Failed to call the tiling v1 function of atomic op [%s, %s].",
           op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
  }
  return ret ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

static ge::graphStatus TurnToOpAtomicCalculateV1(const ge::OpDescPtr &op_desc_ptr, OpRunInfoV2 &run_info,
                                          const OpTilingFunc &tiling_func) {
  OpRunInfo run_info_struct;
  run_info_struct.block_dim = run_info.GetBlockDim();
  run_info_struct.clear_atomic = run_info.GetClearAtomic();
  run_info_struct.tiling_key = run_info.GetTilingKey();
  if (OpAtomicCalculateV1(op_desc_ptr, run_info_struct, tiling_func) != ge::GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Do OpAtomicCalculateV1 failed, op_type[%s], op_name[%s]",
                         op_desc_ptr->GetType().c_str(), op_desc_ptr->GetName().c_str());
    return ge::GRAPH_FAILED;
  }
  run_info.InternelSetTiling(run_info_struct.tiling_data);
  run_info.SetBlockDim(run_info_struct.block_dim);
  run_info.SetClearAtomic(run_info_struct.clear_atomic);
  run_info.SetTilingKey(run_info_struct.tiling_key);
  if (!run_info_struct.workspaces.empty()) {
    for (const int64_t &workspace : run_info_struct.workspaces) {
      run_info.AddWorkspace(workspace);
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AssembleWorkspaceList(const ge::OpDescPtr &op_desc_ptr,
                                      std::vector<int64_t> &workspace_list,
                                      std::vector<int64_t> &workspace_size_list) {
  std::vector<int64_t> atomic_output_indices;
  (void) ge::AttrUtils::GetListInt(op_desc_ptr, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indices);
  std::map<string, std::map<int64_t, int64_t>> atomic_workspace_info;
  atomic_workspace_info = op_desc_ptr->TryGetExtAttr(ge::EXT_ATTR_ATOMIC_WORKSPACE_INFO, atomic_workspace_info);
  const bool atomic_flag = atomic_output_indices.empty() && atomic_workspace_info.empty();
  if (atomic_flag) {
    REPORT_INNER_ERR_MSG("E19999",
                         "No ATOMIC_ATTR_OUTPUT_INDEX and EXT_ATTR_ATOMIC_WORKSPACE_INFO found,op_type:%s, op_name:%s",
                         OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN.c_str(), op_desc_ptr->GetName().c_str());
    return ge::GRAPH_FAILED;
  }

  if (!atomic_output_indices.empty()) {
    bool is_first_index = true;
    for (const int64_t &atomic_output_indice : atomic_output_indices) {
      const ge::ConstGeTensorDescPtr tensor =
          op_desc_ptr->GetOutputDescPtr(static_cast<uint32_t>(atomic_output_indice));
      if (tensor == nullptr) {
        REPORT_INNER_ERR_MSG("E19999", "Get MutableOutputDesc failed. op_type:%s, op_name:%s",
                             OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN.c_str(), op_desc_ptr->GetName().c_str());
        return ge::GRAPH_FAILED;
      }
      int64_t clean_size = 0;
      if (ge::TensorUtils::GetSize(*tensor, clean_size) != ge::GRAPH_SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Get size of tensor desc failed. op_type:%s, op_name:%s",
                             OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN.c_str(), op_desc_ptr->GetName().c_str());
        return ge::GRAPH_FAILED;
      }
      workspace_size_list.push_back(clean_size);
      if (is_first_index) {
        workspace_list.push_back(clean_size);
        is_first_index = false;
      }
    }
  }

  if (!atomic_workspace_info.empty()) {
    const std::vector<int64_t> workspace_bytes = op_desc_ptr->GetWorkspaceBytes();
    const std::map<int64_t, int64_t> workspace_bytes_map = atomic_workspace_info[op_desc_ptr->GetName()];
    for (auto &workspace_idxs : workspace_bytes_map) {
      if (workspace_idxs.first < static_cast<int64_t>(workspace_bytes.size())) {
        workspace_size_list.push_back(workspace_bytes[workspace_idxs.first]);
        workspace_list.push_back(static_cast<int64_t>(workspace_bytes[static_cast<size_t>(workspace_idxs.first)]));
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TurnToOpAtomicCalculateV2(const ge::OpDescPtr &op_desc_ptr, OpRunInfoV2 &run_info,
                                          const OpTilingFuncV2 &tiling_func) {
  GELOGI("Begin atomic optiling V2 for op [%s, %s].",
         op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
  std::vector<int64_t> workspace_list;
  std::vector<int64_t> workspace_size_list;
  if (AssembleWorkspaceList(op_desc_ptr, workspace_list, workspace_size_list) != ge::GRAPH_SUCCESS) {
    GE_LOGE("Failed to retrieve the workspace size list from op[%s, %s].",
            op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
    return ge::GRAPH_FAILED;
  }
  ge::Operator op_param(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN.c_str());
  (void)op_param.SetAttr(ATTR_NAME_ATOMIC_CLEAN_WORKSPACE.c_str(), workspace_list);

  std::string op_compile_info_key;
  if (!ge::AttrUtils::GetStr(op_desc_ptr, ATOMIC_COMPILE_INFO_KEY, op_compile_info_key)) {
    GE_LOGE("Op [%s] does not have attribute [%s].", op_desc_ptr->GetName().c_str(), ATOMIC_COMPILE_INFO_KEY.c_str());
    return ge::GRAPH_FAILED;
  }
  std::string op_compile_info_json;
  if (!ge::AttrUtils::GetStr(op_desc_ptr, ATOMIC_COMPILE_INFO_JSON, op_compile_info_json)) {
    GE_LOGE("Op [%s] does not have attribute [%s].", op_desc_ptr->GetName().c_str(), ATOMIC_COMPILE_INFO_JSON.c_str());
    return ge::GRAPH_FAILED;
  }
  GenerateCompileInfoKey(workspace_size_list, op_compile_info_key);
  if (AssembleCompileInfoJson(op_desc_ptr, workspace_size_list, op_compile_info_json) != ge::GRAPH_SUCCESS) {
    GE_LOGE("Failed to assemble compile info json for op [%s, %s].",
            op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
    return ge::GRAPH_FAILED;
  }
  const OpCompileInfoV2 op_compile_info(op_compile_info_key, op_compile_info_json);
  const bool ret = (tiling_func)(op_param, op_compile_info, run_info);
  if (ret) {
    GELOGI("Atomic optiling v2 operation succeeded. op_type: %s, op_name: %s.",
           op_desc_ptr->GetType().c_str(), op_desc_ptr->GetName().c_str());
  } else {
    GELOGW("Failed to call the tiling v2 function of atomic op [%s, %s].",
           op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
  }
  op_param.BreakConnect();
  return ret ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

static ge::graphStatus TurnToOpAtomicCalculateV3(const ge::OpDescPtr &op_desc_ptr, OpRunInfoV2 &run_info,
                                          const OpTilingFuncV3 &tiling_func, const OpParseFuncV3 &parse_func) {
  GELOGI("Begin Atomic optiling V3 for op [%s, %s].",
         op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
  std::vector<int64_t> workspace_list;
  std::vector<int64_t> workspace_size_list;
  if (AssembleWorkspaceList(op_desc_ptr, workspace_list, workspace_size_list) != ge::GRAPH_SUCCESS) {
    GE_LOGE("Failed to retrieve workspace list from op[%s, %s].",
            op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
    return ge::GRAPH_FAILED;
  }
  ge::Operator op_param(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN.c_str());
  (void)op_param.SetAttr(ATTR_NAME_ATOMIC_CLEAN_WORKSPACE.c_str(), workspace_list);

  std::string op_compile_info_key;
  if (!ge::AttrUtils::GetStr(op_desc_ptr, ATOMIC_COMPILE_INFO_KEY, op_compile_info_key)) {
    GE_LOGE("Op [%s] does not have attribute [%s].", op_desc_ptr->GetName().c_str(), ATOMIC_COMPILE_INFO_KEY.c_str());
    return ge::GRAPH_FAILED;
  }
  GenerateCompileInfoKey(workspace_size_list, op_compile_info_key);
  void* op_compile_json_ptr = CompileInfoCache::Instance().GetCompileInfo(op_compile_info_key);
  if (op_compile_json_ptr == nullptr) {
    std::string op_compile_info_json;
    if (!ge::AttrUtils::GetStr(op_desc_ptr, ATOMIC_COMPILE_INFO_JSON, op_compile_info_json)) {
      GE_LOGE("Op [%s] does not have attribute [%s].", op_desc_ptr->GetName().c_str(), ATOMIC_COMPILE_INFO_JSON.c_str());
      return ge::GRAPH_FAILED;
    }
    if (AssembleCompileInfoJson(op_desc_ptr, workspace_size_list, op_compile_info_json) != ge::GRAPH_SUCCESS) {
      GE_LOGE("Failed to assemble compile info json for op [%s, %s].",
              op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
      return ge::GRAPH_FAILED;
    }

    const ge::AscendString compile_info_json_str = op_compile_info_json.c_str();
    op_compile_json_ptr = (parse_func)(op_param, compile_info_json_str);
    if (op_compile_json_ptr == nullptr) {
      GE_LOGE("Failed to parse compile json [%s] for op [%s, %s].", op_compile_info_json.c_str(),
              op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
      return ge::GRAPH_FAILED;
    }
    CompileInfoCache::Instance().SetCompileInfo(op_compile_info_key, op_compile_json_ptr);
  }

  const bool ret = (tiling_func)(op_param, op_compile_json_ptr, run_info);
  if (ret) {
    GELOGI("Atomic optiling v3 succeeded. op_type: %s, op_name: %s.",
           op_desc_ptr->GetType().c_str(), op_desc_ptr->GetName().c_str());
  } else {
    GELOGW("Failed to call the tiling v3 function for atomic op [%s, %s].",
           op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
  }
  op_param.BreakConnect();
  return ret ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

static ge::graphStatus TurnToOpAtomicCalculateV4(const ge::OpDescPtr &op_desc_ptr, OpRunInfoV2 &run_info,
                                          const OpTilingFuncV4 &tiling_func, const OpParseFuncV4 &parse_func) {
  GELOGI("Begin Atomic optiling V4 for op [%s, %s].",
         op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
  std::vector<int64_t> workspace_list;
  std::vector<int64_t> workspace_size_list;
  if (AssembleWorkspaceList(op_desc_ptr, workspace_list, workspace_size_list) != ge::GRAPH_SUCCESS) {
    GE_LOGE("Failed to retrieve workspace list from op[%s, %s].",
            op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
    return ge::GRAPH_FAILED;
  }
  ge::Operator op_param(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN.c_str());
  (void)op_param.SetAttr(ATTR_NAME_ATOMIC_CLEAN_WORKSPACE.c_str(), workspace_list);

  std::string op_compile_info_key;
  if (!ge::AttrUtils::GetStr(op_desc_ptr, ATOMIC_COMPILE_INFO_KEY, op_compile_info_key)) {
    GE_LOGE("Op [%s] does not have attribute [%s].", op_desc_ptr->GetName().c_str(), ATOMIC_COMPILE_INFO_KEY.c_str());
    return ge::GRAPH_FAILED;
  }
  GenerateCompileInfoKey(workspace_size_list, op_compile_info_key);
  CompileInfoPtr op_compile_info_ptr = CompileInfoManager::Instance().GetCompileInfo(op_compile_info_key);
  if (op_compile_info_ptr == nullptr) {
    std::string op_compile_info_json;
    if (!ge::AttrUtils::GetStr(op_desc_ptr, ATOMIC_COMPILE_INFO_JSON, op_compile_info_json)) {
      GE_LOGE("Op [%s] does not have attribute [%s].", op_desc_ptr->GetName().c_str(), ATOMIC_COMPILE_INFO_JSON.c_str());
      return ge::GRAPH_FAILED;
    }
    if (AssembleCompileInfoJson(op_desc_ptr, workspace_size_list, op_compile_info_json) != ge::GRAPH_SUCCESS) {
      GE_LOGE("Failed to assemble compile info json for op [%s, %s].",
              op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
      return ge::GRAPH_FAILED;
    }

    const ge::AscendString compile_info_json_str = op_compile_info_json.c_str();
    op_compile_info_ptr = (parse_func)(op_param, compile_info_json_str);
    if (op_compile_info_ptr == nullptr) {
      GE_LOGE("Failed to parse compile json [%s] for op [%s, %s].", op_compile_info_json.c_str(),
              op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
      return ge::GRAPH_FAILED;
    }
    CompileInfoManager::Instance().SetCompileInfo(op_compile_info_key, op_compile_info_ptr);
  }

  const bool ret = (tiling_func)(op_param, op_compile_info_ptr, run_info);
  if (ret) {
    GELOGI("Atomic optiling v4 succeeded. op_type: %s, op_name: %s.",
           op_desc_ptr->GetType().c_str(), op_desc_ptr->GetName().c_str());
  } else {
    GELOGW("Failed to call the tiling v4 function of atomic op[%s, %s].",
           op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
  }
  op_param.BreakConnect();
  return ret ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

OpTilingFuncInfo *GetOpAtomicTilingInfo(const ge::OpDescPtr &op_desc) {
  if (op_desc == nullptr) {
    return nullptr;
  }
  if (op_desc->GetAtomicTilingFuncInfo() == nullptr) {
    auto &op_func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
    const auto iter = op_func_map.find(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
    if (iter == op_func_map.end()) {
      GE_LOGE("Atomic optiling func not found of op[%s, %s].",
              op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return nullptr;
    }
    op_desc->SetAtomicTilingFuncInfo(::ge::PtrToPtr<OpTilingFuncInfo, void>(&(iter->second)));
    return &(iter->second);
  }
  return ::ge::PtrToPtr<void, OpTilingFuncInfo>(op_desc->GetAtomicTilingFuncInfo());
}

extern "C" ge::graphStatus OpAtomicCalculateV2(const ge::Node &node, OpRunInfoV2 &run_info) {
  const ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  OpTilingFuncInfo *op_func_info = GetOpAtomicTilingInfo(op_desc_ptr);
  GE_CHECK_NOTNULL(op_func_info);
  ge::graphStatus status = ge::GRAPH_FAILED;
  if (op_func_info->IsFunctionV4()) {
    const OpTilingFuncV4 &tiling_func = op_func_info->GetOpTilingFuncV4();
    const OpParseFuncV4 &parse_func = op_func_info->GetOpParseFuncV4();
    status = TurnToOpAtomicCalculateV4(op_desc_ptr, run_info, tiling_func, parse_func);
  } else if (op_func_info->IsFunctionV3()) {
    const OpTilingFuncV3 &tiling_func = op_func_info->GetOpTilingFuncV3();
    const OpParseFuncV3 &parse_func = op_func_info->GetOpParseFuncV3();
    status = TurnToOpAtomicCalculateV3(op_desc_ptr, run_info, tiling_func, parse_func);
  } else if (op_func_info->IsFunctionV2()) {
    const OpTilingFuncV2 &tiling_func = op_func_info->GetOpTilingFuncV2();
    status = TurnToOpAtomicCalculateV2(op_desc_ptr, run_info, tiling_func);
  } else if (op_func_info->IsFunctionV1()) {
    const OpTilingFunc &tiling_func = op_func_info->GetOpTilingFunc();
    status = TurnToOpAtomicCalculateV1(op_desc_ptr, run_info, tiling_func);
  } else {
    GE_LOGE("Optiling function for op type [%s] is entirely empty.", op_desc_ptr->GetType().c_str());
  }
  return status;
}

ge::graphStatus UpDateNodeShapeBySliceInfo(const ffts::ThreadSliceMapDyPtr slice_info_ptr, const ge::OpDescPtr op_desc,
                                           const uint32_t thread_id, vector<int64_t> &ori_shape,
                                           bool &same_shape)
{
  if ((thread_id >= slice_info_ptr->input_tensor_slice.size())
      || (thread_id >= slice_info_ptr->output_tensor_slice.size())) {
    REPORT_INNER_ERR_MSG("E19999", "Update node shape thread id(%u) err.", thread_id);
    return ge::GRAPH_FAILED;
  }
  ge::GeTensorDescPtr tensor_ptr = nullptr;
  for (auto &index : slice_info_ptr->input_tensor_indexes) {
    tensor_ptr = op_desc->MutableInputDesc(index);
    GE_CHECK_NOTNULL(tensor_ptr);
    ge::GeShape& shape = tensor_ptr->MutableShape();
    auto &tmp_dim = slice_info_ptr->input_tensor_slice[static_cast<size_t>(thread_id)][index];
    if (tmp_dim.empty()) {
      return ge::GRAPH_FAILED;
    }
    if (thread_id == 0U) {
      (void) ori_shape.emplace_back(shape.GetDim(0));
      auto &tail_dim = slice_info_ptr->input_tensor_slice[slice_info_ptr->slice_instance_num - 1][index];
      if (tail_dim.empty()) {
        return ge::GRAPH_FAILED;
      }
      if (tail_dim[0] != tmp_dim[0]) {
        same_shape = false;
      }
    }
    (void)shape.SetDim(0, tmp_dim[0]);
  }
  for (auto &index : slice_info_ptr->output_tensor_indexes) {
    tensor_ptr = op_desc->MutableOutputDesc(index);
    GE_CHECK_NOTNULL(tensor_ptr);
    ge::GeShape& shape = tensor_ptr->MutableShape();
    if (thread_id == 0U) {
      (void) ori_shape.emplace_back(shape.GetDim(0));
    }
    auto &tmp_dim = slice_info_ptr->output_tensor_slice[static_cast<size_t>(thread_id)][index];
    if (tmp_dim.empty()) {
      return ge::GRAPH_FAILED;
    }
    (void)shape.SetDim(0, tmp_dim[0]);
    GELOGD("Output anchor: %u set dim 0 to %ld", index, tmp_dim[0]);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpDateNodeShapeBack(const ge::OpDescPtr op_desc, const ffts::ThreadSliceMapDyPtr slice_info_ptr,
                                    vector<int64_t> &ori_shape)
{
  if (slice_info_ptr == nullptr ||
      ori_shape.size() != (slice_info_ptr->input_tensor_indexes.size() +
                           slice_info_ptr->output_tensor_indexes.size())) {
    REPORT_INNER_ERR_MSG("E19999", "Update back node shape size err.");
    return ge::GRAPH_FAILED;
  }
  size_t idx = 0;
  for (auto &index : slice_info_ptr->input_tensor_indexes) {
    ge::GeTensorDescPtr tensor_ptr = op_desc->MutableInputDesc(index);
    GE_CHECK_NOTNULL(tensor_ptr);
    ge::GeShape& shape = tensor_ptr->MutableShape();
    (void)shape.SetDim(0, ori_shape[idx++]);
  }
  for (auto &index : slice_info_ptr->output_tensor_indexes) {
    ge::GeTensorDescPtr tensor_ptr = op_desc->MutableOutputDesc(index);
    GE_CHECK_NOTNULL(tensor_ptr);
    ge::GeShape& shape = tensor_ptr->MutableShape();
    (void)shape.SetDim(0, ori_shape[idx++]);
  }
  GELOGD("Node shape update reverted successfully.");
  return ge::GRAPH_SUCCESS;
}

// For FFTS+ dynamic shape
extern "C" ge::graphStatus OpFftsPlusCalculate(const ge::Operator &op, std::vector<OpRunInfoV2> &op_run_info)
{
  const auto node = ge::NodeUtilsEx::GetNodeFromOperator(op);
  GE_CHECK_NOTNULL(node);
  const auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  GELOGD("[OpFftsPlusCalculate]Op_type:%s, op_name:%s", op_desc->GetType().c_str(), op_desc->GetName().c_str());
  ffts::ThreadSliceMapDyPtr slice_info_ptr = nullptr;
  slice_info_ptr = op_desc->TryGetExtAttr(ffts::kAttrSgtStructInfoDy, slice_info_ptr);
  GE_CHECK_NOTNULL(slice_info_ptr);
  if (slice_info_ptr->slice_instance_num != slice_info_ptr->input_tensor_slice.size() ||
      slice_info_ptr->slice_instance_num != slice_info_ptr->output_tensor_slice.size()) {
    REPORT_INNER_ERR_MSG("E19999", "Slice num not equal.");
    return ge::GRAPH_FAILED;
  }
  vector<int64_t> ori_shape; // save original shape
  uint32_t thread_id = 0U;
  op_run_info.resize(ffts::kSgtTillingNum);
  bool same_shape = true;
  for (size_t i = 0U; i < static_cast<size_t>(ffts::kSgtTillingNum); i++) {
    // update node shape by thread slice info
    if (UpDateNodeShapeBySliceInfo(slice_info_ptr, op_desc, thread_id, ori_shape, same_shape) == ge::GRAPH_FAILED) {
      REPORT_INNER_ERR_MSG("E19999", "Update shape failed.");
      return ge::GRAPH_FAILED;
    }
    // call original interface
    const ge::graphStatus rc = OpParaCalculateV2(op, op_run_info[i]);
    if (rc != ge::GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "OpParaCalculateV2 failed, op_type:%s, op_name:%s", op_desc->GetType().c_str(),
                           op_desc->GetName().c_str());
      return rc;
    }
    if (same_shape) {
      op_run_info[1] = op_run_info[0];
      break;
    }
    thread_id = slice_info_ptr->slice_instance_num - 1U;
  }
  // node shape write_back
  (void)UpDateNodeShapeBack(op_desc, slice_info_ptr, ori_shape);
  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling
