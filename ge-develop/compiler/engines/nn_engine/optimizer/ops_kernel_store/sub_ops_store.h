/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_OPS_KERNEL_STORE_SUB_OPS_STORE_H_
#define FUSION_ENGINE_OPTIMIZER_OPS_KERNEL_STORE_SUB_OPS_STORE_H_
#include <map>
#include <memory>
#include <functional>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "common/math_util.h"
#include "common/fe_op_info_common.h"
#include "common/util/json_util.h"
#include "common/fe_context_utils.h"
#include "format_selector/manager/format_dtype_querier.h"
#include "ops_store/op_kernel_info_constructor.h"
#include "ops_store/op_kernel_info.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/node_utils.h"

namespace fe {
struct SupportedFormatAndDtype {
  OpKernelInfoPtr op_kernel_info_ptr;
  IndexNameMap input_index_name_map;
  IndexNameMap output_index_name_map;
  map<string, vector<ge::Format>> suppport_formats_map;
  map<string, vector<uint32_t>> suppport_sub_formats_map;
  map<string, vector<ge::DataType>> support_data_types_map;
  string reason;
  bool is_input;
  uint32_t cur_idx = 0;
  bool promote_flag = false;
  vector<vector<int>> promote_input_list;
  vector<ge::DataType> promote_target_type;

  SupportedFormatAndDtype(OpKernelInfoPtr op_kernel_info_ptr_param, string reason_param)
      : op_kernel_info_ptr(std::move(op_kernel_info_ptr_param)),
        reason(std::move(reason_param)),
        is_input(false) {}
};

enum class PrecisionReduceType {
  FP32_TO_BF16,
  FP32_TO_FP16,
  BF16_TO_FP16,
  INVALID
};
using PrecisonReduceCheckFunc = std::function<bool(const bool &, const fe::PrecisionMode &)>;
using PrecisonReduceCheckFuncArray =
    std::array<PrecisonReduceCheckFunc, static_cast<size_t>(PrecisionReduceType::INVALID)>;
using FormatDtypeQuerierPtr = std::shared_ptr<FormatDtypeQuerier>;

/**
 * @brief Store ops kernel info of only one specific implement.
 * FEOpsKernelInfoStore consists of at most 8 SubOpsStore
 */
class SubOpsStore {
 public:
  friend class FEOpsKernelInfoStore;
  explicit SubOpsStore(const std::string &engine_name);

  virtual ~SubOpsStore();

  SubOpsStore(const SubOpsStore &) = delete;

  SubOpsStore &operator=(const SubOpsStore &) = delete;

  /*
   * @ingroup fe
   * @brief : Initialize the SubOpsStore, load the op info from the specific.json file
   * @param[in] options: none
   * @return : SUCCESS/FAILED
   */
  virtual Status InitializeSubStore();

  /*
   * @ingroup fe
   * @brief : finalize the SubOpsStore, clear all op info and op kernel info;
   * @param[in] None
   * @return : SUCCESS/FAILED
   */
  Status FinalizeSubStore();

  /*
   * @ingroup fe
   * @brief : set the the FEOpsStoreInfo of this SubOpsStore;
   * @param[in|out] store_info : the FEOpsStoreInfo of this SubOpsStore;
   * @return : void
   */
  void SetSubStoreInfo(const FEOpsStoreInfo &sub_store_info);

  /*
   * @brief : Check whether the input op_desc is supported in this sub ops store
   */
  virtual bool CheckSubStoreSupported(const ge::NodePtr &node, const CheckSupportMode &check_mode,
                                      const bool &check_unknown_shape, CheckSupportParam &check_param) const;

  Status GetSupportFormatAndDtype(const ge::NodePtr &node,
                                  const OpKernelInfoPtr &op_kernel_info_ptr,
                                  const bool &is_dynamic_check,
                                  FormatDtypeInfo &format_dtype_info) const;

 protected:
  /*
   *  @ingroup fe
   *  @brief   check all attr value specified in OpKernelInfo to OpDesc,
   *           each Value of OpDesc Attr should containing in OpKernelInfo
   *  @param   [in] op_desc       : OpDesc from graph_ir node
   *  @param   [in] op_kernel_info : OpKernelInfo(om_optype) from sub ops store
   *  @return  true or false
   */
  bool CheckAttrSupport(const ge::NodePtr &node, const OpKernelInfo &op_kernel_info, std::string &reason) const;

  bool CheckParamType(const ge::NodePtr &node, OpKernelInfoPtr op_kernel_info_ptr,
                      const std::map<uint32_t, string> &input_index_name_map,
                      const std::map<uint32_t, string> &output_index_name_map, std::string &reason) const;

  bool CheckAllTensorsParamType(const ge::NodePtr &node, bool is_input,
                                const vector<InputOrOutputInfoPtr> &all_tesnors_info,
                                const map<uint32_t, string> &index_name_map, std::string &reason) const;

  bool CheckInputSupported(const ge::NodePtr &node, uint32_t input_size, const bool &is_force_dtype_support,
                           SupportedFormatAndDtype &info) const;

  bool CheckAllTensorsSupportedAccurateMode(const ge::NodePtr &node, SupportedFormatAndDtype &info) const;

  bool CheckOutputSupported(const ge::NodePtr &node, uint32_t output_size, const bool &is_force_dtype_support,
                            SupportedFormatAndDtype &info) const;

  bool CheckOpSupported(const ge::NodePtr &node_ptr, const bool &is_dynamic_impl,
                        CheckSupportParam &check_param) const;

 private:
  bool IsInputConst(const ge::NodePtr &node, const int32_t &index) const;

  bool CheckSingleTensorAccurateMode(uint32_t size, uint32_t type_index, const ge::NodePtr &node,
                                     SupportedFormatAndDtype &info, bool &need_continue) const;

  bool CheckSingleTensorSupportedAccurateMode(const ge::NodePtr &node, uint32_t index,
                                              uint32_t type_index, SupportedFormatAndDtype &info,
                                              bool &check_flag) const;

  bool CheckFormatAndDtypeNormalMode(const ge::NodePtr &node, const std::string &name,
                                     const ge::GeTensorDescPtr &tensor_desc,
                                     SupportedFormatAndDtype &info,
                                     const bool &is_force_dtype_support,
                                     std::string &reason) const;

  /*
   * @ingroup fe
   * @brief: check whether the dtype is supported by the FEOpsKernelInfoStore;
   * @param[in] tensor_desc_ptr    : a GeTensorDescPtr of the input/output tensor;
   * @param[in] tensor_desc_info_ptr : a InputOrOutputInfoPtr of the input
   * @return: true if supported, else false;
   */
  bool CheckDtypeSupported(const ge::NodePtr &node, const ge::GeTensorDescPtr &tensor_desc_ptr,
                           InputOrOutputInfoPtr input_or_output_info_ptr,
                           const vector<ge::DataType> &support_data_types,
                           const SupportedFormatAndDtype &info) const;

  /*
   * @ingroup fe
   * @brief: check whether the dtype is supported by the FEOpsKernelInfoStore;
   * @param[in] tensor_desc_ptr    : a GeTensorDescPtr of the input/output tensor;
   * @param[in] tensor_desc_info_ptr : a InputOrOutputInfoPtr of the input
   * @return: true if supported, else false;
   */
  bool CheckAccuracyDtypeSupported(ge::ConstGeTensorDescPtr c_tensor_desc_ptr,
                                   uint32_t type_index, InputOrOutputInfoPtr input_or_output_info_ptr,
                                   const vector<ge::DataType> &support_data_types) const;

  /*
   *  @ingroup fe
   *  @brief   check op_desc_attr.Value() is in info_op_attr.std::vector<Values>
   *  @param   [in] op_desc_attr : one of GeAttrValue from OpDesc
   *  @param   [in] info_op_attr : one of std::vector<GeAttrValue> from OpKernelInfo
   *  @param   [in] attr_type   : enum of ValueType, from OpKernelInfo
   *  @return  true or false
   */
  bool CheckAttrValue(const ge::GeAttrValue &op_desc_attr, const std::vector<ge::GeAttrValue> &info_op_attr,
                      ge::GeAttrValue::ValueType attr_type) const;

  /*
   * @ingroup fe
   * @brief: check whether the format is supported by this sub ops store
   * @param[in] c_tensor_desc_ptr    : a ConstGeTensorDescPtr of the input/output tensor;
   * @param[in] tensor_desc_info_ptr : a InputOrOutputInfoPtr of the input
   * @return: true if supported, else false;
   */
  bool CheckFormatSupported(const ge::NodePtr &node, ge::ConstGeTensorDescPtr c_tensor_desc_ptr,
                            InputOrOutputInfoPtr input_or_output_info_ptr,
                            const vector<ge::Format> &support_formats) const;

  /*
   * @ingroup fe
   * @brief: check whether the groups is supported by this sub ops store
   * @param[in] info : a SupportedFormatAndDtype of the input/output tensor;
   * @param[in] tensor_desc_info_ptr : a InputOrOutputInfoPtr of the input
   * @return: true if supported, else false;
   */
  bool CheckSubformatSupported(const ge::NodePtr &node, int64_t groups,
                               const InputOrOutputInfoPtr &input_or_output_info_ptr,
                               const string &tensor_name, SupportedFormatAndDtype &info) const;
  /*
   * @ingroup fe
   * @brief: check whether the format is supported by this sub ops store
   * @param[in] c_tensor_desc_ptr    : a ConstGeTensorDescPtr of the input/output tensor;
   * @param[in] type_index         : the index of format in op_store
   * @param[in] tensor_desc_info_ptr : a InputOrOutputInfoPtr of the input
   * @return: true if supported, else false;
   */
  bool CheckAccuracyFormatSupported(ge::ConstGeTensorDescPtr c_tensor_desc_ptr,
                                    uint32_t type_index, InputOrOutputInfoPtr input_or_output_info_ptr,
                                    const vector<ge::Format> &support_formats) const;

  /*
   * @ingroup fe
   * @brief: check whether the input whose value depend is required links with a const or constant node.
   * @param[in] info           : a struct object about op kernel info
   * @param[in] input_name     : the name of input desc
   * @param[in] is_input_const : the value from GetIsInputConst
   * @return: true if supported, else false;
   */
  bool CheckInputConstValueDepend(const ge::NodePtr &node, const string &input_name,
                                  const uint32_t &in_index, SupportedFormatAndDtype &info) const;

  void LogAllFormatAndDtype(const SupportedFormatAndDtype &info, const string &tensor_name,
                            std::ostringstream &reason_oss, string &reason) const;

  bool IsAllDtypeExpected(ge::DataType dtype_expected, ge::DataType dtype_unexpected,
                          const vector<ge::DataType> &dtypes) const;

  void GetIsInputConstNodeVec(const ge::NodePtr &node, vector<bool> &is_input_const_vec) const;

  bool IsSupportExpectedDtype(ge::DataType dtype_expected, const vector<ge::DataType> &dtypes) const;

  bool CheckCustomizeDtype(const ge::OpDescPtr &op_desc, const SupportedFormatAndDtype &info,
                           bool &is_force_dtype_support) const;

  void FilterDtypesByCustom(const vector<ge::DataType> &cust_dtypes,
                            const IndexNameMap &index_map,
                            const SupportedFormatAndDtype &info,
                            const bool &is_input,
                            vector<bool> &filer_index) const;

  void SetAttrParamTypeList(const ge::OpDescPtr &op_desc, const OpKernelInfoPtr &op_kernel_info_ptr,
                            const SupportedFormatAndDtype &info) const;
  PrecisionReduceType GetPrecisionReduceType(const ge::DataType &desc_dtype, const ge::DataType &store_dtype) const;

  bool PrecisionReduceToBf16(const bool &black_list_op, const fe::PrecisionMode &precision_mode) const;

  bool PrecisionReduceToFp16(const bool &black_list_op, const fe::PrecisionMode &precision_mode) const;

  bool CheckPrecisionReduce(const ge::DataType &desc_dtype, const ge::DataType &stored_dtype, const int64_t &keep_dtype,
                            const bool &black_list_op, const fe::PrecisionMode &precision_mode) const;

  static bool VerifyFormatC0Val(const ge::OpDescPtr &op_desc_ptr, const ge::ConstGeTensorDescPtr &tensor_desc_ptr);

  void JudgeNeedUpdateDtype(const fe::PrecisionMode &precision_mode, const ge::GeTensorDescPtr &tensor_desc_ptr,
      const vector<ge::DataType> &support_data_types, const ge::OpDescPtr &op_desc_ptr,
      const InputOrOutputInfoPtr &input_or_output_info_ptr) const;

  void FeedPromoteInfo(const ge::NodePtr &node, SupportedFormatAndDtype &info) const;

  void GetPromoteFlag(const ge::NodePtr &node, SupportedFormatAndDtype &info) const;

  void GetPromoteInputList(const ge::NodePtr &node, SupportedFormatAndDtype &info) const;

  bool ParsePromoteStr(const PromoteTypeVal &promote_type_val, const ge::NodePtr &node,
      const OpKernelInfoPtr &op_kernel_info_ptr, std::vector<std::vector<int>> &promote_list) const;

  void TransInputNameToIdx(const std::vector<std::string> &promote_item, const ge::NodePtr &node,
                           std::vector<std::vector<int>> &promote_list) const;

  static bool CheckTensorNotNull(const ge::GeShape &ge_shape);

  static bool CheckPromoteTypeSupport(const vector<ge::DataType> &support_data_types,
                                      const SupportedFormatAndDtype &info);

  protected:
  bool init_flag_;

  std::string engine_name_;

  FEOpsStoreInfo sub_store_info_;

  FormatDtypeQuerierPtr format_dtype_querier_ptr_;

  const PrecisonReduceCheckFuncArray precison_reduce_checkfuncs_ = {
      std::bind(&SubOpsStore::PrecisionReduceToBf16, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&SubOpsStore::PrecisionReduceToFp16, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&SubOpsStore::PrecisionReduceToFp16, this, std::placeholders::_1, std::placeholders::_2)};
};
}  // namespace fe
#endif  // FUSION_ENGINE_OPTIMIZER_OPS_KERNEL_STORE_SUB_OPS_STORE_H_
