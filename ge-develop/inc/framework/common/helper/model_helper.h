/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_COMMON_HELPER_MODEL_HELPER_H_
#define INC_FRAMEWORK_COMMON_HELPER_MODEL_HELPER_H_

#include <memory>
#include <string>

#include "framework/common/helper/om_file_helper.h"
#include "framework/common/helper/model_save_helper.h"
#include "framework/common/types.h"
#include "graph/model.h"
#include "platform/platform_info.h"
#include "common/op_so_store/op_so_store.h"
#include "common/host_resource_center/host_resource_serializer.h"

namespace ge {
using NodeRefreshInfo = std::map<NodePtr, std::map<NodePtr, std::vector<std::pair<size_t, int64_t>>>>;
class GeModel;
class GeRootModel;
class GE_FUNC_VISIBILITY ModelHelper : public ModelSaveHelper {
 public:
  ModelHelper() noexcept = default;
  virtual ~ModelHelper() override = default;
  ModelHelper(const ModelHelper &) = default;
  ModelHelper &operator=(const ModelHelper &) & = default;

  Status SaveToOmModel(const GeModelPtr &ge_model, const std::string &output_file, ge::ModelBufferData &model,
                       const GeRootModelPtr &ge_root_model = nullptr) override;
  Status GenerateGeModel(const OmFileLoadHelper &om_load_helper, GeModelPtr &cur_model, GeModelPtr &first_ge_model,
                         const size_t mode_index, const bool is_dyn_root) const;
  Status SaveToOmRootModel(const GeRootModelPtr &ge_root_model, const std::string &output_file, ModelBufferData &model,
                           const bool is_unknown_shape) override;
  Status SaveOriginalGraphToOmModel(const ge::Graph &graph, const std::string &output_file) const;

  Status LoadModel(const ge::ModelData &model_data);
  Status LoadRootModel(const ge::ModelData &model_data);
  static Status GetModelFileHead(const ge::ModelData &model_data, const ModelFileHeader *&file_header);
  static Status SetModelToGeModel(const GeModelPtr &ge_model, const GeModelPtr &first_ge_model, Model &model);
  static Status SaveBundleModelBufferToMem(const std::vector<ModelBufferData> &model_buffers, uint64_t var_size,
                                           ModelBufferData &output_buffer);
  static std::string GetOutputFileName() {
    return output_file_name_;
  }
  Status LoadPartInfoFromModel(const ge::ModelData &model_data, ModelPartition &partition);

  GeModelPtr GetGeModel();
  GeRootModelPtr GetGeRootModel();
  virtual void SetSaveMode(const bool val) override {
    is_offline_ = val;
  }

  void SetSharedWeightFlag(const bool val) {
    is_shared_weight_ = val;
  }

  bool GetModelType() const {
    return is_unknown_shape_model_;
  }

  Status GetBaseNameFromFileName(const std::string &file_name, std::string &base_name) const;

  // for soft sync op
  Status GetHardwareInfo(std::map<std::string, std::string> &options) const;
  Status HandleDeviceInfo(fe::PlatFormInfos &platform_infos) const;
  Status HandleDeviceInfo(fe::PlatFormInfos &platform_infos, fe::PlatformInfo &origin_platform_info) const;
  static Status InitRuntimePlatform();

  Status InitRuntimeAndGetDevicePlatformInfos(int32_t device_id, const std::string &soc_version,fe::PlatFormInfos &platform_infos_device) const;

  Status UpdateCoreCountWithOption(const std::string &key, const std::string &context_key, uint32_t core_num_ini,
                                 uint32_t &platform_info_count, std::map<std::string, std::string> &options) const;

  Status UpdateCoreCountWithDevice(const std::string &key, uint32_t core_num_ini,
                                            const std::string &core_num_str, uint32_t &platform_info_count) const;

  void UpdateCoreCountWithRuntime(const std::string &key, uint32_t platform_count,
                                            const int64_t core_num_rts, uint32_t &platform_info_count) const;

  Status GetPlatformInfo(int32_t device_id, const std::string &soc_version, fe::PlatformInfo &platform_info,
                         int32_t &virtual_type) const;
  Status GetPlatformInfo(int32_t device_id, const std::string &soc_version, fe::PlatformInfo &platform_info,
                         int32_t &virtual_type, std::map<std::string, std::string> &options) const;
  Status SetPlatformInfos(const std::string &soc_version, const fe::PlatformInfo &platform_info,
                          fe::PlatFormInfos &platform_infos) const;

  Status UpdatePlatfromInfoWithOption(std::map<std::string, std::string> &options, const uint32_t ai_core_cnt_ini,
                          const uint32_t vector_core_cnt_ini, fe::PlatformInfo &platform_info) const;

  Status UpdatePlatfromInfoWithDevice(fe::PlatFormInfos &platformInfos_device, const uint32_t ai_core_cnt_ini,
                          const uint32_t vector_core_cnt_ini, fe::PlatformInfo &platform_info) const;

  Status UpdatePlatfromInfoWithRuntime(const int32_t device_id, const uint32_t ai_core_cnt_ini, const uint32_t vector_core_cnt_ini,
                                    fe::PlatformInfo &platform_info, int32_t &virtual_type) const;

  Status CheckOsCpuInfoAndOppVersion();

  Status GetSoBinData(const string &cpu_info, const string &os_info);

  const uint8_t *GetOpSoStoreData() const;

  size_t GetOpStoreDataSize() const;

  void SetRepackSoFlag(const bool val);

  Status PackSoToModelData(const ModelData &model_data, const std::string &output_file, ModelBufferData &model_buffer,
                           const bool save_to_file = true);
  bool IsSoStore() const {
    return is_so_store_;
  }

  // Configure attribute compression enabled flag (default true for backward compatibility)
  void SetAttrCompressionEnabled(bool enabled) {
    attr_compression_enabled_ = enabled;
  }

  // configuration attr compression mode
  Status ConfigureAttrCompressionMode(const string &mode) override;
  static Status UpdateGeRootModelTaskAddr(const GeRootModelPtr &ge_root_model,
                                          const ComputeGraphPtr &root_graph,
                                          std::set<ComputeGraph *> &refreshed_graphs,
                                          const bool is_cache);
  static Status UpdateSessionGraphId(const ComputeGraphPtr &graph,
                                     const std::string &session_graph_id,
                                     bool &refreshed);
  static constexpr const char_t *kFilePreffix = ".exeom";
  static constexpr const char_t *kDebugPreffix = ".dbg";

 protected:
  Status SaveModelCustAICPU(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                            const size_t model_index = 0U) const;
  static Status GetOppVersion(std::string &version);
  static Status EnsureKernelBuilt(const GeModelPtr &model);
  Status SaveSoStoreModelPartitionInfo(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper,
                                       const GeRootModelPtr &ge_root_model, string &output_file_name,
                                       const GeModelPtr &first_ge_model);
  Status SaveModelHeader(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                         const size_t model_num = 1U, const bool need_check_os_cpu = false,
                         const bool is_unknow_shape = false) const;
  Status SaveModelPartition(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const ModelPartitionType type,
                            const uint8_t *const data, const size_t size, const size_t model_index) const;
  Status SaveModelWeights(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                          const size_t model_index = 0U) const;
  Status SaveModelIntroduction(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                               const bool is_dynamic = false) const;
  Status SaveModelTbeKernel(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                            const size_t model_index = 0U) const;
  GeModelPtr model_;
  bool is_so_store_ = false;

 private:
  static Status RecordOffsetsRefreshInfo(const ComputeGraphPtr &graph,
                                         const std::map<int64_t, NodePtr> &unrefreshed_offsets,
                                         NodeRefreshInfo &inputs_need_refresh, NodeRefreshInfo &outputs_need_refresh);
  static Status RefreshNodeOffset(const NodeRefreshInfo &inputs_need_refresh,
                                  const NodeRefreshInfo &outputs_need_refresh,
                                  std::map<int64_t, int64_t> &logical_addr_mapping);
  static Status RefreshAndGetAddrMapping(const ComputeGraphPtr &graph, const uint64_t session_id, const bool is_cache,
                                         std::map<int64_t, int64_t> &logical_addr_mapping,
                                         std::set<ComputeGraph *> &refreshed_graphs);
  static Status UpdateModelTaskAddr(const GeRootModelPtr &ge_root_model, const uint64_t session_id,
                                    const std::map<int64_t, int64_t> &logical_addr_mapping,
                                    std::set<ComputeGraph *> &refreshed_graphs);
 private:
  bool is_assign_model_ = false;
  bool is_offline_ = true;
  bool save_to_file_ = true;
  bool is_unknown_shape_model_ = false;
  bool is_shared_weight_ = false;
  const ModelFileHeader *file_header_ = nullptr;
  GeRootModelPtr root_model_;
  OpSoStore op_so_store_;
  bool is_repack_so_ = false;
  bool is_need_compress_ = true;
  static std::string output_file_name_;
  std::unordered_set<std::string> custom_compiler_versions_{};
  HostResourceSerializer host_serializer_;

  // Attribute compression configuration (only supports ATC command line)
  bool attr_compression_enabled_ = true;

  // Calculate final compression decision based on enabled flag
  bool ShouldCompress() const {
    return attr_compression_enabled_ && is_offline_ && is_need_compress_;
  }

  bool IsPartitionedGraph(const GeModelPtr &cur_model) const;

  Status GenerateGeRootModel(const OmFileLoadHelper &om_load_helper, const ModelData &model_data);

  Status CheckIfWeightPathValid(const ge::ComputeGraphPtr &graph, const ge::ModelData &model_data) const;
  Status LoadModelData(const OmFileLoadHelper &om_load_helper, const GeModelPtr &cur_model,
                       const GeModelPtr &first_ge_model, const size_t mode_index) const;
  virtual Status LoadWeights(const OmFileLoadHelper &om_load_helper, const GeModelPtr &cur_model,
                             const size_t mode_index) const;
  Status LoadTask(const OmFileLoadHelper &om_load_helper, const GeModelPtr &cur_model, const size_t mode_index) const;
  Status LoadTBEKernelStore(const OmFileLoadHelper &om_load_helper, const GeModelPtr &cur_model,
                            const size_t mode_index) const;
  Status LoadCustAICPUKernelStore(const OmFileLoadHelper &om_load_helper, const GeModelPtr &cur_model,
                                  const size_t mode_index) const;

  Status SaveModelDef(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                      Buffer &model_buffer, const size_t model_index = 0U) const;
  Status SaveSizeToModelDef(const GeModelPtr &ge_model, const size_t model_index) const;

  Status SaveModelTaskDef(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                          Buffer &task_buffer, const size_t model_index = 0U) const;
  Status SaveAllModelPartiton(shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeModelPtr &ge_model,
                              Buffer &model_buffer, Buffer &task_buffer, const size_t model_index = 0U) const;

  Status LoadOpSoBin(const OmFileLoadHelper &om_load_helper, const GeRootModelPtr &ge_root_model) const;
  Status LoadTilingData(const OmFileLoadHelper &om_load_helper, const GeRootModelPtr &ge_root_model) const;
  Status SaveTilingData(std::shared_ptr<OmFileSaveHelper> &om_file_save_helper, const GeRootModelPtr &ge_root_model);
  void SaveOpSoInfo(const GeRootModelPtr &ge_root_model) const;
  Status SetModelCompilerVersion(const GeModelPtr &first_ge_model);
  Status LoadAndStoreOppSo(const string &path, bool is_split, bool is_sub_pkg = false);
  void SaveOutNodesFromRootGraph(const GeRootModelPtr &ge_root_model, GeModelPtr &first_ge_model) const;
  Status LoadAndStoreOppSo(const std::unordered_set<std::string> &op_so_set, const SoBinType so_bin_type);
  Status SaveSpaceRegistrySoBin(const GeRootModelPtr &ge_root_model, const GeModelPtr &first_ge_model,
                                string &output_file_name);
  Status SaveOpMasterDeviceSoBin(const GeRootModelPtr &ge_root_model);
  Status SaveAutofuseSoBin(const GeRootModelPtr &ge_root_model);
};
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_HELPER_MODEL_HELPER_H_
