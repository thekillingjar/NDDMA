/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_DEPLOY_RESOURCE_DEVICE_INFO_H_
#define AIR_RUNTIME_HETEROGENEOUS_DEPLOY_RESOURCE_DEVICE_INFO_H_

#include <cstdint>
#include <string>
#include <vector>
#include "graph/types.h"
#include "graph_metadef/common/ge_common/util.h"
#include "framework/common/string_util.h"

namespace ge {
class DeviceInfo {
 public:
  DeviceInfo() = default;
  DeviceInfo(int32_t node_id, int32_t device_id) noexcept;
  DeviceInfo(int32_t node_id, DeviceType device_type, int32_t device_id) noexcept;

  int32_t GetNodeId() const;
  inline void SetNodeId(int32_t node_id) {
    node_id_ =  node_id;
  }

  int32_t GetDeviceId() const;
  inline void SetDeviceId(int32_t device_id) {
    device_id_ =  device_id;
  }

  int32_t GetPhyDeviceId() const {
    return phy_device_id_;
  }

  inline void SetPhyDeviceId(int32_t phy_device_id) {
    phy_device_id_ =  phy_device_id;
  }

  int32_t GetHcomDeviceId() const {
    return hcom_device_id_;
  }

  inline void SetHcomDeviceId(int32_t hcom_device_id) {
    hcom_device_id_ =  hcom_device_id;
  }

  int32_t GetOsId() const {
    return os_id_;
  }

  inline void SetOsId(int32_t os_id) {
    os_id_ =  os_id;
  }

  int32_t GetDeviceIndex() const {
    return device_index_;
  }

  inline void SetDeviceIndex(int32_t device_index) {
    device_index_ =  device_index;
  }

  const std::string &GetHostIp() const;
  void SetHostIp(const std::string &host_ip);

  inline int32_t GetNodePort() const {
    return node_port_;
  }

  inline void SetNodePort(int32_t node_port) {
    node_port_ = node_port;
  }

  const std::string &GetDeviceIp() const;
  void SetDeviceIp(const std::string &device_ip);

  int32_t GetDgwPort() const;
  void SetDgwPort(int32_t dgw_port);

  DeviceType GetDeviceType() const;
  inline void SetDeviceType(DeviceType device_type) {
    device_type_ = device_type;
  }

  inline const std::string &GetResourceType() const {
    return resource_type_;
  }

  inline void SetResourceType(const std::string &resource_type) {
    resource_type_ = resource_type;
  }

  inline const std::string &GetAvailablePorts() const {
    return available_ports_;
  }

  inline void SetAvailablePorts(const std::string &available_ports) {
    available_ports_ = available_ports;
  }

  inline void SetSupportHcom(bool support_hcom) {
    support_hcom_ = support_hcom;
  }

  inline bool SupportHcom() const {
    return support_hcom_;
  }

  inline void SetNodeMeshIndex(const std::vector<int32_t> &node_mesh_index) {
    node_mesh_index_.clear();
    for (auto index : node_mesh_index) {
      node_mesh_index_.emplace_back(index);
    }
    const uint8_t ai_server_node_mesh_size = 2;
    if (node_mesh_index.size() == ai_server_node_mesh_size) {
      node_mesh_index_.emplace_back(device_index_);
      node_mesh_index_.emplace_back(0);
    } else {
      node_mesh_index_.emplace_back(device_id_);
    }
  }

  inline std::vector<int32_t> GetNodeMeshIndex() const {
    return node_mesh_index_;
  }

  inline std::string GetKey() const {
    return std::to_string(static_cast<int32_t>(device_type_)) + "_" +
           std::to_string(node_id_) + "_" + std::to_string(device_id_);
  }

  inline bool SupportFlowgw() const {
    return support_flowgw_;
  }

  inline void SetSupportFlowgw(bool support_flowgw) {
    support_flowgw_ = support_flowgw;
  }

  inline int64_t GetSuperDeviceId() const {
    return super_device_id_;
  }

  inline void SetSuperDeviceId(int64_t super_device_id) {
    super_device_id_ = super_device_id;
  }

  inline std::string ToIndex() const {
    return StrJoin(node_mesh_index_.begin(), node_mesh_index_.end(), ":");
  }

  inline std::string DebugString() const {
    auto debug_string = "node_id = " + std::to_string(node_id_) + ", " +
                        "device_id = " + std::to_string(device_id_) + ", " +
                        "device_type = " + std::to_string(static_cast<int32_t>(device_type_)) + ", " +
                        "node_ip = " + host_ip_ + ", " +
                        "device_ip = " + device_ip_ + ", " +
                        "resource_type = " + resource_type_ + ", " +
                        "phy_device_id = " + std::to_string(phy_device_id_) + ", " +
                        "hcom_device_id = " + std::to_string(hcom_device_id_) + ", " +
                        "device_index = " + std::to_string(device_index_) + ", " +
                        "node_mesh_index = " + ToString(node_mesh_index_) + ", " +
                        "os_id = " + std::to_string(os_id_) + ", " +
                        "node_port = " + std::to_string(node_port_) + ", " +
                        "available_ports = " + available_ports_;
    return debug_string;
  }

 private:
  int32_t node_id_ = 0;
  DeviceType device_type_ = NPU;
  int32_t device_id_ = -1;
  int32_t phy_device_id_ = -1;
  int32_t hcom_device_id_ = -1;
  int32_t device_index_ = -1;
  int32_t dgw_port_ = -1;
  std::string host_ip_;
  std::string device_ip_;
  std::string resource_type_;
  std::vector<int32_t> node_mesh_index_;
  int32_t node_port_ = -1;
  bool support_flowgw_ = true;
  bool support_hcom_ = true;
  int32_t os_id_ = 0;
  std::string available_ports_;
  int64_t super_device_id_ = -1;
};
}  // namespace ge

#endif  // AIR_RUNTIME_HETEROGENEOUS_DEPLOY_RESOURCE_DEVICE_INFO_H_
