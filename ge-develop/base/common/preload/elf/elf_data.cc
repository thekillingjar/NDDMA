/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/preload/elf/elf_data.h"
#include <functional>
#include <securec.h>
#include "graph/ge_error_codes.h"
#include "common/checker.h"
#include "graph/def_types.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace {
constexpr size_t kElfClassIdx = 4UL;
constexpr size_t kElfDataIdx = 5UL;
constexpr int32_t kElfClass64 = 2;
constexpr size_t k2ByteSize = 2UL;
constexpr size_t k4ByteSize = 4UL;
constexpr size_t k8ByteSize = 8UL;
constexpr size_t k16ByteSize = 16UL;
constexpr uint32_t kShfAlloc = 0x2U; // 表示在进程运行时，占用内存

using GetByteFunc = std::function<uint64_t(const uint8_t[], const int32_t)>;

enum class ElfDataFormat : int32_t {
  kElfDataNone = 0,
  kElfData2Lsb = 1,
  kElfData2Msb = 2,
  kEnd = 3
};

struct Elf64ExternalShdr final {
  std::array<uint8_t, k4ByteSize> sh_name;
  std::array<uint8_t, k4ByteSize> sh_type;
  std::array<uint8_t, k8ByteSize> sh_flags;
  std::array<uint8_t, k8ByteSize> sh_addr;
  std::array<uint8_t, k8ByteSize> sh_offset;
  std::array<uint8_t, k8ByteSize> sh_size;
  std::array<uint8_t, k4ByteSize> sh_link;
  std::array<uint8_t, k4ByteSize> sh_info;
  std::array<uint8_t, k8ByteSize> sh_addralign;
  std::array<uint8_t, k8ByteSize> sh_entsize;
};

struct Elf64ExternalEhdr {
  std::array<uint8_t, k16ByteSize> e_ident;
  std::array<uint8_t, k2ByteSize> e_type;
  std::array<uint8_t, k2ByteSize> e_machine;
  std::array<uint8_t, k4ByteSize> e_version;
  std::array<uint8_t, k8ByteSize> e_entry;
  std::array<uint8_t, k8ByteSize> e_phoff;
  std::array<uint8_t, k8ByteSize> e_shoff;
  std::array<uint8_t, k4ByteSize> e_flags;
  std::array<uint8_t, k2ByteSize> e_ehsize;
  std::array<uint8_t, k2ByteSize> e_phentsize;
  std::array<uint8_t, k2ByteSize> e_phnum;
  std::array<uint8_t, k2ByteSize> e_shentsize;
  std::array<uint8_t, k2ByteSize> e_shnum;
  std::array<uint8_t, k2ByteSize> e_shstrndx;
};

struct ElfInternalEhdr {
  std::array<uint8_t, k16ByteSize> e_ident;
  uint64_t e_entry;
  uint64_t e_phoff;
  uint64_t e_shoff;
  uint64_t e_version;
  uint64_t e_flags;
  uint16_t e_type;
  uint16_t e_machine;
  uint32_t e_ehsize;
  uint32_t e_phentsize;
  uint32_t e_phnum;
  uint32_t e_shentsize;
  uint32_t e_shnum;
  uint32_t e_shstrndx;
};

struct ElfInternalShdr {
  uint32_t sh_name;
  uint32_t sh_type;
  uint64_t sh_flags;
  uint64_t sh_addr;
  uint64_t sh_offset;
  uint64_t sh_size;
  uint32_t sh_link;
  uint32_t sh_info;
  uint64_t sh_addralign;
  uint64_t sh_entsize;
};

struct ElfData {
  uint8_t *obj_ptr = nullptr;
  uint8_t *obj_ptr_origin = nullptr;
  uint64_t obj_size;
  struct ElfInternalEhdr elf_header = {};
  std::vector<ElfInternalShdr> section_headers;
  uint64_t text_offset;
  uint64_t text_size;
  uint32_t kernel_num;
  uint32_t func_num;
  const char_t *so_name = nullptr;
  bool degenerate_flag;
  uint64_t stack_size;
  bool data_flag;
  bool contains_ascend_meta;
};
  
uint64_t ByteGetBigEndian(const uint8_t field[], const size_t size) {
  uint64_t output_byte = 0UL;
  constexpr size_t kMaxShift = 63;
  for (size_t i = 0UL; i < static_cast<size_t>(size); i++) {
    const size_t shift = k8ByteSize * (size - i - 1UL);
    if (shift <= kMaxShift) {
      output_byte |= (static_cast<uint64_t>(field[i]) << shift);
    }
  }
  return output_byte;
}
  
uint64_t ByteGetLittleEndian(const uint8_t field[], const int32_t size) {
  uint64_t output_byte = 0UL;
  for (size_t i = 0UL; i < static_cast<size_t>(size); i++) {
    output_byte |= (static_cast<uint64_t>(field[i]) << (k8ByteSize * i));
  }
  return output_byte;
}
  
GetByteFunc SwitchGetByteFunc(const int32_t elf_format) {
  GetByteFunc func;
  switch (elf_format) {
    case static_cast<int32_t>(ElfDataFormat::kElfData2Msb):
      func = &ByteGetBigEndian;
      break;
    case static_cast<int32_t>(ElfDataFormat::kElfDataNone):
    case static_cast<int32_t>(ElfDataFormat::kElfData2Lsb):
      func = &ByteGetLittleEndian;
      break;
    default:
      func = &ByteGetLittleEndian;
      break;
  }
  return func;
}

Status GetFileHeader(ElfData &elf_data) {
  GE_ASSERT_TRUE(memcpy_s(elf_data.elf_header.e_ident.data(), k16ByteSize, elf_data.obj_ptr, k16ByteSize) == EOK);
  elf_data.obj_ptr += k16ByteSize;
  auto get_byte_func = SwitchGetByteFunc(static_cast<int32_t>(elf_data.elf_header.e_ident[kElfDataIdx]));
  GE_ASSERT_TRUE(static_cast<int32_t>(elf_data.elf_header.e_ident[kElfClassIdx]) == kElfClass64, "Elf can not be 32 bit.");
  size_t offset = 0UL;
  elf_data.elf_header.e_type = static_cast<uint16_t>(get_byte_func(static_cast<const uint8_t *>(&elf_data.obj_ptr[offset]), static_cast<int32_t>(k2ByteSize)));
  offset += k2ByteSize;
  elf_data.elf_header.e_machine = static_cast<uint16_t>(get_byte_func(static_cast<const uint8_t *>(&elf_data.obj_ptr[offset]), static_cast<int32_t>(k2ByteSize)));
  offset += k2ByteSize;
  elf_data.elf_header.e_version = get_byte_func(static_cast<const uint8_t *>(&elf_data.obj_ptr[offset]), static_cast<int32_t>(k4ByteSize));
  offset += k4ByteSize;
  elf_data.elf_header.e_entry = get_byte_func(static_cast<const uint8_t *>(&elf_data.obj_ptr[offset]), static_cast<int32_t>(k8ByteSize));
  offset += k8ByteSize;
  elf_data.elf_header.e_phoff = get_byte_func(static_cast<const uint8_t *>(&elf_data.obj_ptr[offset]), static_cast<int32_t>(k8ByteSize));
  offset += k8ByteSize;
  elf_data.elf_header.e_shoff = get_byte_func(static_cast<const uint8_t *>(&elf_data.obj_ptr[offset]), static_cast<int32_t>(k8ByteSize));
  offset += k8ByteSize;
  elf_data.elf_header.e_flags = get_byte_func(static_cast<const uint8_t *>(&elf_data.obj_ptr[offset]), static_cast<int32_t>(k4ByteSize));
  offset += k4ByteSize;
  elf_data.elf_header.e_ehsize = static_cast<uint32_t>(get_byte_func(static_cast<const uint8_t *>(&elf_data.obj_ptr[offset]), static_cast<int32_t>(k2ByteSize)));
  offset += k2ByteSize;
  elf_data.elf_header.e_phentsize = static_cast<uint32_t>(get_byte_func(static_cast<const uint8_t *>(&elf_data.obj_ptr[offset]), static_cast<int32_t>(k2ByteSize)));
  offset += k2ByteSize;
  elf_data.elf_header.e_phnum = static_cast<uint32_t>(get_byte_func(static_cast<const uint8_t *>(&elf_data.obj_ptr[offset]), static_cast<int32_t>(k2ByteSize)));
  offset += k2ByteSize;
  elf_data.elf_header.e_shentsize = static_cast<uint32_t>(get_byte_func(static_cast<const uint8_t *>(&elf_data.obj_ptr[offset]), static_cast<int32_t>(k2ByteSize)));
  offset += k2ByteSize;
  elf_data.elf_header.e_shnum = static_cast<uint32_t>(get_byte_func(static_cast<const uint8_t *>(&elf_data.obj_ptr[offset]), static_cast<int32_t>(k2ByteSize)));
  offset += k2ByteSize;
  elf_data.elf_header.e_shstrndx = static_cast<uint32_t>(get_byte_func(static_cast<const uint8_t *>(&elf_data.obj_ptr[offset]), static_cast<int32_t>(k2ByteSize)));
  offset += k2ByteSize;
  elf_data.obj_ptr += offset;
  return GRAPH_SUCCESS;
}

Status Get64BitSectionHeaders(ElfData &elf_data) {
  const uint32_t sh_size = elf_data.elf_header.e_shentsize;
  const uint32_t sh_num = elf_data.elf_header.e_shnum;
  GE_ASSERT_TRUE((sh_size > 0U) && (sh_num > 0U), "The value of e_shentsize: %u field or e_shnum: %u should more than 0.", sh_size, sh_num);
  GE_ASSERT_TRUE((sh_num <= (~(0UL) / sh_size)), "The value of e_shentsize: %u and e_shnum: %u is invalid.", sh_size, sh_num);
  GE_ASSERT_TRUE((static_cast<size_t>(sh_size) == sizeof(Elf64ExternalShdr)),
      "The value of e_shentsize: %zu should equal to size of elf section header: %zu.", sh_size, sizeof(Elf64ExternalShdr));
  
  Elf64ExternalShdr *shdrs = PtrToPtr<uint8_t, Elf64ExternalShdr>(&elf_data.obj_ptr_origin[elf_data.elf_header.e_shoff]);
  GE_ASSERT_NOTNULL(shdrs);
  elf_data.section_headers.resize(static_cast<uint64_t>(sh_num));
  auto get_byte_func = SwitchGetByteFunc(static_cast<int32_t>(elf_data.elf_header.e_ident[kElfDataIdx]));
  for (size_t i = 0U; i < sh_num; i++) {
    const uint64_t obj_offset = sizeof(Elf64ExternalShdr) * static_cast<uint64_t>(i + 1U) + elf_data.elf_header.e_shoff;
    GE_ASSERT_TRUE(obj_offset <= elf_data.obj_size,
        "Section %u is out of obj, num:%u, e_shoff:%lu, obj size: %lu, Elf64ExternalShdr size: %zu.",
        i, sh_num, elf_data.elf_header.e_shoff, elf_data.obj_size, sizeof(Elf64ExternalShdr));
    elf_data.section_headers[i].sh_name = static_cast<uint32_t>(get_byte_func(static_cast<const uint8_t *>(shdrs[i].sh_name.data()), static_cast<int32_t>(k4ByteSize)));
    elf_data.section_headers[i].sh_type = static_cast<uint32_t>(get_byte_func(static_cast<const uint8_t *>(shdrs[i].sh_type.data()), static_cast<int32_t>(k4ByteSize)));
    elf_data.section_headers[i].sh_flags = get_byte_func(static_cast<const uint8_t *>(shdrs[i].sh_flags.data()), static_cast<int32_t>(k8ByteSize));
    elf_data.section_headers[i].sh_addr = get_byte_func(static_cast<const uint8_t *>(shdrs[i].sh_addr.data()), static_cast<int32_t>(k8ByteSize));
    elf_data.section_headers[i].sh_size = get_byte_func(static_cast<const uint8_t *>(shdrs[i].sh_size.data()), static_cast<int32_t>(k8ByteSize));
    elf_data.section_headers[i].sh_entsize = get_byte_func(static_cast<const uint8_t *>(shdrs[i].sh_entsize.data()), static_cast<int32_t>(k8ByteSize));
    elf_data.section_headers[i].sh_link = static_cast<uint32_t>(get_byte_func(static_cast<const uint8_t *>(shdrs[i].sh_link.data()), static_cast<int32_t>(k8ByteSize)));
    elf_data.section_headers[i].sh_info = static_cast<uint32_t>(get_byte_func(static_cast<const uint8_t *>(shdrs[i].sh_info.data()), static_cast<int32_t>(k8ByteSize)));
    elf_data.section_headers[i].sh_offset = get_byte_func(static_cast<const uint8_t *>(shdrs[i].sh_offset.data()), static_cast<int32_t>(k8ByteSize));
    elf_data.section_headers[i].sh_addralign = get_byte_func(static_cast<const uint8_t *>(shdrs[i].sh_addralign.data()), static_cast<int32_t>(k8ByteSize));
    GE_ASSERT_TRUE(elf_data.section_headers[i].sh_link <= sh_num, "Section %u sh_link value:%lu invalid, which should in range [0, %u].",
        i, elf_data.section_headers[i].sh_link, sh_num);
  }
  return GRAPH_SUCCESS;
}

Status ProcessSectionTableGetOffset(ElfData &elf_data, uint32_t &offset) {
  GE_ASSERT_SUCCESS(Get64BitSectionHeaders(elf_data));
  for (uint32_t i = 0U; i < elf_data.elf_header.e_shnum; i++) {
    const uint64_t index = static_cast<uint64_t>(i);
    if (((elf_data.section_headers[index].sh_flags & kShfAlloc) != 0U) &&
        (elf_data.section_headers[index].sh_size > 0U) && (elf_data.text_offset == 0UL)) {
      elf_data.text_offset = elf_data.section_headers[index].sh_offset;
    }
  }
  GE_CHECK_LE(elf_data.text_offset, std::numeric_limits<uint32_t>::max());
  offset = static_cast<uint32_t>(elf_data.text_offset);
  return GRAPH_SUCCESS;
}
}

Status Elf::GetElfOffset(uint8_t *const bin_data, const uint32_t bin_len, uint32_t &bin_data_offset) {
  GE_ASSERT_NOTNULL(bin_data);
  ElfData elf_data{};
  elf_data.obj_ptr = bin_data;
  elf_data.obj_ptr_origin = bin_data;
  elf_data.obj_size = bin_len;

  GE_ASSERT_SUCCESS(GetFileHeader(elf_data));
  GE_ASSERT_SUCCESS(ProcessSectionTableGetOffset(elf_data, bin_data_offset));
  GELOGI("Get elf offset: %u", bin_data_offset);
  return GRAPH_SUCCESS;
}
} // namespace ge