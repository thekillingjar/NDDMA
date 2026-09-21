/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * @file constant_folding_ops_stub.cpp
 * @brief Stub library for constant_folding_ops (used only at build time).
 *
 * This file intentionally provides **no real implementations**.
 * It only exists so that the AIR repository can be built independently
 * without depending on the real constant_folding_ops.so produced by OPP.
 *
 * At runtime:
 *   ✅ The real libconstant_folding_ops.so (from OPP) will be loaded first.
 *   ✅ This stub library will NOT be used.
 *
 * If this stub is accidentally loaded (extremely unlikely):
 *   ✅ It still won't crash, because it exports zero symbols.
 *
 * This file is safe, minimal, and production-friendly.
 */
 
namespace aicpu_stub {
// --------------------------------------------------------------
// Optional: provide a weak dummy symbol as a safety fallback
// --------------------------------------------------------------
// If your linker prefers at least one exported symbol to build a shared lib,
// this dummy weak function will satisfy it.
// If not needed, it will never be used.
// --------------------------------------------------------------
__attribute__((weak))
void constant_folding_ops_stub_marker() {}

} // namespace aicpu_stub