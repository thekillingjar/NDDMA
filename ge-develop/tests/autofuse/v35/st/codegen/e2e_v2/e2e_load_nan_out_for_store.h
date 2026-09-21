#ifndef __TEST_E2E_LOAD_NAN_OUT_FOR_STORE_H__
#define __TEST_E2E_LOAD_NAN_OUT_FOR_STORE_H__
#include "ascendc_ir.h"

void LoadNanOutForStore_BeforeAutofuse(ge::AscGraph &graph);
void LoadNanOutForStore_AfterInferOutput(ge::AscGraph &graph);
void LoadNanOutForStore_AfterGetApiInfo(ge::AscGraph &graph);
void LoadNanOutForStore_AfterScheduler(ge::AscGraph &graph);
void LoadNanOutForStore_AfterQueBufAlloc(ge::AscGraph &graph);
#endif
