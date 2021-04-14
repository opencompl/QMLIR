//===- PassDetail.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_QASMTOSCF_PASSDETAIL_H_
#define CONVERSION_QASMTOSCF_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_CLASSES
#include "Conversion/QASMToSCF/Passes.h.inc"

} // end namespace mlir

#endif // CONVERSION_QASMTOSCF_PASSDETAIL_H_
