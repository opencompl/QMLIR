#ifndef QUANTUM_TRANSFORMS_CONVERT1QTOU_TD
#define QUANTUM_TRANSFORMS_CONVERT1QTOU_TD

include "mlir/IR/OpBase.td"
include "mlir/Pass/PassBase.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "Dialect/Quantum/QuantumOps.td"

def ConstantZero : NativeCodeCall<"$_builder.getF64FloatAttr(0.0)">;
def ConstantPi : NativeCodeCall<"$_builder.getF64FloatAttr(M_PI)">;

def : Pat<(PauliXGateOp $q),
          (UniversalRotationGateOp
             (ConstantOp (ConstantPi)),
             (ConstantOp (ConstantZero)),
             (ConstantOp (ConstantPi)),
             $q)>;

#endif // QUANTUM_TRANSFORMS_CONVERT1QTOU_TD
