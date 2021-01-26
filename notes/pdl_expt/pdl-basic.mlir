// pdl.pattern contains metadata similarly to a `RewritePattern`.
pdl.pattern : benefit(100) {
  %type = pdl.type : i32
  %a = pdl.input : %type
  %zattr = pdl.attr
  %cop, %0 = pdl.operation "std.constant" {value: 0} -> %type
  %addop, %sum = pdl.operation "std.addi"(%a, %0) -> %type
  pdl.rewrite %addop {
    pdl.replace %addop with %a
  }
}

func @test(%a : i32) -> i32 {
  %0 = constant 0 : i32
  %b = addi %a, %0 : i32
  return %b : i32
}

// Available Dialects: acc, affine, arm_neon, arm_sve, async, avx512, gpu, linalg, llvm, llvm_arm_neon, llvm_arm_sve, llvm_avx512, nvvm, omp, pdl, pdl_interp, quant, rocdl, scf, sdbm, shape, spv, std, tensor, test, tosa, vector
// --convert-pdl-to-pdl-interp - Convert PDL ops to PDL interpreter ops
// --test-pdl-bytecode-pass    - Test PDL ByteCode functionality
