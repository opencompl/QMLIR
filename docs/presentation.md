# QMLIR
### An MLIR Dialect for Quantum progamming languages

### Presentation
0. **About MLIR**
1. **QMLIR Types**
    - Qubit Type
2. **QMLIR Operations**
    - Allocation
    - Manipulation
    - Measurement
    - Transformation
3. **Examples**
    - Teleportation
    - Deutsch-Josza
3. **Current features**
    - Simulation
    - Lowering to LLVMIR
4. **Ideas for further work**
    - Register Allocation
    - Choice of Universal Gate Set
    - More Qubit Types
    - Circuit Optimizations

# 0. About MLIR
> The [MLIR](https://mlir.llvm.org/) project is a novel approach to building reusable and extensible compiler infrastructure. MLIR aims to address software fragmentation, improve compilation for heterogeneous hardware, significantly reduce the cost of building domain specific compilers, and aid in connecting existing compilers together.

For the purpose of this presentation, the key features of MLIR that are used can be summarized as follows:
- Co-existing dialects, each introducing some operations and types.
- SSA based IR.
- Basic blocks of instructions, and Regions containing them.

# 1. QMLIR Types

## 1.1. Qubit Type
Linear qubit arrays. Can have dynamic sizes.
```mlir
qubit<10> // static
qubit<?> // dynamic
```

# 2. QMLIR Operations

## 2.1. Allocation

Allocate a qubit array of a given size.
```mlir
// static size
%q = allocate() : qubit<10>

// dynamic size
// %n : index
%q = allocate(%n) : qubit<?>
```

## 2.2. Manipulation

### Split and Merge
```mlir
// %q : qubit<4>
%q1, %q2 = split %q : qubit<4> -> (qubit<2>, qubit<2>)

%q3 = merge %q1, %q2 : (qubit<2>, qubit<2>) -> qubit<4>
```

**Rationale:**
Providing array addressing to access qubits make dataflow analysis hard, and adds a few concerns:
- Operations like `CNOT(a, b)` could have `a = b`, which is undesirable.
- We cannot tell the range in which a qubit is _live_. Knowing live ranges makes register allocation simpler.

Therefore, we use the split-and-merge operations to manipulate the arrays.  
And enforce a constraint: a qubit SSA can only be used _once_.

### Dimension and Cast
The `cast` operation supports conversions between static and dynamic qubit arrays.
```mlir
// %q0 : qubit<5>
%q1 = cast %q0 : qubit<5> to qubit<?>
```

The `dim` operation is used to extract the size of a dynamic qubit array.
```mlir
// %q0 : qubit<?>
%q1, %n = dim %q0 : qubit<?>
```

## 2.3. Measurement

Measure (and deallocate) qubits in the standard bases (Pauli-X) and return an array of bits.
```mlir
// %q0 : qubit<?>
%res = measure %q0 : qubit<?> -> memref<? x i1>
```

## 2.4. Transformations

Currently, a simple gate set is supported - the Pauli gates, Hadamard, CNOT. 
```mlir
// %q0 : qubit<?>
%q1 = pauliX %q0 : qubit<?> // applies X on each qubit
%q2 = H %q1 : qubit<?> // applies H on each qubit
```
```mlir
// %q0 : qubit<2>
%q1 = CNOT %q0 : qubit<2>
```

It also supports unregistered transformations - which can be converted by optimizations, or lowered based on target.
```mlir
// %q0 : qubit<?>
%q1 = transform(...) %q0 : qubit<?> // pass floating point parameters

//// example
// %alpha : f32
%q1 = transform(%alpha : f32) { name = "Rx" } %q0 : qubit<?>

//// a controlled gate
// %qc : qubit<1> // control qubit
%q1 = controlled(%alpha : f32) { name = "C-Rx" } [qc : qubit<1>] %q0 : qubit<?>
```

# 3. Examples

## 3.1. Teleportation
```mlir
func @std_to_bell(%qs: !quantum.qubit<2>) -> !quantum.qubit<2> {
  // H(qs[0])
  %q0, %q1 = quantum.split %qs : !quantum.qubit<2> -> (!quantum.qubit<1>, !quantum.qubit<1>)
  %q2 = quantum.H %q0 : !quantum.qubit<1>

  // CNOT(qs[0], qs[1])
  %q3 = quantum.concat %q2, %q1 : (!quantum.qubit<1>, !quantum.qubit<1>) -> !quantum.qubit<2>
  %q4 = quantum.CNOT %q3 : !quantum.qubit<2>

  return %q4 : !quantum.qubit<2>
}

func @bell_to_std(%qs : !quantum.qubit<2>) -> !quantum.qubit<2> {
  // CNOT(qs[0], qs[1])
  %q0 = quantum.CNOT %qs : !quantum.qubit<2>

  // H(qs[0])
  %q1, %q2 = quantum.split %q0 : !quantum.qubit<2> -> (!quantum.qubit<1>, !quantum.qubit<1>)
  %q3 = quantum.H %q1 : !quantum.qubit<1>

  %q4 = quantum.concat %q3, %q2 : (!quantum.qubit<1>, !quantum.qubit<1>) -> !quantum.qubit<2>
  return %q4 : !quantum.qubit<2>
}

func @teleport(%psiA: !quantum.qubit<1>, %eb: !quantum.qubit<2>) -> (!quantum.qubit<1>) {
  %ebA, %psiB0 = quantum.split %eb : !quantum.qubit<2> -> (!quantum.qubit<1>, !quantum.qubit<1>)

  // Alice's qubits
  %qsA0 = quantum.concat %psiA, %ebA : (!quantum.qubit<1>, !quantum.qubit<1>) -> !quantum.qubit<2>

  // Measure in Bell basis
  %qsA1 = call @bell_to_std(%qsA0) : (!quantum.qubit<2>) -> !quantum.qubit<2>
  %resA = quantum.measure %qsA1 : !quantum.qubit<2> -> memref<2xi1>

  // Apply corrections

  // 1. Apply X correction, if resA[0] == 1
  %idx0 = constant 0 : index
  %corrX = load %resA[%idx0] : memref<2xi1>

  %psiB1 = scf.if %corrX -> !quantum.qubit<1> {
    %temp = quantum.pauliX %psiB0 : !quantum.qubit<1>
    scf.yield %temp : !quantum.qubit<1>
  } else {
    scf.yield %psiB0 : !quantum.qubit<1>
  }

  // 2. Apply Z correction, if resA[1] == 1
  %idx1 = constant 1 : index
  %corrZ = load %resA[%idx1] : memref<2xi1>

  %psiB2 = scf.if %corrZ -> !quantum.qubit<1> {
    %temp = quantum.pauliZ %psiB1 : !quantum.qubit<1>
    scf.yield %temp : !quantum.qubit<1>
  } else {
    scf.yield %psiB1 : !quantum.qubit<1>
  }

  return %psiB2 : !quantum.qubit<1>
}

func @prepare_bell(%qa : !quantum.qubit<1>, %qb : !quantum.qubit<1>) -> !quantum.qubit<2> {
  %q0 = quantum.concat %qa, %qb : (!quantum.qubit<1>, !quantum.qubit<1>) -> !quantum.qubit<2>
  %q1 = call @std_to_bell(%q0) : (!quantum.qubit<2>) -> !quantum.qubit<2>
  return %q1 : !quantum.qubit<2>
}

func @main() {
  // Alice's qubits
  %psiA = quantum.allocate() : !quantum.qubit<1>
  %ebA = quantum.allocate() : !quantum.qubit<1>

  // Bob's qubits
  %ebB = quantum.allocate() : !quantum.qubit<1>

  // Entangle the qubits
  %eb = call @prepare_bell(%ebA, %ebB) : (!quantum.qubit<1>, !quantum.qubit<1>) -> !quantum.qubit<2>

  // Teleport |psi> from Alice to Bob
  %psiB = call @teleport(%psiA, %eb) : (!quantum.qubit<1>, !quantum.qubit<2>) -> !quantum.qubit<1>

  return
}
```

## 3.2. Deutsch-Josza
```mlir

// implements U|x⟩|y⟩ = |x⟩|y ⊕ f(x)⟩
func @oracle(%x : !quantum.qubit<?>, %y : !quantum.qubit<1>)
  -> (!quantum.qubit<?>, !quantum.qubit<1>)

// implements U|x⟩ = (-1)^{f(x)} |x⟩
func @phase_flip_oracle(%x : !quantum.qubit<?>)
  -> !quantum.qubit<?> {
  %y0 = quantum.allocate() : !quantum.qubit<1>
  %y1 = quantum.pauliX %y0 : !quantum.qubit<1>
  %y2 = quantum.H %y1 : !quantum.qubit<1>
  %x1, %y3 = call @oracle(%x, %y2)
    : (!quantum.qubit<?>, !quantum.qubit<1>) -> (!quantum.qubit<?>, !quantum.qubit<1>)

  %0 = quantum.measure %y3 : !quantum.qubit<1> -> memref<1xi1>

  return %x1: !quantum.qubit<?>
}

// return false for constant, true for balanced
func @deutsch_josza(%n : index) -> i1 { // %n : number of input bits
  %x0 = quantum.allocate(%n) : !quantum.qubit<?>
  %x1 = quantum.H %x0 : !quantum.qubit<?>
  %x2 = call @phase_flip_oracle(%x1) : (!quantum.qubit<?>) -> !quantum.qubit<?>
  %x3 = quantum.H %x2 : !quantum.qubit<?>
  %res = quantum.measure %x3 : !quantum.qubit<?> -> memref<?xi1>

  // compute bitwise-OR of all the bits in %res
  %false = constant 0 : i1
  %0 = constant 0 : index
  %1 = constant 1 : index
  %lst = subi %n, %1 : index

  %ans = scf.for %i = %0 to %lst step %1
    iter_args(%out = %false) -> i1 {
    %v = load %res[%i] : memref<?xi1>
    %cur = or %out, %v : i1
    scf.yield %cur : i1
  }

  return %ans : i1
}
```

# 4. Current Features

### 4.1. Simulation
Implemented a barebones simulation library which can handle around 10 qubits efficiently.
Only supports the basic gates mentioned above (`X, Y, Z, CNOT, H`)

### 4.2. Lowering to LLVMIR
A pass currently can lower our dialect into the standard dialect, which is in-turn lowered into LLVMIR and can be executed (by calling simulator functions)

# 5. Ideas for further work

### 5.1. Register Allocation
Can implement different register allocation schemes, based on existing literature.  
A few constraints in mind:
- Hardware specific restrictions
    * Multiqubit gates on nearby qubits
    * Only supports certain multiqubit gates
- Allocation/Deallocation strategies
    * Currently: Deallocate on measure

### 5.2. Choice of Universal Gate Set
- Multiple works on different universal gates. Can see which works best with our framework.  
- Conversions between different sets.

### 5.3. More Qubit Types
Google Cirq has a few different processors with different layouts. Maybe we can factor that into our qubit type representation. 

### 5.4. Circuit Optimizations
Similar to Cirq, can implement various circuit/gate optimizations and rewriting.  
One idea was to implement gate factoring: break down a large multiqubit gate into smaller (fundamental) gates.
