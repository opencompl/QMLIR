
%Range = type { i64, i64, i64 }
%Array = type opaque
%Qubit = type opaque
%Result = type opaque
%String = type opaque

@PauliI = constant i2 0
@PauliX = constant i2 1
@PauliY = constant i2 -1
@PauliZ = constant i2 -2
@EmptyRange = internal constant %Range { i64 0, i64 1, i64 -1 }
@0 = internal constant [3 x i8] c"()\00"

@testground__HelloQ__Interop = alias void (), void ()* @testground__HelloQ__body

define void @testground__HelloQ__body() #0 {
entry:
  %q = call %Array* @__quantum__rt__qubit_allocate_array(i64 2)
  call void @__quantum__rt__array_update_alias_count(%Array* %q, i32 1)
  %0 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %q, i64 0)
  %1 = bitcast i8* %0 to %Qubit**
  %qubit = load %Qubit*, %Qubit** %1, align 8
  call void @__quantum__qis__h__body(%Qubit* %qubit)
  %2 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %q, i64 0)
  %3 = bitcast i8* %2 to %Qubit**
  %4 = load %Qubit*, %Qubit** %3, align 8
  %res = call %Result* @Microsoft__Quantum__Intrinsic__M__body(%Qubit* %4)
  %5 = call %Result* @__quantum__rt__result_get_one()
  %6 = call i1 @__quantum__rt__result_equal(%Result* %res, %Result* %5)
  call void @testground__tryopt__body(%Array* %q, i1 %6)
  call void @__quantum__rt__array_update_alias_count(%Array* %q, i32 -1)
  call void @__quantum__rt__result_update_reference_count(%Result* %res, i32 -1)
  call void @__quantum__rt__qubit_release_array(%Array* %q)
  ret void
}

declare %Qubit* @__quantum__rt__qubit_allocate()

declare %Array* @__quantum__rt__qubit_allocate_array(i64)

declare void @__quantum__rt__array_update_alias_count(%Array*, i32)

declare i8* @__quantum__rt__array_get_element_ptr_1d(%Array*, i64)

declare void @__quantum__qis__h__body(%Qubit*)

define %Result* @Microsoft__Quantum__Intrinsic__M__body(%Qubit* %qubit) {
entry:
  %bases = call %Array* @__quantum__rt__array_create_1d(i32 1, i64 1)
  %0 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %bases, i64 0)
  %1 = bitcast i8* %0 to i2*
  %2 = load i2, i2* @PauliZ, align 1
  store i2 %2, i2* %1, align 1
  call void @__quantum__rt__array_update_alias_count(%Array* %bases, i32 1)
  %qubits = call %Array* @__quantum__rt__array_create_1d(i32 8, i64 1)
  %3 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %qubits, i64 0)
  %4 = bitcast i8* %3 to %Qubit**
  store %Qubit* %qubit, %Qubit** %4, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %qubits, i32 1)
  %5 = call %Result* @__quantum__qis__measure__body(%Array* %bases, %Array* %qubits)
  call void @__quantum__rt__array_update_alias_count(%Array* %bases, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %qubits, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %bases, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %qubits, i32 -1)
  ret %Result* %5
}

define void @testground__tryopt__body(%Array* %q, i1 %b) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %q, i32 1)
  %0 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %q, i64 0)
  %1 = bitcast i8* %0 to %Qubit**
  %qubit = load %Qubit*, %Qubit** %1, align 8
  call void @__quantum__qis__h__body(%Qubit* %qubit)
  br i1 %b, label %then0__1, label %else__1

then0__1:                                         ; preds = %entry
  %2 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %q, i64 1)
  %3 = bitcast i8* %2 to %Qubit**
  %qubit__1 = load %Qubit*, %Qubit** %3, align 8
  call void @__quantum__qis__x__body(%Qubit* %qubit__1)
  %4 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %q, i64 0)
  %5 = bitcast i8* %4 to %Qubit**
  %qubit__2 = load %Qubit*, %Qubit** %5, align 8
  call void @__quantum__qis__h__body(%Qubit* %qubit__2)
  %6 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %q, i64 1)
  %7 = bitcast i8* %6 to %Qubit**
  %qubit__3 = load %Qubit*, %Qubit** %7, align 8
  call void @__quantum__qis__x__body(%Qubit* %qubit__3)
  br label %continue__1

else__1:                                          ; preds = %entry
  %8 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %q, i64 1)
  %9 = bitcast i8* %8 to %Qubit**
  %qubit__4 = load %Qubit*, %Qubit** %9, align 8
  call void @__quantum__qis__z__body(%Qubit* %qubit__4)
  %10 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %q, i64 0)
  %11 = bitcast i8* %10 to %Qubit**
  %qubit__5 = load %Qubit*, %Qubit** %11, align 8
  call void @__quantum__qis__h__body(%Qubit* %qubit__5)
  %12 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %q, i64 1)
  %13 = bitcast i8* %12 to %Qubit**
  %qubit__6 = load %Qubit*, %Qubit** %13, align 8
  call void @__quantum__qis__z__body(%Qubit* %qubit__6)
  br label %continue__1

continue__1:                                      ; preds = %else__1, %then0__1
  call void @__quantum__rt__array_update_alias_count(%Array* %q, i32 -1)
  ret void
}

declare %Result* @__quantum__rt__result_get_one()

declare i1 @__quantum__rt__result_equal(%Result*, %Result*)

declare void @__quantum__rt__result_update_reference_count(%Result*, i32)

declare void @__quantum__rt__qubit_release_array(%Array*)

declare void @__quantum__qis__x__body(%Qubit*)

declare void @__quantum__qis__z__body(%Qubit*)

define void @Microsoft__Quantum__Intrinsic__H__body(%Qubit* %qubit) {
entry:
  call void @__quantum__qis__h__body(%Qubit* %qubit)
  ret void
}

define void @Microsoft__Quantum__Intrinsic__H__adj(%Qubit* %qubit) {
entry:
  call void @__quantum__qis__h__body(%Qubit* %qubit)
  ret void
}

define void @Microsoft__Quantum__Intrinsic__H__ctl(%Array* %__controlQubits__, %Qubit* %qubit) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 1)
  call void @__quantum__qis__h__ctl(%Array* %__controlQubits__, %Qubit* %qubit)
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 -1)
  ret void
}

declare void @__quantum__qis__h__ctl(%Array*, %Qubit*)

define void @Microsoft__Quantum__Intrinsic__H__ctladj(%Array* %__controlQubits__, %Qubit* %qubit) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 1)
  call void @__quantum__qis__h__ctl(%Array* %__controlQubits__, %Qubit* %qubit)
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 -1)
  ret void
}

define %Result* @Microsoft__Quantum__Intrinsic__Measure__body(%Array* %bases, %Array* %qubits) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %bases, i32 1)
  call void @__quantum__rt__array_update_alias_count(%Array* %qubits, i32 1)
  %0 = call %Result* @__quantum__qis__measure__body(%Array* %bases, %Array* %qubits)
  call void @__quantum__rt__array_update_alias_count(%Array* %bases, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %qubits, i32 -1)
  ret %Result* %0
}

declare %Result* @__quantum__qis__measure__body(%Array*, %Array*)

define void @Microsoft__Quantum__Intrinsic__Z__body(%Qubit* %qubit) {
entry:
  call void @__quantum__qis__z__body(%Qubit* %qubit)
  ret void
}

define void @Microsoft__Quantum__Intrinsic__Z__adj(%Qubit* %qubit) {
entry:
  call void @__quantum__qis__z__body(%Qubit* %qubit)
  ret void
}

define void @Microsoft__Quantum__Intrinsic__Z__ctl(%Array* %__controlQubits__, %Qubit* %qubit) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 1)
  call void @__quantum__qis__z__ctl(%Array* %__controlQubits__, %Qubit* %qubit)
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 -1)
  ret void
}

declare void @__quantum__qis__z__ctl(%Array*, %Qubit*)

define void @Microsoft__Quantum__Intrinsic__Z__ctladj(%Array* %__controlQubits__, %Qubit* %qubit) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 1)
  call void @__quantum__qis__z__ctl(%Array* %__controlQubits__, %Qubit* %qubit)
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 -1)
  ret void
}

define void @Microsoft__Quantum__Intrinsic__X__body(%Qubit* %qubit) {
entry:
  call void @__quantum__qis__x__body(%Qubit* %qubit)
  ret void
}

define void @Microsoft__Quantum__Intrinsic__X__adj(%Qubit* %qubit) {
entry:
  call void @__quantum__qis__x__body(%Qubit* %qubit)
  ret void
}

define void @Microsoft__Quantum__Intrinsic__X__ctl(%Array* %__controlQubits__, %Qubit* %qubit) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 1)
  call void @__quantum__qis__x__ctl(%Array* %__controlQubits__, %Qubit* %qubit)
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 -1)
  ret void
}

declare void @__quantum__qis__x__ctl(%Array*, %Qubit*)

define void @Microsoft__Quantum__Intrinsic__X__ctladj(%Array* %__controlQubits__, %Qubit* %qubit) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 1)
  call void @__quantum__qis__x__ctl(%Array* %__controlQubits__, %Qubit* %qubit)
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 -1)
  ret void
}

declare %Array* @__quantum__rt__array_create_1d(i32, i64)

declare void @__quantum__rt__array_update_reference_count(%Array*, i32)

define void @testground__HelloQ() #1 {
entry:
  call void @testground__HelloQ__body()
  %0 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @0, i32 0, i32 0))
  call void @__quantum__rt__message(%String* %0)
  call void @__quantum__rt__string_update_reference_count(%String* %0, i32 -1)
  ret void
}

declare void @__quantum__rt__message(%String*)

declare %String* @__quantum__rt__string_create(i8*)

declare void @__quantum__rt__string_update_reference_count(%String*, i32)

attributes #0 = { "InteropFriendly" }
attributes #1 = { "EntryPoint" }
