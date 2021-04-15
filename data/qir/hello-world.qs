namespace testground {
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;
    
    operation cx(qs : Qubit[]) : Unit {
        CNOT(qs[0], qs[1]);
    }
    @EntryPoint()
    operation main() : Unit {
        use qs = Qubit[3];
        cx(qs[0..2]);
        let res = Measure([PauliZ, PauliZ, PauliZ], qs);
    }
}

