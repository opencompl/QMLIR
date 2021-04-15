namespace testground {

    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;
    
    operation tryopt(q : Qubit[], b : Bool) : Unit {
        H(q[0]);
        if (b) {
            X(q[1]);
            H(q[0]);
            X(q[1]);
        } else {
            Z(q[1]);
            H(q[0]);
            Z(q[1]);
        }
    }

    @EntryPoint()
    operation HelloQ() : Unit {
        use q = Qubit[2];
        H(q[0]);
        let res = M(q[0]);
        tryopt(q, res == One);
    }
}

