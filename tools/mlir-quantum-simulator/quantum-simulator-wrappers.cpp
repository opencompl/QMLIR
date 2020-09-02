#include <iostream>
using namespace std;

class Qubit {
public:
  Qubit(int n) {
    cerr << "> {allocate} " << n << endl;
  }
};

using QubitPtr = void*;

extern "C" QubitPtr acquire_qubits(int size) {
  Qubit *qs = new Qubit(size);
  return (void*) qs;
}

extern "C" void printLn(int n) {
  cout << n << endl;
}
