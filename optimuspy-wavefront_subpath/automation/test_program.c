

volatile int foo(int r) {
    int x = r +2;
    return x;
}

void optimus() {
    int A[10], B[10], C[10];

    int b = 0;
    int c = 1;
    for(int i = 0; i < 10; i++) {
        b = 5;
        c = 3+b-i*2;
        C[i] = A[i] * B[i] + c;
    }
    int s = 0;
    for(int i = 0; i < 10; i++) {
        s = s + C[i];
    }
    foo(s);


}

// int main() {
//   optimus();
// }