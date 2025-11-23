#include <stdio.h>

int main() {
    int A, B;
    scanf("%d %d", &A, &B);

    int sum = A + B;
    int diff = A - B;
    int prod = A * B;

    printf("%d\n%d\n%d\n", sum, diff, prod);

    return 0;
}
