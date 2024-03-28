#include <stdio.h>
void hanoi_tower(int n, char from, char tmp, char to){
    if(n == 1) {
        printf("원판 1을 %c에서 %c으로 옮긴다.\n", from, to);
    }
    else{
        hanoi_tower(n-1, from, to, tmp);       // A -> B를 가기위해 to를 거치는 행위
        printf("원판 %d을 %c에서 %c으로 옮긴다.\n", n, from, to); // A -> C로 이동을 했다.
        hanoi_tower(n-1, tmp, from, to);     // B -> C로  A를 겨처 이동을 했다.
    }
}

int main(void){
    hanoi_tower(4, 'A', 'B', 'C');
    return 0;
}


/*
 * n = 3을 기준으로 코드를 구현한다.
 * n = 3인 경우 경로가 크게
 * 1. A -> B
 * 2. A -> C
 * 3. B -> C
 * 이렇게 존재하는데 위의 경우로 나눠서 생각을 한다.
 *
 */
