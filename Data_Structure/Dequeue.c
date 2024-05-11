#include <stdio.h>
#include <stdlib.h>
#define MAX_SIZE 5

typedef int element;
typedef struct {
    element data[MAX_SIZE];
    int front, rear;
}DequeType;

void error(char *message)
{
    fprintf(stderr, "%s\n", message);
    exit(1);
}
void init(DequeType *q){
    q->front = q->rear = 0;
}
int is_empty(DequeType *q){
    return (q->front == q-> rear);
}
int is_full(DequeType *q){
    return (q->front % MAX_SIZE == (q->rear+1) % MAX_SIZE);
}
void deque_print(DequeType *q){
    printf("DEQUE(front=%d rear=%d) = ", q->front, q->rear);
    if(!is_empty(q)){
        int i = q->front;
        do{
            i = (i + 1) % MAX_SIZE;
            printf("%d | ", q->data[i]);
            if (i == q->rear)
                break;
        } while (i != q->front);
    }
    printf("\n");
}
void add_rear(DequeType *q, element item){
    if(is_full(q)){
        error("큐가 포화상태입니다.");
    }
    q->rear = (q->rear + 1) % MAX_SIZE;
    q->data[q->rear] = item;
}
element delete_front(DequeType *q){
    if (is_empty(q))
        error("큐가 공백상태입니다.");
    q->front = (q->front + 1) % MAX_SIZE;
    return q->data[q->front];
}
element get_front(DequeType *q)
{
    if (is_empty(q))
        error("큐가 공백상태입니다");
    return q->data[(q->front + 1) % MAX_SIZE];
}
void add_front(DequeType *q, element val) {
    if (is_full(q))
        error("큐가 포화상태입니다");
    q->data[q->front] = val;
    q->front = (q->front - 1 + MAX_SIZE) % MAX_SIZE;
}
element delete_rear(DequeType *q)
{
    int prev = q->rear;
    if (is_empty(q))
        error("큐가 공백상태입니다");
    q->rear = (q->rear - 1 + MAX_SIZE) % MAX_SIZE;
    return q->data[prev];
}
element get_rear(DequeType *q)
{
    if (is_empty(q))
        error("큐가 공백상태입니다");
    return q->data[q->rear];
}
int main(){
    DequeType queue;
    init(&queue);
    int num;
    printf("원형덱의 사이즈를 입력하세요 : ");
    scanf("%d", &num);
    while(num != 0)
    {
        printf("덱의 동작을 선택하시오.'\n");
        printf("1. 스택\n");
        printf("2. 큐\n");
        printf("0. 종료'\n");
        printf(">>> ");
        int order;
        scanf("%d", &order);
        if (order == 1)
        {
            printf("*** 스택 ***\n");
            printf("수행할 연산을 선택하시오.\n");
            printf("1. push\n");
            printf("2. pop\n");
            printf("0. 종료(출력=전체 삭제)'\n");
            printf(">>> ");
            int n;
            scanf("%d", &n);
        }


        else if (num == 2){
            printf("*** 스택 ***\n");
            printf("수행할 연산을 선택하시오.\n");
            printf("1. push\n");
            printf("2. pop\n");
            printf("0. 종료(출력=전체 삭제)'\n");
            printf(">>> ");
            int n;
            scanf("%d", &n);
        }
    }
}
