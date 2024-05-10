#include <stdio.h>
#define MAX_SIZE 5

typedef int element;
typedef struct {
    element data[MAX_SIZE];
    int front, rear;
}QueueType;

void init(QueueType *q){
    q->rear = q->front = 0;
}
void error(char *message){
    printf(stderr, "%s\n", message);
}
int is_empty(QueueType *q){
    return q->front == q->rear;
}
int is_full(QueueType *q){
    return q->front == (q->rear + 1) % MAX_SIZE;
}
void show_Queue(QueueType *q){
    printf("Queue(front = %d, rear = %d) = ", q->front, q->rear);
    int idx = (q->front + 1) % MAX_SIZE;
    while (idx != (q->rear + 1) % MAX_SIZE) {
        printf("%d |", q->data[idx]);
        idx = (idx + 1) % MAX_SIZE;
    }
    printf("\n");
}
void enqueue(QueueType *q, element item){
    if(is_full(q)){
        error("포화 상태입니다.");
    }

    q->rear = (q->rear + 1) % MAX_SIZE;
    q->data[q->rear] = item;

}
int dequeue(QueueType *q){
    int get;
    if(is_empty(q)){
        error("공백 상태입니다.");
    }

    q->front = (q->front+1) % MAX_SIZE;
    get = q->data[q->front];
    return get;

}

int main(){
    QueueType p;
    init(&p);
    int num;
    while(!is_full(&p)){

        printf("큐에 채울 정수를 입력하세요 : ");
        scanf("%d", &num);
        enqueue(&p, num);
        show_Queue(&p);
    }
    printf("큐는 포화상태입니다.\n");

    while(!is_empty(&p)){
        printf("큐에서 배낸 정수 : ");
        int n;
        n = dequeue(&p);
        printf("%d\n", n);
        show_Queue(&p);
    }
    printf("큐는 공백상태입니다.");


}
