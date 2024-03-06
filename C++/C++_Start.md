# C++

- cout : 객체
- << : 출력연산자
- endl : 줄바꿈 문자를 결과로 내는 객체

```cpp
#include <iostream>
using namespace std;

int main() {

    cout << "Hello World" << endl; // cout:객체, <<:출력연산자

    return 0;
}
```

```cpp
#include <iostream>
using namespace std;

int main(void)
{
    string name; // 문자열을 담을 수 있는 string 클래스 변수 or 객체

    cout << "Enter Your Name: ";

    cin >> name; // 이름을 입력받음

    cout << "Hello " << name << endl;

    return 0;

}
```
