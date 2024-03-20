#include <iostream>
using namespace std;
#include "Circle.h"
#include "Circle.cpp" // cpp 하나 더 추가해주면 됨!

int main() {
    Circle donut;
    double area = donut.getArea();
    cout << "donut 면적은 ";
    cout << area << endl;
    Circle pizza(30);
    area = pizza.getArea();
    cout << "pizza 면적은 ";
    cout << area << endl;
}


// .main에 .cpp 소스파일 추가하면됨.
