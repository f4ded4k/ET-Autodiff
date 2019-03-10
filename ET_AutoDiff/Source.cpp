#include <iostream>
#include <chrono>
#include "et_autodiff.h"
#include "tensor.h"



double Do() {

	Et::ConstantExpr x = Et::Double(5.0), y = Et::Double(3.4), z = Et::Double(1.2);
	
	auto b = (x - y) + (x * y) + (x / z) + -x + sin(cos(y) + tan(log(z))); // 17.06

	return b();
}

int main() { 

	auto begin = std::chrono::high_resolution_clock::now();
	
	std::cout << Do() << std::endl;

	auto end = std::chrono::high_resolution_clock::now();

	std::cout << "Time elapsed : " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "us" << std::endl;
	
	return 0;
}