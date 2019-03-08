#include <iostream>
#include <chrono>
#include "et_autodiff.h"


auto Do() {

	constexpr Et::ConstantExpr x = 4.0, y = 0.5, z = 1.0;

	auto b = x + (y * y) / y + Et::pow(x, y) + (-y);

	return b();
}



int main() { 

	auto begin = std::chrono::high_resolution_clock::now();
	
	std::cout << Do() << std::endl;

	auto end = std::chrono::high_resolution_clock::now();

	std::cout << "Time elapsed : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
	
	return 0;
}
