#include <iostream>
#include <chrono>
#include "et_autodiff.h"


constexpr auto Do() {

	constexpr et::ConstantExpr x = 4, z = 1;
	constexpr et::ConstantExpr y = 0.5;
	auto b = et::pow(x, y) + (-z);
	return b();
}

int main() { 

	auto begin = std::chrono::high_resolution_clock::now();
	
	std::cout << Do() << std::endl;

	auto end = std::chrono::high_resolution_clock::now();

	std::cout << "Time elapsed : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
	
	return 0;
}