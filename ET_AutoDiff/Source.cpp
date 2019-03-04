#include <iostream>
#include "et_autodiff.h"


constexpr auto Do() {
	et::ConstantExpr x = 5, z = 6;
	et::ConstantExpr y = 8.;
	auto b = (x+y) - y * z + y / z;
	return b();
}

int main() { 

	std::cout << Do() << std::endl;
	return 0;
}