#include <iostream>
#include <chrono>
#include "et_autodiff.h"



double Do() 
{
	Et::ConstantExpr c1 = Et::Double(5.0), c2 = Et::Double(3.4);
	Et::VariableExpr v1 = Et::Double(5.0);
	Et::PlaceholderExpr<Et::Double> p1;
	p1.FeedValue(3.4);

	auto b = c1 + sin(v1 - p1);
	return Et::Evaluate(b);
}

int main() 
{ 
	auto begin = std::chrono::high_resolution_clock::now();
	
	std::cout << Do() << std::endl;

	auto end = std::chrono::high_resolution_clock::now();

	std::cout << "Time elapsed : " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "us" << std::endl;
	
	return 0;
}