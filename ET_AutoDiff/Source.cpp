#include <iostream>
#include <chrono>
#include "et_autodiff.h"



void Test() 
{
	Et::ConstantExpr C1 = Et::Double(4), C2 = Et::Double(2);
	Et::VariableExpr X1 = Et::Double(5.53), X2 = Et::Double(-3.12);
	Et::PlaceholderExpr<Et::Double> P;

	// min{f(x1,x2) = x1^2 + x2^2 + 4*x1 + 2*x2 - 6.3} = -11.3
	auto Y = X1 * X1 + X2 * X2 + C1 * X1 + C2 * X2 + P;

	int Iterations = 1000;
	for (int i = 0; i < Iterations; i++)
	{
		std::cout << "Value at #" << i + 1 << " : " << Et::Evaluate(Y, Et::PlFeed(P, -6.3)) << std::endl;
		Et::ApplyGrediants(0.01, X1, X2);
	}

	std::cout << std::endl;

	std::cout << "Final Value :" << Y() << std::endl;
}

int main() 
{
	auto begin = std::chrono::high_resolution_clock::now();
	
	Test();

	auto end = std::chrono::high_resolution_clock::now();

	std::cout << "Time elapsed : " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "us" << std::endl;
	
	return 0;
}