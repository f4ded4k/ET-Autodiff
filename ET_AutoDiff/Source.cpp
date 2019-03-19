#pragma once

#include <iostream>
#include <chrono>
#include "et_autodiff.h"

void Test() 
{
	Et::ConstantExpr C1{ 4 }, C2{ 2 };
	Et::VariableExpr X1{ 5.53 }, X2{ -3.12 };
	Et::PlaceholderExpr P;

	auto Y = X1 * X1 + X2 * X2 + C1 * X1 + C2 * X2 + P;

	Et::GradientDescentOptimizer Optimizer{ Y };

	int Iterations = 500;
	for (int i = 0; i < Iterations; i++)
	{
		std::cout << "Value at #" << i + 1 << " : " <<

		Optimizer
			.FeedPlaceholders(Et::PlFeed(P, -6.3))
			.ForwardPass()
			.Minimize(0.01)
			.GetPreResult()

		<< std::endl;
	}
	std::cout << std::endl;
	std::cout << "Final Value : " << Optimizer.GetPostResult() << std::endl;
}

int main()
{
	auto begin = std::chrono::high_resolution_clock::now();
	
	Test();

	auto end = std::chrono::high_resolution_clock::now();

	std::cout << "Time elapsed : " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "us" << std::endl;
	
	return 0;
}