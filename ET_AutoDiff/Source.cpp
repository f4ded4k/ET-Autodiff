#pragma once

#include <iostream>
#include <chrono>
#include "et_autodiff.h"

#if defined(_DEBUG)
template <class T> constexpr std::string_view
type_name()
{
	using namespace std;
	string_view p = __FUNCSIG__;
	return string_view(p.data() + 84, p.size() - 84 - 7);
}
#endif 

void AutodiffTest() 
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
			.ForwardPass(Et::H(P, -6.3))
			.Minimize(0.01)
			.GetPreResult()

		<< std::endl;
	}
	std::cout << std::endl;
	std::cout << "Final Value : " << Optimizer.GetPostResult() << std::endl;
}

void TensorTests()
{
	auto x = TTest::TensorFactory::MakeRandomTensor<double, 10000, 10000>(-1.0, 1.0);
}

int main()
{
	auto begin = std::chrono::high_resolution_clock::now();

	AutodiffTest();
	//TensorTests();

	auto end = std::chrono::high_resolution_clock::now();

	std::cout << "Time elapsed : " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "us" << std::endl;
	
	return 0;
}