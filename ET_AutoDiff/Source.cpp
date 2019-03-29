#pragma once

#include <iostream>
#include <chrono>
#include "et_autodiff.h"
#include "et_tensor.h"

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

void AutodiffTestTest()
{
	using namespace Et_test;
	ConstantExpr C1{ Tensor<double,123>(4) }, C2{ Tensor<double,123>(2.4) };
	VariableExpr X1{ Tensor<double,123>(5.53) }, X2{ Tensor<double,123>(-3.12) };
	PlaceholderExpr<Tensor<double, 123>> P;

	auto Y = C1 + X1 + P;
	
	GradientDescentOptimizer Optimizer{ Y };
	Optimizer.FeedPlaceholders(H{ P,Tensor<double,123>(4.3) });
	
	size_t Iterations = 100;

	for (size_t i = 0; i < Iterations; i++)
	{
		std::cout << "Value at #" << i + 1 << " : " <<

			Optimizer
			.ForwardPass()
			.Minimize(0.01)
			.GetPreResult()(5)

			<< std::endl;
	}

	Optimizer.Terminate();

	std::cout << std::endl << "Final Value : " << Optimizer.GetPostResult()(5) << std::endl;
}

void TensorTestTest()
{
	using namespace Et_test;

	Tensor<int, 4, 5> a(4);
	Tensor<int, 5, 2> b(2);
	auto r = matmul(a, b);
	std::cout << r(0, 1) << std::endl; 
}

void TensorTests()
{
	auto x = TTest::TensorFactory::MakeTensorWithInitValue<double, 100, 10>(5.0);
	auto y = TTest::TensorFactory::MakeTensorWithInitValue<double, 100, 10>(1.2);
	auto a = TTest::TensorFactory::MakeTensorWithInitValue<double, 100, 10>(1.2);
	auto z = 4 * x * y - tan(a) + a + log(a / y);
	std::cout << z(3, 4) << std::endl;
}

int main()
{
	auto begin = std::chrono::high_resolution_clock::now();

	AutodiffTestTest();
	//TensorTestTest();
	//AutodiffTest();
	//TensorTests();

	auto end = std::chrono::high_resolution_clock::now();

	std::cout << "Time elapsed : " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "us" << std::endl;
	
	return 0;
}