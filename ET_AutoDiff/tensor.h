#pragma once

#include <cmath>

namespace Num 
{

	template <typename D>
	class Tensor 
	{

	public:
		constexpr D const& GetSelf() const 
		{
			return static_cast<D const&>(*this);
		}
	};

	template <typename>
	class Scaler;

	template<>
	class Scaler<double> : public Tensor<Scaler<double>> 
	{

	private:
		double value_;

	public:
		constexpr Scaler(double value = 0.0) : value_(value) {}

		constexpr double GetValue() const 
		{ 
			return value_; 
		}
		constexpr operator double() const 
		{ 
			return value_; 
		}
	};

	template <typename T>
	constexpr Scaler<T> operator+(Scaler<T> const& first, Scaler<T> const& second) 
	{
		return Scaler<T>(first.GetValue() + second.GetValue());
	}

	template <typename T>
	constexpr Scaler<T> operator-(Scaler<T> const& first, Scaler<T> const& second) 
	{
		return Scaler<T>(first.GetValue() - second.GetValue());
	}

	template <typename T>
	constexpr Scaler<T> operator-(Scaler<T> const& first) 
	{
		return Scaler<T>(-first.GetValue());
	}

	template <typename T>
	constexpr Scaler<T> operator*(Scaler<T> const& first, Scaler<T> const& second) 
	{
		return Scaler<T>(first.GetValue() * second.GetValue());
	}

	template <typename T>
	constexpr Scaler<T> operator/(Scaler<T> const& first, Scaler<T> const& second) 
	{
		return Scaler<T>(first.GetValue() / second.GetValue());
	}

	template <typename T>
	constexpr Scaler<T> pow(Scaler<T> const& first, Scaler<T> const& second) 
	{
		return Scaler<T>(std::pow(first.GetValue(), second.GetValue()));
	}

	template <typename T>
	constexpr Scaler<T> sin(Scaler<T> const& first) 
	{
		return Scaler<T>(std::sin(first.GetValue()));
	}

	template <typename T>
	constexpr Scaler<T> cos(Scaler<T> const& first) 
	{
		return Scaler<T>(std::cos(first.GetValue()));
	}

	template <typename T>
	constexpr Scaler<T> tan(Scaler<T> const& first) 
	{
		return Scaler<T>(std::tan(first.GetValue()));
	}

	template <typename T>
	constexpr Scaler<T> log(Scaler<T> const& first) 
	{
		return Scaler<T>(std::log(first.GetValue()));
	}
}