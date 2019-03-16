#pragma once

#include <cmath>
#include <type_traits>

namespace Num
{

	struct Tensor {};

	template<typename T>
	constexpr bool is_tensor_v = std::is_base_of_v<Tensor, T>;

	template <typename>
	class Scaler;

	template<>
	class Scaler<double> : public Tensor
	{

	private:
		double value_;

	public:
		constexpr Scaler(double value = 0.0) : value_(value) {}

		constexpr Scaler<double> Inverse() const
		{
			return Scaler<double>(1.0 / value_);
		}

		constexpr double GetValue() const 
		{ 
			return value_; 
		}
		constexpr operator double() const 
		{ 
			return value_; 
		}

		constexpr Scaler<double>& operator+=(const Scaler<double>& other)
		{
			this->value_ += other.GetValue();
			return *this;
		}

		constexpr Scaler<double>& operator-=(const Scaler<double>& other)
		{
			this->value_ -= other.GetValue();
			return *this;
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
	constexpr Scaler<T> sec(Scaler<T> const& first)
	{
		return Scaler<T>(1.0 / std::cos(first.GetValue()));
	}

	template <typename T>
	constexpr Scaler<T> log(Scaler<T> const& first) 
	{
		return Scaler<T>(std::log(first.GetValue()));
	}
}