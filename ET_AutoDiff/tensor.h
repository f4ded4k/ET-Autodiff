#pragma once

#include <cmath>
#include <type_traits>

namespace Num
{

	struct Tensor {};

	template<typename T>
	constexpr bool is_tensor_v = std::is_base_of_v<Tensor, T>;

	template <typename V1, typename V2>
	struct num_result
	{
		using type = long double;
	};

	template <>
	struct num_result<double, double>
	{
		using type = double;
	};

	template <typename V1, typename V2>
	using num_result_t = typename num_result<V1, V2>::type;

	template<typename V>
	class Scalar : private Tensor
	{
	public:
		using num_type = V;

	private:
		V _value;

	public:
		constexpr Scalar(V const& value = 0.0) : _value(value) {}

		constexpr auto Inverse() const -> Scalar<V>
		{
			return Scalar<V>(1.0 / _value);
		}

		constexpr auto GetValue() const -> V const&
		{ 
			return _value; 
		}
		constexpr operator V() const 
		{ 
			return _value; 
		}

		constexpr auto operator+=(Scalar<V> const& other) -> Scalar<V> &
		{
			this->_value += other.GetValue();
			return *this;
		}

		constexpr auto operator-=(Scalar<V> const& other) -> Scalar<V> &
		{
			this->_value -= other.GetValue();
			return *this;
		}
	};

	Scalar(int const&)->Scalar<double>;
	Scalar(float const&)->Scalar<double>;
	Scalar(double const&)->Scalar<double>;
	Scalar(long const&)->Scalar<long double>;
	Scalar(long long const&)->Scalar<long double>;
	Scalar(long double const&)->Scalar<long double>;

	template <typename T, typename = std::enable_if_t<is_tensor_v<T>>>
	constexpr T zero_v = T{};

	template <typename T, typename = std::enable_if_t<is_tensor_v<T>>>
	constexpr T identity_v = T{ 1.0 };

	template <typename V1, typename V2>
	constexpr auto operator+(Scalar<V1> const& first, Scalar<V2> const& second) -> Scalar<typename num_result_t<V1, V2>>
	{
		return { first.GetValue() + second.GetValue() };
	}

	template <typename V1, typename V2>
	constexpr auto operator-(Scalar<V1> const& first, Scalar<V2> const& second) -> Scalar<typename num_result_t<V1, V2>>
	{
		return { first.GetValue() - second.GetValue() };
	}

	template <typename T>
	constexpr auto operator-(Scalar<T> const& first) -> Scalar<T>
	{
		return { -first.GetValue() };
	}

	template <typename V1, typename V2>
	constexpr auto operator*(Scalar<V1> const& first, Scalar<V2> const& second) -> Scalar<typename num_result_t<V1, V2>>
	{
		return { first.GetValue() * second.GetValue() };
	}

	template <typename V1, typename V2>
	constexpr auto operator/(Scalar<V1> const& first, Scalar<V2> const& second) -> Scalar<typename num_result_t<V1, V2>>
	{
		return { first.GetValue() / second.GetValue() };
	}

	template <typename V1, typename V2>
	constexpr auto pow(Scalar<V1> const& first, Scalar<V2> const& second) -> Scalar<typename num_result_t<V1, V2>>
	{
		return { std::pow(first.GetValue(),second.GetValue()) };
	}

	template <typename T>
	constexpr auto sin(Scalar<T> const& first) -> Scalar<T>
	{
		return { std::sin(first.GetValue()) };
	}

	template <typename T>
	constexpr auto cos(Scalar<T> const& first) -> Scalar<T>
	{
		return { std::cos(first.GetValue()) };
	}

	template <typename T>
	constexpr auto tan(Scalar<T> const& first) -> Scalar<T>
	{
		return { std::tan(first.GetValue()) };
	}

	template <typename T>
	constexpr auto sec(Scalar<T> const& first) -> Scalar<T>
	{
		return { 1.0 / std::cos(first.GetValue()) };
	}

	template <typename T>
	constexpr auto log(Scalar<T> const& first) -> Scalar<T>
	{
		return { std::log(first.GetValue()) };
	}
}