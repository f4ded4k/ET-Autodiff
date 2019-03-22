#pragma once

#include <cmath>
#include <array>
#include <type_traits>
#include <random>

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

namespace TTest
{
	template <size_t ... Args>
	struct total_size;

	template <size_t F, size_t... Fs>
	struct total_size<F, Fs...>
	{
		constexpr static size_t value = F * total_size<Fs...>::value;
	};

	template <size_t F>
	struct total_size<F>
	{
		constexpr static size_t value = F;
	};

	template <size_t... Args>
	constexpr static size_t total_size_v = total_size<Args...>::value;

	template <size_t I>
	struct i_integrals
	{
		using type = decltype(std::tuple_cat(
			std::declval<std::tuple<size_t>>(),
			std::declval<typename i_integrals<I - 1>::type>()
		));
	};

	template <>
	struct i_integrals<1>
	{
		using type = std::tuple<size_t>;
	};

	template <size_t I>
	using i_integrals_t = typename i_integrals<I>::type;

	template <size_t... Ss>
	struct value_list {};

	template <size_t... S1s, size_t... S2s>
	value_list<S1s..., S2s...> value_list_cat(value_list<S1s...> const&, value_list<S2s...> const&)
	{
		return value_list<S1s..., S2s...>();
	}

	template <size_t I>
	struct i_zeros
	{
		using type = decltype(value_list_cat(
			std::declval<value_list<0>>(),
			std::declval<typename i_zeros<I - 1>::type>()
		));
	};

	template <>
	struct i_zeros<0>
	{
		using type = value_list<>;
	};

	template <size_t I>
	using i_zeros_t = typename i_zeros<I>::type;
	
	template <typename T>
	struct reverse_list;

	template <size_t F, size_t... Fs>
	struct reverse_list<value_list<F, Fs...>>
	{
		using type = decltype(value_list_cat(
			std::declval<typename reverse_list<value_list<Fs...>>::type>(),
			std::declval<value_list<F>>()
		));
	};

	template <size_t F>
	struct reverse_list<value_list<F>>
	{
		using type = value_list<F>;
	};

	template <typename T>
	using reverse_list_t = typename reverse_list<T>::type;

	template <typename T, typename N>
	struct _impl_nD_array;

	template <typename T, size_t N, size_t... Ns>
	struct _impl_nD_array<T, value_list<N, Ns...>>
	{
		using type = std::array<typename _impl_nD_array<T, value_list<Ns...>>::type, N>;
	};

	template <typename T, size_t N>
	struct _impl_nD_array<T, value_list<N>>
	{
		using type = std::array<T, N>;
	};

	template <typename T, typename N>
	using _impl_nD_array_t = typename _impl_nD_array<T, N>::type;

	template <typename T, size_t... Ns>
	using nD_array_t = _impl_nD_array_t<T, reverse_list_t<value_list<Ns...>>>;

	template <typename V, typename Tup, size_t... Ds>
	class Tensor;

	template <typename V, typename... Indices, size_t... Ds>
	class Tensor<V, std::tuple<Indices...>, Ds...>
	{
	public:
		constexpr static size_t n_dims_v = sizeof...(Ds);
		constexpr static size_t n_elems_v = total_size_v<Ds...>;

	private:
		using array_t = nD_array_t<V, Ds...>;

		template <typename VList>
		struct _impl_GetVals;

		template <size_t... Vs>
		struct _impl_GetVals<value_list<Vs...>>
		{
			constexpr static V* _impl_cbegin(Tensor<V, std::tuple<Indices...>, Ds...>& encloser)
			{
				return &encloser(Vs...);
			}
		};

		constexpr V* cbegin()
		{
			return _impl_GetVals<i_zeros_t<n_dims_v>>::_impl_cbegin(*this);
		}


		template <typename T, typename... Ts>
		constexpr auto& _impl_get_at_index(T N, Ts... Ns)
		{
			if constexpr (sizeof...(Ts) > 0)
			{
				return _impl_get_at_index(Ns...)[N];
			}
			else
			{
				return _data[N];
			}
		}

		template <typename T, typename... Ts>
		constexpr auto& _impl_get_at_index(T N, Ts... Ns) const
		{
			if constexpr (sizeof...(Ts) > 0)
			{
				return _impl_get_at_index(Ns...)[N];
			}
			else
			{
				return _data[N];
			}
		}

	private:
		array_t& _data;

	public:
		constexpr Tensor() : _data{ *(new array_t) } {}

		constexpr Tensor(V const& init_value) : Tensor()
		{
			std::uninitialized_fill_n(cbegin(), n_elems_v, init_value);
		}

		template <typename Generator>
		constexpr Tensor(Generator const& g) : Tensor()
		{
			std::generate_n(cbegin(), n_elems_v, g);
		}

		constexpr V& operator()(Indices... indices)
		{
			return _impl_get_at_index<Indices...>(indices...);
		}

		constexpr V const& operator()(Indices... indices) const
		{
			return _impl_get_at_index<Indices...>(indices...);
		}

		constexpr auto begin()
		{
			return _data.begin();
		}

		constexpr auto end()
		{
			return _data.end();
		}
	};

	struct TensorFactory
	{
		template <typename V, size_t... Ds>
		constexpr static auto MakeZeroTensor()
		{
			return MakeUniformTensor<V, Ds...>(V{ 0 });
		}

		template <typename V, size_t... Ds>
		constexpr static auto MakeUniformTensor(V const& init_value)
		{
			return Tensor<V, i_integrals_t<sizeof...(Ds)>, Ds...>{init_value};
		}

		template <typename V, size_t... Ds>
		constexpr static auto MakeRandomTensor(V const& min_value, V const& max_value)
		{
			std::random_device random_device;
			std::mt19937 rng{ random_device() };
			std::uniform_real_distribution<V> distribution{ min_value, max_value };
			return Tensor<V, i_integrals_t<sizeof...(Ds)>, Ds...>{[&]() {return distribution(rng); }};
		}
	};
}