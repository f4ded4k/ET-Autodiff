#pragma once

#include <type_traits>
#include <array>
#include "my_traits.h"

namespace Et_test
{
	// Base class for all tensors.
	struct TensorBase {};

	// Returns true if all types are of the specific type.
	template <typename... T>
	constexpr bool is_tensor_v = std::conjunction_v<std::is_base_of<TensorBase, std::decay_t<T>>...>;

	// Forward declaration.
	template <typename D, typename IdxTup>
	class TensorImpl;

	// Forward declaration.
	template <typename V, size_t... Ds>
	class Tensor;

	// Class that implements operator() of Tensor.
	template <typename V, size_t... Dims, typename... Indices>
	class TensorImpl<Tensor<V, Dims...>, std::tuple<Indices...>>
	{
	private:
		using D = Tensor<V, Dims...>;
	public:
		V& operator()(Indices... indices)
		{
			return _GetElem(indices...);
		}

		V const& operator()(Indices... indices) const
		{
			return _GetElem(indices...);
		}
	private:
		template <typename T, typename... Ts>
		auto& _GetElem(T n, Ts... ns)
		{
			if constexpr (sizeof...(Ts) > 0)
			{
				return _GetElem(ns...)[n];
			}
			else
			{
				return (*static_cast<D&>(*this)._data)[n];
			}
		}

		template <typename T, typename... Ts>
		auto const& _GetElem(T n, Ts... ns) const
		{
			if constexpr (sizeof...(Ts) > 0)
			{
				return _GetElem(ns...)[n];
			}
			else
			{
				return (*static_cast<D const&>(*this)._data)[n];
			}
		}
	};

	// Primary class that implements a tensor of any dimension.
	template <typename V, size_t... Ds>
	class Tensor : 
		public TensorImpl<Tensor<V, Ds...>, uniform_tuple_t<size_t, sizeof...(Ds)>>, 
		private TensorBase
	{
	private:
		friend class TensorImpl<Tensor<V, Ds...>, uniform_tuple_t<size_t, sizeof...(Ds)>>;

		constexpr static size_t ndims_v = sizeof...(Ds);
		constexpr static size_t nelems_v = list_product_v<value_list<size_t, Ds...>>;
		using elem_t = V;
		using data_t = nd_array_t<V, reverse_list_t<value_list<size_t, Ds...>>>;

		data_t* _data = nullptr;
		V* _cbegin = nullptr;
		V* _cend = nullptr;
	public:
		explicit Tensor() : _data{ new data_t }
		{
			_cbegin = _ListCapture<uniform_list_t<size_t, 0, ndims_v>>::_cbegin(*this);
			_cend = _cbegin + nelems_v;
		}

		Tensor(V const& value) : Tensor()
		{
			std::uninitialized_fill(cbegin(), cend(), value);
		}

		Tensor(Tensor<V, Ds...> const& other) : Tensor()
		{
			*_data = *other._data;
		}

		Tensor(Tensor<V, Ds...>&& other)
		{
			_data = other._data;
			_cbegin = _ListCapture<uniform_list_t<size_t, 0, ndims_v>>::_cbegin(*this);
			_cend = _cbegin + nelems_v;
			other._data = nullptr;
		}

		Tensor<V, Ds...>& operator=(Tensor<V, Ds...> const& other)
		{
			*_data = *other._data;
			return *this;
		}

		Tensor<V, Ds...>& operator=(Tensor<V, Ds...>&& other)
		{
			_cbegin = other._cbegin;
			_cend = other._cend;
			_data = other._data;
			other._data = nullptr;
			return *this;
		}

		~Tensor()
		{
			delete _data;
			_cbegin = nullptr;
			_cend = nullptr;
		}

		void Fill(V const& value)
		{
			std::fill(cbegin(), cend(), value);
		}

		V* const cbegin() { return _cbegin; }

		V const* const cbegin() const { return _cbegin; }

		V* const cend() { return _cend; }

		V const* const cend() const { return _cend; }
	private:
		template <typename VL>
		struct _ListCapture;

		template <typename _V, _V... _Vs>
		struct _ListCapture<value_list<_V, _Vs...>>
		{
			template <typename Tensor>
			static V* _cbegin(Tensor& tensor)
			{
				return std::addressof(tensor(_Vs...));
			}
		};
	};

	// Utility function to create a tensor with all elements set to 0.
	template <typename V, size_t... Ds>
	auto Zeros()
	{
		Tensor<V, Ds...> tensor;
		std::uninitialized_fill(tensor.cbegin(), tensor.cend(), static_cast<V>(0));
		return std::move(tensor);
	}
	
	// Utility function to create a tensor with all elements set to 1.
	template <typename V, size_t... Ds>
	auto Ones()
	{
		Tensor<V, Ds...> tensor;
		std::uninitialized_fill(tensor.cbegin(), tensor.cend(), static_cast<V>(1));
		return std::move(tensor);
	}

	// Operator Overloads.
	template <typename V1, typename V2, size_t... Ds>
	auto operator+(Tensor<V1, Ds...> const& first, Tensor<V2, Ds...> const& second)
	{
		using elem_t = decltype(std::declval<V1>() + std::declval<V2>());
		Tensor<elem_t, Ds...> result;
		auto iter_a = first.cbegin();
		auto iter_b = second.cbegin();
		for (auto iter_r = result.cbegin(); iter_r != result.cend(); iter_r++, iter_a++, iter_b++)
		{
			*iter_r = *iter_a + *iter_b;
		}
		return result;
	}

	template <typename V1, typename V2, size_t... Ds>
	auto& operator+=(Tensor<V1, Ds...>& first, Tensor<V2, Ds...> const& second)
	{
		auto iter_b = second.cbegin();
		for (auto iter_r = first.cbegin(); iter_r != first.cend(); iter_r++, iter_b++)
		{
			*iter_r += *iter_b;
		}
		return first;
	}

	template <typename V1, typename V2, size_t... Ds>
	auto operator-(Tensor<V1, Ds...> const& first, Tensor<V2, Ds...> const& second)
	{
		using elem_t = decltype(std::declval<V1>() - std::declval<V2>());
		Tensor<elem_t, Ds...> result;
		auto iter_a = first.cbegin();
		auto iter_b = second.cbegin();
		for (auto iter_r = result.cbegin(); iter_r != result.cend(); iter_r++, iter_a++, iter_b++)
		{
			*iter_r = *iter_a - *iter_b;
		}
		return result;
	}

	template <typename V1, typename V2, size_t... Ds>
	auto& operator-=(Tensor<V1, Ds...>& first, Tensor<V2, Ds...> const& second)
	{
		auto iter_b = second.cbegin();
		for (auto iter_r = first.cbegin(); iter_r != first.cend(); iter_r++, iter_b++)
		{
			*iter_r -= *iter_b;
		}
		return first;
	}

	template <typename V, size_t... Ds>
	auto operator-(Tensor<V, Ds...> const& first)
	{
		Tensor<V, Ds...> result;
		auto iter_a = first.cbegin();
		for (auto iter_r = result.cbegin(); iter_r != result.cend(); iter_r++, iter_a++)
		{
			*iter_r = -(*iter_a);
		}
		return result;
	}

	template <typename V1, typename V2, size_t... Ds>
	auto operator*(Tensor<V1, Ds...> const& first, Tensor<V2, Ds...> const& second)
	{
		using elem_t = decltype(std::declval<V1>() * std::declval<V2>());
		Tensor<elem_t, Ds...> result;
		auto iter_a = first.cbegin();
		auto iter_b = second.cbegin();
		for (auto iter_r = result.cbegin(); iter_r != result.cend(); iter_r++, iter_a++, iter_b++)
		{
			*iter_r = *iter_a * *iter_b;
		}
		return result;
	}

	template <typename V1, typename V2, size_t... Ds>
	auto& operator*=(Tensor<V1, Ds...>& first, Tensor<V2, Ds...> const& second)
	{
		auto iter_b = second.cbegin();
		for (auto iter_r = first.cbegin(); iter_r != first.cend(); iter_r++, iter_b++)
		{
			*iter_r *= *iter_b;
		}
		return first;
	}

	template <typename V1, typename V2, size_t... Ds>
	auto operator/(Tensor<V1, Ds...> const& first, Tensor<V2, Ds...> const& second)
	{
		using elem_t = decltype(std::declval<V1>() / std::declval<V2>());
		Tensor<elem_t, Ds...> result;
		auto iter_a = first.cbegin();
		auto iter_b = second.cbegin();
		for (auto iter_r = result.cbegin(); iter_r != result.cend(); iter_r++, iter_a++, iter_b++)
		{
			*iter_r = *iter_a / *iter_b;
		}
		return result;
	}

	template <typename V1, typename V2, size_t... Ds>
	auto& operator/=(Tensor<V1, Ds...>& first, Tensor<V2, Ds...> const& second)
	{
		auto iter_b = second.cbegin();
		for (auto iter_r = first.cbegin(); iter_r != first.cend(); iter_r++, iter_b++)
		{
			*iter_r /= *iter_b;
		}
		return first;
	}

	template <typename V1, typename V2, size_t DA, size_t DB, size_t DC>
	auto matmul(Tensor<V1, DA, DB> const& first, Tensor<V2, DB, DC> const& second)
	{
		using elem_t = decltype(std::declval<V1>() * std::declval<V2>());
		Tensor<elem_t, DA, DC> result = Zeros<elem_t, DA, DC>();
		for (int i = 0; i < DA; i++)
		{
			for (int j = 0; j < DC; j++)
			{
				for (int k = 0; k < DB; k++)
				{
					result(i, j) += first(i, k) * second(k, j);
				}
			}
		}
		return result;
	}
}