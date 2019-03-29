#pragma once

namespace Et_test
{
	// A container type for variable number of values analogous to std::tuple for types.
	template <typename V, V... Vs>
	struct value_list
	{
		constexpr static size_t length = sizeof...(Vs);
		using value_t = V;
	};

	// Utility metafunction to concatenate two value_lists of same type.
	template <typename V, V... V1s, V... V2s>
	constexpr auto value_list_cat(value_list<V, V1s...> const&, value_list<V, V2s...> const&)
	{
		return value_list<V, V1s..., V2s...>{};
	}

	// Implementation of value_list_element_v.
	template <typename VL, size_t Curr, size_t Query>
	struct _impl_value_list_element;

	template <typename V, V F, V... Vs, size_t I>
	struct _impl_value_list_element<value_list<V, F, Vs...>, I, I>
	{
		constexpr static V value = F;
	};

	template <typename V, V F, V... Vs, size_t Curr, size_t Query>
	struct _impl_value_list_element<value_list<V, F, Vs...>, Curr, Query>
	{
		constexpr static V value = _impl_value_list_element<value_list<V, Vs...>, Curr + 1, Query>::value;
	};

	// Alias for _impl_value_list_element::value.
	template <typename VL, size_t Curr, size_t Query>
	constexpr typename VL::value_t _impl_value_list_element_v = _impl_value_list_element<VL, Curr, Query>::value;

	// Returns the Query-th value from the value_list VL.
	template <typename VL, size_t Query>
	constexpr typename VL::value_t value_list_element_v = _impl_value_list_element_v<VL, 0, Query>;

	// Implementation of list_product_t.
	template <typename VL>
	struct list_product;

	template <typename V, V F, V... Vs>
	struct list_product<value_list<V, F, Vs...>>
	{
		constexpr static V value = F * list_product<value_list<V, Vs...>>::value;
	};

	template <typename V, V F>
	struct list_product<value_list<V, F>>
	{
		constexpr static V value = F;
	};

	// Returns the product of all the values in VL.
	template <typename VL>
	constexpr typename VL::value_t list_product_v = list_product<VL>::value;

	// Implementation of list_sum_t.
	template <typename VL>
	struct list_sum;

	template <typename V, V F, V... Vs>
	struct list_sum<value_list<V, F, Vs...>>
	{
		constexpr static V value = F + list_sum<value_list<V, Vs...>>::value;
	};

	template <typename V, V F>
	struct list_sum<value_list<V, F>>
	{
		constexpr static V value = F;
	};
	
	// Returns the sum of all the values in VL.
	template <typename VL>
	constexpr typename VL::value_t list_sum_v = list_sum<VL>::value;

	// Utility metafunction to get J-th value from I-th element in the tuple.
	template <typename Tup, size_t I, size_t J>
	struct child_index_utility
	{
		constexpr static typename std::tuple_element_t<I, Tup>::child_list_t::value_t value =
			value_list_element_v<typename std::tuple_element_t<I, Tup>::child_list_t, J>;
	};

	// Returns the J-th value from the I-th element in the tuple.
	template <typename Tup, size_t I, size_t J>
	constexpr auto child_index_utility_v = child_index_utility<Tup, I, J>::value;

	// Implementation of nd_array_t.
	template <typename V, typename VL>
	struct nd_array;

	template <typename V, size_t N, size_t...Ns>
	struct nd_array<V, value_list<size_t, N, Ns...>>
	{
		using type = std::array<typename nd_array<V, value_list<size_t, Ns...>>::type, N>;
	};

	template <typename V, size_t N>
	struct nd_array<V, value_list<size_t, N>>
	{
		using type = std::array<V, N>;
	};

	// Returns a nested std::array of type V with the given dimensions in VL.
	template <typename V, typename VL>
	using nd_array_t = typename nd_array<V, VL>::type;

	// Implementation of reverse_list_t.
	template <typename VL>
	struct reverse_list;

	template <typename V, V F, V... Vs>
	struct reverse_list<value_list<V, F, Vs...>>
	{
		using type = decltype(value_list_cat(
			std::declval<typename reverse_list<value_list<V, Vs...>>::type>(),
			std::declval<value_list<V, F>>()
		));
	};
	
	template <typename V, V F>
	struct reverse_list<value_list<V, F>>
	{
		using type = value_list<V, F>;
	};

	// Returns a value_list with entries in reverse order.
	template <typename VL>
	using reverse_list_t = typename reverse_list<VL>::type;

	// Implementation of uniform_list_t.
	template <typename V, V Val, size_t Left>
	struct uniform_list
	{
		using type = decltype(value_list_cat(
			std::declval<value_list<V, Val>>(),
			std::declval<typename uniform_list<V, Val, Left - 1>::type>()
		));
	};

	template <typename V, V Val>
	struct uniform_list<V, Val, 0>
	{
		using type = value_list<V>;
	};

	// Returns a value_list with N elements having value Val
	template <typename V, V Val, size_t N>
	using uniform_list_t = typename uniform_list<V, Val, N>::type;

	// Implementation of uniform_tuple_t.
	template <typename V, size_t Left>
	struct uniform_tuple
	{
		using type = decltype(std::tuple_cat(
			std::declval<std::tuple<V>>(),
			std::declval<typename uniform_tuple<V, Left - 1>::type>()
		));
	};

	template <typename V>
	struct uniform_tuple<V, 0>
	{
		using type = std::tuple<>;
	};

	// Returns a tuple containing N types T.
	template <typename V, size_t N>
	using uniform_tuple_t = typename uniform_tuple<V, N>::type;
}