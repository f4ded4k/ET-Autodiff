#pragma once

#include <type_traits>
#include <tuple>
#include <cmath>
#include "tensor.h"

namespace Et {

	using ScalarD = Num::Scaler<double>;

	template <typename T1, typename T2>
	constexpr auto PlFeed(T1& first, T2 const& second)
	{
		return std::make_pair(&first, second);
	}

	struct Expr {};

	struct TerminalExpr {};
	struct BinaryExpr {};
	struct UnaryExpr {};
	struct TrainableExpr {};

	template<typename... T>
	constexpr bool is_expr_v = std::conjunction_v<std::is_base_of<Expr, std::decay_t<T>>...>;

	template <typename V>
	class ConstantExpr : public Expr, private TerminalExpr
	{
	public:
		static_assert(Num::is_tensor_v<V>);
		using value_t = std::decay_t<V>;

	private:
		V const _value;

	public:
		constexpr ConstantExpr(V const& value) : _value(value) {}

		constexpr auto operator()() const -> auto&
		{
			return _value;
		}

		template <int I, typename T>
		constexpr auto Eval(T& tuple) const -> auto&
		{
			std::get<I>(tuple).SetLocalGrads(this);
			return _value;
		}
	};

	ConstantExpr(int const&)->ConstantExpr<ScalarD>;
	ConstantExpr(double const&)->ConstantExpr<ScalarD>;

	template <typename V>
	class PlaceholderExpr : public Expr, private TerminalExpr
	{
	public:
		static_assert(Num::is_tensor_v<V>);
		using value_t = std::decay_t<V>;

	private:
		bool _is_default;
		V _value;

	public:
		constexpr PlaceholderExpr() : _is_default{ true }, _value{ Num::default_v<value_t> } {}

		constexpr auto operator()() const -> auto&
		{
			return _value;
		}

		template <int I, typename T>
		constexpr auto Eval(T& tuple) const -> auto&
		{
			std::get<I>(tuple).SetLocalGrads(this);
			return _value;
		}

		constexpr auto FeedValue(value_t const& value) -> void
		{
			_value = value;
		}
	};

	PlaceholderExpr()->PlaceholderExpr<ScalarD>;

	template <typename V>
	class VariableExpr : public Expr, private TerminalExpr, private TrainableExpr
	{
	public:
		static_assert(Num::is_tensor_v<V>);
		using value_t = std::decay_t<V>;

	private:
		V _value;

	public:
		mutable V _cache;

		constexpr VariableExpr(V const& value) : _value{ value } {}

		constexpr auto operator()() const -> auto&
		{
			return _value;
		}

		template <int I, typename T>
		constexpr auto Eval(T& tuple) const -> auto&
		{
			std::get<I>(tuple).SetLocalGrads(this);
			return _value;
		}

		constexpr auto SetCache(value_t const& delta) const -> void
		{
			_cache = delta;
		}

		constexpr auto SetValueAndResetCache(double learning_rate) -> void
		{
			_value -= learning_rate * _cache;
			_cache = 0.0;
		}
	};

	VariableExpr(int const&)->VariableExpr<ScalarD>;
	VariableExpr(double const&)->VariableExpr<ScalarD>;

	template <typename E1, typename E2>
	class AddExpr : public Expr, private BinaryExpr
	{
	public:
		static_assert(is_expr_v<E1, E2>);
		using first_expr_t = std::decay_t<E1>;
		using second_expr_t = std::decay_t<E2>;
		using first_value_t = std::decay_t<decltype(std::declval<E1>()())>;
		using second_value_t = std::decay_t<decltype(std::declval<E2>()())>;
		using first_local_grad_t = first_value_t;
		using second_local_grad_t = second_value_t;
		using value_t = std::decay_t<decltype(std::declval<E1>()() + std::declval<E2>()())>;

	private:
		E1 _first_expr;
		E2 _second_expr;

	public:
		constexpr AddExpr(E1&& first_expr, E2&& second_expr)
			: _first_expr{ std::forward<E1>(first_expr) }, _second_expr{ std::forward<E2>(second_expr) } {}

		constexpr auto operator()() const -> auto
		{
			return _first_expr() + _second_expr();
		}

		template <int I, typename T>
		constexpr auto Eval(T& tuple) const -> auto
		{
			first_value_t first_value = _first_expr.template Eval<std::tuple_element_t<I, T>::child_one_v>(tuple);
			second_value_t second_value = _second_expr.template Eval<std::tuple_element_t<I, T>::child_two_v>(tuple);
			std::get<I>(tuple).SetLocalGrads(this, first_local_grad_t(1.0), second_local_grad_t(1.0));
			return first_value + second_value;
		}
	};

	template<typename E1, typename E2>
	AddExpr(E1&&, E2&&)->AddExpr<E1, E2>;

	template <typename E1, typename E2>
	class MultiplyExpr : public Expr, private BinaryExpr
	{
	public:
		static_assert(is_expr_v<E1, E2>);
		using first_expr_t = std::decay_t<E1>;
		using second_expr_t = std::decay_t<E2>;
		using first_value_t = std::decay_t<decltype(std::declval<E1>()())>;
		using second_value_t = std::decay_t<decltype(std::declval<E2>()())>;
		using first_local_grad_t = second_value_t;
		using second_local_grad_t = first_value_t;
		using value_t = std::decay_t<decltype(std::declval<E1>()() * std::declval<E2>()())>;

	private:
		E1 _first_expr;
		E2 _second_expr;

	public:
		constexpr MultiplyExpr(E1&& first_expr, E2&& second_expr)
			: _first_expr{ std::forward<E1>(first_expr) }, _second_expr{ std::forward<E2>(second_expr) } {}

		constexpr auto operator()() const -> auto
		{
			return _first_expr() * _second_expr();
		}

		template <int I, typename T>
		constexpr auto Eval(T& tuple) const -> auto
		{
			first_value_t first_value = _first_expr.template Eval<std::tuple_element_t<I, T>::child_one_v>(tuple);
			second_value_t second_value = _second_expr.template Eval<std::tuple_element_t<I, T>::child_two_v>(tuple);
			std::get<I>(tuple).SetLocalGrads(this, second_value, first_value);
			return first_value * second_value;
		}
	};

	template<typename E1, typename E2>
	MultiplyExpr(E1&&, E2&&)->MultiplyExpr<E1, E2>;

	template <typename E1, typename E2>
	class SubtractExpr : public Expr, private BinaryExpr
	{
	public:
		static_assert(is_expr_v<E1, E2>);
		using first_expr_t = std::decay_t<E1>;
		using second_expr_t = std::decay_t<E2>;
		using first_value_t = std::decay_t<decltype(std::declval<E1>()())>;
		using second_value_t = std::decay_t<decltype(std::declval<E2>()())>;
		using first_local_grad_t = first_value_t;
		using second_local_grad_t = second_value_t;
		using value_t = std::decay_t<decltype(std::declval<E1>()() - std::declval<E2>()())>;

	private:
		E1 _first_expr;
		E2 _second_expr;

	public:
		constexpr SubtractExpr(E1&& first_expr, E2&& second_expr)
			: _first_expr{ std::forward<E1>(first_expr) }, _second_expr{ std::forward<E2>(second_expr) } {}

		constexpr auto operator()() const -> auto
		{
			return _first_expr() - _second_expr();
		}

		template <int I, typename T>
		constexpr auto Eval(T& tuple) const ->auto
		{
			first_value_t first_value = _first_expr.template Eval<std::tuple_element_t<I, T>::child_one_v>(tuple);
			second_value_t second_value = _second_expr.template Eval<std::tuple_element_t<I, T>::child_two_v>(tuple);
			std::get<I>(tuple).SetLocalGrads(this, first_local_grad_t(1.0), second_local_grad_t(-1.0));
			return first_value - second_value;
		}
	};

	template<typename E1, typename E2>
	SubtractExpr(E1&&, E2&&)->SubtractExpr<E1, E2>;

	template <typename E1, typename E2>
	class DivideExpr : public Expr, private BinaryExpr
	{
	public:
		static_assert(is_expr_v<E1, E2>);
		using first_expr_t = std::decay_t<E1>;
		using second_expr_t = std::decay_t<E2>;
		using first_value_t = std::decay_t<decltype(std::declval<E1>()())>;
		using second_value_t = std::decay_t<decltype(std::declval<E2>()())>;
		using first_local_grad_t = first_value_t;
		using second_local_grad_t = second_value_t;
		using value_t = std::decay_t<decltype(std::declval<E1>()() / std::declval<E2>()())>;

	private:
		E1 _first_expr;
		E2 _second_expr;

	public:
		constexpr DivideExpr(E1&& first_expr, E2&& second_expr)
			: _first_expr{ std::forward<E1>(first_expr) }, _second_expr{ std::forward<E2>(second_expr) } {}

		constexpr auto operator()() const -> auto
		{
			return _first_expr() / _second_expr();
		}

		template <int I, typename T>
		constexpr auto Eval(T& tuple) const -> auto
		{
			first_value_t first_value = _first_expr.template Eval<std::tuple_element_t<I, T>::child_one_v>(tuple);
			second_value_t second_value = _second_expr.template Eval<std::tuple_element_t<I, T>::child_two_v>(tuple);
			auto second_value_inverse = second_value.Inverse();
			std::get<I>(tuple).SetLocalGrads(this, second_value_inverse, -first_value * second_value_inverse * second_value_inverse);
			return first_value / second_value;
		}
	};

	template<typename E1, typename E2>
	DivideExpr(E1&&, E2&&)->DivideExpr<E1, E2>;

	template <typename E1, typename E2>
	class PowerExpr : public Expr, private BinaryExpr
	{
	public:
		static_assert(is_expr_v<E1, E2>);
		using first_expr_t = std::decay_t<E1>;
		using second_expr_t = std::decay_t<E2>;
		using first_value_t = std::decay_t<decltype(std::declval<E1>()())>;
		using second_value_t = std::decay_t<decltype(std::declval<E2>()())>;
		using first_local_grad_t = first_value_t;
		using second_local_grad_t = second_value_t;
		using value_t = std::decay_t<decltype(Num::pow(std::declval<E1>()(), std::declval<E2>()()))>;

	private:
		E1 _first_expr;
		E2 _second_expr;

	public:
		constexpr PowerExpr(E1&& first_expr, E2&& second_expr)
			: _first_expr{ std::forward<E1>(first_expr) }, _second_expr{ std::forward<E2>(second_expr) } {}

		constexpr auto operator()() const -> auto
		{
			return Num::pow(_first_expr(), _second_expr());
		}

		template <int I, typename T>
		constexpr auto Eval(T& tuple) const -> auto
		{
			first_value_t first_value = _first_expr.template Eval<std::tuple_element_t<I, T>::child_one_v>(tuple);
			second_value_t second_value = _second_expr.template Eval<std::tuple_element_t<I, T>::child_two_v>(tuple);
			value_t value = Num::pow(first_value, second_value);
			std::get<I>(tuple).SetLocalGrads(this, second_value * value * first_value.Inverse(), value * Num::log(first_value));
			return value;
		}
	};

	template<typename E1, typename E2>
	PowerExpr(E1&&, E2&&)->PowerExpr<E1, E2>;

	template <typename E1>
	class NegateExpr : public Expr, private UnaryExpr
	{
	public:
		static_assert(is_expr_v<E1>);
		using first_expr_t = std::decay_t<E1>;
		using first_value_t = std::decay_t<decltype(std::declval<E1>()())>;
		using first_local_grad_t = first_value_t;
		using value_t = std::decay_t<decltype(-std::declval<E1>()())>;

	private:
		E1 _first_expr;

	public:
		constexpr NegateExpr(E1&& first_expr)
			: _first_expr{ std::forward<E1>(first_expr) } {}

		constexpr auto operator()() const -> auto
		{
			return -_first_expr();
		}

		template <int I, typename T>
		constexpr auto Eval(T& tuple) const -> auto
		{
			first_value_t first_value = _first_expr.template Eval<std::tuple_element_t<I, T>::child_one_v>(tuple);
			std::get<I>(tuple).SetLocalGrads(this, first_local_grad_t(-1.0));
			return -first_value;
		}
	};

	template<typename E1>
	NegateExpr(E1&&)->NegateExpr<E1>;

	template <typename E1>
	class LogExpr : public Expr, private UnaryExpr
	{
	public:
		static_assert(is_expr_v<E1>);
		using first_expr_t = std::decay_t<E1>;
		using first_value_t = std::decay_t<decltype(std::declval<E1>()())>;
		using first_local_grad_t = first_value_t;
		using value_t = std::decay_t<decltype(Num::log(std::declval<E1>()()))>;

	private:
		E1 _first_expr;

	public:
		constexpr LogExpr(E1&& first_expr)
			: _first_expr{ std::forward<E1>(first_expr) } {}

		constexpr auto operator()() const -> auto
		{
			return Num::log(_first_expr());
		}

		template <int I, typename T>
		constexpr auto Eval(T& tuple) const -> auto
		{
			first_value_t first_value = _first_expr.template Eval<std::tuple_element_t<I, T>::child_one_v>(tuple);
			std::get<I>(tuple).SetLocalGrads(this, first_value.Inverse());
			return Num::log(first_value);
		}
	};

	template<typename E1>
	LogExpr(E1&&)->LogExpr<E1>;

	template <typename E1>
	class SinExpr : public Expr, private UnaryExpr
	{
	public:
		static_assert(is_expr_v<E1>);
		using first_expr_t = std::decay_t<E1>;
		using first_value_t = std::decay_t<decltype(std::declval<E1>()())>;
		using first_local_grad_t = first_value_t;
		using value_t = std::decay_t<decltype(Num::sin(std::declval<E1>()()))>;

	private:
		E1 _first_expr;

	public:
		constexpr SinExpr(E1&& first_expr)
			: _first_expr{ std::forward<E1>(first_expr) } {}

		constexpr auto operator()() const -> auto
		{
			return Num::sin(_first_expr());
		}

		template <int I, typename T>
		constexpr auto Eval(T& tuple) const -> auto
		{
			first_value_t first_value = _first_expr.template Eval<std::tuple_element_t<I, T>::child_one_v>(tuple);
			std::get<I>(tuple).SetLocalGrads(this, Num::cos(first_value));
			return Num::sin(first_value);
		}
	};

	template<typename E1>
	SinExpr(E1&&)->SinExpr<E1>;

	template <typename E1>
	class CosExpr : public Expr, private UnaryExpr
	{
	public:
		static_assert(is_expr_v<E1>);
		using first_expr_t = std::decay_t<E1>;
		using first_value_t = std::decay_t<decltype(std::declval<E1>()())>;
		using first_local_grad_t = first_value_t;
		using value_t = std::decay_t<decltype(Num::cos(std::declval<E1>()()))>;

	private:
		E1 _first_expr;

	public:
		constexpr CosExpr(E1&& first_expr)
			: _first_expr{ std::forward<E1>(first_expr) } {}

		constexpr auto operator()() const -> auto
		{
			return Num::cos(_first_expr());
		}

		template <int I, typename T>
		constexpr auto Eval(T& tuple) const -> auto
		{
			first_value_t first_value = _first_expr.template Eval<std::tuple_element_t<I, T>::child_one_v>(tuple);
			std::get<I>(tuple).SetLocalGrads(this, -Num::sin(first_value));
			return Num::cos(first_value);
		}
	};

	template<typename E1>
	CosExpr(E1&&)->CosExpr<E1>;

	template <typename E1>
	class TanExpr : public Expr, private UnaryExpr
	{
	public:
		static_assert(is_expr_v<E1>);
		using first_expr_t = std::decay_t<E1>;
		using first_value_t = std::decay_t<decltype(std::declval<E1>()())>;
		using first_local_grad_t = first_value_t;
		using value_t = std::decay_t<decltype(Num::tan(std::declval<E1>()()))>;

	private:
		E1 _first_expr;

	public:
		constexpr TanExpr(E1&& first_expr)
			: _first_expr{ std::forward<E1>(first_expr) } {}

		constexpr auto operator()() const -> auto
		{
			return Num::tan(_first_expr());
		}

		template <int I, typename T>
		constexpr auto Eval(T& tuple) const -> auto
		{
			first_value_t first_value = _first_expr.template Eval<std::tuple_element_t<I, T>::child_one_v>(tuple);
			auto sec_value = Num::sec(first_value);
			std::get<I>(tuple).SetLocalGrads(this, sec_value * sec_value);
			return Num::tan(first_value);
		}
	};

	template<typename E1>
	TanExpr(E1&&)->TanExpr<E1>;

	template <typename E1, typename E2, typename = std::enable_if_t<is_expr_v<E1, E2>>>
	constexpr auto operator+(E1&& first_expr, E2&& second_expr) -> AddExpr<E1, E2>
	{
		return { std::forward<E1>(first_expr), std::forward<E2>(second_expr) };
	}

	template <typename E1, typename E2, typename = std::enable_if_t<is_expr_v<E1, E2>>>
	constexpr auto operator*(E1&& first_expr, E2&& second_expr) -> MultiplyExpr<E1, E2>
	{
		return { std::forward<E1>(first_expr), std::forward<E2>(second_expr) };
	}

	template <typename E1, typename E2, typename = std::enable_if_t<is_expr_v<E1, E2>>>
	constexpr auto operator-(E1&& first_expr, E2&& second_expr) -> SubtractExpr<E1, E2>
	{
		return { std::forward<E1>(first_expr), std::forward<E2>(second_expr) };
	}

	template <typename E1, typename E2, typename = std::enable_if_t<is_expr_v<E1, E2>>>
	constexpr auto operator/(E1&& first_expr, E2&& second_expr) -> DivideExpr<E1, E2>
	{
		return { std::forward<E1>(first_expr), std::forward<E2>(second_expr) };
	}

	template <typename E1, typename E2, typename = std::enable_if_t<is_expr_v<E1, E2>>>
	constexpr auto pow(E1&& first_expr, E2&& second_expr) -> PowerExpr<E1, E2>
	{
		return { std::forward<E1>(first_expr), std::forward<E2>(second_expr) };
	}

	template <typename E1, typename = std::enable_if_t<is_expr_v<E1>>>
	constexpr auto operator-(E1&& first_expr) -> NegateExpr<E1>
	{
		return { std::forward<E1>(first_expr) };
	}

	template <typename E1, typename = std::enable_if_t<is_expr_v<E1>>>
	constexpr auto log(E1&& first_expr) -> LogExpr<E1>
	{
		return { std::forward<E1>(first_expr) };
	}

	template <typename E1, typename = std::enable_if_t<is_expr_v<E1>>>
	constexpr auto sin(E1&& first_expr) -> SinExpr<E1>
	{
		return { std::forward<E1>(first_expr) };
	}

	template <typename E1, typename = std::enable_if_t<is_expr_v<E1>>>
	constexpr auto cos(E1&& first_expr) -> CosExpr<E1>
	{
		return { std::forward<E1>(first_expr) };
	}

	template <typename E1, typename = std::enable_if_t<is_expr_v<E1>>>
	constexpr auto tan(E1&& first_expr) -> TanExpr<E1>
	{
		return { std::forward<E1>(first_expr) };
	}

	template <typename E, int... Ints>
	class Node;

	template <typename E>
	class Node<E> : TerminalExpr
	{
	public:
		using expr_t = E;

	private:
		E const* _expr;

	public:
		typename E::value_t _gradient;

		constexpr Node() : _expr(nullptr), _gradient(0.0) {}

		constexpr void SetLocalGrads(E const* const expr)
		{
			_expr = expr;
		}

		constexpr void SetVariableCache() const 
		{
			if constexpr (std::is_base_of_v<TrainableExpr, expr_t>)
			{
				_expr->_cache += _gradient;
			}
		}

		constexpr void ResetGradient()
		{
			_gradient = static_cast<typename E::value_t>(0.0);
		}
	};

	template <typename E, int I>
	class Node<E, I> : UnaryExpr
	{
	public:
		using expr_t = E;
		constexpr static int child_one_v = I;
		
	public:
		E const* _expr;
		typename E::value_t _gradient;
		typename E::first_local_grad_t _first_local_grad;
		
		constexpr Node() : _expr(nullptr), _gradient(0.0), _first_local_grad(0.0) {}

		constexpr void SetLocalGrads(E const* const expr, typename E::first_local_grad_t first_local_grad)
		{
			_expr = expr;
			_first_local_grad = first_local_grad;
		}

		template <typename T>
		constexpr void SetGrads(T& tuple) const
		{
			std::get<I>(tuple)._gradient += _gradient * _first_local_grad;
		}

		constexpr void ResetGradient()
		{
			_gradient = static_cast<typename E::value_t>(0.0);
		}
	};

	template <typename E, int I1, int I2>
	class Node<E, I1, I2> : BinaryExpr
	{
	public:
		using expr_t = E;
		constexpr static int child_one_v = I1;
		constexpr static int child_two_v = I2;

	public:
		E const* _expr;
		typename E::value_t _gradient;
		typename E::first_local_grad_t _first_local_grad;
		typename E::second_local_grad_t _second_local_grad;

		constexpr Node() : _expr(nullptr), _gradient(0.0), _first_local_grad(0.0), _second_local_grad(0.0) {}

		constexpr void SetLocalGrads(E const* const expr, typename E::first_local_grad_t first_local_grad, typename E::second_local_grad_t second_local_grad)
		{
			_expr = expr;
			_first_local_grad = first_local_grad;
			_second_local_grad = second_local_grad;
		}

		template <typename T>
		constexpr void SetGrads(T& tuple) const
		{
			std::get<I1>(tuple)._gradient += _gradient * _first_local_grad;
			std::get<I2>(tuple)._gradient += _gradient * _second_local_grad;
		}

		constexpr void ResetGradient()
		{
			_gradient = static_cast<typename E::value_t>(0.0);
		}
	};

	template <typename E, typename = void>
	struct dfs_tuple;

	template <typename E>
	struct dfs_tuple<E, std::enable_if_t<std::is_base_of_v<TerminalExpr, E>>> {

		using type = std::tuple<E>;
	};

	template <typename E>
	using dfs_tuple_t = typename dfs_tuple<E>::type;

	template <typename E>
	struct dfs_tuple<E, std::enable_if_t<std::is_base_of_v<UnaryExpr, E>>> {

		using type = decltype(std::tuple_cat(
			std::declval<dfs_tuple_t<typename E::first_expr_t>>(),
			std::declval<std::tuple<E>>()
		));
	};

	template <typename E>
	struct dfs_tuple<E, std::enable_if_t<std::is_base_of_v<BinaryExpr, E>>> {

		using type = decltype(std::tuple_cat(
			std::declval<dfs_tuple_t<typename E::first_expr_t>>(),
			std::declval<dfs_tuple_t<typename E::second_expr_t>>(),
			std::declval<std::tuple<E>>()
		));
	};

	template <typename E>
	constexpr int dfs_tuple_size_v = std::tuple_size_v<dfs_tuple_t<E>>;

	template <typename E, int I, typename = void>
	struct _impl_dfs_final_tuple;

	template <typename E, int I>
	struct _impl_dfs_final_tuple<E, I, std::enable_if_t<std::is_base_of_v<TerminalExpr, E>>> {

		using type = std::tuple<Node<E>>;
	};

	template <typename E, int I>
	using _impl_dfs_final_tuple_t = typename _impl_dfs_final_tuple<E, I>::type;

	template <typename E, int I>
	struct _impl_dfs_final_tuple<E, I, std::enable_if_t<std::is_base_of_v<UnaryExpr, E>>> {

		using type = decltype(std::tuple_cat(
			std::declval<_impl_dfs_final_tuple_t<typename E::first_expr_t, I - 1>>(),
			std::declval<std::tuple<Node<E, I - 2>>>()
		));
	};

	template <typename E, int I>
	struct _impl_dfs_final_tuple<E, I, std::enable_if_t<std::is_base_of_v<BinaryExpr, E>>> {

		using type = decltype(std::tuple_cat(
			std::declval<_impl_dfs_final_tuple_t<typename E::first_expr_t, I - 1 - dfs_tuple_size_v<typename E::second_expr_t>>>(),
			std::declval<_impl_dfs_final_tuple_t<typename E::second_expr_t, I - 1>>(),
			std::declval<std::tuple<Node<E, I - 2 - dfs_tuple_size_v<typename E::second_expr_t>, I - 2>>>()
		));
	};

	template <typename E>
	using dfs_final_tuple_t = _impl_dfs_final_tuple_t<E, dfs_tuple_size_v<E>>;


	template <typename E>
	class GradientDescentOptimizer
	{
	private:
		static_assert(is_expr_v<E>);
		using tuple_t = typename dfs_final_tuple_t<E>;
		using result_t = typename std::decay_t<decltype(std::declval<E>()())>;

		tuple_t _tup;
		E& _expr;
		result_t _result;

		template <typename... Args>
		constexpr void _Helper_FeedPlaceholders(Args&& ... args)
		{
			((args.first->FeedValue(args.second)), ...);
		}

		template <int I>
		constexpr void _Helper_Backprop()
		{
			if constexpr (I == std::tuple_size_v<tuple_t> -1)
			{
				std::get<I>(_tup)._gradient = (typename std::tuple_element_t<I, tuple_t>::expr_t::value_t)(1.0);
			}
			if constexpr (std::is_base_of_v<TerminalExpr, std::tuple_element_t<I, tuple_t>>)
			{
				std::get<I>(_tup).SetVariableCache();
			}
			else
			{
				std::get<I>(_tup).SetGrads(_tup);
			}
			std::get<I>(_tup).ResetGradient();
			if constexpr (I > 0)
			{
				_Helper_Backprop<I - 1>();
			}
		}

		template <typename... Vars>
		constexpr void _Helper_ApplyGradients(double learning_rate, Vars& ... vars)
		{
			((vars.SetValueAndResetCache(learning_rate)), ...);
		}

	public:
		
		constexpr GradientDescentOptimizer(E& expr) : _expr(expr) {}
		template <typename... Args>
		constexpr GradientDescentOptimizer& FeedPlaceholders(Args&& ... args)
		{
			_Helper_FeedPlaceholders(std::forward<Args...>(args...));
			return *this;
		}

		constexpr GradientDescentOptimizer& Eval()
		{
			_result = _expr.template Eval<dfs_tuple_size_v<E> - 1>(_tup);
			return *this;
		}

		template <typename... Vars>
		constexpr GradientDescentOptimizer& Backpass(double learning_rate, Vars& ... vars)
		{
			_Helper_Backprop<dfs_tuple_size_v<E> - 1>();
			_Helper_ApplyGradients(learning_rate, vars...);
			return *this;
		}

		constexpr result_t GetPreResult()
		{
			return _result;
		}

		constexpr result_t GetPostResult()
		{
			return _expr();
		}
	};
}