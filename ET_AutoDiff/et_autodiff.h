#pragma once

#include <type_traits>
#include <tuple>
#include <cmath>
#include "tensor.h"

namespace Et {

	using ScalarD = Num::Scalar<double>;
	using ScalarL = Num::Scalar<long double>;

	struct BaseExpr {};

	struct _impl_TerminalExpr {};
	struct _impl_BinaryExpr {};
	struct _impl_UnaryExpr {};
	struct _impl_TrainableExpr {};

	template<typename... T>
	constexpr bool is_expr_v = std::conjunction_v<std::is_base_of<BaseExpr, std::decay_t<T>>...>;

	template <typename V>
	class ConstantExpr : private BaseExpr, private _impl_TerminalExpr
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
		constexpr auto Eval(T& tuple) -> auto&
		{
			std::get<I>(tuple).SetLocalGrads(this);
			return _value;
		}
	};

	ConstantExpr(int const&)->ConstantExpr<ScalarD>;
	ConstantExpr(double const&)->ConstantExpr<ScalarD>;
	ConstantExpr(float const&)->ConstantExpr<ScalarD>;
	ConstantExpr(long const&)->ConstantExpr<ScalarL>;
	ConstantExpr(long long const&)->ConstantExpr<ScalarL>;
	ConstantExpr(long double const&)->ConstantExpr<ScalarL>;

	template <typename V>
	class PlaceholderExpr : private BaseExpr, private _impl_TerminalExpr
	{
	public:
		static_assert(Num::is_tensor_v<V>);
		using value_t = std::decay_t<V>;

	private:
		bool _is_default;
		V _value;

	public:
		constexpr PlaceholderExpr() : _is_default{ true }, _value{ Num::zero_v<value_t> } {}

		constexpr auto operator()() const -> auto&
		{
			return _value;
		}

		template <int I, typename T>
		constexpr auto Eval(T& tuple) -> auto&
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
	class VariableExpr : private BaseExpr, private _impl_TerminalExpr, private _impl_TrainableExpr
	{
	public:
		static_assert(Num::is_tensor_v<V>);
		using value_t = std::decay_t<V>;

	private:
		V _value;

	public:
		constexpr VariableExpr(V const& value) : _value{ value } {}

		constexpr auto operator()() const -> auto&
		{
			return _value;
		}

		template <int I, typename T>
		constexpr auto Eval(T& tuple) -> auto&
		{
			std::get<I>(tuple).SetLocalGrads(this);
			return _value;
		}

		constexpr auto AddDelta(value_t delta)
		{
			_value += delta;
		}
	};

	VariableExpr(int const&)->VariableExpr<ScalarD>;
	VariableExpr(double const&)->VariableExpr<ScalarD>;
	VariableExpr(float const&)->VariableExpr<ScalarD>;
	VariableExpr(long const&)->VariableExpr<ScalarL>;
	VariableExpr(long long const&)->VariableExpr<ScalarL>;
	VariableExpr(long double const&)->VariableExpr<ScalarL>;

	template <typename E1, typename E2>
	class AddExpr : private BaseExpr, private _impl_BinaryExpr
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
		constexpr auto Eval(T& tuple) -> auto
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
	class MultiplyExpr : private BaseExpr, private _impl_BinaryExpr
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
		constexpr auto Eval(T& tuple) -> auto
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
	class SubtractExpr : private BaseExpr, private _impl_BinaryExpr
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
		constexpr auto Eval(T& tuple) -> auto
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
	class DivideExpr : private BaseExpr, private _impl_BinaryExpr
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
		constexpr auto Eval(T& tuple) -> auto
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
	class PowerExpr : private BaseExpr, private _impl_BinaryExpr
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
		constexpr auto Eval(T& tuple) -> auto
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
	class NegateExpr : private BaseExpr, private _impl_UnaryExpr
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
		constexpr auto Eval(T& tuple) -> auto
		{
			first_value_t first_value = _first_expr.template Eval<std::tuple_element_t<I, T>::child_one_v>(tuple);
			std::get<I>(tuple).SetLocalGrads(this, first_local_grad_t(-1.0));
			return -first_value;
		}
	};

	template<typename E1>
	NegateExpr(E1&&)->NegateExpr<E1>;

	template <typename E1>
	class LogExpr : private BaseExpr, private _impl_UnaryExpr
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
		constexpr auto Eval(T& tuple) -> auto
		{
			first_value_t first_value = _first_expr.template Eval<std::tuple_element_t<I, T>::child_one_v>(tuple);
			std::get<I>(tuple).SetLocalGrads(this, first_value.Inverse());
			return Num::log(first_value);
		}
	};

	template<typename E1>
	LogExpr(E1&&)->LogExpr<E1>;

	template <typename E1>
	class SinExpr : private BaseExpr, private _impl_UnaryExpr
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
		constexpr auto Eval(T& tuple) -> auto
		{
			first_value_t first_value = _first_expr.template Eval<std::tuple_element_t<I, T>::child_one_v>(tuple);
			std::get<I>(tuple).SetLocalGrads(this, Num::cos(first_value));
			return Num::sin(first_value);
		}
	};

	template<typename E1>
	SinExpr(E1&&)->SinExpr<E1>;

	template <typename E1>
	class CosExpr : private BaseExpr, private _impl_UnaryExpr
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
		constexpr auto Eval(T& tuple) -> auto
		{
			first_value_t first_value = _first_expr.template Eval<std::tuple_element_t<I, T>::child_one_v>(tuple);
			std::get<I>(tuple).SetLocalGrads(this, -Num::sin(first_value));
			return Num::cos(first_value);
		}
	};

	template<typename E1>
	CosExpr(E1&&)->CosExpr<E1>;

	template <typename E1>
	class TanExpr : private BaseExpr, private _impl_UnaryExpr
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
		constexpr auto Eval(T& tuple) -> auto
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

	struct _impl_BinaryNode {};
	struct _impl_UnaryNode {};
	struct _impl_UntrainableNode {};
	struct _impl_TrainableNode {};

	template <typename E, int I1, int I2>
	class BinaryNode : private _impl_BinaryNode
	{
	public:
		using expr_t = std::decay<E>;
		constexpr static int child_one_v = I1;
		constexpr static int child_two_v = I2;

	private:
		E* _expr;
		typename E::value_t _gradient;
		typename E::first_local_grad_t _first_local_grad;
		typename E::second_local_grad_t _second_local_grad;

	public:
		constexpr BinaryNode() : _expr{ nullptr }, _gradient{ Num::zero_v<typename E::value_t> },
			_first_local_grad{ Num::zero_v<typename E::first_local_grad_t> }, _second_local_grad{ Num::zero_v<typename E::second_local_grad_t> } {}

		constexpr auto AddMyGrad(typename E::value_t const& addition) -> void
		{
			_gradient += addition;
		}

		constexpr auto SetLocalGrads(E* const expr, typename E::first_local_grad_t first_local_grad, typename E::second_local_grad_t second_local_grad) -> void
		{
			_expr = expr;
			_first_local_grad = first_local_grad;
			_second_local_grad = second_local_grad;
		}

		template <typename T>
		constexpr auto SetChildGrads(T& tuple) const -> void
		{
			std::get<I1>(tuple).AddMyGrad(_gradient * _first_local_grad);
			std::get<I2>(tuple).AddMyGrad(_gradient * _second_local_grad);
		}

		constexpr auto ResetGrad() -> void
		{
			_gradient = Num::zero_v<typename E::value_t>;
		}
	};

	template <typename E, int I1>
	class UnaryNode : private _impl_UnaryNode
	{
	public:
		using expr_t = std::decay<E>;
		constexpr static int child_one_v = I1;

	private:
		E* _expr;
		typename E::value_t _gradient;
		typename E::first_local_grad_t _first_local_grad;

	public:
		constexpr UnaryNode() : _expr{ nullptr }, _gradient{ Num::zero_v<typename E::value_t> }, 
			_first_local_grad{ Num::zero_v<typename E::first_local_grad_t> } {}

		constexpr auto AddMyGrad(typename E::value_t const& addition) -> void
		{
			_gradient += addition;
		}

		constexpr auto SetLocalGrads(E* const expr, typename E::first_local_grad_t first_local_grad) -> void
		{
			_expr = expr;
			_first_local_grad = first_local_grad;
		}

		template <typename T>
		constexpr auto SetChildGrads(T& tuple) const -> void
		{
			std::get<I1>(tuple).AddMyGrad(_gradient * _first_local_grad);
		}

		constexpr auto ResetGrad() -> void
		{
			_gradient = Num::zero_v<typename E::value_t>;
		}
	};

	template <typename E, typename = void>
	class TerminalNode : private _impl_UntrainableNode
	{
	public:
		using expr_t = std::decay<E>;

	private:
		E* _expr;
		typename E::value_t _gradient;

	public:
		constexpr TerminalNode() : _expr{ nullptr }, _gradient{ Num::zero_v<typename E::value_t> } {}

		constexpr auto SetLocalGrads(E* const expr) -> void
		{
			_expr = expr;
		}

		constexpr auto AddMyGrad(typename E::value_t const& addition) -> void
		{
			_gradient += addition;
		}

		constexpr auto ResetGrad() -> void
		{
			_gradient = Num::zero_v<typename E::value_t>;
		}
	};

	template <typename E>
	class TerminalNode<E, std::enable_if_t<std::is_base_of_v<_impl_TrainableExpr, E>>> : private _impl_TrainableNode
	{
	public:
		using expr_t = std::decay<E>;

	private:
		E* _expr;
		typename E::value_t _gradient;

	public:
		constexpr TerminalNode() : _expr{ nullptr }, _gradient{ Num::zero_v<typename E::value_t> } {}

		constexpr auto SetLocalGrads(E* const expr) -> void
		{
			_expr = expr;
		}

		constexpr auto AddMyGrad(typename E::value_t const& addition)
		{
			_gradient += addition;
		}

		constexpr auto UpdateVariable(double learning_rate) const -> void
		{
			_expr->AddDelta(learning_rate * _gradient);
		}
		
		constexpr auto ResetGrad() -> void
		{
			_gradient = Num::zero_v<typename E::value_t>;
		}
	};

	template <typename E, typename = void>
	struct dfs_tuple;

	template <typename E>
	struct dfs_tuple<E, std::enable_if_t<std::is_base_of_v<_impl_TerminalExpr, E>>> {

		using type = std::tuple<E>;
	};

	template <typename E>
	using dfs_tuple_t = typename dfs_tuple<E>::type;

	template <typename E>
	struct dfs_tuple<E, std::enable_if_t<std::is_base_of_v<_impl_UnaryExpr, E>>> {

		using type = decltype(std::tuple_cat(
			std::declval<dfs_tuple_t<typename E::first_expr_t>>(),
			std::declval<std::tuple<E>>()
		));
	};

	template <typename E>
	struct dfs_tuple<E, std::enable_if_t<std::is_base_of_v<_impl_BinaryExpr, E>>> {

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
	struct _impl_dfs_final_tuple<E, I, std::enable_if_t<std::is_base_of_v<_impl_TerminalExpr, E>>> {

		using type = std::tuple<TerminalNode<E>>;
	};

	template <typename E, int I>
	using _impl_dfs_final_tuple_t = typename _impl_dfs_final_tuple<E, I>::type;

	template <typename E, int I>
	struct _impl_dfs_final_tuple<E, I, std::enable_if_t<std::is_base_of_v<_impl_UnaryExpr, E>>> {

		using type = decltype(std::tuple_cat(
			std::declval<_impl_dfs_final_tuple_t<typename E::first_expr_t, I - 1>>(),
			std::declval<std::tuple<UnaryNode<E, I - 2>>>()
		));
	};

	template <typename E, int I>
	struct _impl_dfs_final_tuple<E, I, std::enable_if_t<std::is_base_of_v<_impl_BinaryExpr, E>>> {

		using type = decltype(std::tuple_cat(
			std::declval<_impl_dfs_final_tuple_t<typename E::first_expr_t, I - 1 - dfs_tuple_size_v<typename E::second_expr_t>>>(),
			std::declval<_impl_dfs_final_tuple_t<typename E::second_expr_t, I - 1>>(),
			std::declval<std::tuple<BinaryNode<E, I - 2 - dfs_tuple_size_v<typename E::second_expr_t>, I - 2>>>()
		));
	};

	template <typename E>
	using dfs_final_tuple_t = _impl_dfs_final_tuple_t<E, dfs_tuple_size_v<E>>;


	template <typename V>
	struct H
	{
		PlaceholderExpr<V>& _placeholder;
		V const& _value;

		H(PlaceholderExpr<V>& placeholder, V const& value) : _placeholder{ placeholder }, _value{ value } {}
	};

	template <typename V, typename S>
	H(PlaceholderExpr<V>&, S const&)->H<V>;

	template <typename E>
	class GradientDescentOptimizer
	{
	private:
		static_assert(is_expr_v<E>);
		using tuple_t = typename dfs_final_tuple_t<E>;
		using result_t = typename E::value_t;

		tuple_t _tuple;
		E& _expr;
		result_t _result;

		template <typename... Vs>
		constexpr auto _impl_FeedPlaceholders(H<Vs>&& ... hs) -> void
		{
			((hs._placeholder.FeedValue(hs._value)), ...);
		}

		template <int I>
		constexpr auto _impl_BackwardPass(double learning_rate) -> void
		{
			using node_t = typename std::tuple_element_t<I, tuple_t>;

			if constexpr (std::is_base_of_v<_impl_TrainableNode, node_t>)
			{
				std::get<I>(_tuple).UpdateVariable(learning_rate);
			}
			else if constexpr (std::is_base_of_v<_impl_UnaryNode, node_t> || std::is_base_of_v<_impl_BinaryNode, node_t>)
			{
				std::get<I>(_tuple).SetChildGrads(_tuple);
			}

			std::get<I>(_tuple).ResetGrad();

			if constexpr (I > 0)
			{
				_impl_BackwardPass<I - 1>(learning_rate);
			}
		}

	public:
		constexpr GradientDescentOptimizer(E& expr) : _expr{ expr } {}

		template <typename... Vs>
		constexpr auto ForwardPass(H<Vs>&& ... hs) -> GradientDescentOptimizer &
		{
			_impl_FeedPlaceholders(std::forward<H<Vs>...>(hs...));
			_result = _expr.template Eval<dfs_tuple_size_v<E> -1>(_tuple);
			return *this;
		}

		constexpr auto Minimize(double learning_rate) -> GradientDescentOptimizer &
		{
			std::get<dfs_tuple_size_v<E>-1>(_tuple).AddMyGrad(Num::identity_v<typename E::value_t>);
			_impl_BackwardPass<dfs_tuple_size_v<E> - 1>(-learning_rate);
			return *this;
		}

		constexpr auto Maximize(double learning_rate) -> GradientDescentOptimizer &
		{
			std::get<dfs_tuple_size_v<E>-1>(_tuple).AddMyGrad(Num::identity_v<typename E::value_t>);
			_impl_BackwardPass<dfs_tuple_size_v<E> - 1>(learning_rate);
			return *this;
		}

		constexpr auto GetPreResult() -> result_t
		{
			return _result;
		}

		constexpr auto GetPostResult() -> result_t
		{
			return _expr();
		}
	};
}

#pragma once

#include <type_traits>
#include <optional>
#include "tensor.h"

namespace Et_test
{
	// Aliases for basic scalar types.
	using ScalarD = Num::Scalar<double>;
	using ScalarL = Num::Scalar<long double>;

	// Base class for all expressions.
	struct BaseExpr {};

	// Identifier for special kinds of expressions.
	struct TerminalExpr {};
	struct OperationExpr {};
	struct TrainableExpr {};
	struct BinaryExpr {};
	struct UnaryExpr {};

	// Returns true if all types are of the specific kind.
	template<typename... T>
	constexpr bool is_expr_v = std::conjunction_v<std::is_base_of<BaseExpr, std::decay_t<T>>...>;

	template <typename... T>
	constexpr bool is_terminal_expr_v = std::conjunction_v<std::is_base_of<TerminalExpr, std::decay_t<T>>...>;

	template <typename... T>
	constexpr bool is_operation_expr_v = std::conjunction_v<std::is_base_of<OperationExpr, std::decay_t<T>>...>;

	template <typename... T>
	constexpr bool is_trainable_operation_expr_v = std::conjunction_v<std::is_base_of<TrainableExpr, std::decay_t<T>>...>;

	template <typename... T>
	constexpr bool is_binary_operation_expr_v = std::conjunction_v<std::is_base_of<BinaryExpr, std::decay_t<T>>...>;

	template <typename... T>
	constexpr bool is_unary_operation_expr_v = std::conjunction_v<std::is_base_of<UnaryExpr, std::decay_t<T>>...>;

	// Terminal expression containing a constant value that can't be learnt.
	template <typename V, typename = std::enable_if_t<Num::is_tensor_v<V>>>
	class ConstantExpr : private BaseExpr, private TerminalExpr
	{
	public:
		using value_t = std::decay_t<V>;
	private:
		value_t const _value;
	public:
		constexpr explicit ConstantExpr(V& value) : _value{ value } {}
		constexpr explicit ConstantExpr(V&& value) : _value{ std::move(value) } {}

		auto operator()() const
		{
			return _value;
		}

		template <size_t I, typename Tup>
		auto ForwardPass(Tup& container)
		{
			std::get<I>(container) = { this };
			return _value;
		}
	};

	// Deduction guides for ConstantExpr.
	template <typename V, typename = std::enable_if_t<std::is_arithmetic_v<V>>>
	ConstantExpr(V&&)->ConstantExpr<ScalarD>;

	ConstantExpr(long double&)->ConstantExpr<ScalarL>;

	ConstantExpr(long double&&)->ConstantExpr<ScalarL>;

	// Terminal expression containing a placeholder value that can't be learnt.
	template <typename V, typename = std::enable_if_t<Num::is_tensor_v<V>>>
	class PlaceholderExpr : private BaseExpr, private TerminalExpr
	{
	public:
		using value_t = std::decay_t<V>;
	private:
		std::optional<value_t> _value = std::nullopt;
	public:
		explicit PlaceholderExpr() {}
		explicit PlaceholderExpr(V& value) : _value{ value } {}
		explicit PlaceholderExpr(V&& value) : _value{ std::move(value) } {}

		auto operator()() const
		{
			return _value.value();
		}

		template <size_t I, typename Tup>
		auto ForwardPass(Tup& container)
		{
			std::get<I>(container) = { this };
			return _value.value();
		}

		void FeedValue(value_t const& value)
		{
			_value = value;
		}
	};

	// Deduction guides for PlaceholderExpr.
	template <typename V, typename = std::enable_if_t<std::is_arithmetic_v<V>>>
	PlaceholderExpr(V&&)->PlaceholderExpr<ScalarD>;

	PlaceholderExpr(long double&)->PlaceholderExpr<ScalarL>;

	PlaceholderExpr(long double&&)->PlaceholderExpr<ScalarL>;

	PlaceholderExpr()->PlaceholderExpr<ScalarD>;

	// Terminal expression containing a variable value that can be learnt.
	template <typename V, typename = std::enable_if_t<Num::is_tensor_v<V>>>
	class VariableExpr : private BaseExpr, private TerminalExpr, private TrainableExpr
	{
	public:
		using value_t = std::decay_t<V>;
	private:
		V _value;
	public:
		explicit VariableExpr(V& value) : _value{ value } {}
		explicit VariableExpr(V&& value) : _value{ std::move(value) } {}

		auto operator()() const
		{
			return _value;
		}

		template <size_t I, typename Tup>
		auto ForwardPass(Tup& container)
		{
			std::get<I>(container) = { this };
			return _value;
		}

		void AddGradient(value_t const& addition)
		{
			_value += addition;
		}
	};

	// Deduction guides for VariableExpr.
	template <typename V, typename = std::enable_if_t<std::is_arithmetic_v<V>>>
	VariableExpr(V&&)->VariableExpr<ScalarD>;

	VariableExpr(long double&)->VariableExpr<ScalarL>;

	VariableExpr(long double&&)->VariableExpr<ScalarL>;

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

	// Container for pointer to an expression and it's child indices.
	template <typename E, typename VL>
	class ExprHolder;

	template <typename E, size_t... Vs>
	class ExprHolder<E, value_list<size_t, Vs...>>
	{
	public:
		using expr_t = E;
		using child_list_t = value_list<size_t, Vs...>;
	private:
		E* _expr = nullptr;
	public:
		explicit ExprHolder() {}
		explicit ExprHolder(E* expr) : _expr{ expr } {}

		E& GetExpr()
		{
			return *_expr;
		}
	};

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

	// Expression representing addition between two expressions.
	template <typename E1, typename E2, typename = std::enable_if_t<is_expr_v<E1, E2>>>
	class AddExpr : private BaseExpr, private OperationExpr, private TrainableExpr, private BinaryExpr
	{
	public:
		using first_expr_t = std::decay_t<E1>;
		using second_expr_t = std::decay_t<E2>;
		using first_value_t = std::decay_t<decltype(std::declval<E1>()())>;
		using second_value_t = std::decay_t<decltype(std::declval<E2>()())>;
		using value_t = std::decay_t<decltype(std::declval<E1>()() + std::declval<E2>()())>;
	private:
		E1 _first_expr;
		E2 _second_expr;

		value_t _global_gradient;
	public:
		AddExpr(E1&& first_expr, E2&& second_expr) :
			_first_expr{ std::forward<E1>(first_expr) }, _second_expr{ std::forward<E2>(second_expr) } {}

		auto operator()() const
		{
			return _first_expr() + _second_expr();
		}

		template <size_t I, typename Tup>
		auto ForwardPass(Tup& container)
		{
			std::get<I>(container) = { this };
			return _first_expr.template ForwardPass<child_index_utility_v<Tup, I, 0>>(container)
				+ _second_expr.template ForwardPass<child_index_utility_v<Tup, I, 1>>(container);
		}

		void BackwardPass()
		{
			if constexpr (is_trainable_operation_expr_v<first_expr_t>)
				_first_expr.AddGradient(_global_gradient);

			if constexpr (is_trainable_operation_expr_v<second_expr_t>)
				_second_expr.AddGradient(_global_gradient);
		}

		void AddGradient(value_t const& addition)
		{
			_global_gradient += addition;
		}

		void InitializeGradient()
		{
			_global_gradient = 0.0;
		}

		void ResetGradient()
		{
			_global_gradient = 0.0;
		}

		void TerminateGradient()
		{
			_global_gradient = 0.0;
		}
	};

	// Deduction guide for AddExpr.
	template <typename E1, typename E2>
	AddExpr(E1&&, E2&&)->AddExpr<E1, E2>;

	// Operator overload for AddExpr.
	template <typename E1, typename E2, typename = std::enable_if_t<is_expr_v<E1, E2>>>
	auto operator+(E1&& first_expr, E2&& second_expr) -> AddExpr<E1, E2>
	{
		return { std::forward<E1>(first_expr), std::forward<E2>(second_expr) };
	}

	// Utility metafunction that computes a tuple containing all expression types in DFS order.
	template <typename E, typename = void>
	struct dfs_tuple;

	template <typename E>
	struct dfs_tuple<E, std::enable_if_t<is_terminal_expr_v<E>>>
	{
		using type = std::tuple<E>;
	};

	template <typename E>
	struct dfs_tuple<E, std::enable_if_t<is_unary_operation_expr_v<E>>>
	{
		using type = decltype(std::tuple_cat(
			std::declval<typename dfs_tuple<typename E::first_expr_t>::type>(),
			std::declval<std::tuple<E>>()
		));
	};

	template <typename E>
	struct dfs_tuple<E, std::enable_if_t<is_binary_operation_expr_v<E>>>
	{
		using type = decltype(std::tuple_cat(
			std::declval<typename dfs_tuple<typename E::first_expr_t>::type>(),
			std::declval<typename dfs_tuple<typename E::second_expr_t>::type>(),
			std::declval<std::tuple<E>>()
		));
	};

	// Returns a tuple containing expression types in DFS order.
	template <typename E>
	using dfs_tuple_t = typename dfs_tuple<E>::type;

	// Returns the number of non-unique expressions visited in DFS order.
	template <typename E>
	constexpr size_t dfs_tuple_size_v = std::tuple_size_v<dfs_tuple_t<E>>;

	// Implementation of dfs_exprholder_tuple.
	template <typename E, size_t I, typename = void>
	struct _impl_dfs_exprholder_tuple;

	template <typename E, size_t I>
	struct _impl_dfs_exprholder_tuple<E, I, std::enable_if_t<is_terminal_expr_v<E>>>
	{
		using type = std::tuple<ExprHolder<E, value_list<size_t>>>;
	};

	template <typename E, size_t I>
	struct _impl_dfs_exprholder_tuple<E, I, std::enable_if_t<is_unary_operation_expr_v<E>>>
	{
		using type = decltype(std::tuple_cat(
			std::declval<typename _impl_dfs_exprholder_tuple<typename E::first_expr_t, I - 1>::type>(),
			std::declval<std::tuple<ExprHolder<E, value_list<size_t, I - 2>>>>()
		));
	};

	template <typename E, size_t I>
	struct _impl_dfs_exprholder_tuple<E, I, std::enable_if_t<is_binary_operation_expr_v<E>>>
	{
		using type = decltype(std::tuple_cat(
			std::declval<typename _impl_dfs_exprholder_tuple<typename E::first_expr_t, I - 1 - dfs_tuple_size_v<typename E::second_expr_t>>::type>(),
			std::declval<typename _impl_dfs_exprholder_tuple<typename E::second_expr_t, I - 1>::type>(),
			std::declval<std::tuple<ExprHolder<E, value_list<size_t, I - 2 - dfs_tuple_size_v<typename E::second_expr_t>, I - 2>>>>()
		));
	};

	// Returns tuple of ExprHolders containing expression types in DFS order and indices of child expressions.
	template <typename E>
	using dfs_exprholder_tuple_t = typename _impl_dfs_exprholder_tuple<E, dfs_tuple_size_v<E>>::type;

	// Container for pair of placeholder and value to feed.
	template <typename V>
	struct H
	{
		PlaceholderExpr<std::decay_t<V>>& _pl;
		V&& _value;

		H(PlaceholderExpr<std::decay_t<V>>& pl, V&& value)
			: _pl{ pl }, _value{ std::forward<V>(value) } {}
	};

	// Deduction guide for H
	template <typename V, typename S>
	H(PlaceholderExpr<V>&, S&&)->H<V>;

	// Optimizer that implements a typical backpropagation algorithm.
	template <typename E, typename = std::enable_if_t<is_operation_expr_v<E>>>
	class GradientDescentOptimizer
	{
	private:
		using tuple_t = dfs_exprholder_tuple_t<E>;
		constexpr static size_t tuple_size_v = dfs_tuple_size_v<E>;
		using result_t = typename E::value_t;

		E& _expr;
		tuple_t _tuple;
		result_t _result;
	public:
		GradientDescentOptimizer(E& expr) : _expr{ expr } {}

		template <typename... Vs>
		GradientDescentOptimizer& ForwardPass()
		{
			_result = _expr.template ForwardPass<dfs_tuple_size_v<E> -1>(_tuple);
			return *this;
		}

		template <typename... Vs>
		GradientDescentOptimizer& FeedPlaceholders(H<Vs>&& ... hs)
		{
			_FeedPlaceholders(std::forward<H<Vs>...>(hs...));
			return *this;
		}

		GradientDescentOptimizer& Minimize(long double learning_rate)
		{
			_InitializeGradients<0>();
			std::get<tuple_size_v - 1>(_tuple).GetExpr().AddGradient(-learning_rate);
			_BackwardPass<tuple_size_v - 1>();
			return *this;
		}

		GradientDescentOptimizer & Maximize(long double learning_rate)
		{
			_InitializeGradients<0>();
			std::get<tuple_size_v - 1>(_tuple).GetExpr().AddGradient(1.0);
			_BackwardPass<tuple_size_v - 1>();
			return *this;
		}

		void Terminate()
		{
			_TerminateGradients<0>();
		}

		result_t GetPreResult() const
		{
			return _result;
		}

		result_t GetPostResult() const
		{
			return _expr();
		}
	private:
		template <typename... Vs>
		void _FeedPlaceholders(H<Vs> && ... hs)
		{
			((hs._pl.FeedValue(hs._value)), ...);
		}

		template <size_t I>
		void _BackwardPass()
		{
			using expr_t = typename std::tuple_element_t<I, tuple_t>::expr_t;

			if constexpr (is_operation_expr_v<expr_t>)
			{
				std::get<I>(_tuple).GetExpr().BackwardPass();
				std::get<I>(_tuple).GetExpr().ResetGradient();
			}

			if constexpr (I > 0)
			{
				_BackwardPass<I - 1>();
			}
		}

		template <size_t I>
		void _InitializeGradients()
		{
			using expr_t = typename std::tuple_element_t<I, tuple_t>::expr_t;

			if constexpr (is_operation_expr_v<expr_t>)
			{
				std::get<I>(_tuple).GetExpr().InitializeGradient();
			}

			if constexpr (I < tuple_size_v - 1)
			{
				_InitializeGradients<I + 1>();
			}
		}

		template <size_t I>
		void _TerminateGradients()
		{
			using expr_t = typename std::tuple_element_t<I, tuple_t>::expr_t;

			if constexpr (is_operation_expr_v<expr_t>)
			{
				std::get<I>(_tuple).GetExpr().TerminateGradient();
			}

			if constexpr (I < tuple_size_v - 1)
			{
				_TerminateGradients<I + 1>();
			}
		}
	};
}