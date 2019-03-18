#pragma once

#include <type_traits>
#include <tuple>
#include <initializer_list>
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
		constexpr auto Eval(T& tuple) -> auto&
		{
			std::get<I>(tuple).SetLocalGrads(this);
			return _value;
		}
	};

	ConstantExpr(int const&)->ConstantExpr<ScalarD>;
	ConstantExpr(double const&)->ConstantExpr<ScalarD>;
	ConstantExpr(float const&)->ConstantExpr<ScalarD>;

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
	class VariableExpr : public Expr, private TerminalExpr, private TrainableExpr
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
	class TerminalNode<E, std::enable_if_t<std::is_base_of_v<TrainableExpr, E>>> : private _impl_TrainableNode
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

		using type = std::tuple<TerminalNode<E>>;
	};

	template <typename E, int I>
	using _impl_dfs_final_tuple_t = typename _impl_dfs_final_tuple<E, I>::type;

	template <typename E, int I>
	struct _impl_dfs_final_tuple<E, I, std::enable_if_t<std::is_base_of_v<UnaryExpr, E>>> {

		using type = decltype(std::tuple_cat(
			std::declval<_impl_dfs_final_tuple_t<typename E::first_expr_t, I - 1>>(),
			std::declval<std::tuple<UnaryNode<E, I - 2>>>()
		));
	};

	template <typename E, int I>
	struct _impl_dfs_final_tuple<E, I, std::enable_if_t<std::is_base_of_v<BinaryExpr, E>>> {

		using type = decltype(std::tuple_cat(
			std::declval<_impl_dfs_final_tuple_t<typename E::first_expr_t, I - 1 - dfs_tuple_size_v<typename E::second_expr_t>>>(),
			std::declval<_impl_dfs_final_tuple_t<typename E::second_expr_t, I - 1>>(),
			std::declval<std::tuple<BinaryNode<E, I - 2 - dfs_tuple_size_v<typename E::second_expr_t>, I - 2>>>()
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
		using result_t = typename E::value_t;

		tuple_t _tuple;
		E& _expr;
		result_t _result;

		template <typename... Args>
		constexpr auto _Helper_FeedPlaceholders(Args&& ... args) -> void
		{
			((args.first->FeedValue(args.second)), ...);
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

		template <typename... Args>
		constexpr auto FeedPlaceholders(Args&& ... args) -> GradientDescentOptimizer &
		{
			_Helper_FeedPlaceholders(std::forward<Args...>(args...));
			return *this;
		}

		constexpr auto ForwardPass() -> GradientDescentOptimizer &
		{
			_result = _expr.template Eval<dfs_tuple_size_v<E> - 1>(_tuple);
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