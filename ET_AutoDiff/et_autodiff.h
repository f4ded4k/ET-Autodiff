#pragma once

#include <type_traits>
#include <tuple>
#include <cmath>
#include "tensor.h"

namespace Et {

	using Double = Num::Scaler<double>;

	namespace details
	{
		class TerminalExpr {};
		class BinaryExpr {};
		class UnaryExpr {};
	}

	template <typename D>
	class Expr
	{
		using Derived_t = D;

	public:
		constexpr D const& GetSelf() const
		{
			return static_cast<D const&>(*this);
		}

		constexpr decltype(auto) operator()() const
		{
			return GetSelf()();
		}

		template <int I, typename T>
		constexpr decltype(auto) Eval(T&& tuple) const
		{
			return GetSelf().eval(tuple);
		}
	};

	template <typename V>
	class ConstantExpr : public Expr<ConstantExpr<V>>, private details::TerminalExpr
	{
		using Value_t = std::remove_reference_t<V>;

	private:
		V const _value;

	public:
		constexpr ConstantExpr(Num::Tensor<V> const& value) : _value(value.GetSelf()) {}

		constexpr Value_t const& operator()() const
		{
			return _value;
		}

		template <int I, typename T>
		constexpr Value_t const& Eval(T&& tuple) const
		{
			return _value;
		}
	};

	template <typename V>
	class PlaceholderExpr : public Expr<PlaceholderExpr<V>>, private details::TerminalExpr
	{
		using Value_t = std::remove_reference_t<V>;

	private:
		bool _is_default;
		V _value;

	public:
		constexpr PlaceholderExpr() : _value(0.0), _is_default(true) {}


		constexpr Value_t const& operator()() const
		{
			return _value;
		}

		template <int I, typename T>
		constexpr Value_t const& Eval(T&& tuple) const
		{
			return _value;
		}

		constexpr void FeedValue(V const& value)
		{
			_value = value;
		}
	};

	template <typename V>
	class VariableExpr : public Expr<VariableExpr<V>>, private details::TerminalExpr
	{
		using Value_t = std::remove_reference_t<V>;

	private:
		V _value;

	public:
		constexpr VariableExpr(Num::Tensor<V> const& value) : _value(value.GetSelf()) {}

		constexpr Value_t const& operator()() const
		{
			return _value;
		}

		template <int I, typename T>
		constexpr Value_t const& Eval(T&& tuple) const
		{
			return _value;
		}
	};

	template <typename E1, typename E2>
	class AddExpr : public Expr<AddExpr<E1, E2>>, private details::BinaryExpr
	{
		using FirstExpr_t = std::remove_reference_t<E1>;
		using SecondExpr_t = std::remove_reference_t<E2>;
		using FirstValue_t = std::remove_reference_t<decltype(std::declval<E1>()())>;
		using SecondValue_t = std::remove_reference_t<decltype(std::declval<E2>()())>;
		using FirstLocalGrad_t = FirstValue_t;
		using SecondLocalGrad_t = SecondValue_t;
		using Value_t = std::remove_reference_t<decltype(std::declval<E1>()() + std::declval<E2>()())>;

	private:
		E1 const& _first_expr;
		E2 const& _second_expr;

	public:
		constexpr AddExpr(Expr<E1> const& first_expr, Expr<E2> const& second_expr)
			: _first_expr(first_expr.GetSelf()), _second_expr(second_expr.GetSelf()) {}

		constexpr Value_t operator()() const
		{
			return _first_expr() + _second_expr();
		}

		template <int I, typename T>
		constexpr Value_t Eval(T&& tuple) const
		{
			// TODO insert recursion logic and grad calculation
			return 0.0;
		}
	};

	template <typename E1, typename E2>
	class MultiplyExpr : public Expr<MultiplyExpr<E1, E2>>, private details::BinaryExpr
	{
		using FirstExpr_t = std::remove_reference_t<E1>;
		using SecondExpr_t = std::remove_reference_t<E2>;
		using FirstValue_t = std::remove_reference_t<decltype(std::declval<E1>()())>;
		using SecondValue_t = std::remove_reference_t<decltype(std::declval<E2>()())>;
		using FirstLocalGrad_t = SecondValue_t;
		using SecondLocalGrad_t = FirstValue_t;
		using Value_t = std::remove_reference_t<decltype(std::declval<E1>()()* std::declval<E2>()())>;

	private:
		E1 const& _first_expr;
		E2 const& _second_expr;

	public:
		constexpr MultiplyExpr(Expr<E1> const& first_expr, Expr<E2> const& second_expr)
			: _first_expr(first_expr.GetSelf()), _second_expr(second_expr.GetSelf()) {}

		constexpr Value_t operator()() const
		{
			return _first_expr()* _second_expr();
		}

		template <int I, typename T>
		constexpr Value_t Eval(T&& tuple) const
		{
			// TODO insert recursion logic and grad calculation
			return 0.0;
		}
	};

	template <typename E1, typename E2>
	class SubtractExpr : public Expr<SubtractExpr<E1, E2>>, private details::BinaryExpr
	{
		using FirstExpr_t = std::remove_reference_t<E1>;
		using SecondExpr_t = std::remove_reference_t<E2>;
		using FirstValue_t = std::remove_reference_t<decltype(std::declval<E1>()())>;
		using SecondValue_t = std::remove_reference_t<decltype(std::declval<E2>()())>;
		using FirstLocalGrad_t = FirstValue_t;
		using SecondLocalGrad_t = SecondValue_t;
		using Value_t = std::remove_reference_t<decltype(std::declval<E1>()() - std::declval<E2>()())>;

	private:
		E1 const& _first_expr;
		E2 const& _second_expr;

	public:
		constexpr SubtractExpr(Expr<E1> const& first_expr, Expr<E2> const& second_expr)
			: _first_expr(first_expr.GetSelf()), _second_expr(second_expr.GetSelf()) {}

		constexpr Value_t operator()() const
		{
			return _first_expr() - _second_expr();
		}

		template <int I, typename T>
		constexpr Value_t Eval(T&& tuple) const
		{
			// TODO insert recursion logic and grad calculation
			return 0.0;
		}
	};

	template <typename E1, typename E2>
	class DivideExpr : public Expr<DivideExpr<E1, E2>>, private details::BinaryExpr
	{
		using FirstExpr_t = std::remove_reference_t<E1>;
		using SecondExpr_t = std::remove_reference_t<E2>;
		using FirstValue_t = std::remove_reference_t<decltype(std::declval<E1>()())>;
		using SecondValue_t = std::remove_reference_t<decltype(std::declval<E2>()())>;
		using FirstLocalGrad_t = FirstValue_t;
		using SecondLocalGrad_t = SecondValue_t;
		using Value_t = std::remove_reference_t<decltype(std::declval<E1>()() / std::declval<E2>()())>;

	private:
		E1 const& _first_expr;
		E2 const& _second_expr;

	public:
		constexpr DivideExpr(Expr<E1> const& first_expr, Expr<E2> const& second_expr)
			: _first_expr(first_expr.GetSelf()), _second_expr(second_expr.GetSelf()) {}

		constexpr Value_t operator()() const
		{
			return _first_expr() / _second_expr();
		}

		template <int I, typename T>
		constexpr Value_t Eval(T&& tuple) const
		{
			// TODO insert recursion logic and grad calculation
			return 0.0;
		}
	};

	template <typename E1, typename E2>
	class PowerExpr : public Expr<PowerExpr<E1, E2>>, private details::BinaryExpr
	{
		using FirstExpr_t = std::remove_reference_t<E1>;
		using SecondExpr_t = std::remove_reference_t<E2>;
		using FirstValue_t = std::remove_reference_t<decltype(std::declval<E1>()())>;
		using SecondValue_t = std::remove_reference_t<decltype(std::declval<E2>()())>;
		using FirstLocalGrad_t = FirstValue_t;
		using SecondLocalGrad_t = SecondValue_t;
		using Value_t = std::remove_reference_t<decltype(Num::pow(std::declval<E1>()(), std::declval<E2>()()))>;

	private:
		E1 const& _first_expr;
		E2 const& _second_expr;

	public:
		constexpr PowerExpr(Expr<E1> const& first_expr, Expr<E2> const& second_expr)
			: _first_expr(first_expr.GetSelf()), _second_expr(second_expr.GetSelf()) {}

		constexpr Value_t operator()() const
		{
			return Num::pow(_first_expr(), _second_expr());
		}

		template <int I, typename T>
		constexpr Value_t Eval(T&& tuple) const
		{
			// TODO insert recursion logic and grad calculation
			return 0.0;
		}
	};

	template <typename E1>
	class NegateExpr : public Expr<NegateExpr<E1>>, private details::UnaryExpr
	{
		using FirstExpr_t = std::remove_reference_t<E1>;
		using FirstValue_t = std::remove_reference_t<decltype(std::declval<E1>()())>;
		using FirstLocalGrad_t = FirstValue_t;
		using Value_t = std::remove_reference_t<decltype(-std::declval<E1>()())>;

	private:
		E1 const& _first_expr;

	public:
		constexpr NegateExpr(Expr<E1> const& first_expr)
			: _first_expr(first_expr.GetSelf()) {}

		constexpr Value_t operator()() const
		{
			return -_first_expr();
		}

		template <int I, typename T>
		constexpr Value_t Eval(T&& tuple) const
		{
			// TODO insert recursion logic and grad calculation
			return 0.0;
		}
	};

	template <typename E1>
	class LogExpr : public Expr<LogExpr<E1>>, private details::UnaryExpr
	{
		using FirstExpr_t = std::remove_reference_t<E1>;
		using FirstValue_t = std::remove_reference_t<decltype(std::declval<E1>()())>;
		using FirstLocalGrad_t = FirstValue_t;
		using Value_t = std::remove_reference_t<decltype(Num::log(std::declval<E1>()()))>;

	private:
		E1 const& _first_expr;

	public:
		constexpr LogExpr(Expr<E1> const& first_expr)
			: _first_expr(first_expr.GetSelf()) {}

		constexpr Value_t operator()() const
		{
			return Num::log(_first_expr());
		}

		template <int I, typename T>
		constexpr Value_t Eval(T&& tuple) const
		{
			// TODO insert recursion logic and grad calculation
			return 0.0;
		}
	};

	template <typename E1>
	class SinExpr : public Expr<SinExpr<E1>>, private details::UnaryExpr
	{
		using FirstExpr_t = std::remove_reference_t<E1>;
		using FirstValue_t = std::remove_reference_t<decltype(std::declval<E1>()())>;
		using FirstLocalGrad_t = FirstValue_t;
		using Value_t = std::remove_reference_t<decltype(Num::sin(std::declval<E1>()()))>;

	private:
		E1 const& _first_expr;

	public:
		constexpr SinExpr(Expr<E1> const& first_expr)
			: _first_expr(first_expr.GetSelf()) {}

		constexpr Value_t operator()() const
		{
			return Num::sin(_first_expr());
		}

		template <int I, typename T>
		constexpr Value_t Eval(T&& tuple) const
		{
			// TODO insert recursion logic and grad calculation
			return 0.0;
		}
	};

	template <typename E1>
	class CosExpr : public Expr<CosExpr<E1>>, private details::UnaryExpr
	{
		using FirstExpr_t = std::remove_reference_t<E1>;
		using FirstValue_t = std::remove_reference_t<decltype(std::declval<E1>()())>;
		using FirstLocalGrad_t = FirstValue_t;
		using Value_t = std::remove_reference_t<decltype(Num::cos(std::declval<E1>()()))>;

	private:
		E1 const& _first_expr;

	public:
		constexpr CosExpr(Expr<E1> const& first_expr)
			: _first_expr(first_expr.GetSelf()) {}

		constexpr Value_t operator()() const
		{
			return Num::cos(_first_expr());
		}

		template <int I, typename T>
		constexpr Value_t Eval(T&& tuple) const
		{
			// TODO insert recursion logic and grad calculation
			return 0.0;
		}
	};

	template <typename E1>
	class TanExpr : public Expr<TanExpr<E1>>, private details::UnaryExpr
	{
		using FirstExpr_t = std::remove_reference_t<E1>;
		using FirstValue_t = std::remove_reference_t<decltype(std::declval<E1>()())>;
		using FirstLocalGrad_t = FirstValue_t;
		using Value_t = std::remove_reference_t<decltype(Num::tan(std::declval<E1>()()))>;

	private:
		E1 const& _first_expr;

	public:
		constexpr TanExpr(Expr<E1> const& first_expr)
			: _first_expr(first_expr.GetSelf()) {}

		constexpr Value_t operator()() const
		{
			return Num::tan(_first_expr());
		}

		template <int I, typename T>
		constexpr Value_t Eval(T&& tuple) const
		{
			// TODO insert recursion logic and grad calculation
			return 0.0;
		}
	};

	template <typename E1, typename E2>
	constexpr AddExpr<E1, E2> operator+(Expr<E1> const& first_expr, Expr<E2> const& second_expr)
	{
		return AddExpr(first_expr, second_expr);
	}

	template <typename E1, typename E2>
	constexpr MultiplyExpr<E1, E2> operator*(Expr<E1> const& first_expr, Expr<E2> const& second_expr)
	{
		return MultiplyExpr(first_expr, second_expr);
	}

	template <typename E1, typename E2>
	constexpr SubtractExpr<E1, E2> operator-(Expr<E1> const& first_expr, Expr<E2> const& second_expr)
	{
		return SubtractExpr(first_expr, second_expr);
	}

	template <typename E1, typename E2>
	constexpr DivideExpr<E1, E2> operator/(Expr<E1> const& first_expr, Expr<E2> const& second_expr)
	{
		return DivideExpr(first_expr, second_expr);
	}

	template <typename E1, typename E2>
	constexpr PowerExpr<E1, E2> pow(Expr<E1> const& first_expr, Expr<E2> const& second_expr)
	{
		return PowerExpr(first_expr, second_expr);
	}

	template <typename E1>
	constexpr NegateExpr<E1> operator-(Expr<E1> const& first_expr)
	{
		return NegateExpr(first_expr);
	}

	template <typename E1>
	constexpr LogExpr<E1> log(Expr<E1> const& first_expr)
	{
		return LogExpr(first_expr);
	}

	template <typename E1>
	constexpr SinExpr<E1> sin(Expr<E1> const& first_expr)
	{
		return SinExpr(first_expr);
	}

	template <typename E1>
	constexpr CosExpr<E1> cos(Expr<E1> const& first_expr)
	{
		return CosExpr(first_expr);
	}

	template <typename E1>
	constexpr TanExpr<E1> tan(Expr<E1> const& first_expr)
	{
		return TanExpr(first_expr);
	}
}