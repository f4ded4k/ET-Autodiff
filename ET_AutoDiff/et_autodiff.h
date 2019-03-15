#pragma once

#include <type_traits>
#include <tuple>
#include <cmath>
#include "tensor.h"

namespace Et {

	using Double = Num::Scaler<double>;

	template <typename T1, typename T2>
	constexpr auto PlFeed(T1&& first, T2&& second)
	{
		return std::make_pair(&first, second);
	}

	namespace details
	{
		class TerminalExpr {};
		class BinaryExpr {};
		class UnaryExpr {};
		class TrainableExpr {};
	}

	template <typename D>
	class Expr
	{
	public:
		using Derived_t = std::decay<D>;

	public:
		constexpr D const& GetSelf() const
		{
			return static_cast<D const&>(*this);
		}

		constexpr D& GetSelf()
		{
			return static_cast<D&>(*this);
		}
		
		constexpr decltype(auto) operator()() const
		{
			return GetSelf()();
		}

		template <int I, typename T>
		constexpr decltype(auto) Eval(T& tuple) const
		{
			return GetSelf().Eval<I>(tuple);
		}
	};

	template <typename V>
	class ConstantExpr : public Expr<ConstantExpr<V>>, private details::TerminalExpr
	{
	public:
		using Value_t = std::decay_t<V>;

	private:
		V const _value;

	public:
		constexpr ConstantExpr(Num::Tensor<V> const& value) : _value(value.GetSelf()) {}

		constexpr Value_t const& operator()() const
		{
			return _value;
		}

		template <int I, typename T>
		constexpr Value_t const& Eval(T& tuple) const
		{
			std::get<I>(tuple).SetLocalGrads(this);
			return _value;
		}
	};

	template <typename V>
	class PlaceholderExpr : public Expr<PlaceholderExpr<V>>, private details::TerminalExpr
	{
	public:
		using Value_t = std::decay_t<V>;

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
		constexpr Value_t const& Eval(T& tuple) const
		{
			std::get<I>(tuple).SetLocalGrads(this);
			return _value;
		}

		constexpr void FeedValue(V const& value)
		{
			_value = value;
		}
	};

	template <typename V>
	class VariableExpr : public Expr<VariableExpr<V>>, private details::TerminalExpr, private details::TrainableExpr
	{
	public:
		using Value_t = std::decay_t<V>;

	private:
		V _value;

	public:
		mutable V _cache;

		constexpr VariableExpr(Num::Tensor<V> const& value) : _value(value.GetSelf()) {}

		constexpr Value_t const& operator()() const
		{
			return _value;
		}

		template <int I, typename T>
		constexpr Value_t const& Eval(T& tuple) const
		{
			std::get<I>(tuple).SetLocalGrads(this);
			return _value;
		}

		constexpr void SetCache(V const& delta) const
		{
			_cache = delta;
		}

		constexpr void SetValueAndResetCache(double learning_rate)
		{
			_value -= learning_rate * _cache;
			_cache = 0.0;
		}
	};

	template <typename E1, typename E2>
	class AddExpr : public Expr<AddExpr<E1, E2>>, private details::BinaryExpr
	{
	public:
		using FirstExpr_t = std::decay_t<E1>;
		using SecondExpr_t = std::decay_t<E2>;
		using FirstValue_t = std::decay_t<decltype(std::declval<E1>()())>;
		using SecondValue_t = std::decay_t<decltype(std::declval<E2>()())>;
		using FirstLocalGrad_t = FirstValue_t;
		using SecondLocalGrad_t = SecondValue_t;
		using Value_t = std::decay_t<decltype(std::declval<E1>()() + std::declval<E2>()())>;

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
		constexpr Value_t Eval(T& tuple) const
		{
			FirstValue_t first_value = _first_expr.Eval<std::tuple_element_t<I, T>::Child1_v>(tuple);
			SecondValue_t second_value = _second_expr.Eval<std::tuple_element_t<I, T>::Child2_v>(tuple);
			std::get<I>(tuple).SetLocalGrads(this, Double(1.0), Double(1.0));
			return first_value + second_value;
		}
	};

	template <typename E1, typename E2>
	class MultiplyExpr : public Expr<MultiplyExpr<E1, E2>>, private details::BinaryExpr
	{
	public:
		using FirstExpr_t = std::decay_t<E1>;
		using SecondExpr_t = std::decay_t<E2>;
		using FirstValue_t = std::decay_t<decltype(std::declval<E1>()())>;
		using SecondValue_t = std::decay_t<decltype(std::declval<E2>()())>;
		using FirstLocalGrad_t = SecondValue_t;
		using SecondLocalGrad_t = FirstValue_t;
		using Value_t = std::decay_t<decltype(std::declval<E1>()()* std::declval<E2>()())>;

	private:
		E1 const& _first_expr;
		E2 const& _second_expr;

	public:
		constexpr MultiplyExpr(Expr<E1> const& first_expr, Expr<E2> const& second_expr)
			: _first_expr(first_expr.GetSelf()), _second_expr(second_expr.GetSelf()) {}

		constexpr Value_t operator()() const
		{
			return _first_expr() * _second_expr();
		}

		template <int I, typename T>
		constexpr Value_t Eval(T& tuple) const
		{
			FirstValue_t first_value = _first_expr.Eval<std::tuple_element_t<I, T>::Child1_v>(tuple);
			SecondValue_t second_value = _second_expr.Eval<std::tuple_element_t<I, T>::Child2_v>(tuple);
			std::get<I>(tuple).SetLocalGrads(this, second_value, first_value);
			return first_value * second_value;
		}
	};

	template <typename E1, typename E2>
	class SubtractExpr : public Expr<SubtractExpr<E1, E2>>, private details::BinaryExpr
	{
	public:
		using FirstExpr_t = std::decay_t<E1>;
		using SecondExpr_t = std::decay_t<E2>;
		using FirstValue_t = std::decay_t<decltype(std::declval<E1>()())>;
		using SecondValue_t = std::decay_t<decltype(std::declval<E2>()())>;
		using FirstLocalGrad_t = FirstValue_t;
		using SecondLocalGrad_t = SecondValue_t;
		using Value_t = std::decay_t<decltype(std::declval<E1>()() - std::declval<E2>()())>;

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
		constexpr Value_t Eval(T& tuple) const
		{
			FirstValue_t first_value = _first_expr.Eval<std::tuple_element_t<I, T>::Child1_v>(tuple);
			SecondValue_t second_value = _second_expr.Eval<std::tuple_element_t<I, T>::Child2_v>(tuple);
			std::get<I>(tuple).SetLocalGrads(this, Double(1.0), Double(-1.0));
			return first_value - second_value;
		}
	};

	template <typename E1, typename E2>
	class DivideExpr : public Expr<DivideExpr<E1, E2>>, private details::BinaryExpr
	{
	public:
		using FirstExpr_t = std::decay_t<E1>;
		using SecondExpr_t = std::decay_t<E2>;
		using FirstValue_t = std::decay_t<decltype(std::declval<E1>()())>;
		using SecondValue_t = std::decay_t<decltype(std::declval<E2>()())>;
		using FirstLocalGrad_t = FirstValue_t;
		using SecondLocalGrad_t = SecondValue_t;
		using Value_t = std::decay_t<decltype(std::declval<E1>()() / std::declval<E2>()())>;

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
		constexpr Value_t Eval(T& tuple) const
		{
			FirstValue_t first_value = _first_expr.Eval<std::tuple_element_t<I, T>::Child1_v>(tuple);
			SecondValue_t second_value = _second_expr.Eval<std::tuple_element_t<I, T>::Child2_v>(tuple);
			auto second_value_inverse = second_value.Inverse();
			std::get<I>(tuple).SetLocalGrads(this, second_value_inverse, -first_value * second_value_inverse * second_value_inverse);
			return first_value / second_value;
		}
	};

	template <typename E1, typename E2>
	class PowerExpr : public Expr<PowerExpr<E1, E2>>, private details::BinaryExpr
	{
	public:
		using FirstExpr_t = std::decay_t<E1>;
		using SecondExpr_t = std::decay_t<E2>;
		using FirstValue_t = std::decay_t<decltype(std::declval<E1>()())>;
		using SecondValue_t = std::decay_t<decltype(std::declval<E2>()())>;
		using FirstLocalGrad_t = FirstValue_t;
		using SecondLocalGrad_t = SecondValue_t;
		using Value_t = std::decay_t<decltype(Num::pow(std::declval<E1>()(), std::declval<E2>()()))>;

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
		constexpr Value_t Eval(T& tuple) const
		{
			FirstValue_t first_value = _first_expr.Eval<std::tuple_element_t<I, T>::Child1_v>(tuple);
			SecondValue_t second_value = _second_expr.Eval<std::tuple_element_t<I, T>::Child2_v>(tuple);
			Value_t value = Num::pow(first_value, second_value);
			std::get<I>(tuple).SetLocalGrads(this, second_value * value * first_value.Inverse(), value * Num::log(first_value));
			return value;
		}
	};

	template <typename E1>
	class NegateExpr : public Expr<NegateExpr<E1>>, private details::UnaryExpr
	{
	public:
		using FirstExpr_t = std::decay_t<E1>;
		using FirstValue_t = std::decay_t<decltype(std::declval<E1>()())>;
		using FirstLocalGrad_t = FirstValue_t;
		using Value_t = std::decay_t<decltype(-std::declval<E1>()())>;

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
		constexpr Value_t Eval(T& tuple) const
		{
			FirstValue_t first_value = _first_expr.Eval<std::tuple_element_t<I, T>::Child1_v>(tuple);
			std::get<I>(tuple).SetLocalGrads(this, Double(-1.0));
			return -first_value;
		}
	};

	template <typename E1>
	class LogExpr : public Expr<LogExpr<E1>>, private details::UnaryExpr
	{
	public:
		using FirstExpr_t = std::decay_t<E1>;
		using FirstValue_t = std::decay_t<decltype(std::declval<E1>()())>;
		using FirstLocalGrad_t = FirstValue_t;
		using Value_t = std::decay_t<decltype(Num::log(std::declval<E1>()()))>;

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
		constexpr Value_t Eval(T& tuple) const
		{
			FirstValue_t first_value = _first_expr.Eval<std::tuple_element_t<I, T>::Child1_v>(tuple);
			std::get<I>(tuple).SetLocalGrads(this, first_value.Inverse());
			return Num::log(first_value);
		}
	};

	template <typename E1>
	class SinExpr : public Expr<SinExpr<E1>>, private details::UnaryExpr
	{
	public:
		using FirstExpr_t = std::decay_t<E1>;
		using FirstValue_t = std::decay_t<decltype(std::declval<E1>()())>;
		using FirstLocalGrad_t = FirstValue_t;
		using Value_t = std::decay_t<decltype(Num::sin(std::declval<E1>()()))>;

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
		constexpr Value_t Eval(T& tuple) const
		{
			FirstValue_t first_value = _first_expr.Eval<std::tuple_element_t<I, T>::Child1_v>(tuple);
			std::get<I>(tuple).SetLocalGrads(this, Num::cos(first_value));
			return Num::sin(first_value);
		}
	};

	template <typename E1>
	class CosExpr : public Expr<CosExpr<E1>>, private details::UnaryExpr
	{
	public:
		using FirstExpr_t = std::decay_t<E1>;
		using FirstValue_t = std::decay_t<decltype(std::declval<E1>()())>;
		using FirstLocalGrad_t = FirstValue_t;
		using Value_t = std::decay_t<decltype(Num::cos(std::declval<E1>()()))>;

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
		constexpr Value_t Eval(T& tuple) const
		{
			FirstValue_t first_value = _first_expr.Eval<std::tuple_element_t<I, T>::Child1_v>(tuple);
			std::get<I>(tuple).SetLocalGrads(this, -Num::sin(first_value));
			return Num::cos(first_value);
		}
	};

	template <typename E1>
	class TanExpr : public Expr<TanExpr<E1>>, private details::UnaryExpr
	{
	public:
		using FirstExpr_t = std::decay_t<E1>;
		using FirstValue_t = std::decay_t<decltype(std::declval<E1>()())>;
		using FirstLocalGrad_t = FirstValue_t;
		using Value_t = std::decay_t<decltype(Num::tan(std::declval<E1>()()))>;

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
		constexpr Value_t Eval(T& tuple) const
		{
			FirstValue_t first_value = _first_expr.Eval<std::tuple_element_t<I, T>::Child1_v>(tuple);
			auto sec_value = Num::sec(first_value);
			std::get<I>(tuple).SetLocalGrads(this, sec_value * sec_value);
			return Num::tan(first_value);
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

	template <typename E, int... Ints>
	class Node;

	template <typename E>
	class Node<E> : details::TerminalExpr
	{
	public:
		using Expr_t = E;

	private:
		E const* _expr;

	public:
		typename E::Value_t _gradient;

		constexpr Node() : _gradient(0.0), _expr(nullptr) {}

		constexpr void SetLocalGrads(E const* const expr)
		{
			_expr = expr;
		}

		constexpr void SetVariableCache() const 
		{
			if constexpr (std::is_base_of_v<details::TrainableExpr, Expr_t>)
			{
				_expr->_cache += _gradient;
			}
		}

		constexpr void ResetGradient()
		{
			_gradient = static_cast<typename E::Value_t>(0.0);
		}
	};

	template <typename E, int I>
	class Node<E, I> : details::UnaryExpr
	{
	public:
		using Expr_t = E;
		constexpr static int Child1_v = I;
		
	public:
		E const* _expr;
		typename E::Value_t _gradient;
		typename E::FirstLocalGrad_t _first_local_grad;
		
		constexpr Node() : _gradient(0.0), _first_local_grad(0.0), _expr(nullptr) {}

		constexpr void SetLocalGrads(E const* const expr, typename E::FirstLocalGrad_t first_local_grad)
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
			_gradient = static_cast<typename E::Value_t>(0.0);
		}
	};

	template <typename E, int I1, int I2>
	class Node<E, I1, I2> : details::BinaryExpr
	{
	public:
		using Expr_t = E;
		constexpr static int Child1_v = I1;
		constexpr static int Child2_v = I2;

	public: 
		E const* _expr;
		typename E::Value_t _gradient;
		typename E::FirstLocalGrad_t _first_local_grad;
		typename E::SecondLocalGrad_t _second_local_grad;

		constexpr Node() : _gradient(0.0), _first_local_grad(0.0), _second_local_grad(0.0), _expr(nullptr) {}

		constexpr void SetLocalGrads(E const* const expr, typename E::FirstLocalGrad_t first_local_grad, typename E::SecondLocalGrad_t second_local_grad)
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
			_gradient = static_cast<typename E::Value_t>(0.0);
		}
	};

	namespace details
	{
		template <typename E, typename = void>
		struct DfsTuple;

		template <typename E>
		struct DfsTuple<E, std::enable_if_t<std::is_base_of_v<TerminalExpr, E>>> {

			using type = std::tuple<E>;
		};

		template <typename E>
		using DfsTuple_t = typename DfsTuple<E>::type;

		template <typename E>
		struct DfsTuple<E, std::enable_if_t<std::is_base_of_v<UnaryExpr, E>>> {

			using type = decltype(std::tuple_cat(
				std::declval<DfsTuple_t<typename E::FirstExpr_t>>(),
				std::declval<std::tuple<E>>()
			));
		};

		template <typename E>
		struct DfsTuple<E, std::enable_if_t<std::is_base_of_v<BinaryExpr, E>>> {

			using type = decltype(std::tuple_cat(
				std::declval<DfsTuple_t<typename E::FirstExpr_t>>(),
				std::declval<DfsTuple_t<typename E::SecondExpr_t>>(),
				std::declval<std::tuple<E>>()
			));
		};

		template <typename E>
		constexpr int DfsTupleSize_v = std::tuple_size_v<DfsTuple_t<E>>;

		template <typename E, int I, typename = void>
		struct _Helper_DfsFinalTuple;

		template <typename E, int I>
		struct _Helper_DfsFinalTuple<E, I, std::enable_if_t<std::is_base_of_v<TerminalExpr, E>>> {

			using type = std::tuple<Node<E>>;
		};

		template <typename E, int I>
		using _Helper_DfsFinalTuple_t = typename _Helper_DfsFinalTuple<E, I>::type;

		template <typename E, int I>
		struct _Helper_DfsFinalTuple<E, I, std::enable_if_t<std::is_base_of_v<UnaryExpr, E>>> {

			using type = decltype(std::tuple_cat(
				std::declval<_Helper_DfsFinalTuple_t<typename E::FirstExpr_t, I - 1>>(),
				std::declval<std::tuple<Node<E, I - 2>>>()
			));
		};

		template <typename E, int I>
		struct _Helper_DfsFinalTuple<E, I, std::enable_if_t<std::is_base_of_v<BinaryExpr, E>>> {

			using type = decltype(std::tuple_cat(
				std::declval<_Helper_DfsFinalTuple_t<typename E::FirstExpr_t, I - 1 - DfsTupleSize_v<typename E::SecondExpr_t>>>(),
				std::declval<_Helper_DfsFinalTuple_t<typename E::SecondExpr_t, I - 1>>(),
				std::declval<std::tuple<Node<E, I - 2 - DfsTupleSize_v<typename E::SecondExpr_t>, I - 2>>>()
			));
		};

		template <typename E>
		using DfsFinalTuple_t = _Helper_DfsFinalTuple_t<E, DfsTupleSize_v<E>>;
	}


	template <typename E>
	class GradientDescentOptimizer
	{
	private:
		using Tuple_t = typename details::DfsFinalTuple_t<E>;
		using Result_t = typename std::decay_t<decltype(std::declval<E>()())>;

		Tuple_t _tup;
		E& _expr;
		Result_t _result;

		template <typename... Args>
		constexpr void _Helper_FeedPlaceholders(Args&& ... args)
		{
			((args.first->FeedValue(args.second)), ...);
		}

		template <int I>
		constexpr void _Helper_Backprop()
		{
			if constexpr (I == std::tuple_size_v<Tuple_t> -1)
			{
				std::get<I>(_tup)._gradient = (typename std::tuple_element_t<I, Tuple_t>::Expr_t::Value_t)(1.0);
			}
			if constexpr (std::is_base_of_v<details::TerminalExpr, std::tuple_element_t<I, Tuple_t>>)
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
		constexpr GradientDescentOptimizer(Expr<E>& expr) : _expr(expr.GetSelf()) {}
		
		template <typename... Args>
		constexpr GradientDescentOptimizer& FeedPlaceholders(Args&& ... args)
		{
			_Helper_FeedPlaceholders(std::forward<Args...>(args...));
			return *this;
		}

		constexpr GradientDescentOptimizer& Eval()
		{
			_result = _expr.Eval<details::DfsTupleSize_v<E> - 1>(_tup);
			return *this;
		}

		template <typename... Vars>
		constexpr GradientDescentOptimizer& Backpass(double learning_rate, Vars& ... vars)
		{
			_Helper_Backprop<details::DfsTupleSize_v<E> - 1>();
			_Helper_ApplyGradients(learning_rate, vars...);
			return *this;
		}

		constexpr Result_t GetPreResult()
		{
			return _result;
		}

		constexpr Result_t GetPostResult()
		{
			return _expr();
		}
	};
}