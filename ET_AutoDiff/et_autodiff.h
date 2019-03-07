#pragma once

#include <optional>
#include <tuple>

namespace Et
{
	using std::pow;

	template <typename D>
	class Expr {

	public:
		constexpr D const& GetSelf() const {
			return static_cast<D const&>(*this);
		}

		constexpr decltype(auto) operator()() const {
			return GetSelf()();
		}
	};

	template <typename V>
	class ConstantExpr : public Expr<ConstantExpr<V>> {

	private:
		V const value_;

	public:
		using value_type_t = V;

		constexpr ConstantExpr(V const& value) : value_(value) {}

		constexpr V const& operator()() const {
			return value_;
		}
	};

	template <typename V>
	class PlaceholderExpr : public Expr<PlaceholderExpr<V>> {

	private:
		std::optional<V> value_;

	public:
		using value_type_t = V;

		constexpr PlaceholderExpr() : value_(std::nullopt) {}

		constexpr V const& operator()() const {
			return value_.value();
		}

		constexpr void FeedValue(V const& value) {
			value_ = value;
		}
	};

	template <typename V>
	class VariableExpr : public Expr<VariableExpr<V>> {

	private:
		V value_;

	public:
		using value_type_t = V;

		constexpr VariableExpr(V const& init_value) : value_(init_value) {}

		constexpr V const& operator()() const {
			return value_;
		}
	};

	template <typename E1, typename E2>
	class AddExpr : public Expr<AddExpr<E1, E2>> {

	private:
		E1 const& first_expr_;
		E2 const& second_expr_;

	public:
		using value_type_t = decltype(std::declval<E1>()() + std::declval<E2>()());
		using first_expr_type_t = E1;
		using second_expr_type_t = E2;

		constexpr AddExpr(Expr<E1> const& first_expr, Expr<E2> const& second_expr)
			: first_expr_(first_expr.GetSelf()), second_expr_(second_expr.GetSelf()) {}
	
		constexpr value_type_t operator()() const {
			return first_expr_() + second_expr_();
		}
	};

	template <typename E1, typename E2>
	class SubtractExpr : public Expr<SubtractExpr<E1, E2>> {

	private:
		E1 const& first_expr_;
		E2 const& second_expr_;

	public:
		using value_type_t = decltype(std::declval<E1>()() - std::declval<E2>()());
		using first_expr_type_t = E1;
		using second_expr_type_t = E2;

		constexpr SubtractExpr(Expr<E1> const& first_expr, Expr<E2> const& second_expr)
			: first_expr_(first_expr.GetSelf()), second_expr_(second_expr.GetSelf()) {}

		constexpr value_type_t operator()() const {
			return first_expr_() - second_expr_();
		}
	};

	template <typename E1, typename E2>
	class MultiplyExpr : public Expr<MultiplyExpr<E1, E2>> {

	private:
		E1 const& first_expr_;
		E2 const& second_expr_;

	public:
		using value_type_t = decltype(std::declval<E1>()() * std::declval<E2>()());
		using first_expr_type_t = E1;
		using second_expr_type_t = E2;

		constexpr MultiplyExpr(Expr<E1> const& first_expr, Expr<E2> const& second_expr)
			: first_expr_(first_expr.GetSelf()), second_expr_(second_expr.GetSelf()) {}

		constexpr value_type_t operator()() const {
			return first_expr_() * second_expr_();
		}
	};

	template <typename E1, typename E2>
	class DivideExpr : public Expr<DivideExpr<E1, E2 >> {

	private:
		E1 const& first_expr_;
		E2 const& second_expr_;

	public:
		using value_type_t = decltype(std::declval<E1>()() / std::declval<E2>()());
		using first_expr_type_t = E1;
		using second_expr_type_t = E2;

		constexpr DivideExpr(Expr<E1> const& first_expr, Expr<E2> const& second_expr)
			: first_expr_(first_expr.GetSelf()), second_expr_(second_expr.GetSelf()) {}

		constexpr value_type_t operator()() const {
			return first_expr_() /second_expr_();
		}
	};

	template <typename E1>
	class NegateExpr : public Expr<NegateExpr<E1>> {

	private:
		E1 const& first_expr_;

	public:
		using value_type_t = decltype(-(std::declval<E1>()()));
		using first_expr_type_t = E1;

		constexpr NegateExpr(Expr<E1> const& first_expr) : first_expr_(first_expr.GetSelf()) {}

		constexpr value_type_t operator()() const {
			return -(first_expr_());
		}
	};

	template <typename E1, typename E2>
	class ExponentExpr : public Expr<ExponentExpr<E1, E2>> {

	private:
		E1 const& first_expr_;
		E2 const& second_expr_;

	public:
		using value_type_t = decltype(pow(std::declval<E1>()(), std::declval<E2>()()));
		using first_expr_type_t = E1;
		using second_expr_type_t = E2;

		constexpr ExponentExpr(Expr<E1> const& first_expr, Expr<E2> const& second_expr)
			: first_expr_(first_expr.GetSelf()), second_expr_(second_expr.GetSelf()) {}

		constexpr value_type_t operator()() const {
			return pow(first_expr_(), second_expr_());
		}
	};

	template <typename E1, typename E2>
	constexpr AddExpr<E1, E2> operator+(Expr<E1> const& FirstExpr, Expr<E2> const& SecondExpr) {
		return AddExpr(FirstExpr, SecondExpr);
	}

	template <typename E1, typename E2>
	constexpr SubtractExpr<E1, E2> operator-(Expr<E1> const& FirstExpr, Expr<E2> const& SecondExpr) {
		return SubtractExpr(FirstExpr, SecondExpr);
	}

	template <typename E1, typename E2>
	constexpr MultiplyExpr<E1, E2> operator*(Expr<E1> const& FirstExpr, Expr<E2> const& SecondExpr) {
		return MultiplyExpr(FirstExpr, SecondExpr);
	}

	template <typename E1, typename E2>
	constexpr DivideExpr<E1, E2> operator/(Expr<E1> const& FirstExpr, Expr<E2> const& SecondExpr) {
		return DivideExpr(FirstExpr, SecondExpr);
	}

	template <typename E1>
	constexpr NegateExpr<E1> operator-(Expr<E1> const& FirstExpr) {
		return NegateExpr(FirstExpr);
	}

	template <typename E1, typename E2>
	constexpr ExponentExpr<E1, E2> pow(Expr<E1> const& FirstExpr, Expr<E2> const& SecondExpr) {
		return ExponentExpr(FirstExpr, SecondExpr);
	}
}
