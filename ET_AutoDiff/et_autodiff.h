#pragma once

#include <optional>
#include <tuple>
#include <vector>

namespace Et
{
	using std::pow;
	using std::log;

	namespace __util {

		class TerminalExpr {};
		class BinaryExpr {};
		class UnaryExpr {};
	}

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
	class ConstantExpr : public Expr<ConstantExpr<V>>, private __util::TerminalExpr {

	private:
		V const value_;

	public:
		using value_type_t = std::remove_cv_t < std::remove_reference_t <V>>;

		constexpr ConstantExpr(V const& value) : value_(value) {
			static_assert(std::is_floating_point_v<V>, "The values must be decimal for accuracy.");
		}

		constexpr V const& operator()() const {
			return value_;
		}
	};
	
	template <typename V>
	class PlaceholderExpr : public Expr<PlaceholderExpr<V>>, private __util::TerminalExpr {

	private:
		std::optional<V> value_;

	public:
		using value_type_t = std::remove_cv_t < std::remove_reference_t <V>>;

		constexpr PlaceholderExpr() : value_(std::nullopt) {
			static_assert(std::is_floating_point_v<V>, "The values must be decimal for accuracy.");
		}

		constexpr V const& operator()() const {
			return value_.value();
		}

		constexpr void FeedValue(V const& value) {
			value_ = value;
		}
	};

	template <typename V>
	class VariableExpr : public Expr<VariableExpr<V>>, private __util::TerminalExpr {

	private:
		V value_;

	public:
		using value_type_t = std::remove_cv_t < std::remove_reference_t <V>>;

		constexpr VariableExpr(V const& init_value) : value_(init_value) {
			static_assert(std::is_floating_point_v<V>, "The values must be decimal for accuracy.");
		}

		constexpr V const& operator()() const {
			return value_;
		}
	};

	template <typename E1, typename E2>
	class AddExpr : public Expr<AddExpr<E1, E2>>, private __util::BinaryExpr {

	private:
		E1 const& first_expr_;
		E2 const& second_expr_;

	public:
		using value_type_t = std::remove_cv_t < std::remove_reference_t <decltype(std::declval<E1>()() + std::declval<E2>()())>>;
		using first_expr_type_t = E1;
		using second_expr_type_t = E2;
		using first_expr_local_grad_type_t = std::remove_cv_t < std::remove_reference_t<decltype(std::declval<E1>()())>>;
		using second_expr_local_grad_type_t = std::remove_cv_t < std::remove_reference_t<decltype(std::declval<E2>()())>>;

		constexpr AddExpr(Expr<E1> const& first_expr, Expr<E2> const& second_expr)
			: first_expr_(first_expr.GetSelf()), second_expr_(second_expr.GetSelf()) {}
	
		constexpr value_type_t operator()() const {
			return first_expr_() + second_expr_();
		}
	};

	template <typename E1, typename E2>
	class SubtractExpr : public Expr<SubtractExpr<E1, E2>>, private __util::BinaryExpr {

	private:
		E1 const& first_expr_;
		E2 const& second_expr_;

	public:
		using value_type_t = std::remove_cv_t < std::remove_reference_t <decltype(std::declval<E1>()() - std::declval<E2>()())>>;
		using first_expr_type_t = E1;
		using second_expr_type_t = E2;
		using first_expr_local_grad_type_t = std::remove_cv_t < std::remove_reference_t<decltype(std::declval<E1>()())>>;
		using second_expr_local_grad_type_t = std::remove_cv_t < std::remove_reference_t<decltype(std::declval<E2>()())>>;

		constexpr SubtractExpr(Expr<E1> const& first_expr, Expr<E2> const& second_expr)
			: first_expr_(first_expr.GetSelf()), second_expr_(second_expr.GetSelf()) {}

		constexpr value_type_t operator()() const {
			return first_expr_() - second_expr_();
		}
	};

	template <typename E1, typename E2>
	class MultiplyExpr : public Expr<MultiplyExpr<E1, E2>>, private __util::BinaryExpr {

	private:
		E1 const& first_expr_;
		E2 const& second_expr_;

	public:
		using value_type_t = decltype(std::declval<E1>()() * std::declval<E2>()());
		using first_expr_type_t = E1;
		using second_expr_type_t = E2;
		using first_expr_local_grad_type_t = std::remove_cv_t < std::remove_reference_t<decltype(std::declval<E2>()())>>;
		using second_expr_local_grad_type_t = std::remove_cv_t < std::remove_reference_t<decltype(std::declval<E1>()())>>;

		constexpr MultiplyExpr(Expr<E1> const& first_expr, Expr<E2> const& second_expr)
			: first_expr_(first_expr.GetSelf()), second_expr_(second_expr.GetSelf()) {}

		constexpr value_type_t operator()() const {
			return first_expr_() * second_expr_();
		}
	};

	template <typename E1, typename E2>
	class DivideExpr : public Expr<DivideExpr<E1, E2 >>, private __util::BinaryExpr {

	private:
		E1 const& first_expr_;
		E2 const& second_expr_;

	public:
		using value_type_t = decltype(std::declval<E1>()() / std::declval<E2>()());
		using first_expr_type_t = E1;
		using second_expr_type_t = E2;
		using first_expr_local_grad_type_t = std::remove_cv_t < std::remove_reference_t<decltype(1.0 / std::declval<E2>()())>>;
		using second_expr_local_grad_type_t = std::remove_cv_t < std::remove_reference_t <decltype(std::declval<E1>()())>>;

		constexpr DivideExpr(Expr<E1> const& first_expr, Expr<E2> const& second_expr)
			: first_expr_(first_expr.GetSelf()), second_expr_(second_expr.GetSelf()) {}

		constexpr value_type_t operator()() const {
			return first_expr_() /second_expr_();
		}
	};

	template <typename E1>
	class NegateExpr : public Expr<NegateExpr<E1>>, private __util::UnaryExpr {

	private:
		E1 const& first_expr_;

	public:
		using value_type_t = decltype(-(std::declval<E1>()()));
		using first_expr_type_t = E1;
		using first_expr_local_grad_type_t = std::remove_cv_t < std::remove_reference_t <decltype(std::declval<E1>()())>>;

		constexpr NegateExpr(Expr<E1> const& first_expr) : first_expr_(first_expr.GetSelf()) {}

		constexpr value_type_t operator()() const {
			return -(first_expr_());
		}
	};

	template <typename E1, typename E2>
	class ExponentExpr : public Expr<ExponentExpr<E1, E2>>, private __util::BinaryExpr {

	private:
		E1 const& first_expr_;
		E2 const& second_expr_;

	public:
		using value_type_t = decltype(pow(std::declval<E1>()(), std::declval<E2>()()));
		using first_expr_type_t = E1;
		using second_expr_type_t = E2;
		using first_expr_local_grad_type_t = std::remove_cv_t < std::remove_reference_t<decltype(std::declval<E2>()() * pow(std::declval<E1>()(), std::declval<E2>()() - 1))>>;
		using second_expr_local_grad_type_t = std::remove_cv_t < std::remove_reference_t<decltype(log(std::declval<E1>()()) * pow(std::declval<E1>()(), std::declval<E2>()()))>>;

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

	template <typename E, int C1, int C2>
	class BinaryNode {

	public:
		typename E::first_expr_local_grad_type_t first_expr_local_grad_;
		typename E::second_expr_local_grad_type_t second_expr_local_grad_;
		typename E::value_type_t grediant_;

		BinaryNode() : grediant_(0) {}

		template <typename T>
		void UpdateChilds(T const& container) {
			std::get<C1>(container).grediant_ += this->grediant_ * first_expr_local_grad_;
			std::get<C2>(container).grediant_ += this->grediant_ * second_expr_local_grad_;
		}
	};

	template <typename H, int C1>
	class UnaryNode {

	public:
		typename H::first_expr_local_grad_type_t first_expr_cal_grad_;
		typename H::value_type_t grediant_;

		UnaryNode() : grediant_(0) {}

		template <typename T>
		void UpdateChilds(T const& container) {
			std::get<C1>(container).grediant_ += this->grediant_ * first_expr_cal_grad_;
		}
	};

	template <typename H>
	class TerminalNode {

	public:
		typename H::value_type_t grediant_;

		TerminalNode() : grediant_(0) {}
	};

	namespace __util {

		template <typename E, typename = void>
		struct expr_dfs_type;

		template <typename E>
		using expr_dfs_type_t = typename expr_dfs_type<E>::type;

		template <typename E>
		struct expr_dfs_type<E, std::enable_if_t<std::is_base_of_v<TerminalExpr, E>>> {

			using type = std::tuple<E>;
		};

		template <typename E>
		struct expr_dfs_type<E, std::enable_if_t<std::is_base_of_v<UnaryExpr, E>>> {

			using type = decltype(std::tuple_cat(
				std::declval<expr_dfs_type_t<E::first_expr_type_t>>(),
				std::declval<std::tuple<E>>()
			));
		};

		template <typename E>
		struct expr_dfs_type<E, std::enable_if_t<std::is_base_of_v<BinaryExpr, E>>> {

			using type = decltype(std::tuple_cat(
				std::declval<expr_dfs_type_t<E::first_expr_type_t>>(),
				std::declval<expr_dfs_type_t<E::second_expr_type_t>>(),
				std::declval<std::tuple<E>>()
			));
		};

		template <typename E>
		constexpr int expr_dfs_size_v = std::tuple_size_v<expr_dfs_type_t<E>>;

		template <typename, int>
		struct Holder {};

		template <typename, typename>
		struct helper_1;

		template <typename F, int I, typename... Args, int... Ints>
		struct helper_1<std::tuple<F, Args...>, std::integer_sequence<int, I, Ints...>> {

			using type = decltype(std::tuple_cat(
				std::declval<std::tuple<Holder<F, I>>>(),
				std::declval<helper_1<std::tuple<Args...>, std::integer_sequence<int, Ints...>>::type>()
			));
		};

		template <typename F, int I>
		struct helper_1<std::tuple<F>, std::integer_sequence<int, I>> {

			using type = decltype(std::declval<std::tuple<Holder<F, I>>>());
		};

		template <typename T>
		struct next_phase;

		template <typename... Args>
		struct next_phase<std::tuple<Args...>> {

			using type = typename helper_1<std::tuple<Args...>, std::make_integer_sequence<int, sizeof...(Args)>>::type;
		};
	}
}