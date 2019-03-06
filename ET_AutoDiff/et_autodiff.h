#pragma once

#include <optional>

namespace et
{
	using std::pow;

	struct Types {

		template <typename FirstExprType, typename SecondExprType>
		using AddExprReturnType = decltype(std::declval<FirstExprType>()() + std::declval<SecondExprType>()());

		template <typename FirstExprType, typename SecondExprType>
		using SubtractExprReturnType = decltype(std::declval<FirstExprType>()() - std::declval<SecondExprType>()());

		template <typename FirstExprType, typename SecondExprType>
		using MultiplyExprReturnType = decltype(std::declval<FirstExprType>()() * std::declval<SecondExprType>()());

		template <typename FirstExprType, typename SecondExprType>
		using DivideExprReturnType = decltype(std::declval<FirstExprType>()() / std::declval<SecondExprType>()());

		template <typename FirstExprType>
		using NegateExprReturnType = decltype(-std::declval<FirstExprType>()());

		template <typename FirstExprType, typename SecondExprType>
		using ExponentExprReturnType = decltype(pow(std::declval<FirstExprType>()(), std::declval<SecondExprType>()()));
	};

	template <typename Derived>
	class Expr {

	public:
		constexpr Derived const& getSelf() const {
			return static_cast<Derived const&>(*this);
		}

		constexpr decltype(auto) operator()() const {
			return getSelf()();
		}
	};

	template <typename ValueType>
	class ConstantExpr : public Expr<ConstantExpr<ValueType>> {

	private:
		ValueType const mValue;

	public:
		constexpr ConstantExpr(ValueType const& Value) : mValue(Value) {}

		constexpr ValueType const& operator()() const {
			return mValue;
		}
	};

	template <typename ValueType>
	class PlaceholderExpr : public Expr<PlaceholderExpr<ValueType>> {

	private:
		std::optional<ValueType> mValue;

	public:
		constexpr PlaceholderExpr() : mValue(std::nullopt) {}

		constexpr ValueType const& operator()() const {
			return mValue.value();
		}

		constexpr void feedValue(ValueType& Value) {
			mValue = Value;
		}
	};

	template <typename ValueType>
	class VariableExpr : public Expr<VariableExpr<ValueType>> {

	private:
		ValueType mValue;

	public:
		constexpr VariableExpr(ValueType const& InitValue) : mValue(InitValue) {}

		constexpr ValueType const& operator()() const {
			return mValue;
		}
	};

	template <typename FirstExprType, typename SecondExprType>
	class AddExpr : public Expr<AddExpr<FirstExprType, SecondExprType>> {

	private:
		FirstExprType const& mFirstExpr;
		SecondExprType const& mSecondExpr;
		using ReturnType = Types::AddExprReturnType<FirstExprType, SecondExprType>;

	public:
		constexpr AddExpr(Expr<FirstExprType> const& FirstExpr, Expr<SecondExprType> const& SecondExpr)
			: mFirstExpr(FirstExpr.getSelf()), mSecondExpr(SecondExpr.getSelf()) {}
	
		constexpr ReturnType operator()() const {
			return mFirstExpr() + mSecondExpr();
		}
	};

	template <typename FirstExprType, typename SecondExprType>
	class SubtractExpr : public Expr<SubtractExpr<FirstExprType, SecondExprType>> {

	private:
		FirstExprType const& mFirstExpr;
		SecondExprType const& mSecondExpr;
		using ReturnType = Types::SubtractExprReturnType<FirstExprType, SecondExprType>;

	public:
		constexpr SubtractExpr(Expr<FirstExprType> const& FirstExpr, Expr<SecondExprType> const& SecondExpr)
			: mFirstExpr(FirstExpr.getSelf()), mSecondExpr(SecondExpr.getSelf()) {}

		constexpr ReturnType operator()() const {
			return mFirstExpr() - mSecondExpr();
		}
	};

	template <typename FirstExprType, typename SecondExprType>
	class MultiplyExpr : public Expr<MultiplyExpr<FirstExprType, SecondExprType>> {

	private:
		FirstExprType const& mFirstExpr;
		SecondExprType const& mSecondExpr;
		using ReturnType = Types::MultiplyExprReturnType<FirstExprType, SecondExprType>;

	public:
		constexpr MultiplyExpr(Expr<FirstExprType> const& FirstExpr, Expr<SecondExprType> const& SecondExpr)
			: mFirstExpr(FirstExpr.getSelf()), mSecondExpr(SecondExpr.getSelf()) {}

		constexpr ReturnType operator()() const {
			return mFirstExpr()* mSecondExpr();
		}
	};

	template <typename FirstExprType, typename SecondExprType>
	class DivideExpr : public Expr<DivideExpr<FirstExprType, SecondExprType >> {

	private:
		FirstExprType const& mFirstExpr;
		SecondExprType const& mSecondExpr;
		using ReturnType = Types::DivideExprReturnType<FirstExprType, SecondExprType>;

	public:
		constexpr DivideExpr(Expr<FirstExprType> const& FirstExpr, Expr<SecondExprType> const& SecondExpr)
			: mFirstExpr(FirstExpr.getSelf()), mSecondExpr(SecondExpr.getSelf()) {}

		constexpr ReturnType operator()() const {
			return mFirstExpr() / mSecondExpr();
		}
	};

	template <typename FirstExprType>
	class NegateExpr : public Expr<NegateExpr<FirstExprType>> {

	private:
		FirstExprType const& mFirstExpr;
		using ReturnType = Types::NegateExprReturnType<FirstExprType>;

	public:
		constexpr NegateExpr(Expr<FirstExprType> const& FirstExpr) : mFirstExpr(FirstExpr.getSelf()) {}

		constexpr ReturnType operator()() const {
			return -(mFirstExpr());
		}
	};

	template <typename FirstExprType, typename SecondExprType>
	class ExponentExpr : public Expr<ExponentExpr<FirstExprType, SecondExprType>> {

	private:
		FirstExprType const& mFirstExpr;
		SecondExprType const& mSecondExpr;
		using ReturnType = Types::ExponentExprReturnType<FirstExprType, SecondExprType>;

	public:
		constexpr ExponentExpr(Expr<FirstExprType> const& FirstExpr, Expr<SecondExprType> const& SecondExpr)
			: mFirstExpr(FirstExpr.getSelf()), mSecondExpr(SecondExpr.getSelf()) {}

		constexpr ReturnType operator()() const {
			return pow(mFirstExpr(), mSecondExpr());
		}
	};

	template <typename FirstExprType, typename SecondExprType>
	constexpr AddExpr<FirstExprType, SecondExprType> operator+(Expr<FirstExprType> const& FirstExpr, Expr<SecondExprType> const& SecondExpr) {
		return AddExpr(FirstExpr, SecondExpr);
	}

	template <typename FirstExprType, typename SecondExprType>
	constexpr SubtractExpr<FirstExprType, SecondExprType> operator-(Expr<FirstExprType> const& FirstExpr, Expr<SecondExprType> const& SecondExpr) {
		return SubtractExpr(FirstExpr, SecondExpr);
	}

	template <typename FirstExprType, typename SecondExprType>
	constexpr MultiplyExpr<FirstExprType, SecondExprType> operator*(Expr<FirstExprType> const& FirstExpr, Expr<SecondExprType> const& SecondExpr) {
		return MultiplyExpr(FirstExpr, SecondExpr);
	}

	template <typename FirstExprType, typename SecondExprType>
	constexpr DivideExpr<FirstExprType, SecondExprType> operator/(Expr<FirstExprType> const& FirstExpr, Expr<SecondExprType> const& SecondExpr) {
		return DivideExpr(FirstExpr, SecondExpr);
	}

	template <typename FirstExprType>
	constexpr NegateExpr<FirstExprType> operator-(Expr<FirstExprType> const& FirstExpr) {
		return NegateExpr(FirstExpr);
	}

	template <typename FirstExprType, typename SecondExprType>
	constexpr ExponentExpr<FirstExprType, SecondExprType> pow(Expr<FirstExprType> const& FirstExpr, Expr<SecondExprType> const& SecondExpr) {
		return ExponentExpr(FirstExpr, SecondExpr);
	}
}
