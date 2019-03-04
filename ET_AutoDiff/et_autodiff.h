#pragma once

namespace et
{

	template <typename Derived>
	class Expr {

	public:
		constexpr Derived const& getSelf() const {
			return static_cast<Derived const&>(*this);
		}

		constexpr auto operator()() const {
			return getSelf()();
		}
	};

	template <typename ValueType>
	class ConstantExpr : public Expr<ConstantExpr<ValueType>> {

	private:
		const ValueType& mValue;

	public:
		constexpr ConstantExpr(const ValueType& Value) : mValue(Value) {}

		constexpr auto operator()() const {
			return mValue;
		}
	};

	template <typename FirstExprType, typename SecondExprType>
	class AddExpr : public Expr<AddExpr<FirstExprType, SecondExprType>> {

	private:
		const FirstExprType& mFirstExpr;
		const SecondExprType& mSecondExpr;

	public:
		constexpr AddExpr(const Expr<FirstExprType>& FirstExpr, const Expr<SecondExprType>& SecondExpr)
			: mFirstExpr(FirstExpr.getSelf()), mSecondExpr(SecondExpr.getSelf()) {}

		constexpr auto operator()() const {
			return mFirstExpr() + mSecondExpr();
		}
	};

	template <typename FirstExprType, typename SecondExprType>
	class SubtractExpr : public Expr<SubtractExpr<FirstExprType, SecondExprType>> {

	private:
		const FirstExprType& mFirstExpr;
		const SecondExprType& mSecondExpr;

	public:
		constexpr SubtractExpr(const Expr<FirstExprType>& FirstExpr, const Expr<SecondExprType>& SecondExpr)
			: mFirstExpr(FirstExpr.getSelf()), mSecondExpr(SecondExpr.getSelf()) {}

		constexpr auto operator()() const {
			return mFirstExpr() - mSecondExpr();
		}
	};

	template <typename FirstExprType, typename SecondExprType>
	class MultiplyExpr : public Expr<MultiplyExpr<FirstExprType, SecondExprType>> {

	private:
		const FirstExprType& mFirstExpr;
		const SecondExprType& mSecondExpr;

	public:
		constexpr MultiplyExpr(const Expr<FirstExprType>& FirstExpr, const Expr<SecondExprType>& SecondExpr)
			: mFirstExpr(FirstExpr.getSelf()), mSecondExpr(SecondExpr.getSelf()) {}

		constexpr auto operator()() const {
			return mFirstExpr()* mSecondExpr();
		}
	};

	template <typename FirstExprType, typename SecondExprType>
	class DivideExpr : public Expr<DivideExpr<FirstExprType, SecondExprType >> {

	private:
		const FirstExprType& mFirstExpr;
		const SecondExprType& mSecondExpr;

	public:
		constexpr DivideExpr(const Expr<FirstExprType>& FirstExpr, const Expr<SecondExprType>& SecondExpr)
			: mFirstExpr(FirstExpr.getSelf()), mSecondExpr(SecondExpr.getSelf()) {}

		constexpr auto operator()() const {
			return mFirstExpr() / mSecondExpr();
		}
	};

	template <typename FirstExprType, typename SecondExprType>
	constexpr auto operator+(const Expr<FirstExprType>& FirstExpr, const Expr<SecondExprType>& SecondExpr) {
		return AddExpr(FirstExpr, SecondExpr);
	}

	template <typename FirstExprType, typename SecondExprType>
	constexpr auto operator-(const Expr<FirstExprType>& FirstExpr, const Expr<SecondExprType>& SecondExpr) {
		return SubtractExpr(FirstExpr, SecondExpr);
	}

	template <typename FirstExprType, typename SecondExprType>
	constexpr auto operator*(const Expr<FirstExprType>& FirstExpr, const Expr<SecondExprType>& SecondExpr) {
		return MultiplyExpr(FirstExpr, SecondExpr);
	}

	template <typename FirstExprType, typename SecondExprType>
	constexpr auto operator/(const Expr<FirstExprType>& FirstExpr, const Expr<SecondExprType>& SecondExpr) {
		return DivideExpr(FirstExpr, SecondExpr);
	}

}