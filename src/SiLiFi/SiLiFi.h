#pragma once

#include <SiLi/SiLi.h>

namespace SiLiFi {

template <int Dim, typename T = double>
struct Value {
	SiLi::Matrix<Dim, 1, T>   model;
	SiLi::Matrix<Dim, Dim, T> covariance;

	Value()
		: model {0}
		, covariance {SiLi::make_eye<T, Dim, Dim>()}
	{}

	Value(std::array<T, Dim> const& _model,
	      T _variance = 1.)
		: covariance {SiLi::make_eye<T, Dim, Dim>()}
	{
		for (int i(0); i < _model.size(); ++i) {
			model(i) = _model[i];
		}
	}


	template<typename P2>
	Value(std::array<T, Dim> const& _model,
	      SiLi::MatrixView<Dim, Dim, P2, T const> const& _covariance)
		: covariance {_covariance}
	{
		for (int i(0); i < _model.size(); ++i) {
			model(i) = _model[i];
		}
	}

	template<typename P1, typename P2>
	Value(SiLi::MatrixView<Dim, 1, P1, T const> const& _model,
	      SiLi::MatrixView<Dim, Dim, P2, T const> const& _covariance)
		: model {_model}
		, covariance {_covariance}
	{}

	template<typename P1>
	Value(SiLi::MatrixView<Dim, 1, P1, T const> const& _model,
	      T _variance = 1.)
		: model {_model}
		, covariance {SiLi::make_eye<T, Dim, Dim>()}
	{}
};

template <int Dim, typename T = double>
class Process {
private:
	SiLi::Matrix<Dim, Dim, T> F;
	SiLi::Matrix<Dim, Dim, T> Q;
	SiLi::Matrix<Dim, 1, T>   u;
public:
	template <typename P1, typename P2>
	Process(SiLi::MatrixView<Dim, Dim, P1, T const> const& _F, SiLi::MatrixView<Dim, Dim, P2, T const> const& _Q,
	        SiLi::Matrix<Dim, 1, T> _u = {})
		: F {_F}
		, Q {_Q}
		, u {_u}
	{}

	auto predict(Value<Dim, T> const& _model) -> Value<Dim, T> {
		auto x = F * _model.model + u;
		auto P = F * _model.covariance * F.t() + Q;

		return {x, P};
	}
};

template <int Dim1, int Dim2, typename T>
auto kfUpdate(Value<Dim1, T> const& _model, Value<Dim2, T> const& _measurement) -> Value<Dim1, T> {
	return kfUpdate(_model, _measurement, SiLi::make_eye<T, Dim2, Dim1>());
}


template <int Dim1, int Dim2, typename T, typename P1>
auto kfUpdate(Value<Dim1, T> const& _model, Value<Dim2, T> const& _measurement, SiLi::MatrixView<Dim2, Dim1, P1, T const> const& H) -> Value<Dim1, T> {
	auto const& x = _model.model;
	auto const& P = _model.covariance;
	auto const& z = _measurement.model;
	auto const& R = _measurement.covariance;

	auto y = z - H*x;
	auto S = H * P * H.t() + R;
	auto K = P * H.t() * S.inv();
	auto newX     = x + K * y;
	auto newCovar = (SiLi::make_eye<T, Dim1, Dim1>() - K * H) * P;
	return {newX, newCovar};
}

template<int Dim, typename T>
std::ostream& operator<< (std::ostream& stream, Value<Dim, T> const& value) {
	stream << value.model;
	stream << value.covariance;
	return stream;
}


}
