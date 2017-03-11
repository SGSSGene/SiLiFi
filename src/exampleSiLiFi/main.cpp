#include <SiLiFi/SiLiFi.h>

#include <iostream>

using namespace SiLiFi;

int main() {
	Value<3> value1;
	value1.model(2, 0) = 1.;
	for (int i(0); i<10; ++i) {
		std::cout << value1 << "\n";
		Value<3, double> value2({0., 0., 0.}, 1.);


		auto F = SiLi::make_eye<double, 3, 3>();
		auto Q = SiLi::make_eye<double, 3, 3>();
		Process<3, double> process(F, Q);

		value1 = kfUpdate(value1, value2);
		value1 = process.predict(value1);
	}

	return EXIT_SUCCESS;
}
