#include "Monte Carlo Pricer.h"
#include <cmath>

Option::Option(double S0, double sigma, double r, double T, double K)
    : S0_(S0), sigma_(sigma), r_(r), T_(T), K_(K), rng_() {}

double Option::PriceCall(unsigned long M) const {
    double payoffSum = 0.0;
    for (unsigned long i = 0; i < M; ++i) {
        double Z = rng_.generateNormal();
        double ST = S0_ * exp((r_ - 0.5 * sigma_ * sigma_) * T_ + sigma_ * sqrt(T_) * Z);
        payoffSum += std::max(ST - K_, 0.0);
    }
    return (exp(-r_ * T_) * payoffSum) / M;
}

double Option::PricePut(unsigned long M) const {
    double payoffSum = 0.0;
    for (unsigned long i = 0; i < M; ++i) {
        double Z = rng_.generateNormal();
        double ST = S0_ * exp((r_ - 0.5 * sigma_ * sigma_) * T_ + sigma_ * sqrt(T_) * Z);
        payoffSum += std::max(K_ - ST, 0.0);
    }
    return (exp(-r_ * T_) * payoffSum) / M;
}