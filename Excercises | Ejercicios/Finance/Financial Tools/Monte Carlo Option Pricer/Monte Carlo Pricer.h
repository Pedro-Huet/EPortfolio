#ifndef OPTION_MC_H
#define OPTION_MC_H

#include "Random Number Generator.h"

class Option {
public:
    Option(double S0, double sigma, double r, double T, double K);
    double PriceCall(unsigned long M) const;
    double PricePut(unsigned long M) const;

private:
    double S0_;
    double sigma_;
    double r_;
    double T_;
    double K_;
    mutable RandomNumberGenerator rng_; // Make rng_ mutable
};

#endif // OPTION_MC_H