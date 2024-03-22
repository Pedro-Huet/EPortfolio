#include "Monte Carlo Pricer.h"
#include <iostream>

int main() {
    double S0 = 100.0;
    double sigma = 0.3;
    double r = 0.01;
    double T = 2.0;
    double K = 100.0;

    Option option(S0, sigma, r, T, K);

    unsigned long paths[] = { 10000, 100000, 1000000 };
    for (unsigned long M : paths) {
        std::cout << "Number of Paths: " << M << std::endl;
        std::cout << "Call Option Price: " << option.PriceCall(M) << std::endl;
        std::cout << "Put Option Price: " << option.PricePut(M) << std::endl;
        std::cout << std::endl;
    }

    return 0;
}