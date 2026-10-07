// R's gamma functions, ported from R's nmath sources (R 4.3):
//   src/nmath/gamma.c, lgamma.c, lgammacor.c, chebyshev.c, stirlerr.c,
//   and logcf() and log1pmx() of pgamma.c
//
// std::tgamma, std::lgamma, math.gamma and SciPy's gamma all round differently
// from R in the last bit on most arguments. The Generalised Normal density
// takes gamma(1/shape), and the likelihood surfaces it enters are flat enough
// for that bit to move an optimiser, so the Python densities use these.

#ifndef GREYBOX_NMATH_H
#define GREYBOX_NMATH_H

#include <cmath>
#include <limits>

namespace greybox_nmath {

constexpr double kLnSqrt2Pi = 0.918938533204672741780329736406;  // log(sqrt(2*pi))
constexpr double kLnSqrtPid2 = 0.225791352644727432363097614947;  // log(sqrt(pi/2))
constexpr double kPi = 3.141592653589793238462643383280;

// Coefficients of the Stirling series: 1/12, 1/360, 1/1260, 1/1680, 1/1188.
constexpr double kS0 = 0.083333333333333333333;
constexpr double kS1 = 0.00277777777777777777778;
constexpr double kS2 = 0.00079365079365079365079365;
constexpr double kS3 = 0.000595238095238095238095238;
constexpr double kS4 = 0.0008417508417508417508417508;

// stirlerr(n) for n = 0, 0.5, 1.0, ..., 15.0, tabulated because the series is
// least accurate exactly where the arguments are smallest.
constexpr double kSferrHalves[31] = {
    0.0,
    0.1534264097200273452913848,   0.0810614667953272582196702,
    0.0548141210519176538961390,   0.0413406959554092940938221,
    0.03316287351993628748511048,  0.02767792568499833914878929,
    0.02374616365629749597132920,  0.02079067210376509311152277,
    0.01848845053267318523077934,  0.01664469118982119216319487,
    0.01513497322191737887351255,  0.01387612882307074799874573,
    0.01281046524292022692424986,  0.01189670994589177009505572,
    0.01110455975820691732662991,  0.010411265261972096497478567,
    0.009799416126158803298389475, 0.009255462182712732917728637,
    0.008768700134139385462952823, 0.008330563433362871256469318,
    0.007934114564314020547248100, 0.007573675487951840794972024,
    0.007244554301320383179543912, 0.006942840107209529865664152,
    0.006665247032707682442354394, 0.006408994188004207068439631,
    0.006171712263039457647532867, 0.005951370112758847735624416,
    0.005746216513010115682023589, 0.005554733551962801371038690};

// The Chebyshev series of gamma(x) on [1, 2]: the first ngam = 22 of R's 42.
constexpr double kGamcs[22] = {
    +.8571195590989331421920062399942e-2, +.4415381324841006757191315771652e-2,
    +.5685043681599363378632664588789e-1, -.4219835396418560501012500186624e-2,
    +.1326808181212460220584006796352e-2, -.1893024529798880432523947023886e-3,
    +.3606925327441245256578082217225e-4, -.6056761904460864218485548290365e-5,
    +.1055829546302283344731823509093e-5, -.1811967365542384048291855891166e-6,
    +.3117724964715322277790254593169e-7, -.5354219639019687140874081024347e-8,
    +.9193275519859588946887786825940e-9, -.1577941280288339761767423273953e-9,
    +.2707980622934954543266540433089e-10, -.4646818653825730144081661058933e-11,
    +.7973350192007419656460767175359e-12, -.1368078209830916025799499172309e-12,
    +.2347319486563800657233471771688e-13, -.4027432614949066932766570534699e-14,
    +.6910051747372100912138336975257e-15, -.1185584500221992907052387126192e-15};

// The Chebyshev series of lgammacor(): the first nalgm = 5 of R's 15.
constexpr double kAlgmcs[5] = {
    +.1666389480451863247205729650822e+0, -.1384948176067563840732986059135e-4,
    +.9810825646924729426157171547487e-8, -.1809129475572494194263306266719e-10,
    +.6221098041892605227126015543416e-13};

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

inline double chebyshev_eval(double x, const double* a, int n) {
    if (x < -1.1 || x > 1.1) return kNaN;
    const double twox = x * 2;
    double b0 = 0, b1 = 0, b2 = 0;
    for (int i = 1; i <= n; i++) {
        b2 = b1;
        b1 = b0;
        b0 = twox * b1 - b2 + a[n - i];
    }
    return (b0 - b2) * 0.5;
}

// The correction log(gamma(x)) - [(x - 0.5)*log(x) - x + log(sqrt(2*pi))], x >= 10.
inline double lgammacor(double x) {
    constexpr double xbig = 94906265.62425156;
    if (x < 10) return kNaN;
    if (x < xbig) {
        const double tmp = 10 / x;
        return chebyshev_eval(tmp * tmp * 2 - 1, kAlgmcs, 5) / x;
    }
    return 1 / (x * 12);
}

// sin(pi*x), exact at the multiples of 0.5.
inline double sinpi(double x) {
    if (!std::isfinite(x)) return kNaN;
    x = std::fmod(x, 2.);
    if (x <= -1) {
        x += 2.;
    } else if (x > 1.) {
        x -= 2.;
    }
    if (x == 0. || x == 1.) return 0.;
    if (x == 0.5) return 1.;
    if (x == -0.5) return -1.;
    return std::sin(kPi * x);
}

inline double lgammafn(double x);

// log(n!) - [log(sqrt(2*pi*n)) + n*log(n) - n], the error in Stirling's
// approximation.
inline double stirlerr(double n) {
    if (n <= 15.0) {
        const double nn = n + n;
        if (nn == static_cast<double>(static_cast<int>(nn))) {
            return kSferrHalves[static_cast<int>(nn)];
        }
        return lgammafn(n + 1.) - (n + 0.5) * std::log(n) + n - kLnSqrt2Pi;
    }
    const double nn = n * n;
    if (n > 500.0) return (kS0 - kS1 / nn) / n;
    if (n > 80.0) return (kS0 - (kS1 - kS2 / nn) / nn) / n;
    if (n > 35.0) return (kS0 - (kS1 - (kS2 - kS3 / nn) / nn) / nn) / n;
    return (kS0 - (kS1 - (kS2 - (kS3 - kS4 / nn) / nn) / nn) / nn) / n;
}

inline double gammafn(double x) {
    constexpr double xmin = -170.5674972726612;
    constexpr double xmax = 171.61447887182298;
    constexpr double xsml = 2.2474362225598545e-308;
    if (std::isnan(x)) return x;
    if (x == 0 || (x < 0 && x == std::round(x))) return kNaN;

    double y = std::fabs(x);
    double value;
    if (y <= 10) {
        // gamma(1 + y) for 0 <= y < 1, then the recursion
        int n = static_cast<int>(x);
        if (x < 0) --n;
        y = x - n;
        --n;
        value = chebyshev_eval(y * 2 - 1, kGamcs, 22) + .9375;
        if (n == 0) return value;
        if (n < 0) {
            if (y < xsml) return x > 0 ? kInf : -kInf;
            n = -n;
            for (int i = 0; i < n; i++) value /= (x + i);
            return value;
        }
        for (int i = 1; i <= n; i++) value *= (y + i);
        return value;
    }
    if (x > xmax) return kInf;
    if (x < xmin) return 0.;
    if (y <= 50 && y == static_cast<int>(y)) {
        value = 1.;
        for (int i = 2; i < y; i++) value *= i;
    } else {
        // R tests 2*y == (int)2*y, which casts the 2 only, so stirlerr() is
        // always taken here
        value = std::exp((y - 0.5) * std::log(y) - y + kLnSqrt2Pi + stirlerr(y));
    }
    if (x > 0) return value;
    const double sinpiy = sinpi(y);
    if (sinpiy == 0) return kInf;
    return -kPi / (y * sinpiy * value);
}

inline double lgammafn(double x) {
    constexpr double xmax = 2.5327372760800758e+305;
    if (std::isnan(x)) return x;
    if (x <= 0 && x == std::trunc(x)) return kInf;
    const double y = std::fabs(x);
    if (y < 1e-306) return -std::log(y);
    if (y <= 10) return std::log(std::fabs(gammafn(x)));
    if (y > xmax) return kInf;
    if (x > 0) {
        if (x > 1e17) return x * (std::log(x) - 1.);
        if (x > 4934720.) return kLnSqrt2Pi + (x - 0.5) * std::log(x) - x;
        return kLnSqrt2Pi + (x - 0.5) * std::log(x) - x + lgammacor(x);
    }
    const double sinpiy = std::fabs(sinpi(y));
    if (sinpiy == 0) return kNaN;
    return kLnSqrtPid2 + (x - 0.5) * std::log(y) - x - std::log(sinpiy) - lgammacor(y);
}

// R's x^y: R_POW takes x*x for y == 2, and libm's pow otherwise.
inline double r_pow(double x, double y) { return y == 2.0 ? x * x : std::pow(x, y); }

// The continued fraction of sum_{k=0}^Inf x^k/(i+k*d), to a relative tolerance
// eps. R/nmath/pgamma.c.
inline double logcf(double x, double i, double d, double eps) {
    constexpr double scalefactor = 0x1p256;  // 2^256, R's SQR(SQR(SQR(2^32)))
    double c1 = 2 * d;
    double c2 = i + d;
    double c4 = c2 + d;
    double a1 = c2;
    double b1 = i * (c2 - i * x);
    double b2 = d * d * x;
    double a2 = c4 * c2 - b2;
    b2 = c4 * b1 - i * b2;
    while (std::fabs(a2 * b1 - a1 * b2) > std::fabs(eps * b1 * b2)) {
        double c3 = c2 * c2 * x;
        c2 += d;
        c4 += d;
        a1 = c4 * a2 - c3 * a1;
        b1 = c4 * b2 - c3 * b1;

        c3 = c1 * c1 * x;
        c1 += d;
        c4 += d;
        a2 = c4 * a1 - c3 * a2;
        b2 = c4 * b1 - c3 * b2;

        if (std::fabs(b2) > scalefactor) {
            a1 /= scalefactor;
            b1 /= scalefactor;
            a2 /= scalefactor;
            b2 /= scalefactor;
        } else if (std::fabs(b2) < 1 / scalefactor) {
            a1 *= scalefactor;
            b1 *= scalefactor;
            a2 *= scalefactor;
            b2 *= scalefactor;
        }
    }
    return a2 / b2;
}

// log(1+x) - x, accurate also for small x. R/nmath/pgamma.c.
inline double log1pmx(double x) {
    constexpr double minLog1Value = -0.79149064;
    if (x > 1 || x < minLog1Value) return std::log1p(x) - x;
    // -.791 <= x <= 1: expand in [x/(2+x)]^2 =: y
    const double r = x / (2 + x), y = r * r;
    if (std::fabs(x) < 1e-2) {
        constexpr double two = 2;
        return r * ((((two / 9 * y + two / 7) * y + two / 5) * y + two / 3) * y - x);
    }
    constexpr double tol_logcf = 1e-14;
    return r * (2 * y * logcf(y, 3, 2, tol_logcf) - x);
}

}  // namespace greybox_nmath

#endif  // GREYBOX_NMATH_H
