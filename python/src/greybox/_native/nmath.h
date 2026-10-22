// R's gamma functions, ported from R's nmath sources (R 4.6.1):
//   src/nmath/gamma.c, lgamma.c, lgammacor.c, chebyshev.c, stirlerr.c,
//   and logcf() and log1pmx() of pgamma.c,
// and, at the end, the distribution functions that R's noncentral t and
// chi-squared need, taken from R 4.6.1's nmath sources.
//
// std::tgamma, std::lgamma, math.gamma and SciPy's gamma all round differently
// from R in the last bit on most arguments. The Generalised Normal density
// takes gamma(1/shape), and the likelihood surfaces it enters are flat enough
// for that bit to move an optimiser, so the Python densities use these.

#ifndef GREYBOX_NMATH_H
#define GREYBOX_NMATH_H

#include <cfloat>
#include <climits>
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
// approximation, as R >= 4.4: lgamma() of the C library for 1 <= n <= 5.25,
// lgamma1p() below 1 and up to 17 terms of the series up to 23.5.
inline double lgamma1p(double a);

inline double stirlerr(double n) {
    constexpr double kS5 = 0.0019175269175269175269175262;
    constexpr double kS6 = 0.0064102564102564102564102561;
    constexpr double kS7 = 0.029550653594771241830065352;
    constexpr double kS8 = 0.17964437236883057316493850;
    constexpr double kS9 = 1.3924322169059011164274315;
    constexpr double kS10 = 13.402864044168391994478957;
    constexpr double kS11 = 156.84828462600201730636509;
    constexpr double kS12 = 2193.1033333333333333333333;
    constexpr double kS13 = 36108.771253724989357173269;
    constexpr double kS14 = 691472.26885131306710839498;
    constexpr double kS15 = 15238221.539407416192283370;
    constexpr double kS16 = 382900751.39141414141414141;
    constexpr double kLn2Pi = 1.837877066409345483560659472811;
    double nn;
    if (n <= 23.5) {
        nn = n + n;
        if (n <= 15. && nn == static_cast<double>(static_cast<int>(nn))) {
            return kSferrHalves[static_cast<int>(nn)];
        }
        if (n <= 5.25) {
            if (n >= 1.) {
                const double l_n = std::log(n);
                return std::lgamma(n) + n * (1 - l_n) + std::ldexp(l_n - kLn2Pi, -1);
            }
            return lgamma1p(n) - (n + 0.5) * std::log(n) + n - kLnSqrt2Pi;
        }
        nn = n * n;
        // clang-format off
        if (n > 12.8) return (kS0-(kS1-(kS2-(kS3-(kS4-(kS5 -kS6/nn)/nn)/nn)/nn)/nn)/nn)/n;
        if (n > 12.3) return (kS0-(kS1-(kS2-(kS3-(kS4-(kS5-(kS6 -kS7/nn)/nn)/nn)/nn)/nn)/nn)/nn)/n;
        if (n > 8.9) return (kS0-(kS1-(kS2-(kS3-(kS4-(kS5-(kS6-(kS7 -kS8/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/n;
        if (n > 7.3) return (kS0-(kS1-(kS2-(kS3-(kS4-(kS5-(kS6-(kS7-(kS8-(kS9-kS10/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/n;
        if (n > 6.6) return (kS0-(kS1-(kS2-(kS3-(kS4-(kS5-(kS6-(kS7-(kS8-(kS9-(kS10-(kS11-kS12/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/n;
        if (n > 6.1) return (kS0-(kS1-(kS2-(kS3-(kS4-(kS5-(kS6-(kS7-(kS8-(kS9-(kS10-(kS11-(kS12-(kS13-kS14/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/n;
        return (kS0-(kS1-(kS2-(kS3-(kS4-(kS5-(kS6-(kS7-(kS8-(kS9-(kS10-(kS11-(kS12-(kS13-(kS14-(kS15-kS16/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/n;
        // clang-format on
    }
    nn = n * n;
    if (n > 15.7e6) return kS0 / n;
    if (n > 6180) return (kS0 - kS1 / nn) / n;
    if (n > 205) return (kS0 - (kS1 - kS2 / nn) / nn) / n;
    if (n > 86) return (kS0 - (kS1 - (kS2 - kS3 / nn) / nn) / nn) / n;
    if (n > 27) return (kS0 - (kS1 - (kS2 - (kS3 - kS4 / nn) / nn) / nn) / nn) / n;
    return (kS0 - (kS1 - (kS2 - (kS3 - (kS4 - kS5 / nn) / nn) / nn) / nn) / nn) / n;
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

// ---------------------------------------------------------------------------
// R's distribution functions, from R 4.6.1's src/nmath: the normal (dnorm.c,
// pnorm.c, qnorm.c), the gamma and chi-squared (dpois.c, bd0.c, dgamma.c,
// pgamma.c, qgamma.c, [dpq]chisq.c), the Beta (pbeta.c, toms708.c, lbeta.c),
// the t (dt.c, pt.c, qt.c, cospi.c) and the noncentral t and chi-squared
// (dnt.c, pnt.c, qnt.c, dnchisq.c, pnchisq.c, qnchisq.c). The code is R's, with
// its order of operations and its long double accumulators, so that the
// results are R's to the bit; only the debugging output and the warnings are
// left out. The macros of R's nmath.h, dpq.h and Rmath.h that it uses are
// defined here and undefined at the end of this section.

#define LDOUBLE long double
#define Rboolean int
#define TRUE 1
#define FALSE 0
#define ISNAN(x) (std::isnan(x) != 0)
#define R_FINITE(x) std::isfinite(x)
#define R_forceint(x) std::nearbyint(x)
#define ML_POSINF kInf
#define ML_NEGINF (-kInf)
#define ML_NAN kNaN
#define ML_WARNING(x, s) \
    do {                 \
    } while (0)
#define ML_WARN_return_NAN \
    { return ML_NAN; }
#define MATHLIB_WARNING(...) \
    do {                     \
    } while (0)
#define MATHLIB_WARNING3 MATHLIB_WARNING
#define MATHLIB_WARNING4 MATHLIB_WARNING
#define MATHLIB_WARNING5 MATHLIB_WARNING
#define MATHLIB_WARNING6 MATHLIB_WARNING
#define _(String) (String)
#define Rf_d1mach(i) (0.5 * DBL_EPSILON)                        // d1mach(3) only
#define Rf_i1mach(i) ((i) == 16 ? DBL_MAX_EXP : DBL_MIN_EXP)  // i1mach(15:16)
#define M_SQRT_32 5.656854249492380195206754896838
#define M_1_SQRT_2PI 0.398942280401432677939946059934
#define M_SQRT_2dPI 0.797884560802865355879892119869
#define M_LN_SQRT_PI 0.572364942924700087071713675677
#define M_LN_SQRT_2PI 0.918938533204672741780329736406
#define M_2PI 6.283185307179586476925286766559
#define M_SQRT_PI 1.772453850905516027298167483341
#define R_M_PI 3.141592653589793238462643383280
#define R_M_1_PI 0.318309886183790671537767526745
#define R_M_PI_2 1.570796326794896619231321691640
#define R_M_LN2 0.693147180559945309417232121458
#define R_M_LN10 2.302585092994045684017991454684
#define R_M_SQRT2 1.414213562373095048801688724210

inline double fmax2(double x, double y) {
    if (ISNAN(x) || ISNAN(y)) return x + y;
    return (x < y) ? y : x;
}

inline double fmin2(double x, double y) {
    if (ISNAN(x) || ISNAN(y)) return x + y;
    return (x < y) ? x : y;
}

// ---- R/nmath/dpq.h
/* Utilities for `dpq' handling (density/probability/quantile) */

/* give_log in "d";  log_p in "p" & "q" : */
#define give_log log_p
							/* "DEFAULT" */
							/* --------- */
#define R_D__0	(log_p ? ML_NEGINF : 0.)		/* 0 */
#define R_D__1	(log_p ? 0. : 1.)			/* 1 */
#define R_DT_0	(lower_tail ? R_D__0 : R_D__1)		/* 0 */
#define R_DT_1	(lower_tail ? R_D__1 : R_D__0)		/* 1 */
#define R_D_half (log_p ? -R_M_LN2 : 0.5)		// 1/2 (lower- or upper tail)


/* Use 0.5 - p + 0.5 to perhaps gain 1 bit of accuracy */
#define R_D_Lval(p)	(lower_tail ? (p) : (0.5 - (p) + 0.5))	/*  p  */
#define R_D_Cval(p)	(lower_tail ? (0.5 - (p) + 0.5) : (p))	/*  1 - p */

#define R_D_val(x)	(log_p	? log(x) : (x))		/*  x  in pF(x,..) */
#define R_D_qIv(p)	(log_p	? exp(p) : (p))		/*  p  in qF(p,..) */
#define R_D_exp(x)	(log_p	?  (x)	 : exp(x))	/* exp(x) */
#define R_D_log(p)	(log_p	?  (p)	 : log(p))	/* log(p) */
#define R_D_Clog(p)	(log_p	? log1p(-(p)) : (0.5 - (p) + 0.5)) /* [log](1-p) */

// log(1 - exp(x))  in more stable form than log1p(- R_D_qIv(x)) :
#define R_Log1_Exp(x)   ((x) > -R_M_LN2 ? log(-expm1(x)) : log1p(-exp(x)))

/* log(1-exp(x)):  R_D_LExp(x) == (log1p(- R_D_qIv(x))) but even more stable:*/
#define R_D_LExp(x)     (log_p ? R_Log1_Exp(x) : log1p(-x))

#define R_DT_val(x)	(lower_tail ? R_D_val(x)  : R_D_Clog(x))
#define R_DT_Cval(x)	(lower_tail ? R_D_Clog(x) : R_D_val(x))

/*#define R_DT_qIv(p)	R_D_Lval(R_D_qIv(p))		 *  p  in qF ! */
#define R_DT_qIv(p)	(log_p ? (lower_tail ? exp(p) : - expm1(p)) \
			       : R_D_Lval(p))

/*#define R_DT_CIv(p)	R_D_Cval(R_D_qIv(p))		 *  1 - p in qF */
#define R_DT_CIv(p)	(log_p ? (lower_tail ? -expm1(p) : exp(p)) \
			       : R_D_Cval(p))

#define R_DT_exp(x)	R_D_exp(R_D_Lval(x))		/* exp(x) */
#define R_DT_Cexp(x)	R_D_exp(R_D_Cval(x))		/* exp(1 - x) */

#define R_DT_log(p)	(lower_tail? R_D_log(p) : R_D_LExp(p))/* log(p) in qF */
#define R_DT_Clog(p)	(lower_tail? R_D_LExp(p): R_D_log(p))/* log(1-p) in qF*/
#define R_DT_Log(p)	(lower_tail? (p) : R_Log1_Exp(p))
// ==   R_DT_log when we already "know" log_p == TRUE


#define R_Q_P01_check(p)			\
    if ((log_p	&& p > 0) ||			\
	(!log_p && (p < 0 || p > 1)) )		\
	ML_WARN_return_NAN

/* Do the boundaries exactly for q*() functions :
 * Often  _LEFT_ = ML_NEGINF , and very often _RIGHT_ = ML_POSINF;
 *
 * R_Q_P01_boundaries(p, _LEFT_, _RIGHT_)  :<==>
 *
 *     R_Q_P01_check(p);
 *     if (p == R_DT_0) return _LEFT_ ;
 *     if (p == R_DT_1) return _RIGHT_;
 *
 * the following implementation should be more efficient (less tests):
 */
#define R_Q_P01_boundaries(p, _LEFT_, _RIGHT_)		\
    if (log_p) {					\
	if(p > 0)					\
	    ML_WARN_return_NAN;				\
	if(p == 0) /* upper bound*/			\
	    return lower_tail ? _RIGHT_ : _LEFT_;	\
	if(p == ML_NEGINF)				\
	    return lower_tail ? _LEFT_ : _RIGHT_;	\
    }							\
    else { /* !log_p */					\
	if(p < 0 || p > 1)				\
	    ML_WARN_return_NAN;				\
	if(p == 0)					\
	    return lower_tail ? _LEFT_ : _RIGHT_;	\
	if(p == 1)					\
	    return lower_tail ? _RIGHT_ : _LEFT_;	\
    }

#define R_P_bounds_01(x, x_min, x_max)	\
    if(x <= x_min) return R_DT_0;		\
    if(x >= x_max) return R_DT_1
/* is typically not quite optimal for (-Inf,Inf) where
 * you'd rather have */
#define R_P_bounds_Inf_01(x)			\
    if(!R_FINITE(x)) {				\
	if (x > 0) return R_DT_1;		\
	/* x < 0 */return R_DT_0;		\
    }



/* additions for density functions (C.Loader) */
#define R_D_fexp(f,x)     (give_log ? -0.5*log(f)+(x) : exp(x)/sqrt(f))
// version working with rf := sqrt(f) [avoiding overflow in computation of f in the caller]
#define R_D_rtxp(rf,x)    (give_log ? -log(rf)+(x) : exp(x)/(rf))

/* [neg]ative or [non int]eger : */
#define R_D_negInonint(x) (x < 0. || R_nonint(x))

// for discrete d<distr>(x, ...) :
#define R_D_nonint_check(x)				\
   if(R_nonint(x)) {					\
       MATHLIB_WARNING(_("non-integer x = %f"), x);	\
	return R_D__0;					\
   }

inline double bd0(double x, double np) ;
inline void ebd0(double x, double M, double *yh, double *yl) ;
inline double lbeta(double a, double b) ;
inline double Rtanpi(double x) ;
inline double dnorm(double x, double mu, double sigma, int give_log) ;
inline double pnorm(double x, double mu, double sigma, int lower_tail, int log_p) ;
inline void pnorm_both(double x, double *cum, double *ccum, int i_tail, int log_p) ;
inline double qnorm(double p, double mu, double sigma, int lower_tail, int log_p) ;
inline double dpois_raw(double x, double lambda, int give_log) ;
inline double dgamma(double x, double shape, double scale, int give_log) ;
inline double lgamma1p (double a) ;
inline double logspace_add (double logx, double logy) ;
inline double logspace_sub (double logx, double logy) ;
inline double dpois_wrap (double x_plus_1, double lambda, int give_log) ;
inline double pgamma_smallx (double x, double alph, int lower_tail, int log_p) ;
inline double pd_upper_series (double x, double y, int log_p) ;
inline double pd_lower_cf (double y, double d) ;
inline double pd_lower_series (double lambda, double y) ;
inline double dpnorm (double x, int lower_tail, double lp) ;
inline double ppois_asymp (double x, double lambda, int lower_tail, int log_p) ;
inline double pgamma_raw (double x, double alph, int lower_tail, int log_p) ;
inline double pgamma(double x, double alph, double scale, int lower_tail, int log_p) ;
inline double qchisq_appr(double p, double nu, double g , int lower_tail, int log_p, double tol ) ;
inline double qgamma(double p, double alpha, double scale, int lower_tail, int log_p) ;
inline double pchisq(double x, double df, int lower_tail, int log_p) ;
inline double qchisq(double p, double df, int lower_tail, int log_p) ;
inline double dchisq(double x, double df, int give_log) ;
inline void bratio(double a, double b, double x, double y, double *w, double *w1, int *ierr, int log_p) ;
inline double fpser(double a, double b, double x, double eps, int log_p) ;
inline double apser(double a, double b, double x, double eps) ;
inline double bpser(double a, double b, double x, double eps, int log_p) ;
inline double bup(double a, double b, double x, double y, int n, double eps, int give_log) ;
inline double bfrac(double a, double b, double x, double y, double lambda, double eps, int log_p) ;
inline double brcomp(double a, double b, double x, double y, int log_p) ;
inline double brcmp1(int mu, double a, double b, double x, double y, int give_log) ;
inline void bgrat(double a, double b, double x, double y, double *w, double eps, int *ierr, Rboolean log_w) ;
inline double grat_r(double a, double x, double log_r, double eps) ;
inline double basym(double a, double b, double lambda, double eps, int log_p) ;
inline double exparg(int l) ;
inline double esum(int mu, double x, int give_log) ;
inline double rexpm1(double x) ;
inline double alnrel(double a) ;
inline double rlog1(double x) ;
inline double erf__(double x) ;
inline double erfc1(int ind, double x) ;
inline double gam1(double a) ;
inline double gamln1(double a) ;
inline double psi(double x) ;
inline double betaln(double a0, double b0) ;
inline double gsumln(double a, double b) ;
inline double bcorr(double a0, double b0) ;
inline double algdiv(double a, double b) ;
inline double gamln(double a) ;
inline double pbeta_raw(double x, double a, double b, int lower_tail, int log_p) ;
inline double pbeta(double x, double a, double b, int lower_tail, int log_p) ;
inline double dt(double x, double n, int give_log) ;
inline double pt(double x, double n, int lower_tail, int log_p) ;
inline double qt(double p, double ndf, int lower_tail, int log_p) ;
inline double pnt(double t, double df, double ncp, int lower_tail, int log_p) ;
inline double qnt(double p, double df, double ncp, int lower_tail, int log_p) ;
inline double dnt(double x, double df, double ncp, int give_log) ;
inline double pnchisq(double x, double df, double ncp, int lower_tail, int log_p) ;
inline double pnchisq_raw(double x, double f, double theta , double errmax, double reltol, int itrmax, Rboolean lower_tail, Rboolean log_p) ;
inline double qnchisq(double p, double df, double ncp, int lower_tail, int log_p) ;
inline double dnchisq(double x, double df, double ncp, int give_log) ;

// ---- R/nmath/bd0.c

inline double  bd0(double x, double np)
{
    if(!R_FINITE(x) || !R_FINITE(np) || np == 0.0) ML_WARN_return_NAN;

    if (fabs(x-np) < 0.1*(x+np)) {
    	double d = x - np,
	    v = d/(x+np);
	if((d != 0.0)  && (v == 0.0)) {  // v has underflown to 0 (as  x+np = inf)
	    double
		x_ = ldexp(x, -2),
		n_ = ldexp(np,-2);
	    v = (x_ - n_)/(x_ + n_);
	}
	double s = ldexp(d, -1) * v; // was d * v
	if(fabs(ldexp(s, 1)) < DBL_MIN) return ldexp(s, 1);
	double ej = x * v; // as 2*x*v could overflow:  v > 1/2  <==> ej = 2xv > x
	v *= v; // "v = v^2"
	for (int j = 1; j < 1000; j++) { /* Taylor series; 1000: no infinite loop
					    as |v| < .1,  v^2000 is "zero" */
	    ej *= v;// = x v^(2j+1)
	    double s_ = s;
	    s += ej/((j<<1)+1);
	    if (s == s_) { /* last term was effectively 0 */
		return ldexp(s, 1); // 2*s ; as we dropped '2 *' above
	    }
	}
	/* ---- the following should _never_ happen ------------ */
	MATHLIB_WARNING4("bd0(%g, %g): T.series failed to converge in 1000 it.; s=%g, ej/(2j+1)=%g\n",
			 x, np, s, ej/((1000<<1)+1));
    }
    /* else:  | x - np |  is not too small */
    /* NB: x/np |--> inf (overflow)  doesn't happen when called from rpois_raw() */
#define lg_x_n (R_FINITE(x/np) ? log(x/np) : (log(x) - log(np)))
    return (x > np) ? x*(lg_x_n -1.) + np
	            : x* lg_x_n + np -x;
#undef lg_x_n
}


// ebd0(): R Bugzilla PR#15628 -- proposed accuracy improvement by Morten Welinder

/*
 * A table of logs for scaling purposes.  Each value has four parts with
 * 23 bits in each.  That means each part can be multiplied by a double
 * with at most 30 bits set and not have any rounding error.  Note, that
 * the first entry is log(2).
 *
 * Entry i is associated with the value r = 0.5 + i / 256.0.  The
 * argument to log is p/q where q=1024 and p=floor(q / r + 0.5).
 * Thus r*p/q is close to 1.
 */
static const float bd0_scale[128 + 1][4] = {
	{ +0x1.62e430p-1, -0x1.05c610p-29, -0x1.950d88p-54, +0x1.d9cc02p-79 }, /* 128: log(2048/1024.) */
	{ +0x1.5ee02cp-1, -0x1.6dbe98p-25, -0x1.51e540p-50, +0x1.2bfa48p-74 }, /* 129: log(2032/1024.) */
	{ +0x1.5ad404p-1, +0x1.86b3e4p-26, +0x1.9f6534p-50, +0x1.54be04p-74 }, /* 130: log(2016/1024.) */
	{ +0x1.570124p-1, -0x1.9ed750p-25, -0x1.f37dd0p-51, +0x1.10b770p-77 }, /* 131: log(2001/1024.) */
	{ +0x1.5326e4p-1, -0x1.9b9874p-25, -0x1.378194p-49, +0x1.56feb2p-74 }, /* 132: log(1986/1024.) */
	{ +0x1.4f4528p-1, +0x1.aca70cp-28, +0x1.103e74p-53, +0x1.9c410ap-81 }, /* 133: log(1971/1024.) */
	{ +0x1.4b5bd8p-1, -0x1.6a91d8p-25, -0x1.8e43d0p-50, -0x1.afba9ep-77 }, /* 134: log(1956/1024.) */
	{ +0x1.47ae54p-1, -0x1.abb51cp-25, +0x1.19b798p-51, +0x1.45e09cp-76 }, /* 135: log(1942/1024.) */
	{ +0x1.43fa00p-1, -0x1.d06318p-25, -0x1.8858d8p-49, -0x1.1927c4p-75 }, /* 136: log(1928/1024.) */
	{ +0x1.3ffa40p-1, +0x1.1a427cp-25, +0x1.151640p-53, -0x1.4f5606p-77 }, /* 137: log(1913/1024.) */
	{ +0x1.3c7c80p-1, -0x1.19bf48p-34, +0x1.05fc94p-58, -0x1.c096fcp-82 }, /* 138: log(1900/1024.) */
	{ +0x1.38b320p-1, +0x1.6b5778p-25, +0x1.be38d0p-50, -0x1.075e96p-74 }, /* 139: log(1886/1024.) */
	{ +0x1.34e288p-1, +0x1.d9ce1cp-25, +0x1.316eb8p-49, +0x1.2d885cp-73 }, /* 140: log(1872/1024.) */
	{ +0x1.315124p-1, +0x1.c2fc60p-29, -0x1.4396fcp-53, +0x1.acf376p-78 }, /* 141: log(1859/1024.) */
	{ +0x1.2db954p-1, +0x1.720de4p-25, -0x1.d39b04p-49, -0x1.f11176p-76 }, /* 142: log(1846/1024.) */
	{ +0x1.2a1b08p-1, -0x1.562494p-25, +0x1.a7863cp-49, +0x1.85dd64p-73 }, /* 143: log(1833/1024.) */
	{ +0x1.267620p-1, +0x1.3430e0p-29, -0x1.96a958p-56, +0x1.f8e636p-82 }, /* 144: log(1820/1024.) */
	{ +0x1.23130cp-1, +0x1.7bebf4p-25, +0x1.416f1cp-52, -0x1.78dd36p-77 }, /* 145: log(1808/1024.) */
	{ +0x1.1faa34p-1, +0x1.70e128p-26, +0x1.81817cp-50, -0x1.c2179cp-76 }, /* 146: log(1796/1024.) */
	{ +0x1.1bf204p-1, +0x1.3a9620p-28, +0x1.2f94c0p-52, +0x1.9096c0p-76 }, /* 147: log(1783/1024.) */
	{ +0x1.187ce4p-1, -0x1.077870p-27, +0x1.655a80p-51, +0x1.eaafd6p-78 }, /* 148: log(1771/1024.) */
	{ +0x1.1501c0p-1, -0x1.406cacp-25, -0x1.e72290p-49, +0x1.5dd800p-73 }, /* 149: log(1759/1024.) */
	{ +0x1.11cb80p-1, +0x1.787cd0p-25, -0x1.efdc78p-51, -0x1.5380cep-77 }, /* 150: log(1748/1024.) */
	{ +0x1.0e4498p-1, +0x1.747324p-27, -0x1.024548p-51, +0x1.77a5a6p-75 }, /* 151: log(1736/1024.) */
	{ +0x1.0b036cp-1, +0x1.690c74p-25, +0x1.5d0cc4p-50, -0x1.c0e23cp-76 }, /* 152: log(1725/1024.) */
	{ +0x1.077070p-1, -0x1.a769bcp-27, +0x1.452234p-52, +0x1.6ba668p-76 }, /* 153: log(1713/1024.) */
	{ +0x1.04240cp-1, -0x1.a686acp-27, -0x1.ef46b0p-52, -0x1.5ce10cp-76 }, /* 154: log(1702/1024.) */
	{ +0x1.00d22cp-1, +0x1.fc0e10p-25, +0x1.6ee034p-50, -0x1.19a2ccp-74 }, /* 155: log(1691/1024.) */
	{ +0x1.faf588p-2, +0x1.ef1e64p-27, -0x1.26504cp-54, -0x1.b15792p-82 }, /* 156: log(1680/1024.) */
	{ +0x1.f4d87cp-2, +0x1.d7b980p-26, -0x1.a114d8p-50, +0x1.9758c6p-75 }, /* 157: log(1670/1024.) */
	{ +0x1.ee1414p-2, +0x1.2ec060p-26, +0x1.dc00fcp-52, +0x1.f8833cp-76 }, /* 158: log(1659/1024.) */
	{ +0x1.e7e32cp-2, -0x1.ac796cp-27, -0x1.a68818p-54, +0x1.235d02p-78 }, /* 159: log(1649/1024.) */
	{ +0x1.e108a0p-2, -0x1.768ba4p-28, -0x1.f050a8p-52, +0x1.00d632p-82 }, /* 160: log(1638/1024.) */
	{ +0x1.dac354p-2, -0x1.d3a6acp-30, +0x1.18734cp-57, -0x1.f97902p-83 }, /* 161: log(1628/1024.) */
	{ +0x1.d47424p-2, +0x1.7dbbacp-31, -0x1.d5ada4p-56, +0x1.56fcaap-81 }, /* 162: log(1618/1024.) */
	{ +0x1.ce1af0p-2, +0x1.70be7cp-27, +0x1.6f6fa4p-51, +0x1.7955a2p-75 }, /* 163: log(1608/1024.) */
	{ +0x1.c7b798p-2, +0x1.ec36ecp-26, -0x1.07e294p-50, -0x1.ca183cp-75 }, /* 164: log(1598/1024.) */
	{ +0x1.c1ef04p-2, +0x1.c1dfd4p-26, +0x1.888eecp-50, -0x1.fd6b86p-75 }, /* 165: log(1589/1024.) */
	{ +0x1.bb7810p-2, +0x1.478bfcp-26, +0x1.245b8cp-50, +0x1.ea9d52p-74 }, /* 166: log(1579/1024.) */
	{ +0x1.b59da0p-2, -0x1.882b08p-27, +0x1.31573cp-53, -0x1.8c249ap-77 }, /* 167: log(1570/1024.) */
	{ +0x1.af1294p-2, -0x1.b710f4p-27, +0x1.622670p-51, +0x1.128578p-76 }, /* 168: log(1560/1024.) */
	{ +0x1.a925d4p-2, -0x1.0ae750p-27, +0x1.574ed4p-51, +0x1.084996p-75 }, /* 169: log(1551/1024.) */
	{ +0x1.a33040p-2, +0x1.027d30p-29, +0x1.b9a550p-53, -0x1.b2e38ap-78 }, /* 170: log(1542/1024.) */
	{ +0x1.9d31c0p-2, -0x1.5ec12cp-26, -0x1.5245e0p-52, +0x1.2522d0p-79 }, /* 171: log(1533/1024.) */
	{ +0x1.972a34p-2, +0x1.135158p-30, +0x1.a5c09cp-56, +0x1.24b70ep-80 }, /* 172: log(1524/1024.) */
	{ +0x1.911984p-2, +0x1.0995d4p-26, +0x1.3bfb5cp-50, +0x1.2c9dd6p-75 }, /* 173: log(1515/1024.) */
	{ +0x1.8bad98p-2, -0x1.1d6144p-29, +0x1.5b9208p-53, +0x1.1ec158p-77 }, /* 174: log(1507/1024.) */
	{ +0x1.858b58p-2, -0x1.1b4678p-27, +0x1.56cab4p-53, -0x1.2fdc0cp-78 }, /* 175: log(1498/1024.) */
	{ +0x1.7f5fa0p-2, +0x1.3aaf48p-27, +0x1.461964p-51, +0x1.4ae476p-75 }, /* 176: log(1489/1024.) */
	{ +0x1.79db68p-2, -0x1.7e5054p-26, +0x1.673750p-51, -0x1.a11f7ap-76 }, /* 177: log(1481/1024.) */
	{ +0x1.744f88p-2, -0x1.cc0e18p-26, -0x1.1e9d18p-50, -0x1.6c06bcp-78 }, /* 178: log(1473/1024.) */
	{ +0x1.6e08ecp-2, -0x1.5d45e0p-26, -0x1.c73ec8p-50, +0x1.318d72p-74 }, /* 179: log(1464/1024.) */
	{ +0x1.686c80p-2, +0x1.e9b14cp-26, -0x1.13bbd4p-50, -0x1.efeb1cp-78 }, /* 180: log(1456/1024.) */
	{ +0x1.62c830p-2, -0x1.a8c70cp-27, -0x1.5a1214p-51, -0x1.bab3fcp-79 }, /* 181: log(1448/1024.) */
	{ +0x1.5d1bdcp-2, -0x1.4fec6cp-31, +0x1.423638p-56, +0x1.ee3feep-83 }, /* 182: log(1440/1024.) */
	{ +0x1.576770p-2, +0x1.7455a8p-26, -0x1.3ab654p-50, -0x1.26be4cp-75 }, /* 183: log(1432/1024.) */
	{ +0x1.5262e0p-2, -0x1.146778p-26, -0x1.b9f708p-52, -0x1.294018p-77 }, /* 184: log(1425/1024.) */
	{ +0x1.4c9f08p-2, +0x1.e152c4p-26, -0x1.dde710p-53, +0x1.fd2208p-77 }, /* 185: log(1417/1024.) */
	{ +0x1.46d2d8p-2, +0x1.c28058p-26, -0x1.936284p-50, +0x1.9fdd68p-74 }, /* 186: log(1409/1024.) */
	{ +0x1.41b940p-2, +0x1.cce0c0p-26, -0x1.1a4050p-50, +0x1.bc0376p-76 }, /* 187: log(1402/1024.) */
	{ +0x1.3bdd24p-2, +0x1.d6296cp-27, +0x1.425b48p-51, -0x1.cddb2cp-77 }, /* 188: log(1394/1024.) */
	{ +0x1.36b578p-2, -0x1.287ddcp-27, -0x1.2d0f4cp-51, +0x1.38447ep-75 }, /* 189: log(1387/1024.) */
	{ +0x1.31871cp-2, +0x1.2a8830p-27, +0x1.3eae54p-52, -0x1.898136p-77 }, /* 190: log(1380/1024.) */
	{ +0x1.2b9304p-2, -0x1.51d8b8p-28, +0x1.27694cp-52, -0x1.fd852ap-76 }, /* 191: log(1372/1024.) */
	{ +0x1.265620p-2, -0x1.d98f3cp-27, +0x1.a44338p-51, -0x1.56e85ep-78 }, /* 192: log(1365/1024.) */
	{ +0x1.211254p-2, +0x1.986160p-26, +0x1.73c5d0p-51, +0x1.4a861ep-75 }, /* 193: log(1358/1024.) */
	{ +0x1.1bc794p-2, +0x1.fa3918p-27, +0x1.879c5cp-51, +0x1.16107cp-78 }, /* 194: log(1351/1024.) */
	{ +0x1.1675ccp-2, -0x1.4545a0p-26, +0x1.c07398p-51, +0x1.f55c42p-76 }, /* 195: log(1344/1024.) */
	{ +0x1.111ce4p-2, +0x1.f72670p-37, -0x1.b84b5cp-61, +0x1.a4a4dcp-85 }, /* 196: log(1337/1024.) */
	{ +0x1.0c81d4p-2, +0x1.0c150cp-27, +0x1.218600p-51, -0x1.d17312p-76 }, /* 197: log(1331/1024.) */
	{ +0x1.071b84p-2, +0x1.fcd590p-26, +0x1.a3a2e0p-51, +0x1.fe5ef8p-76 }, /* 198: log(1324/1024.) */
	{ +0x1.01ade4p-2, -0x1.bb1844p-28, +0x1.db3cccp-52, +0x1.1f56fcp-77 }, /* 199: log(1317/1024.) */
	{ +0x1.fa01c4p-3, -0x1.12a0d0p-29, -0x1.f71fb0p-54, +0x1.e287a4p-78 }, /* 200: log(1311/1024.) */
	{ +0x1.ef0adcp-3, +0x1.7b8b28p-28, -0x1.35bce4p-52, -0x1.abc8f8p-79 }, /* 201: log(1304/1024.) */
	{ +0x1.e598ecp-3, +0x1.5a87e4p-27, -0x1.134bd0p-51, +0x1.c2cebep-76 }, /* 202: log(1298/1024.) */
	{ +0x1.da85d8p-3, -0x1.df31b0p-27, +0x1.94c16cp-57, +0x1.8fd7eap-82 }, /* 203: log(1291/1024.) */
	{ +0x1.d0fb80p-3, -0x1.bb5434p-28, -0x1.ea5640p-52, -0x1.8ceca4p-77 }, /* 204: log(1285/1024.) */
	{ +0x1.c765b8p-3, +0x1.e4d68cp-27, +0x1.5b59b4p-51, +0x1.76f6c4p-76 }, /* 205: log(1279/1024.) */
	{ +0x1.bdc46cp-3, -0x1.1cbb50p-27, +0x1.2da010p-51, +0x1.eb282cp-75 }, /* 206: log(1273/1024.) */
	{ +0x1.b27980p-3, -0x1.1b9ce0p-27, +0x1.7756f8p-52, +0x1.2ff572p-76 }, /* 207: log(1266/1024.) */
	{ +0x1.a8bed0p-3, -0x1.bbe874p-30, +0x1.85cf20p-56, +0x1.b9cf18p-80 }, /* 208: log(1260/1024.) */
	{ +0x1.9ef83cp-3, +0x1.2769a4p-27, -0x1.85bda0p-52, +0x1.8c8018p-79 }, /* 209: log(1254/1024.) */
	{ +0x1.9525a8p-3, +0x1.cf456cp-27, -0x1.7137d8p-52, -0x1.f158e8p-76 }, /* 210: log(1248/1024.) */
	{ +0x1.8b46f8p-3, +0x1.11b12cp-30, +0x1.9f2104p-54, -0x1.22836ep-78 }, /* 211: log(1242/1024.) */
	{ +0x1.83040cp-3, +0x1.2379e4p-28, +0x1.b71c70p-52, -0x1.990cdep-76 }, /* 212: log(1237/1024.) */
	{ +0x1.790ed4p-3, +0x1.dc4c68p-28, -0x1.910ac8p-52, +0x1.dd1bd6p-76 }, /* 213: log(1231/1024.) */
	{ +0x1.6f0d28p-3, +0x1.5cad68p-28, +0x1.737c94p-52, -0x1.9184bap-77 }, /* 214: log(1225/1024.) */
	{ +0x1.64fee8p-3, +0x1.04bf88p-28, +0x1.6fca28p-52, +0x1.8884a8p-76 }, /* 215: log(1219/1024.) */
	{ +0x1.5c9400p-3, +0x1.d65cb0p-29, -0x1.b2919cp-53, +0x1.b99bcep-77 }, /* 216: log(1214/1024.) */
	{ +0x1.526e60p-3, -0x1.c5e4bcp-27, -0x1.0ba380p-52, +0x1.d6e3ccp-79 }, /* 217: log(1208/1024.) */
	{ +0x1.483bccp-3, +0x1.9cdc7cp-28, -0x1.5ad8dcp-54, -0x1.392d3cp-83 }, /* 218: log(1202/1024.) */
	{ +0x1.3fb25cp-3, -0x1.a6ad74p-27, +0x1.5be6b4p-52, -0x1.4e0114p-77 }, /* 219: log(1197/1024.) */
	{ +0x1.371fc4p-3, -0x1.fe1708p-27, -0x1.78864cp-52, -0x1.27543ap-76 }, /* 220: log(1192/1024.) */
	{ +0x1.2cca10p-3, -0x1.4141b4p-28, -0x1.ef191cp-52, +0x1.00ee08p-76 }, /* 221: log(1186/1024.) */
	{ +0x1.242310p-3, +0x1.3ba510p-27, -0x1.d003c8p-51, +0x1.162640p-76 }, /* 222: log(1181/1024.) */
	{ +0x1.1b72acp-3, +0x1.52f67cp-27, -0x1.fd6fa0p-51, +0x1.1a3966p-77 }, /* 223: log(1176/1024.) */
	{ +0x1.10f8e4p-3, +0x1.129cd8p-30, +0x1.31ef30p-55, +0x1.a73e38p-79 }, /* 224: log(1170/1024.) */
	{ +0x1.08338cp-3, -0x1.005d7cp-27, -0x1.661a9cp-51, +0x1.1f138ap-79 }, /* 225: log(1165/1024.) */
	{ +0x1.fec914p-4, -0x1.c482a8p-29, -0x1.55746cp-54, +0x1.99f932p-80 }, /* 226: log(1160/1024.) */
	{ +0x1.ed1794p-4, +0x1.d06f00p-29, +0x1.75e45cp-53, -0x1.d0483ep-78 }, /* 227: log(1155/1024.) */
	{ +0x1.db5270p-4, +0x1.87d928p-32, -0x1.0f52a4p-57, +0x1.81f4a6p-84 }, /* 228: log(1150/1024.) */
	{ +0x1.c97978p-4, +0x1.af1d24p-29, -0x1.0977d0p-60, -0x1.8839d0p-84 }, /* 229: log(1145/1024.) */
	{ +0x1.b78c84p-4, -0x1.44f124p-28, -0x1.ef7bc4p-52, +0x1.9e0650p-78 }, /* 230: log(1140/1024.) */
	{ +0x1.a58b60p-4, +0x1.856464p-29, +0x1.c651d0p-55, +0x1.b06b0cp-79 }, /* 231: log(1135/1024.) */
	{ +0x1.9375e4p-4, +0x1.5595ecp-28, +0x1.dc3738p-52, +0x1.86c89ap-81 }, /* 232: log(1130/1024.) */
	{ +0x1.814be4p-4, -0x1.c073fcp-28, -0x1.371f88p-53, -0x1.5f4080p-77 }, /* 233: log(1125/1024.) */
	{ +0x1.6f0d28p-4, +0x1.5cad68p-29, +0x1.737c94p-53, -0x1.9184bap-78 }, /* 234: log(1120/1024.) */
	{ +0x1.60658cp-4, -0x1.6c8af4p-28, +0x1.d8ef74p-55, +0x1.c4f792p-80 }, /* 235: log(1116/1024.) */
	{ +0x1.4e0110p-4, +0x1.146b5cp-29, +0x1.73f7ccp-54, -0x1.d28db8p-79 }, /* 236: log(1111/1024.) */
	{ +0x1.3b8758p-4, +0x1.8b1b70p-28, -0x1.20aca4p-52, -0x1.651894p-76 }, /* 237: log(1106/1024.) */
	{ +0x1.28f834p-4, +0x1.43b6a4p-30, -0x1.452af8p-55, +0x1.976892p-80 }, /* 238: log(1101/1024.) */
	{ +0x1.1a0fbcp-4, -0x1.e4075cp-28, +0x1.1fe618p-52, +0x1.9d6dc2p-77 }, /* 239: log(1097/1024.) */
	{ +0x1.075984p-4, -0x1.4ce370p-29, -0x1.d9fc98p-53, +0x1.4ccf12p-77 }, /* 240: log(1092/1024.) */
	{ +0x1.f0a30cp-5, +0x1.162a68p-37, -0x1.e83368p-61, -0x1.d222a6p-86 }, /* 241: log(1088/1024.) */
	{ +0x1.cae730p-5, -0x1.1a8f7cp-31, -0x1.5f9014p-55, +0x1.2720c0p-79 }, /* 242: log(1083/1024.) */
	{ +0x1.ac9724p-5, -0x1.e8ee08p-29, +0x1.a7de04p-54, -0x1.9bba74p-78 }, /* 243: log(1079/1024.) */
	{ +0x1.868a84p-5, -0x1.ef8128p-30, +0x1.dc5eccp-54, -0x1.58d250p-79 }, /* 244: log(1074/1024.) */
	{ +0x1.67f950p-5, -0x1.ed684cp-30, -0x1.f060c0p-55, -0x1.b1294cp-80 }, /* 245: log(1070/1024.) */
	{ +0x1.494accp-5, +0x1.a6c890p-32, -0x1.c3ad48p-56, -0x1.6dc66cp-84 }, /* 246: log(1066/1024.) */
	{ +0x1.22c71cp-5, -0x1.8abe2cp-32, -0x1.7e7078p-56, -0x1.ddc3dcp-86 }, /* 247: log(1061/1024.) */
	{ +0x1.03d5d8p-5, +0x1.79cfbcp-31, -0x1.da7c4cp-58, +0x1.4e7582p-83 }, /* 248: log(1057/1024.) */
	{ +0x1.c98d18p-6, +0x1.a01904p-31, -0x1.854164p-55, +0x1.883c36p-79 }, /* 249: log(1053/1024.) */
	{ +0x1.8b31fcp-6, -0x1.356500p-30, +0x1.c3ab48p-55, +0x1.b69bdap-80 }, /* 250: log(1049/1024.) */
	{ +0x1.3cea44p-6, +0x1.a352bcp-33, -0x1.8865acp-57, -0x1.48159cp-81 }, /* 251: log(1044/1024.) */
	{ +0x1.fc0a8cp-7, -0x1.e07f84p-32, +0x1.e7cf6cp-58, +0x1.3a69c0p-82 }, /* 252: log(1040/1024.) */
	{ +0x1.7dc474p-7, +0x1.f810a8p-31, -0x1.245b5cp-56, -0x1.a1f4f8p-80 }, /* 253: log(1036/1024.) */
	{ +0x1.fe02a8p-8, -0x1.4ef988p-32, +0x1.1f86ecp-57, +0x1.20723cp-81 }, /* 254: log(1032/1024.) */
	{ +0x1.ff00acp-9, -0x1.d4ef44p-33, +0x1.2821acp-63, +0x1.5a6d32p-87 }, /* 255: log(1028/1024.) */
	{ 0, 0, 0, 0 } /* log(1024/1024) = log(1) = 0 */
};


/*
 * Compute x * log (x / M) + (M - x)
 * aka -x * log1pmx ((M - x) / x)
 *
 * Deliver the result back in two parts, *yh and *yl.
 */
inline void  ebd0(double x, double M, double *yh, double *yl)
{
	const int Sb = 10;
	const double S = 1u << Sb; // = 2^10 = 1024
	const int N = 128; // == ? == G_N_ELEMENTS(bd0_scale) - 1; <<<< FIXME:

	*yl = *yh = 0;

	if (x == M) return;
	if (x == 0) { *yh = M;         return; }
	if (M == 0) { *yh = ML_POSINF; return; }

	if (M/x == ML_POSINF) { *yh = M; return; }//  as when (x == 0)

	int e;
	// NB: M/x overflow handled above; underflow should be handled by fg = Inf
	double r = frexp (M / x, &e); // => r in  [0.5, 1) and 'e' (int) such that  M/x = r * 2^e

	// prevent later overflow
	if (R_M_LN2 * ((double) -e)  > 1. + DBL_MAX / x) { *yh = ML_POSINF; return; }

	int i = (int) floor ((r - 0.5) * (2 * N) + 0.5);
	// now,  0 <= i <= N
	double f = floor (S / (0.5 + i / (2.0 * N)) + 0.5);
	double fg = ldexp (f, -(e + Sb)); // ldexp(f, E) := f * 2^E
	if (fg == ML_POSINF) {
	    *yh = fg; return;
	}
	/* We now have (M * fg / x) close to 1.  */

	/*
	 * We need to compute this:
	 * (x/M)^x * exp(M-x) =
	 * (M/x)^-x * exp(M-x) =
	 * (M*fg/x)^-x * (fg)^x * exp(M-x) =
	 * (M*fg/x)^-x * (fg)^x * exp(M*fg-x) * exp(M-M*fg)
	 *
	 * In log terms:
	 * log((x/M)^x * exp(M-x)) =
	 * log((M*fg/x)^-x * (fg)^x * exp(M*fg-x) * exp(M-M*fg)) =
	 * log((M*fg/x)^-x * exp(M*fg-x)) + x*log(fg) + (M-M*fg) =
	 * -x*log1pmx((M*fg-x)/x) + x*log(fg) + M - M*fg =
	 *
	 * Note, that fg has at most 10 bits.  If M and x are suitably
	 * "nice" -- such as being integers or half-integers -- then
	 * we can compute M*fg as well as x * bd0_scale[.][.] without
	 * rounding errors.
	 */

#define ADD1(d_) do {				\
   volatile double d = (d_);			\
	    double d1 = floor (d + 0.5);	\
	    double d2 = d - d1;/* in [-.5,.5) */ \
	    *yh += d1;				\
	    *yl += d2;				\
	} while(0)

	ADD1(-x * log1pmx ((M * fg - x) / x));
        if(fg == 1) return;
	// else (fg != 1) :
	for (int j = 0; j < 4; j++) {
	    ADD1( x * bd0_scale[i][j]);     // handles  x*log(fg*2^e)
	    ADD1(-x * bd0_scale[0][j] * e); // handles  x*log(1/ 2^e)
	    //                        ^^^ at end prevents overflow in  ebd0(1e307, 1e300)
            if(!R_FINITE(*yh)) { *yh = ML_POSINF; *yl = 0; return; }
	}

	ADD1(M);
	ADD1(-M * fg);
}

#undef ADD1

// ---- R/nmath/lbeta.c

inline double lbeta(double a, double b)
{
    double corr, p, q;

    if(ISNAN(a) || ISNAN(b))
	return a + b;
    p = q = a;
    if(b < p) p = b;/* := min(a,b) */
    if(b > q) q = b;/* := max(a,b) */

    /* both arguments must be >= 0 */
    if (p < 0)
	ML_WARN_return_NAN
    else if (p == 0) {
	return ML_POSINF;
    }
    else if (!R_FINITE(q)) { /* q == +Inf */
	return ML_NEGINF;
    }

    if (p >= 10) {
	/* p and q are big. */
	corr = lgammacor(p) + lgammacor(q) - lgammacor(p + q);
	return log(q) * -0.5 + M_LN_SQRT_2PI + corr
		+ (p - 0.5) * log(p / (p + q)) + q * log1p(-p / (p + q));
    }
    else if (q >= 10) {
	/* p is small, but q is big. */
	corr = lgammacor(q) - lgammacor(p + q);
	return lgammafn(p) + corr + p - p * log(p + q)
		+ (q - 0.5) * log1p(-p / (p + q));
    }
    else {
	/* p and q are small: p <= q < 10. */
	/* R change for very small args */
	if (p < 1e-306) return lgamma(p) + (lgamma(q) - lgamma(p+q));
	else return log(gammafn(p) * (gammafn(q) / gammafn(p + q)));
    }
}

// ---- R/nmath/cospi.c




// tan(pi * x)  -- exact when x = k/4  for all integer k and half-values give NaN
// ----------- e.g. used in ../main/arithmetic.c : 
inline double Rtanpi(double x)
{
    if (ISNAN(x)) return x;
    if(!R_FINITE(x)) ML_WARN_return_NAN;

    x = fmod(x, 1.); // tan(pi(x + k)) == tan(pi x)  for all integer k
    // map (-1,1] --> (-1/2, 1/2] :
    if(x <= -0.5) x++; else if(x > 0.5) x--;
    return (x == 0.) ? 0. :
	((x ==  0.5 ) ? ML_NAN :
	((x ==  0.25) ?  1. :
	((x == -0.25) ? -1. :
			tan(R_M_PI * x)
	    )));
}


// ---- R/nmath/dnorm.c

inline double dnorm(double x, double mu, double sigma, int give_log)
{
    if (ISNAN(x) || ISNAN(mu) || ISNAN(sigma))
	return x + mu + sigma;
    if (sigma < 0) ML_WARN_return_NAN;
    if(!R_FINITE(sigma)) return R_D__0;
    if(!R_FINITE(x) && mu == x) return ML_NAN;/* x-mu is NaN */
    if (sigma == 0) 
	return (x == mu) ? ML_POSINF : R_D__0;
    x = (x - mu) / sigma;

    if(!R_FINITE(x)) return R_D__0;

    x = fabs (x);
    if (x >= 2 * sqrt(DBL_MAX)) return R_D__0;
    if (give_log)
	return -(M_LN_SQRT_2PI + 0.5 * x * x + log(sigma));
    //  M_1_SQRT_2PI = 1 / sqrt(2 * pi)
    // more accurate, less fast :
    if (x < 5)    return M_1_SQRT_2PI * exp(-0.5 * x * x) / sigma;

    /* ELSE:

     * x*x  may lose upto about two digits accuracy for "large" x
     * Morten Welinder's proposal for PR#15620
     * https://bugs.r-project.org/show_bug.cgi?id=15620

     * -- 1 --  No hoop jumping when we underflow to zero anyway:

     *  -x^2/2 <         log(2)*.Machine$double.min.exp  <==>
     *     x   > sqrt(-2*log(2)*.Machine$double.min.exp) =IEEE= 37.64031
     * but "thanks" to denormalized numbers, underflow happens a bit later,
     *  effective.D.MIN.EXP <- with(.Machine, double.min.exp + double.ulp.digits)
     * for IEEE, DBL_MIN_EXP is -1022 but "effective" is -1074
     * ==> boundary = sqrt(-2*log(2)*(.Machine$double.min.exp + .Machine$double.ulp.digits))
     *              =IEEE=  38.58601
     * [on one x86_64 platform, effective boundary a bit lower: 38.56804]
     */
    if (x > sqrt(-2*R_M_LN2*(DBL_MIN_EXP + 1-DBL_MANT_DIG))) return 0.;

    /* Now, to get full accuracy, split x into two parts,
     *  x = x1+x2, such that |x2| <= 2^-16.
     * Assuming that we are using IEEE doubles, that means that
     * x1*x1 is error free for x<1024 (but we have x < 38.6 anyway).

     * If we do not have IEEE this is still an improvement over the naive formula.
     */
    double x1 = //  R_forceint(x * 65536) / 65536 =
	ldexp( R_forceint(ldexp(x, 16)), -16);
    double x2 = x - x1;
    return M_1_SQRT_2PI / sigma *
	(exp(-0.5 * x1 * x1) * exp( (-0.5 * x2 - x1) * x2 ) );
}

// ---- R/nmath/pnorm.c
inline double pnorm(double x, double mu, double sigma, int lower_tail, int log_p)
{
    double p, cp;

    /* Note: The structure of these checks has been carefully thought through.
     * For example, if x == mu and sigma == 0, we get the correct answer 1.
     */
    if(ISNAN(x) || ISNAN(mu) || ISNAN(sigma))
	return x + mu + sigma;
    if(!R_FINITE(x) && mu == x) return ML_NAN;/* x-mu is NaN */
    if (sigma <= 0) {
	if(sigma < 0) ML_WARN_return_NAN;
	/* sigma = 0 : */
	return (x < mu) ? R_DT_0 : R_DT_1;
    }
    p = (x - mu) / sigma;
    if(!R_FINITE(p))
	return (x < mu) ? R_DT_0 : R_DT_1;
    x = p;

    pnorm_both(x, &p, &cp, (lower_tail ? 0 : 1), log_p);

    return(lower_tail ? p : cp);
}


inline void pnorm_both(double x, double *cum, double *ccum, int i_tail, int log_p)
{
/* i_tail in {0,1,2} means: "lower", "upper", or "both" :
   if(lower) return  *cum := P[X <= x]
   if(upper) return *ccum := P[X >  x] = 1 - P[X <= x]
*/
    const static double a[5] = {
	2.2352520354606839287,
	161.02823106855587881,
	1067.6894854603709582,
	18154.981253343561249,
	0.065682337918207449113
    };
    const static double b[4] = {
	47.20258190468824187,
	976.09855173777669322,
	10260.932208618978205,
	45507.789335026729956
    };
    const static double c[9] = {
	0.39894151208813466764,
	8.8831497943883759412,
	93.506656132177855979,
	597.27027639480026226,
	2494.5375852903726711,
	6848.1904505362823326,
	11602.651437647350124,
	9842.7148383839780218,
	1.0765576773720192317e-8
    };
    const static double d[8] = {
	22.266688044328115691,
	235.38790178262499861,
	1519.377599407554805,
	6485.558298266760755,
	18615.571640885098091,
	34900.952721145977266,
	38912.003286093271411,
	19685.429676859990727
    };
    const static double p[6] = {
	0.21589853405795699,
	0.1274011611602473639,
	0.022235277870649807,
	0.001421619193227893466,
	2.9112874951168792e-5,
	0.02307344176494017303
    };
    const static double q[5] = {
	1.28426009614491121,
	0.468238212480865118,
	0.0659881378689285515,
	0.00378239633202758244,
	7.29751555083966205e-5
    };

    double xden, xnum, temp, del, eps, xsq, y;
    int i, lower, upper;

    if(ISNAN(x)) { *cum = *ccum = x; return; }

    /* Consider changing these : */
    eps = DBL_EPSILON * 0.5;

    /* i_tail in {0,1,2} =^= {lower, upper, both} */
    lower = i_tail != 1;
    upper = i_tail != 0;

    y = fabs(x);
    if (y <= 0.67448975) { /* qnorm(3/4) = .6744.... -- earlier had 0.66291 */
	if (y > eps) {
	    xsq = x * x;
	    xnum = a[4] * xsq;
	    xden = xsq;
	    for (i = 0; i < 3; ++i) {
		xnum = (xnum + a[i]) * xsq;
		xden = (xden + b[i]) * xsq;
	    }
	} else xnum = xden = 0.0;

	temp = x * (xnum + a[3]) / (xden + b[3]);
	if(lower)  *cum = 0.5 + temp;
	if(upper) *ccum = 0.5 - temp;
	if(log_p) {
	    if(lower)  *cum = log(*cum);
	    if(upper) *ccum = log(*ccum);
	}
    }
    else if (y <= M_SQRT_32) {

	/* Evaluate pnorm for 0.674.. = qnorm(3/4) < |x| <= sqrt(32) ~= 5.657 */

	xnum = c[8] * y;
	xden = y;
	for (i = 0; i < 7; ++i) {
	    xnum = (xnum + c[i]) * y;
	    xden = (xden + d[i]) * y;
	}
	temp = (xnum + c[7]) / (xden + d[7]);

#define d_2(_x_) ldexp(_x_, -1) // == (_x_ / 2 )  "perfectly"

#define do_del(X)						\
	xsq = ldexp(trunc(ldexp(X, 4)), -4);			\
	del = (X - xsq) * (X + xsq);				\
	if(log_p) {						\
	    *cum = (-xsq * d_2(xsq)) -d_2(del) + log(temp);	\
	    if((lower && x > 0.) || (upper && x <= 0.))		\
		  *ccum = log1p(-exp(-xsq * d_2(xsq)) *		\
				exp(-d_2(del)) * temp);		\
	}							\
	else {							\
	    *cum = exp(-xsq * d_2(xsq)) * exp(-d_2(del)) * temp;\
	    *ccum = 1.0 - *cum;					\
	}

#define swap_tail						\
	if (x > 0.) {/* swap  ccum <--> cum */			\
	    temp = *cum; if(lower) *cum = *ccum; *ccum = temp;	\
	}

	do_del(y);
	swap_tail;
    }

/* else	  |x| > sqrt(32) = 5.657 :
 * the next two case differentiations were really for lower=T, log=F
 * Particularly	 *not*	for  log_p !

 * Cody had (-37.5193 < x  &&  x < 8.2924) ; R originally had y < 50
 *
 * Note that we do want symmetry(0), lower/upper -> hence use y
 */
    else if((log_p && y < 1e170) /* avoid underflow below */
	/*  ^^^^^ MM FIXME: could speed up for log_p and  y := |x| >> 5.657 !
	 * Then, make use of  Abramowitz & Stegun, 26.2.13, p.932,  something like

	 * Even smarter: work with   example(pnormAsymp, package="DPQ")

	 xsq = x*x;

	 if(xsq * DBL_EPSILON < 1.)
	    del = (1. - (1. - 5./(xsq+6.)) / (xsq+4.)) / (xsq+2.);
	 else
	    del = 0.;
	 *cum = -.5*xsq - M_LN_SQRT_2PI - log(x) + log1p(-del);
	 *ccum = log1p(-exp(*cum)); /.* ~ log(1) = 0 *./

 	 swap_tail;

	 Yes, but xsq might be infinite;
 	 well, actually  x = -1.34..e154 = -sqrt(DBL_MAX) already overflows x^2
	 The largest x for which  x/2*x is finite is
	 x = +/- 1.89615038e154 ~= sqrt(2) * sqrt(.Machine$double.xmax)

	 NB: allowing "DENORMS" ==> boundaries at +/- 38.4674  <--> qnorm(log(2^-1074), log.p=TRUE)
	 --                               rather than 37.5193 (up to R 4.4.x)
	*/
	    || (lower && -38.4674 < x  &&  x < 8.2924)
	    || (upper && -8.2924  < x  &&  x < 38.4674)
	) {

	/* Evaluate pnorm for x in (-37.5, -5.657) union (5.657, 37.5) */
	xsq = 1.0 / (x * x); /* (1./x)*(1./x) might be better */
	xnum = p[5] * xsq;
	xden = xsq;
	for (i = 0; i < 4; ++i) {
	    xnum = (xnum + p[i]) * xsq;
	    xden = (xden + q[i]) * xsq;
	}
	temp = xsq * (xnum + p[4]) / (xden + q[4]);
	temp = (M_1_SQRT_2PI - temp) / y;

	do_del(x);
	swap_tail;
    } else { /* large |x| such that probs are 0 or 1 */
	if(x > 0) {	*cum = R_D__1; *ccum = R_D__0;	}
	else {	        *cum = R_D__0; *ccum = R_D__1;	}
    }


    return;
}

// ---- R/nmath/qnorm.c

inline double qnorm(double p, double mu, double sigma, int lower_tail, int log_p)
{
    double p_, q, r, val;

    if (ISNAN(p) || ISNAN(mu) || ISNAN(sigma))
	return p + mu + sigma;
    R_Q_P01_boundaries(p, ML_NEGINF, ML_POSINF);

    if(sigma  < 0)	ML_WARN_return_NAN;
    if(sigma == 0)	return mu;

    p_ = R_DT_qIv(p);/* real lower_tail prob. p */
    q = p_ - 0.5;



/*-- use AS 241 --- */
/* double ppnd16_(double *p, long *ifault)*/
/*      ALGORITHM AS241  APPL. STATIST. (1988) VOL. 37, NO. 3

        Produces the normal deviate Z corresponding to a given lower
        tail area of P; Z is accurate to about 1 part in 10**16.

        (original fortran code used PARAMETER(..) for the coefficients
         and provided hash codes for checking them...)
*/
    if (fabs(q) <= .425) {/* |p~ - 0.5| <= .425  <==> 0.075 <= p~ <= 0.925 */
        r = .180625 - q * q; // = .425^2 - q^2  >= 0
	val =
            q * (((((((r * 2509.0809287301226727 +
                       33430.575583588128105) * r + 67265.770927008700853) * r +
                     45921.953931549871457) * r + 13731.693765509461125) * r +
                   1971.5909503065514427) * r + 133.14166789178437745) * r +
                 3.387132872796366608)
            / (((((((r * 5226.495278852854561 +
                     28729.085735721942674) * r + 39307.89580009271061) * r +
                   21213.794301586595867) * r + 5394.1960214247511077) * r +
                 687.1870074920579083) * r + 42.313330701600911252) * r + 1.);
    }
    else { /* closer than 0.075 from {0,1} boundary :
	    *  r := log(p~);  p~ = min(p, 1-p) < 0.075 :  */
	double lp;
	if(log_p && ((lower_tail && q <= 0) || (!lower_tail && q > 0))) {
	    lp = p;
	} else {
	    lp = log( (q > 0) ? R_DT_CIv(p) /* 1-p */ : p_ /* = R_DT_Iv(p) ^=  p */);
	}
	// r = sqrt( - log(min(p,1-p)) )  <==>  min(p, 1-p) = exp( - r^2 ) :
        r = sqrt(-lp);
        if (r <= 5.) { /* <==> min(p,1-p) >= exp(-25) ~= 1.3888e-11 */
            r += -1.6;
            val = (((((((r * 7.7454501427834140764e-4 +
                       .0227238449892691845833) * r + .24178072517745061177) *
                     r + 1.27045825245236838258) * r +
                    3.64784832476320460504) * r + 5.7694972214606914055) *
                  r + 4.6303378461565452959) * r +
                 1.42343711074968357734)
                / (((((((r *
                         1.05075007164441684324e-9 + 5.475938084995344946e-4) *
                        r + .0151986665636164571966) * r +
                       .14810397642748007459) * r + .68976733498510000455) *
                     r + 1.6763848301838038494) * r +
                    2.05319162663775882187) * r + 1.);
        }
	else if(r <= 27) { /* p is very close to  0 or 1: r in (5, 27] :
		*  r >   5 <==> min(p,1-p)  < exp(-25) = 1.3888..e-11
		*  r <= 27 <==> min(p,1-p) >= exp(-27^2) = exp(-729) ~= 2.507972e-317
		* i.e., we are just barely in the range where min(p, 1-p) has not yet underflowed to zero.
		*/
	    // Wichura, p.478: minimax rational approx R_3(t) is for 5 <= t <= 27  (t :== r)
            r += -5.;
            val = (((((((r * 2.01033439929228813265e-7 +
                       2.71155556874348757815e-5) * r +
                      .0012426609473880784386) * r + .026532189526576123093) *
                    r + .29656057182850489123) * r +
                   1.7848265399172913358) * r + 5.4637849111641143699) *
                 r + 6.6579046435011037772)
                / (((((((r *
                         2.04426310338993978564e-15 + 1.4215117583164458887e-7)*
                        r + 1.8463183175100546818e-5) * r +
                       7.868691311456132591e-4) * r + .0148753612908506148525)
                     * r + .13692988092273580531) * r +
                    .59983220655588793769) * r + 1.);
        }
        else { // r > 27: p is *really* close to 0 or 1 .. practically only when log_p =TRUE
	    if(r >= 6.4e8) { // p is *very extremely* close to 0 or 1
		// Using the asymptotical formula ("0-th order"): qn = sqrt(2*s)
		val = r * R_M_SQRT2;
	    } else {
		double s2 = -ldexp(lp, 1), // = -2*lp = 2s
		    x2 = s2 - log(M_2PI * s2); // = xs_1
		// if(r >= 36000.)  # <==> s >= 36000^2   use x2 = xs_1  above
		if(r < 36000.) {
		    x2 = s2 - log(M_2PI * x2) - 2./(2. + x2); // == xs_2
		    if(r < 840.) { // 27 < r < 840
			x2 = s2 - log(M_2PI * x2) + 2*log1p(- (1 - 1/(4 + x2))/(2. + x2)); // == xs_3
			if(r < 109.) { // 27 < r < 109
			  x2 = s2 - log(M_2PI * x2) +
			      2*log1p(- (1 - (1 - 5/(6 + x2))/(4. + x2))/(2. + x2)); // == xs_4
			  if(r < 55.) { // 27 < r < 55
			    x2 = s2 - log(M_2PI * x2) +
			      2*log1p(- (1 - (1 - (5 - 9/(8. + x2))/(6. + x2))/(4. + x2))/(2. + x2)); // == xs_5
			  }
			}
		    }
		}
                val = sqrt(x2);
	    }
	}
	if(q < 0.0)
	    val = -val;
    }
    return mu + sigma * val;
}

// ---- R/nmath/dpois.c

#define M_SQRT_2PI	2.50662827463100050241576528481104525301  /* sqrt(2*pi) */
// sqrt(2 * Rmpfr::Const("pi", 128))
#define x_LRG           2.86111748575702815380240589208115399625e+307 /* = 2^1023 / pi */

// called also from dgamma.c, pgamma.c, dnbeta.c, dnbinom.c, dnchisq.c :
inline double dpois_raw(double x, double lambda, int give_log)
{
    /*       x >= 0 ; integer for dpois(), but not e.g. for pgamma()!
        lambda >= 0
    */
    if (lambda == 0) return( (x == 0) ? R_D__1 : R_D__0 );
    if (!R_FINITE(lambda)) return R_D__0; // including for the case where  x = lambda = +Inf
    if (x < 0) return( R_D__0 );
    if (x <= lambda * DBL_MIN) return(R_D_exp(-lambda) );
    if (lambda < x * DBL_MIN) {
	if (!R_FINITE(x)) // lambda < x = +Inf
	    return R_D__0;
	// else
	return(R_D_exp(-lambda + x*log(lambda) -lgammafn(x+1)));
    }
    // R <= 4.0.x  had   return(R_D_fexp( M_2PI*x, -stirlerr(x)-bd0(x,lambda) ));
    double yh, yl;
    ebd0 (x, lambda, &yh, &yl);
    yl += stirlerr(x);
    bool Lrg_x = (x >= x_LRG); //really large x  <==>  2*pi*x  overflows
    double r = Lrg_x
	? M_SQRT_2PI * sqrt(x) // sqrt(.): avoid overflow for very large x
	: M_2PI * x;
    return give_log
	? -yl - yh - (Lrg_x ? log(r) : 0.5 * log(r))
	: exp(-yl) * exp(-yh) / (Lrg_x ? r : sqrt(r));
}


// ---- R/nmath/dgamma.c

inline double dgamma(double x, double shape, double scale, int give_log)
{
    if (ISNAN(x) || ISNAN(shape) || ISNAN(scale))
        return x + shape + scale;
    if (shape < 0 || scale <= 0) ML_WARN_return_NAN;
    if (x < 0)
	return R_D__0;
    if (shape == 0) /* point mass at 0 */
	return (x == 0)? ML_POSINF : R_D__0;
    if (x == 0) {
	if (shape < 1) return ML_POSINF;
	if (shape > 1) return R_D__0;
	/* else */
	return give_log ? -log(scale) : 1 / scale;
    }

    double pr;
    if (shape < 1) {
	pr = dpois_raw(shape, x/scale, give_log);
	return (
	    give_log/* NB: currently *always*  shape/x > 0  if shape < 1:
		     * -- overflow to Inf happens, but underflow to 0 does NOT : */
	    ? pr + (R_FINITE(shape/x)
		    ? log(shape/x)
		    : /* shape/x overflows to +Inf */ log(shape) - log(x))
	    : pr*shape / x);
    }
    /* else  shape >= 1 */
    pr = dpois_raw(shape-1, x/scale, give_log);
    return give_log ? pr - log(scale) : pr/scale;
}

// ---- R/nmath/pgamma.c

#define SQR(x) ((x)*(x))
static const double scalefactor = SQR(SQR(SQR(4294967296.0)));
#undef SQR
static const double M_cutoff = R_M_LN2 * DBL_MAX_EXP / DBL_EPSILON;/*=3.196577e18*/




/* Compute  log(gamma(a+1))  accurately also for small a (0 < a < 0.5). */
inline double lgamma1p (double a)
{
    if (fabs (a) >= 0.5)
	return lgammafn (a + 1);

    const double eulers_const =	 0.5772156649015328606065120900824024;

    /* coeffs[i] holds (zeta(i+2)-1)/(i+2) , i = 0:(N-1), N = 40 : */
    const int N = 40;
    static const double coeffs[40] = {
	0.3224670334241132182362075833230126e-0,/* = (zeta(2)-1)/2 */
	0.6735230105319809513324605383715000e-1,/* = (zeta(3)-1)/3 */
	0.2058080842778454787900092413529198e-1,
	0.7385551028673985266273097291406834e-2,
	0.2890510330741523285752988298486755e-2,
	0.1192753911703260977113935692828109e-2,
	0.5096695247430424223356548135815582e-3,
	0.2231547584535793797614188036013401e-3,
	0.9945751278180853371459589003190170e-4,
	0.4492623673813314170020750240635786e-4,
	0.2050721277567069155316650397830591e-4,
	0.9439488275268395903987425104415055e-5,
	0.4374866789907487804181793223952411e-5,
	0.2039215753801366236781900709670839e-5,
	0.9551412130407419832857179772951265e-6,
	0.4492469198764566043294290331193655e-6,
	0.2120718480555466586923135901077628e-6,
	0.1004322482396809960872083050053344e-6,
	0.4769810169363980565760193417246730e-7,
	0.2271109460894316491031998116062124e-7,
	0.1083865921489695409107491757968159e-7,
	0.5183475041970046655121248647057669e-8,
	0.2483674543802478317185008663991718e-8,
	0.1192140140586091207442548202774640e-8,
	0.5731367241678862013330194857961011e-9,
	0.2759522885124233145178149692816341e-9,
	0.1330476437424448948149715720858008e-9,
	0.6422964563838100022082448087644648e-10,
	0.3104424774732227276239215783404066e-10,
	0.1502138408075414217093301048780668e-10,
	0.7275974480239079662504549924814047e-11,
	0.3527742476575915083615072228655483e-11,
	0.1711991790559617908601084114443031e-11,
	0.8315385841420284819798357793954418e-12,
	0.4042200525289440065536008957032895e-12,
	0.1966475631096616490411045679010286e-12,
	0.9573630387838555763782200936508615e-13,
	0.4664076026428374224576492565974577e-13,
	0.2273736960065972320633279596737272e-13,
	0.1109139947083452201658320007192334e-13/* = (zeta(40+1)-1)/(40+1) */
    };

    const double c = 0.2273736845824652515226821577978691e-12;/* zeta(N+2)-1 */
    const double tol_logcf = 1e-14;

    /* Abramowitz & Stegun 6.1.33 : for |x| < 2,
     * <==> log(gamma(1+x)) = -(log(1+x) - x) - gamma*x + x^2 * \sum_{n=0}^\infty c_n (-x)^n
     * where c_n := (Zeta(n+2) - 1)/(n+2)  = coeffs[n]
     *
     * Here, another convergence acceleration trick is used to compute
     * lgam(x) :=  sum_{n=0..Inf} c_n (-x)^n
     */
    double lgam = c * logcf(-a / 2, N + 2, 1, tol_logcf);
    for (int i = N - 1; i >= 0; i--)
	lgam = coeffs[i] - a * lgam;

    return (a * lgam - eulers_const) * a - log1pmx (a);
} /* lgamma1p */



/*
 * Compute the log of a sum from logs of terms, i.e.,
 *
 *     log (exp (logx) + exp (logy))
 *
 * without causing overflows and without throwing away large handfuls
 * of accuracy.
 */
inline double logspace_add (double logx, double logy)
{
    return fmax2 (logx, logy) + log1p (exp (-fabs (logx - logy)));
}


/*
 * Compute the log of a difference from logs of terms, i.e.,
 *
 *     log (exp (logx) - exp (logy))
 *
 * without causing overflows and without throwing away large handfuls
 * of accuracy.
 */
inline double logspace_sub (double logx, double logy)
{
    return logx + R_Log1_Exp(logy - logx);
}
# define EXP expl
# define LOG logl




/* dpois_wrap (x__1, lambda) := dpois(x__1 - 1, lambda);  where
 * dpois(k, L) := exp(-L) L^k / gamma(k+1)  {the usual Poisson probabilities}
 *
 * and  dpois*(.., give_log = TRUE) :=  log( dpois*(..) )
*/
inline double
dpois_wrap (double x_plus_1, double lambda, int give_log)
{
    if (!R_FINITE(lambda))
	return R_D__0;
    if (x_plus_1 > 1)
	return dpois_raw (x_plus_1 - 1, lambda, give_log);
    if (lambda > fabs(x_plus_1 - 1) * M_cutoff)
	return R_D_exp(-lambda - lgammafn(x_plus_1));
    else {
	double d = dpois_raw (x_plus_1, lambda, give_log);
	return give_log
	    ? d + log (x_plus_1 / lambda)
	    : d * (x_plus_1 / lambda);
    }
}

/*
 * Abramowitz and Stegun 6.5.29 [right]
 */
inline double
pgamma_smallx (double x, double alph, int lower_tail, int log_p)
{
    double sum = 0, c = alph, n = 0, term;


    /*
     * Relative to 6.5.29 all terms have been multiplied by alph
     * and the first, thus being 1, is omitted.
     */

    do {
	n++;
	c *= -x / n;
	term = c / (alph + n);
	sum += term;
    } while (fabs (term) > DBL_EPSILON * fabs (sum));

    if (lower_tail) {
	double f1 = log_p ? log1p (sum) : 1 + sum;
	double f2;
	if (alph > 1) {
	    f2 = dpois_raw (alph, x, log_p);
	    f2 = log_p ? f2 + x : f2 * exp (x);
	} else if (log_p)
	    f2 = alph * log (x) - lgamma1p (alph);
	else
	    f2 = pow (x, alph) / exp (lgamma1p (alph));
	return log_p ? f1 + f2 : f1 * f2;
    } else {
	double lf2 = alph * log (x) - lgamma1p (alph);
	if (log_p)
	    return R_Log1_Exp (log1p (sum) + lf2);
	else {
	    double f1m1 = sum;
	    double f2m1 = expm1 (lf2);
	    return -(f1m1 + f2m1 + f1m1 * f2m1);
	}
    }
} /* pgamma_smallx() */

inline double
pd_upper_series (double x, double y, int log_p)
{
    double term = x / y;
    double sum = term;

    do {
	y++;
	term *= x / y;
	sum += term;
    } while (term > sum * DBL_EPSILON);

    /* sum =  \sum_{n=1}^ oo  x^n     / (y*(y+1)*...*(y+n-1))
     *	   =  \sum_{n=0}^ oo  x^(n+1) / (y*(y+1)*...*(y+n))
     *	   =  x/y * (1 + \sum_{n=1}^oo	x^n / ((y+1)*...*(y+n)))
     *	   ~  x/y +  o(x/y)   {which happens when alph -> Inf}
     */
    return log_p ? log (sum) : sum;
}

/* Continued fraction for calculation of
 *    scaled upper-tail F_{gamma}
 *  ~=  (y / d) * [1 +  (1-y)/d +  O( ((1-y)/d)^2 ) ]
 */
inline double
pd_lower_cf (double y, double d)
{
    double f= 0.0 /* -Wall */, of, f0;
    double i, c2, c3, c4,  a1, b1,  a2, b2;

#define	NEEDED_SCALE				\
	  (b2 > scalefactor) {			\
	    a1 /= scalefactor;			\
	    b1 /= scalefactor;			\
	    a2 /= scalefactor;			\
	    b2 /= scalefactor;			\
	}

#define max_it 200000

    if (y == 0) return 0;

    f0 = y/d;
    /* Needed, e.g. for  pgamma(10^c(100,295), shape= 1.1, log=TRUE): */
    if(fabs(y - 1) < fabs(d) * DBL_EPSILON) { /* includes y < d = Inf */
	return (f0);
    }

    if(f0 > 1.) f0 = 1.;
    c2 = y;
    c4 = d; /* original (y,d), *not* potentially scaled ones!*/

    a1 = 0; b1 = 1;
    a2 = y; b2 = d;

    while NEEDED_SCALE

    i = 0; of = -1.; /* far away */
    while (i < max_it) {

	i++;	c2--;	c3 = i * c2;	c4 += 2;
	/* c2 = y - i,  c3 = i(y - i),  c4 = d + 2i,  for i odd */
	a1 = c4 * a2 + c3 * a1;
	b1 = c4 * b2 + c3 * b1;

	i++;	c2--;	c3 = i * c2;	c4 += 2;
	/* c2 = y - i,  c3 = i(y - i),  c4 = d + 2i,  for i even */
	a2 = c4 * a1 + c3 * a2;
	b2 = c4 * b1 + c3 * b2;

	if NEEDED_SCALE

	if (b2 != 0) {
	    f = a2 / b2;
	    /* convergence check: relative; "absolute" for very small f : */
	    if (fabs (f - of) <= DBL_EPSILON * fmax2(f0, fabs(f))) {
		return f;
	    }
	    of = f;
	}
    }

    MATHLIB_WARNING(" ** NON-convergence in pgamma()'s pd_lower_cf() f= %g.\n",
		    f);
    return f;/* should not happen ... */
} /* pd_lower_cf() */
#undef NEEDED_SCALE


inline double
pd_lower_series (double lambda, double y)
{
    double term = 1, sum = 0;

    while (y >= 1 && term > sum * DBL_EPSILON) {
	term *= y / lambda;
	sum += term;
	y--;
    }
    /* sum =  \sum_{n=0}^ oo  y*(y-1)*...*(y - n) / lambda^(n+1)
     *	   =  y/lambda * (1 + \sum_{n=1}^Inf  (y-1)*...*(y-n) / lambda^n)
     *	   ~  y/lambda + o(y/lambda)
     */

    if (y != floor (y)) {
	/*
	 * The series does not converge as the terms start getting
	 * bigger (besides flipping sign) for y < -lambda.
	 */
	double f;
	/* FIXME: in quite few cases, adding  term*f  has no effect (f too small)
	 *	  and is unnecessary e.g. for pgamma(4e12, 121.1) */
	f = pd_lower_cf (y, lambda + 1 - y);
	sum += term * f;
    }

    return sum;
} /* pd_lower_series() */

/*
 * Compute the following ratio with higher accuracy that would be had
 * from doing it directly.
 *
 *		 dnorm (x, 0, 1, FALSE)
 *	   ----------------------------------
 *	   pnorm (x, 0, 1, lower_tail, FALSE)
 *
 * Abramowitz & Stegun 26.2.12
 */
inline double
dpnorm (double x, int lower_tail, double lp)
{
    /*
     * So as not to repeat a pnorm call, we expect
     *
     *	 lp == pnorm (x, 0, 1, lower_tail, TRUE)
     *
     * but use it only in the non-critical case where either x is small
     * or p==exp(lp) is close to 1.
     */

    if (x < 0) {
	x = -x;
	lower_tail = !lower_tail;
    }

    if (x > 10 && !lower_tail) {
	double term = 1 / x;
	double sum = term;
	double x2 = x * x;
	double i = 1;

	do {
	    term *= -i / x2;
	    sum += term;
	    i += 2;
	} while (fabs (term) > DBL_EPSILON * sum);

	return 1 / sum;
    } else {
	double d = dnorm (x, 0., 1., FALSE);
	return d / exp (lp);
    }
}

/*
 * Asymptotic expansion to calculate the probability that Poisson variate
 * has value <= x.
 * Various assertions about this are made (without proof) at
 * http://members.aol.com/iandjmsmith/PoissonApprox.htm
 */
inline double
ppois_asymp (double x, double lambda, int lower_tail, int log_p)
{
    static const double coefs_a[8] = {
	-1e99, /* placeholder used for 1-indexing */
	2/3.,
	-4/135.,
	8/2835.,
	16/8505.,
	-8992/12629925.,
	-334144/492567075.,
	698752/1477701225.
    };

    static const double coefs_b[8] = {
	-1e99, /* placeholder */
	1/12.,
	1/288.,
	-139/51840.,
	-571/2488320.,
	163879/209018880.,
	5246819/75246796800.,
	-534703531/902961561600.
    };

    double elfb, elfb_term;
    double res12, res1_term, res1_ig, res2_term, res2_ig;
    double dfm, pt_, s2pt, f, np;
    int i;

    dfm = lambda - x;
    /* If lambda is large, the distribution is highly concentrated
       about lambda.  So representation error in x or lambda can lead
       to arbitrarily large values of pt_ and hence divergence of the
       coefficients of this approximation.
    */
    pt_ = - log1pmx (dfm / x);
    s2pt = sqrt (2 * x * pt_);
    if (dfm < 0) s2pt = -s2pt;

    res12 = 0;
    res1_ig = res1_term = sqrt (x);
    res2_ig = res2_term = s2pt;
    for (i = 1; i < 8; i++) {
	res12 += res1_ig * coefs_a[i];
	res12 += res2_ig * coefs_b[i];
	res1_term *= pt_ / i ;
	res2_term *= 2 * pt_ / (2 * i + 1);
	res1_ig = res1_ig / x + res1_term;
	res2_ig = res2_ig / x + res2_term;
    }

    elfb = x;
    elfb_term = 1;
    for (i = 1; i < 8; i++) {
	elfb += elfb_term * coefs_b[i];
	elfb_term /= x;
    }
    if (!lower_tail) elfb = -elfb;

    f = res12 / elfb;

    np = pnorm (s2pt, 0.0, 1.0, !lower_tail, log_p);

    if (log_p) {
	double n_d_over_p = dpnorm (s2pt, !lower_tail, np);
	return np + log1p (f * n_d_over_p);
    } else {
	double nd = dnorm (s2pt, 0., 1., log_p);

	return np + f * nd;
    }
} /* ppois_asymp() */


inline double pgamma_raw (double x, double alph, int lower_tail, int log_p)
{
/* Here, assume that  (x,alph) are not NA  &  alph > 0 . */

    double res;

    R_P_bounds_01(x, 0., ML_POSINF);

    if (x < 1) {
	res = pgamma_smallx (x, alph, lower_tail, log_p);
    } else if (x <= alph - 1 && x < 0.8 * (alph + 50)) {
	/* incl. large alph compared to x */
	double sum = pd_upper_series (x, alph, log_p);/* = x/alph + o(x/alph) */
	double d = dpois_wrap (alph, x, log_p);
	if (!lower_tail)
	    res = log_p
		? R_Log1_Exp (d + sum)
		: 1 - d * sum;
	else
	    res = log_p ? sum + d : sum * d;
    } else if (alph - 1 < x && alph < 0.8 * (x + 50)) {
	/* incl. large x compared to alph */
	double sum;
	double d = dpois_wrap (alph, x, log_p);
	if (alph < 1) {
	    if (x * DBL_EPSILON > 1 - alph)
		sum = R_D__1;
	    else {
		double f = pd_lower_cf (alph, x - (alph - 1)) * x / alph;
		/* = [alph/(x - alph+1) + o(alph/(x-alph+1))] * x/alph = 1 + o(1) */
		sum = log_p ? log (f) : f;
	    }
	} else {
	    sum = pd_lower_series (x, alph - 1);/* = (alph-1)/x + o((alph-1)/x) */
	    sum = log_p ? log1p (sum) : 1 + sum;
	}
	if (!lower_tail)
	    res = log_p ? sum + d : sum * d;
	else
	    res = log_p
		? R_Log1_Exp (d + sum)
		: 1 - d * sum;
    } else { /* x >= 1 and x fairly near alph. */
	res = ppois_asymp (alph - 1, x, !lower_tail, log_p);
    }

    /*
     * We lose a fair amount of accuracy to underflow in the cases
     * where the final result is very close to DBL_MIN.	 In those
     * cases, simply redo via log space.
     */
    if (!log_p && res < DBL_MIN / DBL_EPSILON) {
	/* with(.Machine, double.xmin / double.eps) #|-> 1.002084e-292 */
	return exp (pgamma_raw (x, alph, lower_tail, 1));
    } else
	return res;
}


inline double pgamma(double x, double alph, double scale, int lower_tail, int log_p)
{
    if (ISNAN(x) || ISNAN(alph) || ISNAN(scale))
	return x + alph + scale;
    if(alph < 0. || scale <= 0.)
	ML_WARN_return_NAN;
    x /= scale;
    if (ISNAN(x)) /* eg. original x = scale = +Inf */
	return x;
    if(alph == 0.) /* limit case; useful e.g. in pnchisq() */
	return (x <= 0) ? R_DT_0: R_DT_1; /* <= assert  pgamma(0,0) ==> 0 */
    return pgamma_raw (x, alph, lower_tail, log_p);
}
/* From: terra@gnome.org (Morten Welinder)
 * To: R-bugs@biostat.ku.dk
 * Cc: maechler@stat.math.ethz.ch
 * Subject: Re: [Rd] pgamma discontinuity (PR#7307)
 * Date: Tue, 11 Jan 2005 13:57:26 -0500 (EST)

 * this version of pgamma appears to be quite good and certainly a vast
 * improvement over current R code.  (I last looked at 2.0.1)  Apart from
 * type naming, this is what I have been using for Gnumeric 1.4.1.

 * This could be included into R as-is, but you might want to benefit from
 * making logcf, log1pmx, lgamma1p, and possibly logspace_add/logspace_sub
 * available to other parts of R.

 * MM: I've not (yet?) taken  logcf(), but the other four
 */

// ---- R/nmath/qgamma.c



inline double qchisq_appr(double p, double nu, double g /* = log Gamma(nu/2) */,
		   int lower_tail, int log_p, double tol /* EPS1 */)
{
#define C7	4.67
#define C8	6.66
#define C9	6.73
#define C10	13.32

    double alpha, a, c, ch, p1;
    double p2, q, t, x;

    /* test arguments and initialise */

    if (ISNAN(p) || ISNAN(nu))
	return p + nu;
    R_Q_P01_check(p);
    if (nu <= 0) ML_WARN_return_NAN;

    alpha = 0.5 * nu;/* = [pq]gamma() shape */
    c = alpha-1;

    if(nu < (-1.24)*(p1 = R_DT_log(p))) {	/* for small chi-squared */
	/* log(alpha) + g = log(alpha) + log(gamma(alpha)) =
	 *        = log(alpha*gamma(alpha)) = lgamma(alpha+1) suffers from
	 *  catastrophic cancellation when alpha << 1
	 */
	double lgam1pa = (alpha < 0.5) ? lgamma1p(alpha) : (log(alpha) + g);
	ch = exp((lgam1pa + p1)/alpha + R_M_LN2);

    } else if(nu > 0.32) {	/*  using Wilson and Hilferty estimate */

	x = qnorm(p, 0, 1, lower_tail, log_p);
	p1 = 2./(9*nu);
	ch = nu*pow(x*sqrt(p1) + 1-p1, 3);

	/* approximation for p tending to 1: */
	if( ch > 2.2*nu + 6 )
	    ch = -2*(R_DT_Clog(p) - c*log(0.5*ch) + g);

    } else { /* "small nu" : 1.24*(-log(p)) <= nu <= 0.32 */

	ch = 0.4;
	a = R_DT_Clog(p) + g + c*R_M_LN2;
	do {
	    q = ch;
	    p1 = 1. / (1+ch*(C7+ch));
	    p2 = ch*(C9+ch*(C8+ch));
	    t = -0.5 +(C7+2*ch)*p1 - (C9+ch*(C10+3*ch))/p2;
	    ch -= (1- exp(a+0.5*ch)*p2*p1)/t;
	} while(fabs(q - ch) > tol * fabs(ch));
    }

    return ch;
}

inline double qgamma(double p, double alpha, double scale, int lower_tail, int log_p)
/*			shape = alpha */
{
#define EPS1 1e-2
#define EPS2 5e-7/* final precision of AS 91 */
#define EPS_N 1e-15/* precision of Newton step / iterations */
#define LN_EPS -36.043653389117156 /* = log(.Machine$double.eps) iff IEEE_754 */

#define MAXIT 1000/* was 20 */

#define pMIN 1e-100   /* was 0.000002 = 2e-6 */
#define pMAX (1-1e-14)/* was (1-1e-12) and 0.999998 = 1 - 2e-6 */

    const static double
	i420  = 1./ 420.,
	i2520 = 1./ 2520.,
	i5040 = 1./ 5040;

    double p_, a, b, c, g, ch, ch0, p1;
    double p2, q, s1, s2, s3, s4, s5, s6, t, x;
    int i, max_it_Newton = 1;

    /* test arguments and initialise */

    if (ISNAN(p) || ISNAN(alpha) || ISNAN(scale))
	return p + alpha + scale;
    R_Q_P01_boundaries(p, 0., ML_POSINF);

    if (alpha < 0 || scale <= 0) ML_WARN_return_NAN;

    if (alpha == 0) /* all mass at 0 : */ return 0.;

    if (alpha < 1e-10) {
    /* Warning seems unnecessary now: */
	max_it_Newton = 7;/* may still be increased below */
    }

    p_ = R_DT_qIv(p);/* lower_tail prob (in any case) */

    g = lgammafn(alpha);/* log Gamma(v/2) */

    /*----- Phase I : Starting Approximation */
    ch = qchisq_appr(p, /* nu= 'df' =  */ 2*alpha, /* lgamma(nu/2)= */ g,
		     lower_tail, log_p, /* tol= */ EPS1);
    if(!R_FINITE(ch)) {
	/* forget about all iterations! */
	max_it_Newton = 0; goto END;
    }
    if(ch < EPS2) {/* Corrected according to AS 91; MM, May 25, 1999 */
	max_it_Newton = 20;
	goto END;/* and do Newton steps */
    }

    /* FIXME: This (cutoff to {0, +Inf}) is far from optimal
     * -----  when log_p or !lower_tail, but NOT doing it can be even worse */
    if(p_ > pMAX || p_ < pMIN) {
	/* did return ML_POSINF or 0.;	much better: */
	max_it_Newton = 20;
	goto END;/* and do Newton steps */
    }


/*----- Phase II: Iteration
 *	Call pgamma() [AS 239]	and calculate seven term taylor series
 */
    c = alpha-1;
    s6 = (120+c*(346+127*c)) * i5040; /* used below, is "const" */

    ch0 = ch;/* save initial approx. */
    for(i=1; i <= MAXIT; i++ ) {
	q = ch;
	p1 = 0.5*ch;
	p2 = p_ - pgamma_raw(p1, alpha, /*lower_tail*/TRUE, /*log_p*/FALSE);
	if(!R_FINITE(p2) || ch <= 0)
	    { ch = ch0; max_it_Newton = 27; goto END; }/*was  return ML_NAN;*/

	t = p2*exp(alpha*R_M_LN2+g+p1-c*log(ch));
	b = t/ch;
	a = 0.5*t - b*c;
	s1 = (210+ a*(140+a*(105+a*(84+a*(70+60*a))))) * i420;
	s2 = (420+ a*(735+a*(966+a*(1141+1278*a)))) * i2520;
	s3 = (210+ a*(462+a*(707+932*a))) * i2520;
	s4 = (252+ a*(672+1182*a) + c*(294+a*(889+1740*a))) * i5040;
	s5 = (84+2264*a + c*(1175+606*a)) * i2520;

	ch += t*(1+0.5*t*s1-b*c*(s1-b*(s2-b*(s3-b*(s4-b*(s5-b*s6))))));
	if(fabs(q - ch) < EPS2*ch)
	    goto END;
	if(fabs(q - ch) > 0.1*ch) {/* diverging? -- also forces ch > 0 */
	    if(ch < q) ch = 0.9 * q; else ch = 1.1 * q;
	}
    }
/* no convergence in MAXIT iterations -- but we add Newton now... */
/* was
 *    ML_WARNING(ME_PRECISION, "qgamma");
 * does nothing in R !*/

END:
/* PR# 2214 :	 From: Morten Welinder <terra@diku.dk>, Fri, 25 Oct 2002 16:50
   --------	 To: R-bugs@biostat.ku.dk     Subject: qgamma precision

   * With a final Newton step, double accuracy, e.g. for (p= 7e-4; nu= 0.9)
   *
   * Improved (MM): - only if rel.Err > EPS_N (= 1e-15);
   *		    - also for lower_tail = FALSE	 or log_p = TRUE
   *		    - optionally *iterate* Newton
   */
    x = 0.5*scale*ch;
    if(max_it_Newton) {
	/* always use log scale */
	if (!log_p) {
	    p = log(p);
	    log_p = TRUE;
	}
	if(x == 0) {
	    const double _1_p = 1. + 1e-7;
	    const double _1_m = 1. - 1e-7;
	    x = DBL_MIN;
	    p_ = pgamma(x, alpha, scale, lower_tail, log_p);
	    if(( lower_tail && p_ > p * _1_p) ||
	       (!lower_tail && p_ < p * _1_m))
		return(0.);
	    /* else:  continue, using x = DBL_MIN instead of  0  */
	}
	else
	    p_ = pgamma(x, alpha, scale, lower_tail, log_p);
	if(p_ == ML_NEGINF) return 0; /* PR#14710 */
	for(i = 1; i <= max_it_Newton; i++) {
	    p1 = p_ - p;
	    if(fabs(p1) < fabs(EPS_N * p))
		break;
	    /* else */
	    if((g = dgamma(x, alpha, scale, log_p)) == R_D__0) {
		break;
	    }
	    /* else :
	     * delta x = f(x)/f'(x);
	     * if(log_p) f(x) := log P(x) - p; f'(x) = d/dx log P(x) = P' / P
	     * ==> f(x)/f'(x) = f*P / P' = f*exp(p_) / P' (since p_ = log P(x))
	     */
	    t = log_p ? p1*exp(p_ - g) : p1/g ;/* = "delta x" */
	    t = lower_tail ? x - t : x + t;
	    p_ = pgamma (t, alpha, scale, lower_tail, log_p);
	    if (fabs(p_ - p) > fabs(p1) ||
		(i > 1 && fabs(p_ - p) == fabs(p1)) /* <- against flip-flop */) {
		/* no improvement */
		break;
	    } /* else : */
	    x = t;
	}
    }

    return x;
}

// ---- R/nmath/pchisq.c

inline double pchisq(double x, double df, int lower_tail, int log_p)
{
    return pgamma(x, df/2., 2., lower_tail, log_p);
}

// ---- R/nmath/qchisq.c

inline double qchisq(double p, double df, int lower_tail, int log_p)
{
    return qgamma(p, 0.5 * df, 2.0, lower_tail, log_p);
}

// ---- R/nmath/dchisq.c

inline double dchisq(double x, double df, int give_log)
{
    return dgamma(x, df / 2., 2., give_log);
}

// ---- R/nmath/toms708.c
#undef min
#define min(a,b) ((a < b)?a:b)
#undef max
#define max(a,b) ((a > b)?a:b)
/* after config.h to avoid warning on Solaris */

/**----------- DEBUGGING -------------
    (cd `R-devel-pbeta-dbg RHOME`/src/nmath ;
 	make -B CFLAGS='-DDEBUG_bratio' toms708.o ; cd ../..; make R )
*/
# define R_ifDEBUG_printf(...)

/* MM added R_D_LExp, so redefine here in terms of rexpm1 */
#undef R_Log1_Exp
#define R_Log1_Exp(x)   ((x) > -R_M_LN2 ? log(-rexpm1(x)) : log1p(-exp(x)))


inline double bfrac(double, double, double, double, double, double, int log_p);
inline void bgrat(double, double, double, double, double *, double, int *, Rboolean log_w);
inline double grat_r(double a, double x, double r, double eps);
inline double apser(double, double, double, double);
inline double bpser(double, double, double, double, int log_p);
inline double basym(double, double, double, double, int log_p);
inline double fpser(double, double, double, double, int log_p);
inline double bup(double, double, double, double, int, double, int give_log);
inline double exparg(int);
inline double psi(double);
inline double gam1(double);
inline double gamln1(double);
inline double betaln(double, double);
inline double algdiv(double, double);
inline double brcmp1(int, double, double, double, double, int give_log);
inline double brcomp(double, double, double, double, int log_p);
inline double rlog1(double);
inline double bcorr(double, double);
inline double gamln(double);
inline double alnrel(double);
inline double esum(int, double, int give_log);
inline double erf__(double);
inline double rexpm1(double);
inline double erfc1(int, double);
inline double gsumln(double, double);

/*      ALGORITHM 708, COLLECTED ALGORITHMS FROM ACM.
 *      This work published in  Transactions On Mathematical Software,
 *      vol. 18, no. 3, September 1992, pp. 360-373.
 */

/* Changes by R Core Team :
 * add log_p  and work towards gaining precision in that case
 */

inline void 
bratio(double a, double b, double x, double y, double *w, double *w1,
       int *ierr, int log_p)
{
/* -----------------------------------------------------------------------

 *	      Evaluation of the Incomplete Beta function I_x(a,b)

 *		       --------------------

 *     It is assumed that a and b are nonnegative, and that x <= 1
 *     and y = 1 - x.  Bratio assigns w and w1 the values

 *			w  = I_x(a,b)
 *			w1 = 1 - I_x(a,b)

 *     ierr is a variable that reports the status of the results.
 *     If no input errors are detected then ierr is set to 0 and
 *     w and w1 are computed. otherwise, if an error is detected,
 *     then w and w1 are assigned the value 0 and ierr is set to
 *     one of the following values ...

 *	  ierr = 1  if a or b is negative
 *	  ierr = 2  if a = b = 0
 *	  ierr = 3  if x < 0 or x > 1
 *	  ierr = 4  if y < 0 or y > 1
 *	  ierr = 5  if x + y != 1
 *	  ierr = 6  if x = a = 0
 *	  ierr = 7  if y = b = 0
 *	  ierr = 8	(not used currently)
 *	  ierr = 9  NaN in a, b, x, or y
 *	  ierr = 10     (not used currently)
 *	  ierr = 11  bgrat() error code 1 [+ warning in bgrat()]
 *	  ierr = 12  bgrat() error code 2   (no warning here)
 *	  ierr = 13  bgrat() error code 3   (no warning here)
 *	  ierr = 14  bgrat() error code 4 [+ WARNING in bgrat()]


 * --------------------
 *     Written by Alfred H. Morris, Jr.
 *	  Naval Surface Warfare Center
 *	  Dahlgren, Virginia
 *     Revised ... Nov 1991
* ----------------------------------------------------------------------- */

    Rboolean do_swap;
    int n, ierr1 = 0;
    double z, a0, b0, x0, y0, lambda;

/*  eps is a machine dependent constant: the smallest
 *      floating point number for which   1. + eps > 1.
 * NOTE: for almost all purposes it is replaced by 1e-15 (~= 4.5 times larger) below */
    double eps = 2. * Rf_d1mach(3); // == DBL_EPSILON (in R, Rmath), but then set to  1e-15  below

/* ----------------------------------------------------------------------- */
    *w  = R_D__0;
    *w1 = R_D__0;

    // safeguard, preventing infinite loops further down
    if (ISNAN(x) || ISNAN(y) ||
	ISNAN(a) || ISNAN(b)) { *ierr = 9; return; }
    if (a < 0. || b < 0.)   { *ierr = 1; return; }
    if (a == 0. && b == 0.) { *ierr = 2; return; }
    if (x < 0. || x > 1.)   { *ierr = 3; return; }
    if (y < 0. || y > 1.)   { *ierr = 4; return; }

    /* check that  'y == 1 - x' : */
    z = x + y - 0.5 - 0.5;

    if (fabs(z) > eps * 3.) { *ierr = 5; return; }

    R_ifDEBUG_printf("bratio(a=%g, b=%g, x=%9g, y=%9g, .., log_p=%d): ",
		     a,b,x,y, log_p);
    *ierr = 0;
    if (x == 0.) goto L200;
    if (y == 0.) goto L210;

    if (a == 0.) goto L211;
    if (b == 0.) goto L201;

    eps = max(eps, 1e-15);
    Rboolean a_lt_b; a_lt_b = (a < b);
    if (/* max(a,b) */ (a_lt_b ? b : a) < eps * .001) { /* procedure for a and b < 0.001 * eps */
	// L230:  -- result *independent* of x (!)
	// *w  = a/(a+b)  and  w1 = b/(a+b) :
	if(log_p) {
	    if(a_lt_b) {
		*w  = log1p(-a/(a+b)); // notably if a << b
		*w1 = log  ( a/(a+b));
	    } else { // b <= a
		*w  = log  ( b/(a+b));
		*w1 = log1p(-b/(a+b));
	    }
	} else {
	    *w	= b / (a + b);
	    *w1 = a / (a + b);
	}

	R_ifDEBUG_printf("a & b very small -> simple ratios (%g,%g)\n", *w,*w1);
	return;
    }

#define SET_0_noswap \
    a0 = a;  x0 = x; \
    b0 = b;  y0 = y;

#define SET_0_swap   \
    a0 = b;  x0 = y; \
    b0 = a;  y0 = x;

    if (min(a,b) <= 1.) { /*------------------------ a <= 1  or  b <= 1 ---- */

	do_swap = (x > 0.5);
	if (do_swap) {
	    SET_0_swap;
	} else {
	    SET_0_noswap;
	}
	/* now have  x0 <= 1/2 <= y0  (still  x0+y0 == 1) */

	R_ifDEBUG_printf(" min(a,b) <= 1, do_swap=%d;", do_swap);

	if (b0 < min(eps, eps * a0)) { /* L80: */
	    *w = fpser(a0, b0, x0, eps, log_p);
	    *w1 = log_p ? R_Log1_Exp(*w) : 0.5 - *w + 0.5;
	    R_ifDEBUG_printf("  b0 small -> w := fpser(*) = %.15g\n", *w);
	    goto L_end;
	}

	if (a0 < min(eps, eps * b0) && b0 * x0 <= 1.) { /* L90: */
	    *w1 = apser(a0, b0, x0, eps);
	    R_ifDEBUG_printf("  a0 small -> w1 := apser(*) = %.15g\n", *w1);
	    goto L_end_from_w1;
	}

	Rboolean did_bup = FALSE;
	if (max(a0,b0) > 1.) { /* L20:  min(a,b) <= 1 < max(a,b)  */
	    R_ifDEBUG_printf("\n L20:  min(a,b) <= 1 < max(a,b); ");
	    if (b0 <= 1.) goto L_w_bpser;

	    if (x0 >= 0.29) /* was 0.3, PR#13786 */	goto L_w1_bpser;

	    if (x0 < 0.1 && pow(x0*b0, a0) <= 0.7)	goto L_w_bpser;

	    if (b0 > 15.) {
		*w1 = 0.;
		goto L131;
	    }
	} else { /*  a, b <= 1 */
	    R_ifDEBUG_printf("\n      both a,b <= 1; ");
	    if (a0 >= min(0.2, b0))	goto L_w_bpser;

	    if (pow(x0, a0) <= 0.9) 	goto L_w_bpser;

	    if (x0 >= 0.3)		goto L_w1_bpser;
	}
	n = 20; /* goto L130; */
	*w1 = bup(b0, a0, y0, x0, n, eps, FALSE); did_bup = TRUE;
	R_ifDEBUG_printf("  ... n=20 and *w1 := bup(*) = %.15g; ", *w1);
	b0 += n;
    L131:
	R_ifDEBUG_printf(" L131: bgrat(*, w1=%.15g) ", *w1);
	bgrat(b0, a0, y0, x0, w1, 15*eps, &ierr1, FALSE);
	if(*w1 == 0 || (0 < *w1 && *w1 < DBL_MIN)) { // w1=0 or very close:
	    // "almost surely" from underflow, try more: [2013-03-04]
// FIXME: it is even better to do this in bgrat *directly* at least for the case
//  !did_bup, i.e., where *w1 = (0 or -Inf) on entry
	    R_ifDEBUG_printf(" denormalized or underflow (?) -> retrying: ");
	    if(did_bup) { // re-do that part on log scale:
		*w1 = bup(b0-n, a0, y0, x0, n, eps, TRUE);
	    }
	    else *w1 = ML_NEGINF; // = 0 on log-scale
	    bgrat(b0, a0, y0, x0, w1, 15*eps, &ierr1, TRUE);
	    if(ierr1) *ierr = 10 + ierr1;
	    goto L_end_from_w1_log;
	}
	// else
	if(ierr1) *ierr = 10 + ierr1;
	if(*w1 < 0)
	    MATHLIB_WARNING4("bratio(a=%g, b=%g, x=%g): bgrat() -> w1 = %g",
			     a,b,x, *w1);
	goto L_end_from_w1;
    }
    else { /* L30: -------------------- both  a, b > 1  {a0 > 1  &  b0 > 1} ---*/

	/* lambda := a y - b x  =  (a + b)y - b  =  a - (a+b)x    {using x + y == 1},
	 * ------ using the numerically best version : */
	lambda = R_FINITE(a+b)
	    ? ((a > b) ? (a + b) * y - b : a - (a + b) * x)
	    : a*y - b*x;
	do_swap = (lambda < 0.);
	if (do_swap) {
	    lambda = -lambda;
	    SET_0_swap;
	} else {
	    SET_0_noswap;
	}

	R_ifDEBUG_printf("  L30:  both  a, b > 1; |lambda| = %#g, do_swap = %d\n",
			 lambda, do_swap);

	if (b0 < 40.) {
	    R_ifDEBUG_printf("  b0 < 40;");
	    if (b0 * x0 <= 0.7
		|| (log_p && lambda > 650.)) // << added 2010-03; svn r51327
		goto L_w_bpser;
	    else
		goto L140;
	}
	else if (a0 > b0) { /* ----  a0 > b0 >= 40  ---- */
	    R_ifDEBUG_printf("  a0 > b0 >= 40;");
	    if (b0 <= 100. || lambda > b0 * 0.03)
		goto L_bfrac;

	} else if (a0 <= 100.) {
	    R_ifDEBUG_printf("  a0 <= 100; a0 <= b0 >= 40;");
	    goto L_bfrac;
	}
	else if (lambda > a0 * 0.03) {
	    R_ifDEBUG_printf("  b0 >= a0 > 100; lambda > a0 * 0.03 ");
	    goto L_bfrac;
	}

	/* else if none of the above    L180: */
	*w = basym(a0, b0, lambda, eps * 100., log_p);
	*w1 = log_p ? R_Log1_Exp(*w) : 0.5 - *w + 0.5;
	R_ifDEBUG_printf("  b0 >= a0 > 100; lambda <= a0 * 0.03: *w:= basym(*) =%.15g\n",
			 *w);
	goto L_end;

    } /* else: a, b > 1 */

/*            EVALUATION OF THE APPROPRIATE ALGORITHM */

L_w_bpser: // was L100
    *w = bpser(a0, b0, x0, eps, log_p);
    *w1 = log_p ? R_Log1_Exp(*w) : 0.5 - *w + 0.5;
    R_ifDEBUG_printf(" L_w_bpser: *w := bpser(*) = %.15g\n", *w);
    goto L_end;

L_w1_bpser:  // was L110
    *w1 = bpser(b0, a0, y0, eps, log_p);
    *w  = log_p ? R_Log1_Exp(*w1) : 0.5 - *w1 + 0.5;
    R_ifDEBUG_printf(" L_w1_bpser: *w1 := bpser(*) = %.15g\n", *w1);
    goto L_end;

L_bfrac:
    *w = bfrac(a0, b0, x0, y0, lambda, eps * 15., log_p);
    *w1 = log_p ? R_Log1_Exp(*w) : 0.5 - *w + 0.5;
    R_ifDEBUG_printf(" L_bfrac: *w := bfrac(*) = %g\n", *w);
    goto L_end;

L140:
    /* b0 := fractional_part( b0 )  in (0, 1]  */
    n = (int) b0;
    b0 -= n;
    if (b0 == 0.) {
	--n; b0 = 1.;
    }

    *w = bup(b0, a0, y0, x0, n, eps, FALSE);

    if(*w < DBL_MIN && log_p) { /* do not believe it; try bpser() : */
	R_ifDEBUG_printf(" L140: bup(b0=%g,..)=%.15g < DBL_MIN - not used;\n --> ", b0, *w);
	/*revert: */ b0 += n;
	/* which is only valid if b0 <= 1 || b0*x0 <= 0.7 */
	goto L_w_bpser;
    } // else :
    R_ifDEBUG_printf(" L140: *w := bup(b0=%g,..) = %.15g; ", b0, *w);
    if (x0 <= 0.7) {
	/* log_p :  TODO:  w = bup(.) + bpser(.)  -- not so easy to use log-scale */
	*w += bpser(a0, b0, x0, eps, /* log_p = */ FALSE);
	R_ifDEBUG_printf(" x0 <= 0.7: *w := *w + bpser(*) = %.15g\n", *w);
	goto L_end_from_w;
    }
    /* L150: */
    if (a0 <= 15.) {
	n = 20;
	*w += bup(a0, b0, x0, y0, n, eps, FALSE);
	R_ifDEBUG_printf("\n a0 <= 15: *w := *w + bup(*) = %.15g;", *w);
	a0 += n;
    }
    R_ifDEBUG_printf(" bgrat(*, w=%.15g) ", *w);
    bgrat(a0, b0, x0, y0, w, 15*eps, &ierr1, FALSE);
    if(ierr1) *ierr = 10 + ierr1;
    goto L_end_from_w;


/* TERMINATION OF THE PROCEDURE */

L200:
    if (a == 0.) { *ierr = 6;    return; }
    // else:
L201: *w  = R_D__0; *w1 = R_D__1; return;

L210:
    if (b == 0.) { *ierr = 7;    return; }
    // else:
L211: *w  = R_D__1; *w1 = R_D__0; return;

L_end_from_w:
    if(log_p) {
	*w1 = log1p(-*w);
	*w  = log(*w);
    } else {
	*w1 = 0.5 - *w + 0.5;
    }
    goto L_end;

L_end_from_w1:
    if(log_p) {
	*w  = log1p(-*w1);
	*w1 = log(*w1);
    } else {
	*w = 0.5 - *w1 + 0.5;
    }
    goto L_end;

L_end_from_w1_log:
    // *w1 = log(w1) already; w = 1 - w1  ==> log(w) = log(1 - w1) = log(1 - exp(*w1))
    if(log_p) {
	*w = R_Log1_Exp(*w1);
    } else {
	*w  = /* 1 - exp(*w1) */ -expm1(*w1);
	*w1 = exp(*w1);
    }
    goto L_end;


L_end:
    if (do_swap) { /* swap */
	double t = *w; *w = *w1; *w1 = t;
    }
    return;

} /* bratio */

#undef SET_0_noswap
#undef SET_0_swap

inline double fpser(double a, double b, double x, double eps, int log_p)
{
/* ----------------------------------------------------------------------- *

 *                 EVALUATION OF I (A,B)
 *                                X

 *          FOR B < MIN(EPS, EPS*A) AND X <= 0.5

 * ----------------------------------------------------------------------- */

    double ans, c, s, t, an, tol;

    /* SET  ans := x^a : */
    if (log_p) {
	ans = a * log(x);
    } else if (a > eps * 0.001) {
	t = a * log(x);
	if (t < exparg(1)) { /* exp(t) would underflow */
	    return 0.;
	}
	ans = exp(t);
    } else
	ans = 1.;

/*                NOTE THAT 1/B(A,B) = B */

    if (log_p)
	ans += log(b) - log(a);
    else
	ans *= b / a;

    tol = eps / a;
    an = a + 1.;
    t = x;
    s = t / an;
    do {
	an += 1.;
	t = x * t;
	c = t / an;
	s += c;
    } while (fabs(c) > tol);

    if (log_p)
	ans += log1p(a * s);
    else
	ans *= a * s + 1.;
    return ans;
} /* fpser */

inline double apser(double a, double b, double x, double eps)
{
/* -----------------------------------------------------------------------
 *     apser() yields the incomplete beta ratio  I_{1-x}(b,a)  for
 *     a <= min(eps,eps*b), b*x <= 1, and x <= 0.5,  i.e., a is very small.
 *     Use only if above inequalities are satisfied.
 * ----------------------------------------------------------------------- */

    static double const g = .577215664901533;

    double tol, c, j, s, t, aj;
    double bx = b * x;

    t = x - bx;
    if (b * eps <= 0.02)
	c = log(x) + psi(b) + g + t;
    else // b > 2e13 : psi(b) ~= log(b)
	c = log(bx) + g + t;

    tol = eps * 5. * fabs(c);
    j = 1.;
    s = 0.;
    do {
	j += 1.;
	t *= x - bx / j;
	aj = t / j;
	s += aj;
    } while (fabs(aj) > tol);

    return -a * (c + s);
} /* apser */

inline double bpser(double a, double b, double x, double eps, int log_p)
{
/* -----------------------------------------------------------------------
 * Power SERies expansion for evaluating I_x(a,b) when
 *	       b <= 1 or b*x <= 0.7.   eps is the tolerance used.
 * NB: if log_p is TRUE, also use it if   (b < 40  & lambda > 650)   where
 *                       lambda := a y - b x = (a + b)y - b  = a - (a+b)x   {x + y == 1}
 * ----------------------------------------------------------------------- */

    if (x == 0.) {
	return R_D__0;
    }
/* ----------------------------------------------------------------------- */
/*	      compute the factor  x^a/(a*Beta(a,b)) */
/* ----------------------------------------------------------------------- */
    double ans, c, z, a0 = min(a,b);
    if (a0 >= 1.) { /*		 ------	 1 <= a0 <= b0  ------ */
	z = a * log(x) - betaln(a, b);
	ans = log_p ? z - log(a) : exp(z) / a;
    }
    else {
	double t, u, apb, b0 = max(a,b);
	if (b0 < 8.) {

	    if (b0 <= 1.) { /*	------  a0 < 1  and  a0 <= b0 <= 1  ------ */

		if(log_p) {
		    ans = a * log(x);
		} else {
		    ans = pow(x, a);
		    if (ans == 0.) { /* once underflow, always underflow .. */
			R_ifDEBUG_printf(" bpser(a=%g, b=%g, x=%g): x^a underflows to 0",
					 a,b,x);
			return ans;
		    }
		}
		apb = a + b;
		if (apb > 1.) {
		    u = a + b - 1.;
		    z = (gam1(u) + 1.) / apb;
		} else {
		    z = gam1(apb) + 1.;
		}
		c = (gam1(a) + 1.) * (gam1(b) + 1.) / z;

		if(log_p) /* FIXME ? -- improve quite a bit for c ~= 1 */
		    ans += log(c * (b / apb));
		else
		    ans *=  c * (b / apb);

	    } else { /* 	------	a0 < 1 < b0 < 8	 ------ */

		u = gamln1(a0);
		int m = (int)(b0 - 1.);
		if (m >= 1) {
		    c = 1.;
		    for (int i = 1; i <= m; ++i) {
			b0 += -1.;
			c *= b0 / (a0 + b0);
		    }
		    u += log(c);
		}

		z = a * log(x) - u;
		b0 += -1.; // => b0 in (0, 7)
		apb = a0 + b0;
		if (apb > 1.) {
		    u = a0 + b0 - 1.;
		    t = (gam1(u) + 1.) / apb;
		} else {
		    t = gam1(apb) + 1.;
		}

		if(log_p) /* FIXME? potential for improving log(t) */
		    ans = z + log(a0 / a) + log1p(gam1(b0)) - log(t);
		else
		    ans = exp(z) * (a0 / a) * (gam1(b0) + 1.) / t;
	    }

	} else { /* 		------  a0 < 1 < 8 <= b0  ------ */

	    u = gamln1(a0) + algdiv(a0, b0);
	    z = a * log(x) - u;

	    if(log_p)
		ans = z + log(a0 / a);
	    else
		ans = a0 / a * exp(z);
	}
    }
    R_ifDEBUG_printf(" bpser(a=%g, b=%g, x=%g, log=%d, eps=%g): %s = %.14g;",
		     a,b,x, log_p, eps,
		     log_p ? "log(x^a/(a*B(a,b)))" : "x^a/(a*B(a,b))",  ans);
    if (ans == R_D__0 || (!log_p && a <= eps * 0.1)) {
	R_ifDEBUG_printf(" = final answer\n");
	return ans;
    }
#ifdef DEBUG_bratio
    else REprintf("\n");
#endif

/* ----------------------------------------------------------------------- */
/*		       COMPUTE THE SERIES */
/* ----------------------------------------------------------------------- */
    double tol = eps / a,
	n = 0.,
	sum = 0., w;
    c = 1.;
    do { // sum is alternating as long as n < b (<==> 1 - b/n < 0)
	n += 1.;
	c *= (0.5 - b / n + 0.5) * x;
	w = c / (a + n);
	sum += w;
    } while (n < 1e7 && fabs(w) > tol);
    if(fabs(w) > tol) { // the series did not converge (in time)
	// warn only when the result seems to matter:
	if(( log_p && !(a*sum > -1. && fabs(log1p(a * sum)) < eps*fabs(ans))) ||
	   (!log_p && fabs(a*sum + 1.) != 1.))
	    MATHLIB_WARNING5(
		" bpser(a=%g, b=%g, x=%g,...) did not converge (n=1e7, |w|/tol=%g > 1; A=%g)",
		a,b,x, fabs(w)/tol, ans);
    }
    R_ifDEBUG_printf("  -> n=%.0f iterations, |w|=%g %s %g=tol:=eps/a ==> a*sum=%g %s -1\n",
		     n, fabs(w), (fabs(w) > tol) ? ">!!>" : "<=", tol,
		     a*sum, (a*sum > -1.) ? ">" : "<=");
    if(log_p) {
	if (a*sum > -1.) ans += log1p(a * sum);
	else {
	    if(ans > ML_NEGINF)
		MATHLIB_WARNING3(
		    "pbeta(*, log.p=TRUE) -> bpser(a=%g, b=%g, x=%g,...) underflow to -Inf",
		    a,b,x);
	    ans = ML_NEGINF;
	}
    } else if (a*sum > -1.)
	ans *= (a * sum + 1.);
    else // underflow to
	ans = 0.;
    return ans;
} /* bpser */

inline double bup(double a, double b, double x, double y, int n, double eps,
		  int give_log)
{
/* ----------------------------------------------------------------------- */
/*     EVALUATION OF I_x(A,B) - I_x(A+N,B) WHERE N IS A POSITIVE INT. */
/*     EPS IS THE TOLERANCE USED. */
/* ----------------------------------------------------------------------- */

    double ret_val;
    int i, k, mu;
    double d, l;

// Obtain the scaling factor exp(-mu) and exp(mu)*(x^a * y^b / beta(a,b))/a

    double apb = a + b,
	ap1 = a + 1.;
    if (n > 1 && a >= 1. && apb >= ap1 * 1.1) {
	mu = (int)fabs(exparg(1));
	k = (int) exparg(0);
	if (mu > k)
	    mu = k;
 	d = exp(-(double) mu); // = exp(-709) = 1.216780751..e-308  nowadays
    }
    else {
	mu = 0;
	d = 1.;
    }

    /* L10: */
    ret_val = give_log
	? brcmp1(mu, a, b, x, y, TRUE) - log(a)
	: brcmp1(mu, a, b, x, y, FALSE)  / a;
    if (n == 1 ||
	(give_log && ret_val == ML_NEGINF) || (!give_log && ret_val == 0.))
	return ret_val;

    int nm1 = n - 1;
    double w = d;

/*          LET K BE THE INDEX OF THE MAXIMUM TERM */

    k = 0;
    if (b > 1.) {
	if (y > 1e-4) {
	    double r = (b - 1.) * x / y - a;
	    if (r >= 1.)
		k = (r < nm1) ? (int) r : nm1;
	} else
	    k = nm1;

//          ADD THE INCREASING TERMS OF THE SERIES - if k > 0
/* L30: */
	for (i = 0; i < k; ++i) {
	    l = (double) i;
	    d *= (apb + l) / (ap1 + l) * x;
	    w += d;
	}
    }

// L40:     ADD THE REMAINING TERMS OF THE SERIES

    for (i = k; i < nm1; ++i) {
	l = (double) i;
	d *= (apb + l) / (ap1 + l) * x;
	w += d;
	if (d <= eps * w) /* relativ convergence (eps) */
	    break;
    }

    // L50: TERMINATE THE PROCEDURE
    if(give_log) {
	ret_val += log(w);
    } else
	ret_val *= w;

    return ret_val;
} /* bup */

inline double bfrac(double a, double b, double x, double y, double lambda,
		    double eps, int log_p)
{
/* -----------------------------------------------------------------------
       Continued fraction expansion for I_x(a,b) when a, b > 1.
       It is assumed that  lambda = (a + b)*y - b.  (x+y = 1)
   -----------------------------------------------------------------------*/

    if(!R_FINITE(lambda)) return ML_NAN;// TODO: can return 0 or 1 (?)
    R_ifDEBUG_printf(" bfrac(a=%g, b=%g, x=%g, y=%g, lambda=%g, eps=%g, log_p=%d):\n",
		     a,b,x,y, lambda, eps, log_p);
    double brc = brcomp(a, b, x, y, log_p);
    if(ISNAN(brc)) { // till 2025-05, from   L <- 1e308; pnbinom(L, L, mu = 5) -- does this still happen?
	R_ifDEBUG_printf(" --> brcomp(a,b,x,y) = NaN; ");
	ML_WARN_return_NAN; // TODO: could we know better?
    }
    if (!log_p && brc == 0.) {
	R_ifDEBUG_printf(" --> brcomp(a,b,x,y) underflowed to 0; ");
	return 0.;
    }
#ifdef DEBUG_bratio
    else
	REprintf("brcomp(a,b, x,y) = %g; ", brc);
#endif

    double
	c = lambda + 1.,
	c0 = b / a,
	c1 = 1. / a + 1.,
	yp1 = y + 1.,
	n = 0.,
	p = 1.,
	s = a + 1.,
	an = 0.,
	bn = 1.,
	anp1 = 1.,
	bnp1 = c / c1,
	r = c1 / c,
	r0; // := prev r;

/*        CONTINUED FRACTION CALCULATION */
#define bfrac_MAXIT 1000 // was 10000, but have never seen 1000 close to needed
    do {
	n += 1.;
	double w = n * x * (b - n); /* overflows when b is almost DBL_MAX ! */
	bool rescale = !R_FINITE(w);
	/* rescale w  <==> rescaled (alpha, beta) with same factor;
	   alpha (proportional) w  is automatically scaled, and beta needs explicit scaling */
	if(rescale) w = n * x * ldexp(b - n, -20);
	double t = n / a,
	    e = a / s,
	    alpha = p * (p + c0) * e * e * (w * x), beta;
	e = (t + 1.) / (c1 + t + t);
	beta = w / s + ((rescale) ? ldexp(n + e * (c + n * yp1), -20)
			          :       n + e * (c + n * yp1));
	p = t + 1.;
	s += 2.;

	/* update an, bn, anp1, and bnp1 */

	t = alpha * an + beta * anp1;	an = anp1;	anp1 = t;
	t = alpha * bn + beta * bnp1;	bn = bnp1;	bnp1 = t;

	r0 = r;
	r = anp1 / bnp1;
	if (fabs(r - r0) <= eps * r)
	    break;

	/* rescale an, bn, anp1, and bnp1 */

	an /= bnp1;
	bn /= bnp1;
	anp1 = r;
	bnp1 = 1.;
    } while (n < bfrac_MAXIT);// arbitrary; had '1' --> infinite loop for  lambda = Inf
    R_ifDEBUG_printf("  in bfrac(): n=%.0f terms cont.frac.; brc=%g, r=%g\n",
		     n, brc, r);
    if(n >= bfrac_MAXIT && fabs(r - r0) > eps * r)
	MATHLIB_WARNING6(
	    " bfrac(a=%g, b=%g, x=%g, y=%g, lambda=%g) did *not* converge (in %d steps)\n",
	    a,b,x,y, lambda, bfrac_MAXIT);
    return (log_p ? brc + log(r) : brc * r);
} /* bfrac */
#undef bfrac_MAXIT

inline double brcomp(double a, double b, double x, double y, int log_p)
{
/* -----------------------------------------------------------------------
 *		 Evaluation of x^a * y^b / Beta(a,b)
 * ----------------------------------------------------------------------- */

    if (x == 0. || y == 0.) {
	return R_D__0;
    }
    double a0 = min(a, b);
    if (a0 < 8.) {
	double lnx, lny;
	if (x <= .375) {
	    lnx = log(x);
	    lny = alnrel(-x);
	}
	else if (y > .375) {
	    lnx = log(x);
	    lny = log(y);
	} else {
	    lnx = alnrel(-y);
	    lny = log(y);
	}

	double z = a * lnx + b * lny;
	if (a0 >= 1.) {
	    z -= betaln(a, b);
	    return R_D_exp(z);
	}
	// else :
	/* ----------------------------------------------------------------------- */
	/*		PROCEDURE FOR a < 1 OR b < 1 */
	/* ----------------------------------------------------------------------- */
	double b0 = max(a, b);
	if (b0 >= 8.) { /* L80: */
	    double u = gamln1(a0) + algdiv(a0, b0);
	    return (log_p ? log(a0) + (z - u)  : a0 * exp(z - u));
	}
	/* else : */
	if (b0 <= 1.) { /*		algorithm for max(a,b) = b0 <= 1 */
	    double e_z = R_D_exp(z);
	    if (!log_p && e_z == 0.) /* exp() underflow */
		return 0.;

	    double apb = a + b;
	    if (apb > 1.) {
		z = (gam1(apb - 1.) + 1.) / apb;
	    } else {
		z = gam1(apb) + 1.;
	    }
	    double c = (gam1(a) + 1.) * (gam1(b) + 1.) / z;
	    R_ifDEBUG_printf(" brcomp(), max(a,b) <= 1: (e_z, z, c) = (%g, %g, %g)\n",
			     e_z, z, c);
	    /* FIXME? log(a0*c)= log(a0)+ log(c) and that is improvable */
	    return (log_p
		    ? e_z + log(a0 * c) - log1p(a0/b0)
		    : e_z *    (a0 * c) / (a0/b0 + 1.));
	}

	/* else : 		  ALGORITHM FOR 1 < b0 < 8 */

	double u = gamln1(a0);
	int n = (int)(b0 - 1.);
	if (n >= 1) {
	    double c = 1.;
	    for (int i = 1; i <= n; ++i) {
		b0 += -1.;
		c *= b0 / (a0 + b0);
	    }
	    u = log(c) + u;
	}
	z -= u;
	b0 += -1.;
	double apb = a0 + b0, t;
	if (apb > 1.) {
	    u = a0 + b0 - 1.;
	    t = (gam1(u) + 1.) / apb;
	} else {
	    t = gam1(apb) + 1.;
	}

	R_ifDEBUG_printf(" brcomp(), 1 < b0 < 8: (z, t) = (%g, %g)\n", z, t);
	return (log_p
		? log(a0) + z + log1p(gam1(b0))  - log(t)
		: a0 * exp(z) * (gam1(b0) + 1.) / t);

    } else {
/* ----------------------------------------------------------------------- */
/*		PROCEDURE FOR a >= 8 AND b >= 8 */
/* ----------------------------------------------------------------------- */
	static double const__ = .398942280401433; /* == 1/sqrt(2*pi); */
	/* R has  M_1_SQRT_2PI , and M_LN_SQRT_2PI = ln(sqrt(2*pi)) = 0.918938.. */
	double h, x0, y0, apb = a+b,
	    lambda = R_FINITE(apb) // be safe
	      ? ((a <= b) ? a - apb * x
 	                  : apb * y - b)
	      : a*y - b*x;
	if (a <= b) {
	    h = a / b;
	    x0 = h  / (h + 1.);
	    y0 = 1. / (h + 1.);
	    R_ifDEBUG_printf(" brcomp(8 <= a <= b): ");
	} else {
	    h = b / a;
	    x0 = 1. / (h + 1.);
	    y0 = h  / (h + 1.);
	    R_ifDEBUG_printf(" brcomp(8 <= b < a): ");
	}

	double e = -lambda / a,  u, v, z;
	if (fabs(e) > .6)
	    u = e - log(x / x0);
	else
	    u = rlog1(e);

	e = lambda / b;
	if (fabs(e) <= .6)
	    v = rlog1(e);
	else
	    v = e - log(y / y0);

	z = log_p ? -(a * u + b * v) : exp(-(a * u + b * v));
	R_ifDEBUG_printf(" brcomp(): (lambda => u, v => z) = (%g =>  %g, %g  => %g)\n",
			 lambda,  u, v,  z);
	return(log_p
	       ? -M_LN_SQRT_2PI + .5*log(b * x0) + z - bcorr(a,b)
	       : const__ * sqrt(b * x0) * z * exp(-bcorr(a, b)));
    }
} /* brcomp */

// called only once from  bup(),  as   r = brcmp1(mu, a, b, x, y, FALSE) / a;
//                        -----
inline double brcmp1(int mu, double a, double b, double x, double y, int give_log)
{
/* -----------------------------------------------------------------------
 *          Evaluation of    exp(mu) * x^a * y^b / Beta(a,b)
 * --------------------------^^^^^^^^^------------------------------------ */

    double a0 = min(a,b);
    if (a0 < 8.) {
	double lnx, lny;
	if (x <= .375) {
	    lnx = log(x);
	    lny = alnrel(-x);
	} else if (y > .375) {
	    // L11:
	    lnx = log(x);
	    lny = log(y);
	} else {
	    lnx = alnrel(-y);
	    lny = log(y);
	}

	// L20:
	double z = a * lnx + b * lny;
	if (a0 >= 1.) {
	    z -= betaln(a, b);
	    return esum(mu, z, give_log);
	}
	// else :
	/* ----------------------------------------------------------------------- */
	/*              PROCEDURE FOR a < 1 OR b < 1 */
	/* ----------------------------------------------------------------------- */
	// L30:
	double b0 = max(a,b);
	if (b0 >= 8.) {
	/* L80:                  ALGORITHM FOR b0 >= 8 */
	    double u = gamln1(a0) + algdiv(a0, b0);
	    R_ifDEBUG_printf(" brcmp1(mu,a,b,*): a0 < 1, b0 >= 8;  z=%.15g\n", z);
	    return give_log
		? log(a0) + esum(mu, z - u, TRUE)
		:     a0  * esum(mu, z - u, FALSE);

	} else if (b0 <= 1.) {
	    //                   a0 < 1, b0 <= 1
	    double ans = esum(mu, z, give_log);
	    if (ans == (give_log ? ML_NEGINF : 0.))
		return ans;

	    double apb = a + b, z;
	    if (apb > 1.) { // L40:
		z = (gam1(apb - 1.) + 1.) / apb;
	    } else {
		z = gam1(apb) + 1.;
	    }
	    // L50:
	    double c = give_log
		? log1p(gam1(a)) + log1p(gam1(b)) - log(z)
		: (gam1(a) + 1.) * (gam1(b) + 1.) / z;
	    R_ifDEBUG_printf(" brcmp1(mu,a,b,*): a0 < 1, b0 <= 1;  c=%.15g\n", c);
	    return give_log
		? ans + log(a0) + c - log1p(a0 / b0)
		: ans * (a0 * c) / (a0 / b0 + 1.);
	}
	// else:               algorithm for	a0 < 1 < b0 < 8
	// L60:
	double u = gamln1(a0);
	int n = (int)(b0 - 1.); // have n <= 6
	if (n >= 1) {
	    double c = 1.;
	    for (int i = 1; i <= n; ++i) {
		b0 += -1.;
		c *= b0 / (a0 + b0);
		/* L61: */
	    }
	    u += log(c); // TODO?: log(c) = log( prod(...) ) =  sum( log(...) )
	}
	// L70:
	z -= u;
	b0 += -1.;
	double apb = a0 + b0, t;
	if (apb > 1.) {
	    // L71:
	    t = (gam1(apb - 1.) + 1.) / apb;
	} else {
	    t = gam1(apb) + 1.;
	}
	R_ifDEBUG_printf(" brcmp1(mu,a,b,*): a0 < 1 < b0 < 8;  t=%.15g\n", t);
	// L72:
	return give_log
	    ? log(a0)+ esum(mu, z, TRUE) + log1p(gam1(b0)) - log(t) // TODO? log(t) = log1p(..)
	    :     a0 * esum(mu, z, FALSE) * (gam1(b0) + 1.) / t;

    } else {

/* ----------------------------------------------------------------------- */
/*              PROCEDURE FOR A >= 8 AND B >= 8 */
/* ----------------------------------------------------------------------- */
	static double const__ = .398942280401433; /* == 1/sqrt(2*pi); */
	/* R has  M_1_SQRT_2PI , and M_LN_SQRT_2PI = ln(sqrt(2*pi)) = 0.918938.. */
	// L100:
	double h, x0, y0, apb = a+b,
	    lambda = R_FINITE(apb) // be safe
	      ? ((a <= b) ? a - apb * x
 	                  : apb * y - b)
	      : a*y - b*x;
	if (a > b) {
	    // L101:
	    h = b / a;
	    x0 = 1. / (h + 1.);// => lx0 := log(x0) = 0 - log1p(h)
	    y0 = h  / (h + 1.);
	} else {
	    h = a / b;
	    x0 = h  / (h + 1.);  // => lx0 := log(x0) = - log1p(1/h)
	    y0 = 1. / (h + 1.);
	}
	double lx0 = -log1p(b/a); // in both cases

	R_ifDEBUG_printf(" brcmp1(mu,a,b,*): a,b >= 8;	x0=%.15g, lx0=log(x0)=%.15g\n",
			 x0, lx0);
	// L110:
	double e = -lambda / a,  u, v, z;
	if (fabs(e) > 0.6) {
	    // L111:
	    u = e - log(x / x0);
	} else {
	    u = rlog1(e);
	}

	// L120:
	e = lambda / b;
	if (fabs(e) > 0.6) {
	    // L121:
	    v = e - log(y / y0);
	} else {
	    v = rlog1(e);
	}

	// L130:
	z = esum(mu, -(a * u + b * v), give_log);
	return give_log
	    ? log(const__)+ (log(b) + lx0)/2. + z      - bcorr(a, b)
	    :     const__ * sqrt(b * x0)      * z * exp(-bcorr(a, b));
    }

} /* brcmp1 */

inline void bgrat(double a, double b, double x, double y, double *w,
		  double eps, int *ierr, Rboolean log_w)
{
/* -----------------------------------------------------------------------
*     Asymptotic Expansion for I_x(a,b)  when a is larger than b.
*     Compute   w := w + I_x(a,b)
*     It is assumed a >= 15 and b <= 1.
*     eps is the tolerance used.
*     ierr is a variable that reports the status of the results.
*
* if(log_w),  *w  itself must be in log-space;
*     compute   w := w + I_x(a,b)  but return *w = log(w):
*          *w := log(exp(*w) + I_x(a,b)) = logspace_add(*w, log( I_x(a,b) ))
* ----------------------------------------------------------------------- */

#define n_terms_bgrat 30
    double c[n_terms_bgrat], d[n_terms_bgrat];
    double bm1 = b - 0.5 - 0.5,
	nu = a + bm1 * 0.5, /* nu = a + (b-1)/2 =: T, in (9.1) of
			     * Didonato & Morris(1992), p.362 */
	lnx = (y > 0.375) ? log(x) : alnrel(-y),
	z = -nu * lnx; // z =: u in (9.1) of D.&M.(1992)

    if (b * z == 0.) { // should not happen, but does, e.g.,
	// for  pbeta(1e-320, 1e-5, 0.5)  i.e., _subnormal_ x,
	// Warning ... bgrat(a=20.5, b=1e-05, x=1, y=9.99989e-321): ..
	MATHLIB_WARNING5(
	    "bgrat(a=%g, b=%g, x=%g, y=%g): z=%g, b*z == 0 underflow, hence inaccurate pbeta()",
	    a,b,x,y, z);
	/* L_Error:    THE EXPANSION CANNOT BE COMPUTED */
	 *ierr = 1;
	 /* TODO (?):  do not yet return, trying further using log-scale below */
	 return;
    }

/*                 COMPUTATION OF THE EXPANSION */
    double
	/* r1 = b * (gam1(b) + 1.) * exp(b * log(z)),// = b/gamma(b+1) z^b = z^b / gamma(b)
	 * set r := exp(-z) * z^b / gamma(b) ;
	 *          gam1(b) = 1/gamma(b+1) - 1 , b in [-1/2, 3/2] */
	// exp(a*lnx) underflows for large (a * lnx); e.g. large a ==> using log_r := log(r):
	// r = r1 * exp(a * lnx) * exp(bm1 * 0.5 * lnx);
	// log(r)=log(b) + log1p(gam1(b)) + b * log(z) + (a * lnx) + (bm1 * 0.5 * lnx),
	log_r = log(b) + log1p(gam1(b)) + b * log(z) + nu * lnx,
	// FIXME work with  log_u = log(u)  also when log_p=FALSE  (??)
	// u is 'factored out' from the expansion {and multiplied back, at the end}:
	log_u = log_r - (algdiv(b, a) + b * log(nu)),// algdiv(b,a) = log(gamma(a)/gamma(a+b))
	/* u = (log_p) ? log_r - u : exp(log_r-u); // =: M  in (9.2) of {reference above} */
	/* u = algdiv(b, a) + b * log(nu);// algdiv(b,a) = log(gamma(a)/gamma(a+b)) */
	// u = (log_p) ? log_u : exp(log_u); // =: M  in (9.2) of {reference above}
	u = exp(log_u);

    if (log_u == ML_NEGINF) {
	R_ifDEBUG_printf(" bgrat(*): underflow log_u = -Inf  = log_r -u', log_r = %g ",
			 log_r);
	/* L_Error:    THE EXPANSION CANNOT BE COMPUTED */ *ierr = 2; return;
    }

    Rboolean u_0 = (u == 0.); // underflow --> do work with log(u) == log_u !
    double l = // := *w/u .. but with care: such that it also works when u underflows to 0:
	log_w
	? ((*w == ML_NEGINF) ? 0. : exp(  *w    - log_u))
	: ((*w == 0.)        ? 0. : exp(log(*w) - log_u));

    R_ifDEBUG_printf(" bgrat(a=%g, b=%g, x=%g, *)\n -> u=%g, l='w/u'=%g, ",
		     a,b,x, u, l);
    double
	q_r = grat_r(b, z, log_r, eps), // = q/r of former grat1(b,z, r, &p, &q)
	v = 0.25 / (nu * nu),
	t2 = lnx * 0.25 * lnx,
	j = q_r,
	sum = j,
	t = 1., cn = 1., n2 = 0.;
    for (int n = 1; n <= n_terms_bgrat; ++n) {
	double bp2n = b + n2;
	j = (bp2n * (bp2n + 1.) * j + (z + bp2n + 1.) * t) * v;
	n2 += 2.;
	t *= t2;
	cn /= n2 * (n2 + 1.);
	int nm1 = n - 1;
	c[nm1] = cn;
	double s = 0.;
	if (n > 1) {
	    double coef = b - n;
	    for (int i = 1; i <= nm1; ++i) {
		s += coef * c[i - 1] * d[nm1 - i];
		coef += b;
	    }
	}
	d[nm1] = bm1 * cn + s / n;
	double dj = d[nm1] * j;
	sum += dj;
	if (sum <= 0.) {
	    R_ifDEBUG_printf(" bgrat(*): sum_n(..) <= 0; should not happen (n=%d)\n", n);
	    /* L_Error:    THE EXPANSION CANNOT BE COMPUTED */ *ierr = 3; return;
	}
	if (fabs(dj) <= eps * (sum + l)) {
	    *ierr = 0;
	    break;
	} else if(n == n_terms_bgrat) { // e.g. from pbeta(..., 0.001, 1e200)
	    *ierr = 4;
	    MATHLIB_WARNING5(
	"bgrat(a=%g, b=%g, x=%.12g) *no* convergence: NOTIFY R-core!\n dj=%g, rel.err=%g\n",
		a,b,x, dj, fabs(dj) /(sum + l));
	}
    } // for(n .. n_terms..)

/*                    ADD THE RESULTS TO W */

    if(log_w) // *w is in log space already:
	*w = logspace_add(*w, log_u + log(sum));
    else
	*w += (u_0 ? exp(log_u + log(sum)) : u * sum);
    return;
} /* bgrat */


// called only from bgrat() , as   q_r = grat_r(b, z, log_r, eps)  :
inline double grat_r(double a, double x, double log_r, double eps)
{
/* -----------------------------------------------------------------------
 *        Scaled complement of incomplete gamma ratio function
 *                   grat_r(a,x,r) :=  Q(a,x) / r
 * where
 *               Q(a,x) = pgamma(x,a, lower.tail=FALSE)
 *     and            r = e^(-x)* x^a / Gamma(a) ==  exp(log_r)
 *
 *     It is assumed that a <= 1.  eps is the tolerance to be used.
 * ----------------------------------------------------------------------- */

    if (a * x == 0.) { /* L130: */
	if (x <= a) {
	    /* L100: */ return exp(-log_r);
	} else {
	    /* L110:*/  return 0.;
	}
    }
    else if (a == 0.5) { // e.g. when called from pt()
	/* L120: */
	if (x < 0.25) {
	    double p = erf__(sqrt(x));
	    R_ifDEBUG_printf(" grat_r(a=%g, x=%g ..)): a=1/2 --> p=erf__(.)= %g\n",
			     a, x, p);
	    return (0.5 - p + 0.5)*exp(-log_r);

        } else { // 2013-02-27: improvement for "large" x: direct computation of q/r:
	    double sx = sqrt(x),
		q_r = erfc1(1, sx)/sx * M_SQRT_PI;
	    R_ifDEBUG_printf(" grat_r(a=%g, x=%g ..)): a=1/2 --> q_r=erfc1(..)/r= %g\n",
			     a,x, q_r);
	    return q_r;
	}

    } else if (x < 1.1) { /* L10:  Taylor series for  P(a,x)/x^a */

	double an = 3.,
	    c = x,
	    sum = x / (a + 3.),
	    tol = eps * 0.1 / (a + 1.), t;
	do {
	    an += 1.;
	    c *= -(x / an);
	    t = c / (a + an);
	    sum += t;
	} while (fabs(t) > tol);

	R_ifDEBUG_printf(" grat_r(a=%g, x=%g, log_r=%g): sum=%g; Taylor w/ %.0f terms",
			 a,x,log_r, sum, an-3.);
	double j = a * x * ((sum/6. - 0.5/(a + 2.)) * x + 1./(a + 1.)),
	    z = a * log(x),
	    h = gam1(a),
	    g = h + 1.;

	if ((x >= 0.25 && (a < x / 2.59)) || (z > -0.13394)) {
	    // L40:
	    double l = rexpm1(z),
		q = ((l + 0.5 + 0.5) * j - l) * g - h;
	    if (q <= 0.) {
		R_ifDEBUG_printf(" => q_r= 0.\n");
		/* L110:*/ return 0.;
	    } else {
		R_ifDEBUG_printf(" => q_r=%.15g\n", q * exp(-log_r));
		return q * exp(-log_r);
	    }

	} else {
	    double p = exp(z) * g * (0.5 - j + 0.5);
	    R_ifDEBUG_printf(" => q_r=%.15g\n", (0.5 - p + 0.5) * exp(-log_r));
	    return /* q/r = */ (0.5 - p + 0.5) * exp(-log_r);
	}

    } else {
	/* L50: ----  (x >= 1.1)  ---- Continued Fraction Expansion */

	double a2n_1 = 1.,
	    a2n = 1.,
	    b2n_1 = x,
	    b2n = x + (1. - a),
	    c = 1., am0, an0;

	do {
	    a2n_1 = x * a2n + c * a2n_1;
	    b2n_1 = x * b2n + c * b2n_1;
	    am0 = a2n_1 / b2n_1;
	    c += 1.;
	    double c_a = c - a;
	    a2n = a2n_1 + c_a * a2n;
	    b2n = b2n_1 + c_a * b2n;
	    an0 = a2n / b2n;
	} while (fabs(an0 - am0) >= eps * an0);

	R_ifDEBUG_printf(" grat_r(a=%g, x=%g, log_r=%g): Cont.frac. %.0f terms => q_r=%.15g\n",
			 a,x, log_r, c-1., an0);
	return /* q/r = (r * an0)/r = */ an0;
    }
} /* grat_r */



inline double basym(double a, double b, double lambda, double eps, int log_p)
{
/* ----------------------------------------------------------------------- */
/*     ASYMPTOTIC EXPANSION FOR I_x(A,B) FOR LARGE A AND B. */
/*    lambda := a y - b x  = (a + b)y - b  =  a - (a+b)x    {using x + y == 1},
 *                             and eps is the tolerance used.
 *     It is assumed that   lambda >= 0 , i.e., x <= a/(a+b), and both  a, b  >= 15   */
/* ----------------------------------------------------------------------- */


/* ------------------------ */
/*     ****** NUM IS THE MAXIMUM VALUE THAT N CAN TAKE IN THE DO LOOP */
/*            ENDING AT STATEMENT 50. IT IS REQUIRED THAT NUM BE EVEN. */
#define num_IT 20
/*            THE ARRAYS A0, B0, C, D HAVE DIMENSION NUM + 1. */

    static double const e0 = 1.12837916709551;/* e0 == 2/sqrt(pi) */
    static double const e1 = .353553390593274;/* e1 == 2^(-3/2)   */
    static double const ln_e0 = 0.120782237635245; /* == ln(e0) */

    double a0[num_IT + 1], b0[num_IT + 1], c[num_IT + 1], d[num_IT + 1];

    double f = a * rlog1(-lambda/a) + b * rlog1(lambda/b), t;
    if(log_p)
	t = -f;
    else {
	t = exp(-f);
	if (t == 0.) {
	    return 0; /* once underflow, always underflow .. */
	}
    }
    double z0 = sqrt(f),
	z = z0 / e1 * 0.5,
	z2 = f + f,
	h, r0, r1, w0;

    if (a < b) {
	h = a / b;
	r0 = 1. / (h + 1.);
	r1 = (b - a) / b;
	w0 = 1. / sqrt(a * (h + 1.));
    } else {
	h = b / a;
	r0 = 1. / (h + 1.);
	r1 = (b - a) / a;
	w0 = 1. / sqrt(b * (h + 1.));
    }

    a0[0] = r1 * .66666666666666663;
    c[0] = a0[0] * -0.5;
    d[0] = -c[0];
    double j0 = 0.5 / e0 * erfc1(1, z0),
	j1 = e1,
	sum = j0 + d[0] * w0 * j1;

    double s = 1.,
	h2 = h * h,
	hn = 1.,
	w = w0,
	znm1 = z,
	zn = z2;
    for (int n = 2; n <= num_IT; n += 2) {
	hn *= h2;
	a0[n - 1] = r0 * 2. * (h * hn + 1.) / (n + 2.);
	int np1 = n + 1;
	s += hn;
	a0[np1 - 1] = r1 * 2. * s / (n + 3.);

	for (int i = n; i <= np1; ++i) {
	    double r = (i + 1.) * -0.5;
	    b0[0] = r * a0[0];
	    for (int m = 2; m <= i; ++m) {
		double bsum = 0.;
		for (int j = 1; j <= m-1; ++j) {
		    int mmj = m - j;
		    bsum += (j * r - mmj) * a0[j - 1] * b0[mmj - 1];
		}
		b0[m - 1] = r * a0[m - 1] + bsum / m;
	    }
	    c[i - 1] = b0[i - 1] / (i + 1.);

	    double dsum = 0.;
	    for (int j = 1; j <= i-1; ++j) {
		dsum += d[i - j - 1] * c[j - 1];
	    }
	    d[i - 1] = -(dsum + c[i - 1]);
	}

	j0 = e1 * znm1 + (n - 1.) * j0;
	j1 = e1 * zn + n * j1;
	znm1 = z2 * znm1;
	zn = z2 * zn;
	w *= w0;
	double t0 = d[n - 1] * w * j0;
	w *= w0;
	double t1 = d[np1 - 1] * w * j1;
	sum += t0 + t1;
	if (fabs(t0) + fabs(t1) <= eps * sum) {
	    break;
	}
    }

    if(log_p)
	return ln_e0 + t - bcorr(a, b) + log(sum);
    else {
	double u = exp(-bcorr(a, b));
	return e0 * t * u * sum;
    }

} /* basym_ */


inline double exparg(int l)
{
/* --------------------------------------------------------------------
 *     If l = 0 then  exparg(l) = The largest positive W for which
 *     exp(W) can be computed. With 0.99999 fuzz  ==> exparg(0) =   709.7756  nowadays

 *     if l = 1 (nonzero) then  exparg(l) = the largest negative W for
 *     which the computed value of exp(W) is nonzero.
 *     With 0.99999 fuzz			  ==> exparg(1) =  -709.0825  nowadays

 *     Note... only an approximate value for exparg(L) is needed.
 * -------------------------------------------------------------------- */

    static double const lnb = .69314718055995;
    int m = (l == 0) ? Rf_i1mach(16) : Rf_i1mach(15) - 1;

    return m * lnb * .99999;
} /* exparg */

inline double esum(int mu, double x, int give_log)
{
/* ----------------------------------------------------------------------- */
/*                    EVALUATION OF EXP(MU + X) */
/* ----------------------------------------------------------------------- */

    if(give_log)
	return x + (double) mu;

    // else :
    double w;
    if (x > 0.) { /* L10: */
	if (mu > 0)  return exp((double) mu) * exp(x);
	w = mu + x;
	if (w < 0.) return exp((double) mu) * exp(x);
    }
    else { /* x <= 0 */
	if (mu < 0)  return exp((double) mu) * exp(x);
	w = mu + x;
	if (w > 0.) return exp((double) mu) * exp(x);
    }
    return exp(w);

} /* esum */

inline double rexpm1(double x)
{
/* ----------------------------------------------------------------------- */
/*            EVALUATION OF THE FUNCTION EXP(X) - 1 */
/* ----------------------------------------------------------------------- */

    static double p1 = 9.14041914819518e-10;
    static double p2 = .0238082361044469;
    static double q1 = -.499999999085958;
    static double q2 = .107141568980644;
    static double q3 = -.0119041179760821;
    static double q4 = 5.95130811860248e-4;

    if (fabs(x) <= 0.15) {
	return x * (((p2 * x + p1) * x + 1.) /
		    ((((q4 * x + q3) * x + q2) * x + q1) * x + 1.));
    }
    else { /* |x| > 0.15 : */
	double w = exp(x);
	if (x > 0.)
	    return w * (0.5 - 1. / w + 0.5);
	else
	    return w - 0.5 - 0.5;
    }

} /* rexpm1 */

inline double alnrel(double a)
{
/* -----------------------------------------------------------------------
 *            Evaluation of the function ln(1 + a)
 * ----------------------------------------------------------------------- */

    if (fabs(a) > 0.375)
	return log(1. + a);
    // else : |a| <= 0.375
    static double
	p1 = -1.29418923021993,
	p2 = .405303492862024,
	p3 = -.0178874546012214,
	q1 = -1.62752256355323,
	q2 = .747811014037616,
	q3 = -.0845104217945565;
    double
	t = a / (a + 2.),
	t2 = t * t,
	w = (((p3 * t2 + p2) * t2 + p1) * t2 + 1.) /
	(((q3 * t2 + q2) * t2 + q1) * t2 + 1.);
    return t * 2. * w;

} /* alnrel */

inline double rlog1(double x)
{
/* -----------------------------------------------------------------------
 *             Evaluation of the function  x - ln(1 + x)
 *                                        ~=~ - log1pmx(x)  in ./pgamma.c
 * ----------------------------------------------------------------------- */

    static double a = .0566749439387324;
    static double b = .0456512608815524;
    static double p0 = .333333333333333;
    static double p1 = -.224696413112536;
    static double p2 = .00620886815375787;
    static double q1 = -1.27408923933623;
    static double q2 = .354508718369557;

    double h, r, t, w, w1;
    if (x < -0.39 || x > 0.57) { /* direct evaluation */
	w = x + 0.5 + 0.5;
	return x - log(w);
    }
    /* else */
    if (x < -0.18) { /* L10: */
	h = x + .3;
	h /= .7;
	w1 = a - h * .3;
    }
    else if (x > 0.18) { /* L20: */
	h = x * .75 - .25;
	w1 = b + h / 3.;
    }
    else { /*		Argument Reduction */
	h = x;
	w1 = 0.;
    }

/* L30:              	Series Expansion */

    r = h / (h + 2.);
    t = r * r;
    w = ((p2 * t + p1) * t + p0) / ((q2 * t + q1) * t + 1.);
    return t * 2. * (1. / (1. - r) - r * w) + w1;

} /* rlog1 */

inline double erf__(double x)
{
/* -----------------------------------------------------------------------
 *             EVALUATION OF THE REAL ERROR FUNCTION
 * ----------------------------------------------------------------------- */

    /* Initialized data */

    static double c = .564189583547756;
    static double a[5] = { 7.7105849500132e-5,-.00133733772997339,
	    .0323076579225834,.0479137145607681,.128379167095513 };
    static double b[3] = { .00301048631703895,.0538971687740286,
	    .375795757275549 };
    static double p[8] = { -1.36864857382717e-7,.564195517478974,
	    7.21175825088309,43.1622272220567,152.98928504694,
	    339.320816734344,451.918953711873,300.459261020162 };
    static double q[8] = { 1.,12.7827273196294,77.0001529352295,
	    277.585444743988,638.980264465631,931.35409485061,
	    790.950925327898,300.459260956983 };
    static double r[5] = { 2.10144126479064,26.2370141675169,
	    21.3688200555087,4.6580782871847,.282094791773523 };
    static double s[4] = { 94.153775055546,187.11481179959,
	    99.0191814623914,18.0124575948747 };

    /* Local variables */
    double t, x2, ax, bot, top;

    ax = fabs(x);
    if (ax <= 0.5) {
	t = x * x;
	top = (((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4] + 1.;
	bot = ((b[0] * t + b[1]) * t + b[2]) * t + 1.;

	return x * (top / bot);
    }

    // else:  |x| > 0.5

    if (ax <= 4.) { //  |x| in (0.5, 4]
	top = ((((((p[0] * ax + p[1]) * ax + p[2]) * ax + p[3]) * ax + p[4]) * ax
		+ p[5]) * ax + p[6]) * ax + p[7];
	bot = ((((((q[0] * ax + q[1]) * ax + q[2]) * ax + q[3]) * ax + q[4]) * ax
		+ q[5]) * ax + q[6]) * ax + q[7];
	double R = 0.5 - exp(-x * x) * top / bot + 0.5;
	return (x < 0) ? -R : R;
    }

    // else:  |x| > 4

    if (ax >= 5.8) {
	return x > 0 ? 1 : -1;
    }

    // else:  4 < |x| < 5.8
    x2 = x * x;
    t = 1. / x2;
    top = (((r[0] * t + r[1]) * t + r[2]) * t + r[3]) * t + r[4];
    bot = (((s[0] * t + s[1]) * t + s[2]) * t + s[3]) * t + 1.;
    t = (c - top / (x2 * bot)) / ax;
    double R = 0.5 - exp(-x2) * t + 0.5;
    return (x < 0) ? -R : R;
} /* erf */

inline double erfc1(int ind, double x)
{
/* ----------------------------------------------------------------------- */
/*         EVALUATION OF THE COMPLEMENTARY ERROR FUNCTION */

/*          ERFC1(ind,X) = ERFC(X)            if ind = 0
 *          ERFC1(ind,X) = EXP(X*X)*ERFC(X)   otherwise, the *scaled* erfc(),
 *	    				      (the only one used here)     */
/* ----------------------------------------------------------------------- */

    /* Initialized data */

    static double c = .564189583547756;
    static double a[5] = { 7.7105849500132e-5,-.00133733772997339,
	    .0323076579225834,.0479137145607681,.128379167095513 };
    static double b[3] = { .00301048631703895,.0538971687740286,
	    .375795757275549 };
    static double p[8] = { -1.36864857382717e-7,.564195517478974,
	    7.21175825088309,43.1622272220567,152.98928504694,
	    339.320816734344,451.918953711873,300.459261020162 };
    static double q[8] = { 1.,12.7827273196294,77.0001529352295,
	    277.585444743988,638.980264465631,931.35409485061,
	    790.950925327898,300.459260956983 };
    static double r[5] = { 2.10144126479064,26.2370141675169,
	    21.3688200555087,4.6580782871847,.282094791773523 };
    static double s[4] = { 94.153775055546,187.11481179959,
	    99.0191814623914,18.0124575948747 };

    double ret_val;
    double e, t, w, bot, top;

    double ax = fabs(x);
    //				|X| <= 0.5 */
    if (ax <= 0.5) {
	double t = x * x,
	    top = (((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4] + 1.,
	    bot = ((b[0] * t + b[1]) * t + b[2]) * t + 1.;
	ret_val = 0.5 - x * (top / bot) + 0.5;
	if (ind != 0) {
	    ret_val = exp(t) * ret_val;
	}
	return ret_val;
    }
    // else (L10:):		0.5 < |X| <= 4
    if (ax <= 4.) {
	top = ((((((p[0] * ax + p[1]) * ax + p[2]) * ax + p[3]) * ax + p[4]) * ax
		+ p[5]) * ax + p[6]) * ax + p[7];
	bot = ((((((q[0] * ax + q[1]) * ax + q[2]) * ax + q[3]) * ax + q[4]) * ax
		+ q[5]) * ax + q[6]) * ax + q[7];
	ret_val = top / bot;

    } else { //			|X| > 4
	// L20:
	if (x <= -5.6) {
	    // L50:            	LIMIT VALUE FOR "LARGE" NEGATIVE X
	    ret_val = 2.;
	    if (ind != 0) {
		ret_val = exp(x * x) * 2.;
	    }
	    return ret_val;
	}
	if (ind == 0 && (x > 100. || x * x > -exparg(1))) {
	    // nowadays: -exparg(1) = 709.0825.. : above <===> |x| > 26.6286
	    // Underflow to limit for large positive x   when ind = 0
	    // L60:
	    return 0.;
	}

	// L30:  -5.6 < x < -4  or  4 < x <= 26.6286..
	t = 1. / (x * x);
	top = (((r[0] * t + r[1]) * t + r[2]) * t + r[3]) * t + r[4];
	bot = (((s[0] * t + s[1]) * t + s[2]) * t + s[3]) * t + 1.;
	ret_val = (c - t * top / bot) / ax;
    }

    // L40:                 FINAL ASSEMBLY
    if (ind != 0) {
	if (x < 0.)
	    ret_val = exp(x * x) * 2. - ret_val;
    } else {
	// L41:  ind == 0 :
	w = x * x;
	t = w;
	e = w - t;
	ret_val = (0.5 - e + 0.5) * exp(-t) * ret_val;
	if (x < 0.)
	    ret_val = 2. - ret_val;
    }
    return ret_val;

} /* erfc1 */

inline double gam1(double a)
{
/*     ------------------------------------------------------------------ */
/*     COMPUTATION OF 1/GAMMA(A+1) - 1  FOR -0.5 <= A <= 1.5 */
/*     ------------------------------------------------------------------ */

    double d = a - 0.5;
    // t := if(a > 1/2)  a-1  else  a  ==>  in [-0.5, 0.5]  <==>  |t| <= 0.5
    double t = (d > 0.) ? d - 0.5 : a;

    double w, bot, top;
    if (t < 0.) { /* L30: */
	static double
	    r[9] = { -.422784335098468,-.771330383816272,
		     -.244757765222226,.118378989872749,9.30357293360349e-4,
		     -.0118290993445146,.00223047661158249,2.66505979058923e-4,
		     -1.32674909766242e-4 },
	    s1 = .273076135303957,
	    s2 = .0559398236957378;

	top = (((((((r[8] * t + r[7]) * t + r[6]) * t + r[5]) * t + r[4]
		     ) * t + r[3]) * t + r[2]) * t + r[1]) * t + r[0];
	bot = (s2 * t + s1) * t + 1.;
	w = top / bot;
	R_ifDEBUG_printf("  gam1(a = %.15g): t < 0: w=%.15g\n", a, w);
	if (d > 0.)
	    return t * w / a;
	else
	    return a * (w + 0.5 + 0.5);

    } else if (t == 0) { // L10: a in {0, 1}
	return 0.;

    } else { /* t > 0;  L20: */
	static double
	    p[7] = { .577215664901533,-.409078193005776,
		     -.230975380857675,.0597275330452234,.0076696818164949,
		     -.00514889771323592,5.89597428611429e-4 },
	    q[5] = { 1.,.427569613095214,.158451672430138,
		     .0261132021441447,.00423244297896961 };

	top = (((((p[6] * t + p[5]) * t + p[4]) * t + p[3]) * t + p[2]
		   ) * t + p[1]) * t + p[0];
	bot = (((q[4] * t + q[3]) * t + q[2]) * t + q[1]) * t + 1.;
	w = top / bot;
	R_ifDEBUG_printf("  gam1(a = %.15g): t > 0: (is a < 1.5 ?)  w=%.15g\n",
			 a, w);
	if (d > 0.) /* L21: */
	    return t / a * (w - 0.5 - 0.5);
	else
	    return a * w;
    }
} /* gam1 */

inline double gamln1(double a)
{
/* ----------------------------------------------------------------------- */
/*     EVALUATION OF LN(GAMMA(1 + A)) FOR -0.2 <= A <= 1.25 */
/* ----------------------------------------------------------------------- */

    double w;
    if (a < 0.6) {
	static double p0 = .577215664901533;
	static double p1 = .844203922187225;
	static double p2 = -.168860593646662;
	static double p3 = -.780427615533591;
	static double p4 = -.402055799310489;
	static double p5 = -.0673562214325671;
	static double p6 = -.00271935708322958;
	static double q1 = 2.88743195473681;
	static double q2 = 3.12755088914843;
	static double q3 = 1.56875193295039;
	static double q4 = .361951990101499;
	static double q5 = .0325038868253937;
	static double q6 = 6.67465618796164e-4;
	w = ((((((p6 * a + p5)* a + p4)* a + p3)* a + p2)* a + p1)* a + p0) /
	    ((((((q6 * a + q5)* a + q4)* a + q3)* a + q2)* a + q1)* a + 1.);
	return -(a) * w;
    }
    else { /* 0.6 <= a <= 1.25 */
	static double r0 = .422784335098467;
	static double r1 = .848044614534529;
	static double r2 = .565221050691933;
	static double r3 = .156513060486551;
	static double r4 = .017050248402265;
	static double r5 = 4.97958207639485e-4;
	static double s1 = 1.24313399877507;
	static double s2 = .548042109832463;
	static double s3 = .10155218743983;
	static double s4 = .00713309612391;
	static double s5 = 1.16165475989616e-4;
	double x = a - 0.5 - 0.5;
	w = (((((r5 * x + r4) * x + r3) * x + r2) * x + r1) * x + r0) /
	    (((((s5 * x + s4) * x + s3) * x + s2) * x + s1) * x + 1.);
	return x * w;
    }
} /* gamln1 */

inline double psi(double x)
{
/* ---------------------------------------------------------------------

 *                 Evaluation of the Digamma function psi(x)

 *                           -----------

 *     Psi(xx) is assigned the value 0 when the digamma function cannot
 *     be computed.

 *     The main computation involves evaluation of rational Chebyshev
 *     approximations published in Math. Comp. 27, 123-127(1973) by
 *     Cody, Strecok and Thacher. */

/* --------------------------------------------------------------------- */
/*     Psi was written at Argonne National Laboratory for the FUNPACK */
/*     package of special function subroutines. Psi was modified by */
/*     A.H. Morris (NSWC). */
/* --------------------------------------------------------------------- */

    static double piov4 = .785398163397448; /* == pi / 4 */
/*     dx0 = zero of psi() to extended precision : */
    static double dx0 = 1.461632144968362341262659542325721325;

/* --------------------------------------------------------------------- */
/*     COEFFICIENTS FOR RATIONAL APPROXIMATION OF */
/*     PSI(X) / (X - X0),  0.5 <= X <= 3. */
    static double p1[7] = { .0089538502298197,4.77762828042627,
	    142.441585084029,1186.45200713425,3633.51846806499,
	    4138.10161269013,1305.60269827897 };
    static double q1[6] = { 44.8452573429826,520.752771467162,
	    2210.0079924783,3641.27349079381,1908.310765963,
	    6.91091682714533e-6 };
/* --------------------------------------------------------------------- */


/* --------------------------------------------------------------------- */
/*     COEFFICIENTS FOR RATIONAL APPROXIMATION OF */
/*     PSI(X) - LN(X) + 1 / (2*X),  X > 3. */

    static double p2[4] = { -2.12940445131011,-7.01677227766759,
	    -4.48616543918019,-.648157123766197 };
    static double q2[4] = { 32.2703493791143,89.2920700481861,
	    54.6117738103215,7.77788548522962 };
/* --------------------------------------------------------------------- */

    int i, m, n, nq;
    double d2;
    double w, z;
    double den, aug, sgn, xmx0, xmax1, upper, xsmall;

/* --------------------------------------------------------------------- */


/*     MACHINE DEPENDENT CONSTANTS ... */

/* --------------------------------------------------------------------- */
/*	  XMAX1	 = THE SMALLEST POSITIVE FLOATING POINT CONSTANT
		   WITH ENTIRELY INT REPRESENTATION.  ALSO USED
		   AS NEGATIVE OF LOWER BOUND ON ACCEPTABLE NEGATIVE
		   ARGUMENTS AND AS THE POSITIVE ARGUMENT BEYOND WHICH
		   PSI MAY BE REPRESENTED AS LOG(X).
 * Originally:  xmax1 = amin1(ipmpar(3), 1./spmpar(1))  */
    xmax1 = (double) INT_MAX;
    d2 = 0.5 / Rf_d1mach(3); /*= 0.5 / (0.5 * DBL_EPS) = 1/DBL_EPSILON = 2^52 */
    if(xmax1 > d2) xmax1 = d2;

/* --------------------------------------------------------------------- */
/*        XSMALL = ABSOLUTE ARGUMENT BELOW WHICH PI*COTAN(PI*X) */
/*                 MAY BE REPRESENTED BY 1/X. */
    xsmall = 1e-9;
/* --------------------------------------------------------------------- */
    aug = 0.;
    if (x < 0.5) {
/* --------------------------------------------------------------------- */
/*     X < 0.5,  USE REFLECTION FORMULA */
/*     PSI(1-X) = PSI(X) + PI * COTAN(PI*X) */
/* --------------------------------------------------------------------- */
	if (fabs(x) <= xsmall) {

	    if (x == 0.) {
		goto L_err;
	    }
/* --------------------------------------------------------------------- */
/*     0 < |X| <= XSMALL.  USE 1/X AS A SUBSTITUTE */
/*     FOR  PI*COTAN(PI*X) */
/* --------------------------------------------------------------------- */
	    aug = -1. / x;
	} else { /* |x| > xsmall */
/* --------------------------------------------------------------------- */
/*     REDUCTION OF ARGUMENT FOR COTAN */
/* --------------------------------------------------------------------- */
	    /* L100: */
	    w = -x;
	    sgn = piov4;
	    if (w <= 0.) {
		w = -w;
		sgn = -sgn;
	    }
/* --------------------------------------------------------------------- */
/*     MAKE AN ERROR EXIT IF |X| >= XMAX1 */
/* --------------------------------------------------------------------- */
	    if (w >= xmax1) {
		goto L_err;
	    }
	    nq = (int) w;
	    w -= (double) nq;
	    nq = (int) (w * 4.);
	    w = (w - (double) nq * 0.25) * 4.;
/* --------------------------------------------------------------------- */
/*     W IS NOW RELATED TO THE FRACTIONAL PART OF  4. * X. */
/*     ADJUST ARGUMENT TO CORRESPOND TO VALUES IN FIRST */
/*     QUADRANT AND DETERMINE SIGN */
/* --------------------------------------------------------------------- */
	    n = nq / 2;
	    if (n + n != nq) {
		w = 1. - w;
	    }
	    z = piov4 * w;
	    m = n / 2;
	    if (m + m != n) {
		sgn = -sgn;
	    }
/* --------------------------------------------------------------------- */
/*     DETERMINE FINAL VALUE FOR  -PI*COTAN(PI*X) */
/* --------------------------------------------------------------------- */
	    n = (nq + 1) / 2;
	    m = n / 2;
	    m += m;
	    if (m == n) {
/* --------------------------------------------------------------------- */
/*     CHECK FOR SINGULARITY */
/* --------------------------------------------------------------------- */
		if (z == 0.) {
		    goto L_err;
		}
/* --------------------------------------------------------------------- */
/*     USE COS/SIN AS A SUBSTITUTE FOR COTAN, AND */
/*     SIN/COS AS A SUBSTITUTE FOR TAN */
/* --------------------------------------------------------------------- */
		aug = sgn * (cos(z) / sin(z) * 4.);

	    } else { /* L140: */
		aug = sgn * (sin(z) / cos(z) * 4.);
	    }
	}

	x = 1. - x;

    }
    /* L200: */
    if (x <= 3.) {
/* --------------------------------------------------------------------- */
/*     0.5 <= X <= 3. */
/* --------------------------------------------------------------------- */
	den = x;
	upper = p1[0] * x;

	for (i = 1; i <= 5; ++i) {
	    den = (den + q1[i - 1]) * x;
	    upper = (upper + p1[i]) * x;
	}

	den = (upper + p1[6]) / (den + q1[5]);
	xmx0 = x - dx0;
	return den * xmx0 + aug;
    }

/* --------------------------------------------------------------------- */
/*     IF X >= XMAX1, PSI = LN(X) */
/* --------------------------------------------------------------------- */
    if (x < xmax1) {
/* --------------------------------------------------------------------- */
/*     3. < X < XMAX1 */
/* --------------------------------------------------------------------- */
	w = 1. / (x * x);
	den = w;
	upper = p2[0] * w;

	for (i = 1; i <= 3; ++i) {
	    den = (den + q2[i - 1]) * w;
	    upper = (upper + p2[i]) * w;
	}

	aug = upper / (den + q2[3]) - 0.5 / x + aug;
    }
    return aug + log(x);

/* --------------------------------------------------------------------- */
/*     ERROR RETURN */
/* --------------------------------------------------------------------- */
L_err:
    return 0.;
} /* psi */

inline double betaln(double a0, double b0)
{
/* -----------------------------------------------------------------------
 *     Evaluation of the logarithm of the beta function  ln(beta(a0,b0))   -- in R:  lbeta(a0, b0)
 * ----------------------------------------------------------------------- */

    double
	a = min(a0 ,b0),
	b = max(a0, b0);

    if (a < 8.) {
	if (a < 1.) {
/* ----------------------------------------------------------------------- */
//                    		A < 1
/* ----------------------------------------------------------------------- */
	    if (b < 8.)
		return gamln(a) + (gamln(b) - gamln(a+b));
	    else
		return gamln(a) + algdiv(a, b);
	}
	/* else */
/* ----------------------------------------------------------------------- */
//				1 <= A < 8
/* ----------------------------------------------------------------------- */
	double w;
	if (a < 2.) {
	    if (b <= 2.) {
		return gamln(a) + gamln(b) - gsumln(a, b);
	    }
	    /* else */

	    if (b < 8.) {
		w = 0.;
		goto L40;
	    }
	    return gamln(a) + algdiv(a, b);
	}
	// else L30:    REDUCTION OF A WHEN B <= 1000

	if (b <= 1e3) {
	    int n; n = (int)(a - 1.);
	    w = 1.;
	    for (int i = 1; i <= n; ++i) {
		a += -1.;
		double h = a / b;
		w *= h / (h + 1.);
	    }
	    w = log(w);

	    if (b >= 8.)
		return w + gamln(a) + algdiv(a, b);

	    // else
	L40:
	    // 	1 < A <= B < 8 :  reduction of B
	    n = (int)(b - 1.);
	    double z = 1.;
	    for (int i = 1; i <= n; ++i) {
		b += -1.;
		z *= b / (a + b);
	    }
	    return w + log(z) + (gamln(a) + (gamln(b) - gsumln(a, b)));
	}
	else { // L50:	reduction of A when  B > 1000
	    int n = (int)(a - 1.);
	    w = 1.;
	    for (int i = 1; i <= n; ++i) {
		a += -1.;
		w *= a / (a / b + 1.);
	    }
	    return log(w) - n * log(b) + (gamln(a) + algdiv(a, b));
	}

    } else {
/* ----------------------------------------------------------------------- */
	// L60:			A >= 8
/* ----------------------------------------------------------------------- */
	static double e = .918938533204673;/* e == 0.5*LN(2*PI) */
	double
	    w = bcorr(a, b),
	    h = a / b,
	    u = -(a - 0.5) * log(h / (h + 1.)),
	    v = b * alnrel(h);
	if (u > v)
	    return log(b) * -0.5 + e + w - v - u;
	else
	    return log(b) * -0.5 + e + w - u - v;
    }

} /* betaln */

inline double gsumln(double a, double b)
{
/* ----------------------------------------------------------------------- */
/*          EVALUATION OF THE FUNCTION LN(GAMMA(A + B)) */
/*          FOR 1 <= A <= 2  AND  1 <= B <= 2 */
/* ----------------------------------------------------------------------- */

    double x = a + b - 2.;/* in [0, 2] */

    if (x <= 0.25)
	return gamln1(x + 1.);

    /* else */
    if (x <= 1.25)
	return gamln1(x) + alnrel(x);
    /* else x > 1.25 : */
    return gamln1(x - 1.) + log(x * (x + 1.));

} /* gsumln */

inline double bcorr(double a0, double b0)
{
/* ----------------------------------------------------------------------- */

/*     EVALUATION OF  DEL(A0) + DEL(B0) - DEL(A0 + B0)  WHERE */
/*     LN(GAMMA(A)) = (A - 0.5)*LN(A) - A + 0.5*LN(2*PI) + DEL(A). */
/*     IT IS ASSUMED THAT A0 >= 8 AND B0 >= 8. */

/* ----------------------------------------------------------------------- */
    /* Initialized data */

    static double c0 = .0833333333333333;
    static double c1 = -.00277777777760991;
    static double c2 = 7.9365066682539e-4;
    static double c3 = -5.9520293135187e-4;
    static double c4 = 8.37308034031215e-4;
    static double c5 = -.00165322962780713;

    /* Local variables */
    double a, b, c, h, t, w, x, s3, s5, x2, s7, s9, s11;
/* ------------------------ */
    a = min(a0, b0);
    b = max(a0, b0);

    h = a / b;
    c = h / (h + 1.);
    x = 1. / (h + 1.);
    x2 = x * x;

/*                SET s<n> := (1 - x^n)/(1 - x) */
    s3 = x + x2 + 1.;
    s5 = x + x2 * s3 + 1.;
    s7 = x + x2 * s5 + 1.;
    s9 = x + x2 * s7 + 1.;
    s11= x + x2 * s9 + 1.;

/*                SET W = DEL(B) - DEL(A + B) */

    t = 1. / b; t *= t; // t := 1 / b^2
    w = ((((c5 * s11 * t + c4 * s9) * t + c3 * s7) * t + c2 * s5) * t + c1 *
	    s3) * t + c0;
    w *= c / b;

/*                   COMPUTE  DEL(A) + W */
    t = 1. / a; t *= t; // t:= 1 / a^2
    return (((((c5 * t + c4) * t + c3) * t + c2) * t + c1) * t + c0) / a + w;
} /* bcorr */

inline double algdiv(double a, double b)
{
/* ----------------------------------------------------------------------- */

/*     COMPUTATION OF LN(GAMMA(B)/GAMMA(A+B)) WHEN B >= 8 */

/*                         -------- */

/*     IN THIS ALGORITHM, DEL(X) IS THE FUNCTION DEFINED BY */
/*     LN(GAMMA(X)) = (X - 0.5)*LN(X) - X + 0.5*LN(2*PI) + DEL(X). */

/* ----------------------------------------------------------------------- */

    /* Initialized data */

    static double c0 = .0833333333333333;
    static double c1 = -.00277777777760991;
    static double c2 = 7.9365066682539e-4;
    static double c3 = -5.9520293135187e-4;
    static double c4 = 8.37308034031215e-4;
    static double c5 = -.00165322962780713;

    double c, d, h, t, u, v, w, x, s3, s5, x2, s7, s9, s11;

/* ------------------------ */
    if (a > b) {
	h = b / a;
	c = 1. / (h + 1.);
	x = h / (h + 1.);
	d = a + (b - 0.5);
    }
    else {
	h = a / b;
	c = h / (h + 1.);
	x = 1. / (h + 1.);
	d = b + (a - 0.5);
    }

/* Set s<n> = (1 - x^n)/(1 - x) : */

    x2 = x * x;
    s3 = x + x2 + 1.;
    s5 = x + x2 * s3 + 1.;
    s7 = x + x2 * s5 + 1.;
    s9 = x + x2 * s7 + 1.;
    s11 = x + x2 * s9 + 1.;

/* w := Del(b) - Del(a + b) */

    t = 1./ (b * b);
    w = ((((c5 * s11 * t + c4 * s9) * t + c3 * s7) * t + c2 * s5) * t + c1 *
	    s3) * t + c0;
    w *= c / b;

/*                    COMBINE THE RESULTS */

    u = d * alnrel(a / b);
    v = a * (log(b) - 1.);
    if (u > v)
	return w - v - u;
    else
	return w - u - v;
} /* algdiv */

inline double gamln(double a)
{
/* -----------------------------------------------------------------------
 *            Evaluation of  ln(gamma(a))  for positive a
 * ----------------------------------------------------------------------- */
/*     Written by Alfred H. Morris */
/*          Naval Surface Warfare Center */
/*          Dahlgren, Virginia */
/* ----------------------------------------------------------------------- */

    static double d = .418938533204673;/* d == 0.5*(LN(2*PI) - 1) */

    static double c0 = .0833333333333333;
    static double c1 = -.00277777777760991;
    static double c2 = 7.9365066682539e-4;
    static double c3 = -5.9520293135187e-4;
    static double c4 = 8.37308034031215e-4;
    static double c5 = -.00165322962780713;

    if (a <= 0.8)
	return gamln1(a) - log(a); /* ln(G(a+1)) - ln(a) == ln(G(a+1)/a) = ln(G(a)) */
    else if (a <= 2.25)
	return gamln1(a - 0.5 - 0.5);

    else if (a < 10.) {
	int i, n = (int)(a - 1.25);
	double t = a;
	double w = 1.;
	for (i = 1; i <= n; ++i) {
	    t += -1.;
	    w *= t;
	}
	return gamln1(t - 1.) + log(w);
    }
    else { /* a >= 10 */
	double t = 1. / (a * a);
	double w = (((((c5 * t + c4) * t + c3) * t + c2) * t + c1) * t + c0) / a;
	return d + w + (a - 0.5) * (log(a) - 1.);
    }
} /* gamln */

#undef R_Log1_Exp
#define R_Log1_Exp(x)   ((x) > -R_M_LN2 ? log(-expm1(x)) : log1p(-exp(x)))
#undef min
#undef max

// ---- R/nmath/pbeta.c


inline double pbeta_raw(double x, double a, double b, int lower_tail, int log_p)
{
    if (x >= 1) // may happen when called from qbeta()
	return R_DT_1;
    // treat limit cases correctly here:
    if(a == 0 || b == 0 || !R_FINITE(a) || !R_FINITE(b)) {
	// NB:  0 <= x < 1 :
	if(a == 0 && b == 0) // point mass 1/2 at each of {0,1} :
	    return (log_p ? -R_M_LN2 : 0.5);
	if (a == 0 || a/b == 0) // point mass 1 at 0 ==> P(X <= x) = 1, all x >= 0
	    return R_DT_1;
	if (b == 0 || b/a == 0) // point mass 1 at 1 ==> P(X <= x) = 0, all x < 1
	    return R_DT_0;
	// else, remaining case:  a = b = Inf : point mass 1 at 1/2
	if (x < 0.5) return R_DT_0; else return R_DT_1;
    }
    if (x <= 0)
	return R_DT_0;

    // Now:  0 < a < Inf;  0 < b < Inf  and  0 < x < 1

    double x1 = 0.5 - x + 0.5, w, wc;
    int ierr;
    //====
    bratio(a, b, x, x1, &w, &wc, &ierr, log_p); /* -> ./toms708.c */
    //====
    // ierr in {10,14} <==> bgrat() error code ierr-10 in 1:4; for 1 and 4, warned *there*
    if(ierr && ierr != 11 && ierr != 14)
	MATHLIB_WARNING6(_("pbeta_raw(%g, a=%g, b=%g, lower=%d, log=%d) -> bratio() gave error code %d"),
			 x, a,b, lower_tail, log_p, ierr);
    return lower_tail ? w : wc;
} /* pbeta_raw() */

inline double pbeta(double x, double a, double b, int lower_tail, int log_p)
{
    if (ISNAN(x) || ISNAN(a) || ISNAN(b)) return x + a + b;

    if (a < 0 || b < 0) ML_WARN_return_NAN;
    // allowing a==0 and b==0  <==> treat as one- or two-point mass

    return pbeta_raw(x, a, b, lower_tail, log_p);
}

// ---- R/nmath/dt.c

inline double dt(double x, double n, int give_log)
{
    if (ISNAN(x) || ISNAN(n))
	return x + n;
    if (n <= 0) ML_WARN_return_NAN;
    if(!R_FINITE(x))
	return R_D__0;
    if(!R_FINITE(n))
	return dnorm(x, 0., 1., give_log);

    double u, t = -bd0(n/2.,(n+1)/2.) + stirlerr((n+1)/2.) - stirlerr(n/2.),
	x2n = x*x/n, // in  [0, Inf]
	ax = 0., // <- -Wpedantic
	l_x2n; // := log(sqrt(1 + x2n)) = log(1 + x2n)/2
    bool lrg_x2n =  (x2n > 1./DBL_EPSILON);
    if (lrg_x2n) { // large x^2/n :
	ax = fabs(x);
	l_x2n = log(ax) - log(n)/2.; // = log(x2n)/2 = 1/2 * log(x^2 / n)
	u = //  log(1 + x2n) * n/2 =  n * log(1 + x2n)/2 =
	    n * l_x2n;
    }
    else if (x2n > 0.2) {
	l_x2n = log(1 + x2n)/2.;
	u = n * l_x2n;
    } else {
	l_x2n = log1p(x2n)/2.;
	u = -bd0(n/2.,(n+x*x)/2.) + x*x/2.;
    }

    //old: return  R_D_fexp(M_2PI*(1+x2n), t-u);

    // R_D_fexp(f,x) :=  (give_log ? -0.5*log(f)+(x) : exp(x)/sqrt(f))
    // f = 2pi*(1+x2n)
    //  ==> 0.5*log(f) = log(2pi)/2 + log(1+x2n)/2 = log(2pi)/2 + l_x2n
    //	     1/sqrt(f) = 1/sqrt(2pi * (1+ x^2 / n))
    //		       = 1/sqrt(2pi)/(|x|/sqrt(n)*sqrt(1+1/x2n))
    //		       = M_1_SQRT_2PI * sqrt(n)/ (|x|*sqrt(1+1/x2n))
    if(give_log)
	return t-u - (M_LN_SQRT_2PI + l_x2n);

    // else :  if(lrg_x2n) : sqrt(1 + 1/x2n) ='= sqrt(1) = 1
    double I_sqrt_ = (lrg_x2n ? sqrt(n)/ax : exp(-l_x2n));
    return exp(t-u) * M_1_SQRT_2PI * I_sqrt_;
}

// ---- R/nmath/pt.c

inline double pt(double x, double n, int lower_tail, int log_p)
{
/* return  P[ T <= x ]	where
 * T ~ t_{n}  (t distrib. with n degrees of freedom).

 *	--> ./pnt.c for NON-central
 */
    double val, nx;
    if (ISNAN(x) || ISNAN(n))
	return x + n;
    if (n <= 0.0) ML_WARN_return_NAN;

    if(!R_FINITE(x))
	return (x < 0) ? R_DT_0 : R_DT_1;
    if(!R_FINITE(n))
	return pnorm(x, 0.0, 1.0, lower_tail, log_p);


    nx = 1 + (x/n)*x;
    /* FIXME: This test is probably losing rather than gaining precision,
     * now that pbeta(*, log_p = TRUE) is much better.
     * Note however that a version of this test *is* needed for x*x > D_MAX */
    if(nx > 1e100) { /* <==>  x*x > 1e100 * n  */
	/* Danger of underflow. So use Abramowitz & Stegun 26.5.4
	   pbeta(z, a, b) ~ z^a(1-z)^b / aB(a,b) ~ z^a / aB(a,b),
	   with z = 1/nx,  a = n/2,  b= 1/2 :
	*/
	double lval;
	lval = -0.5*n*(2*log(fabs(x)) - log(n))
		- lbeta(0.5*n, 0.5) - log(0.5*n);
	val = log_p ? lval : exp(lval);
    } else {
	val = (n > x * x)
	    ? pbeta (x * x / (n + x * x), 0.5, n / 2., /*lower_tail*/0, log_p)
	    : pbeta (1. / nx,             n / 2., 0.5, /*lower_tail*/1, log_p);
    }

    /* Use "1 - v"  if	lower_tail  and	 x > 0 (but not both):*/
    if(x <= 0.)
	lower_tail = !lower_tail;

    if(log_p) {
	if(lower_tail) return log1p(-0.5*exp(val));
	else return val - R_M_LN2; /* = log(.5* pbeta(....)) */
    }
    else {
	val /= 2.;
	return R_D_Cval(val);
    }
}

// ---- R/nmath/qt.c

inline double qt(double p, double ndf, int lower_tail, int log_p)
{
    const static double eps = 1.e-12;

    double P, q;

    if (ISNAN(p) || ISNAN(ndf))
	return p + ndf;

    R_Q_P01_boundaries(p, ML_NEGINF, ML_POSINF);

    if (ndf <= 0) ML_WARN_return_NAN;

    if (ndf < 1) { /* based on qnt */
	const static double accu = 1e-13;
	const static double Eps = 1e-11; /* must be > accu */

	double ux, lx, nx, pp;

	int iter = 0;

	p = R_DT_qIv(p);

	/* Invert pt(.) :
	 * 1. finding an upper and lower bound */
	if(p > 1 - DBL_EPSILON) return ML_POSINF;
	pp = fmin2(1 - DBL_EPSILON, p * (1 + Eps));
	for(ux = 1.; ux < DBL_MAX && pt(ux, ndf, TRUE, FALSE) < pp; ux *= 2);
	pp = p * (1 - Eps);
	for(lx =-1.; lx > -DBL_MAX && pt(lx, ndf, TRUE, FALSE) > pp; lx *= 2);

	/* 2. interval (lx,ux)  halving
	   regula falsi failed on qt(0.1, 0.1)
	 */
	do {
	    nx = 0.5 * (lx + ux);
	    if (pt(nx, ndf, TRUE, FALSE) > p) ux = nx; else lx = nx;
	} while ((ux - lx) / fabs(nx) > accu && ++iter < 1000);

	if(iter >= 1000) ML_WARNING(ME_PRECISION, "qt");

	return 0.5 * (lx + ux);
    }

    /* Old comment:
     * FIXME: "This test should depend on  ndf  AND p  !!
     * -----  and in fact should be replaced by
     * something like Abramowitz & Stegun 26.7.5 (p.949)"
     *
     * That would say that if the qnorm value is x then
     * the result is about x + (x^3+x)/4df + (5x^5+16x^3+3x)/96df^2
     * The differences are tiny even if x ~ 1e5, and qnorm is not
     * that accurate in the extreme tails.
     */
    if (ndf > 1e20) return qnorm(p, 0., 1., lower_tail, log_p);

    P = R_D_qIv(p); /* if exp(p) underflows, we fix below */

    bool neg = (!lower_tail || P < 0.5) && (lower_tail || P > 0.5),
	is_neg_lower = (lower_tail == neg); /* both TRUE or FALSE == !xor */
    if(neg)
	P = 2 * (log_p ? (lower_tail ? P : -expm1(p)) : R_D_Lval(p));
    else
	P = 2 * (log_p ? (lower_tail ? -expm1(p) : P) : R_D_Cval(p));
    /* 0 <= P <= 1 ; P = 2*min(P', 1 - P')  in all cases */

     if (fabs(ndf - 2) < eps) {	/* df ~= 2 */
	if(P > DBL_MIN) {
	    if(3* P < DBL_EPSILON) /* P ~= 0 */
		q = 1 / sqrt(P);
	    else if (P > 0.9)	   /* P ~= 1 */
		q = (1 - P) * sqrt(2 /(P * (2 - P)));
	    else /* eps/3 <= P <= 0.9 */
		q = sqrt(2 / (P * (2 - P)) - 2);
	}
	else { /* P << 1, q = 1/sqrt(P) = ... */
	    if(log_p)
		q = is_neg_lower ? exp(- p/2) / R_M_SQRT2 : 1/sqrt(-expm1(p));
	    else
		q = ML_POSINF;
	}
    }
    else if (ndf < 1 + eps) { /* df ~= 1  (df < 1 excluded above): Cauchy */
	if(P == 1.) q = 0; // some versions of tanpi give Inf, some NaN
	else if(P > 0)
	    q = 1/Rtanpi(P/2.);/* == - tan((P+1) * R_M_PI_2) -- suffers for P ~= 0 */

	else { /* P = 0, but maybe = 2*exp(p) ! */
	    if(log_p) /* 1/tan(e) ~ 1/e */
		q = is_neg_lower ? R_M_1_PI * exp(-p) : -1./(R_M_PI * expm1(p));
	    else
		q = ML_POSINF;
	}
    }
    else {		/*-- usual case;  including, e.g.,  df = 1.1 */
	double x = 0., y, log_P2 = 0./* -Wall */,
	    a = 1 / (ndf - 0.5),
	    b = 48 / (a * a),
	    c = ((20700 * a / b - 98) * a - 16) * a + 96.36,
	    d = ((94.5 / (b + c) - 3) / b + 1) * sqrt(a * R_M_PI_2) * ndf;
	bool
	    P_ok1 = P > DBL_MIN || !log_p,
	    P_ok  = P_ok1; // when true (after check below), use "normal scale": log_p=FALSE
	if(P_ok1) {
	    y = pow(d * P, 2.0 / ndf);
	    P_ok = (y >= DBL_EPSILON);
	}
	if(!P_ok) {// log.p && P very.small  ||  (d*P)^(2/df) =: y < eps_c
	    log_P2 = is_neg_lower ? R_D_log(p) : R_D_LExp(p); /* == log(P / 2) */
	    x = (log(d) + R_M_LN2 + log_P2) / ndf;
	    y = exp(2 * x);
	}

	if ((ndf < 2.1 && P > 0.5) || y > 0.05 + a) { /* P > P0(df) */
	    /* Asymptotic inverse expansion about normal */
	    if(P_ok)
		x = qnorm(0.5 * P, 0., 1., /*lower_tail*/TRUE,  /*log_p*/FALSE);
	    else /* log_p && P underflowed */
		x = qnorm(log_P2,  0., 1., lower_tail,	        /*log_p*/ TRUE);

	    y = x * x;
	    if (ndf < 5)
		c += 0.3 * (ndf - 4.5) * (x + 0.6);
	    c = (((0.05 * d * x - 5) * x - 7) * x - 2) * x + b + c;
	    y = (((((0.4 * y + 6.3) * y + 36) * y + 94.5) / c
		  - y - 3) / b + 1) * x;
	    y = expm1(a * y * y);
	    q = sqrt(ndf * y);
	} else if(!P_ok && x < - R_M_LN2 * DBL_MANT_DIG) {/* 0.5* log(DBL_EPSILON) */
	    /* y above might have underflown */
	    q = sqrt(ndf) * exp(-x);
	}
	else { /* re-use 'y' from above */
	    y = ((1 / (((ndf + 6) / (ndf * y) - 0.089 * d - 0.822)
		       * (ndf + 2) * 3) + 0.5 / (ndf + 4))
		 * y - 1) * (ndf + 1) / (ndf + 2) + 1 / y;
	    q = sqrt(ndf * y);
	}


	/* Now apply 2-term Taylor expansion improvement (1-term = Newton):
	 * as by Hill (1981) [ref.above] */

	/* FIXME: This can be far from optimal when log_p = TRUE
	 *      but is still needed, e.g. for qt(-2, df=1.01, log=TRUE).
	 *	Probably also improvable when  lower_tail = FALSE */

	if(P_ok1) {
	    double M = fabs(sqrt(DBL_MAX/2.) - ndf);
	    int it=0;
	    while(it++ < 10 && (y = dt(q, ndf, FALSE)) > 0 &&
		  R_FINITE(x = (pt(q, ndf, FALSE, FALSE) - P/2) / y) &&
		  fabs(x) > 1e-14*fabs(q)) {
		/* Newton (=Taylor 1 term):
		 *  q += x;
		 * Taylor 2-term : */
		double F = (fabs(q) < M) ?
		    q * (ndf + 1) / (2 * (q * q + ndf)) :
		        (ndf + 1) / (2 * (q     + ndf/q)),
		    del_q = x * (1. + x * F);
		if(R_FINITE(del_q) && R_FINITE(q + del_q))
		    q += del_q;
		else if(R_FINITE(x) && R_FINITE(q + x))
		    q += x;
		else // FIXME??  if  q+x = +/-Inf is *better* than q should use it
		    break; // cannot improve  q  with a Newton/Taylor step
	    }
	}
    }
     return neg ? -q : q;
}

// ---- R/nmath/pnt.c
/*  Algorithm AS 243  Lenth,R.V. (1989). Appl. Statist., Vol.38, 185-189.
 *  ----------------
 *  Cumulative probability at t of the non-central t-distribution
 *  with df degrees of freedom (may be fractional) and non-centrality
 *  parameter delta.
 *
 *  NOTE
 *
 *    Requires the following auxiliary routines:
 *
 *	lgammafn(x)	- log gamma function
 *	pbeta(x, a, b)	- incomplete beta function
 *	pnorm(x)	- normal distribution function
 *
 *  CONSTANTS
 *
 *    M_SQRT_2dPI  = 1/ {gamma(1.5) * sqrt(2)} = sqrt(2 / pi)
 *    M_LN_SQRT_PI = ln(sqrt(pi)) = ln(pi)/2
 */

/*----------- DEBUGGING -------------
 *
 *	make CFLAGS='-DDEBUG_pnt -g'

 * -- Feb.3, 1999; M.Maechler:
	- For 't > ncp > 20' (or so)	the result is completely WRONG!  <== no longer true
	- but for ncp > 100
 */

inline double pnt(double t, double df, double ncp, int lower_tail, int log_p)
{
    double albeta, a, b, del, errbd, lambda, rxb, tt, x;
    LDOUBLE geven, godd, p, q, s, tnc, xeven, xodd;
    int it, negdel;

    /* note - itrmax and errmax may be changed to suit one's needs. */

    const int itrmax = 1000;
    const static double errmax = 1.e-12;

    if (df <= 0.0) ML_WARN_return_NAN;
    if(ncp == 0.0) return pt(t, df, lower_tail, log_p);

    if(!R_FINITE(t))
	return (t < 0) ? R_DT_0 : R_DT_1;
    if (t >= 0.) {
	negdel = FALSE; tt = t;	 del = ncp;
    }
    else {
	/* We deal quickly with left tail if extreme,
	   since pt(q, df, ncp) <= pt(0, df, ncp) = \Phi(-ncp) */
	if (ncp > 40 && (!log_p || !lower_tail)) return R_DT_0;
	negdel = TRUE;	tt = -t; del = -ncp;
    }

    if (df > 4e5 || del*del > 2*R_M_LN2*(-(DBL_MIN_EXP))) {
	/*-- 2nd part: if del > 37.62, then p=0 below
	  FIXME: test should depend on `df', `tt' AND `del' ! */
	/* Approx. from	 Abramowitz & Stegun 26.7.10 (p.949) */
	s = 1./(4.*df);

	return pnorm((double)(tt*(1. - s)), del,
		     sqrt((double) (1. + tt*tt*2.*s)),
		     lower_tail != negdel, log_p);
    }

    /* initialize twin series */
    /* Guenther, J. (1978). Statist. Computn. Simuln. vol.6, 199. */

    x = t * t;
    rxb = df/(x + df);/* := (1 - x) {x below} -- but more accurately */
    x = x / (x + df);/* in [0,1) */
    if (x > 0.) {/* <==>  t != 0 */
	lambda = del * del;
	p = .5 * exp(-.5 * lambda);
	if(p == 0.) { /* underflow! */

	    /*========== really use an other algorithm for this case !!! */
	    ML_WARNING(ME_UNDERFLOW, "pnt");
	    ML_WARNING(ME_RANGE, "pnt"); /* |ncp| too large */
	    return R_DT_0;
	}
	q = M_SQRT_2dPI * p * del;
	s = .5 - p;
	/* s = 0.5 - p = 0.5*(1 - exp(-.5 L)) =  -0.5*expm1(-.5 L)) */
	if(s < 1e-7)
	    s = -0.5 * expm1(-0.5 * lambda);
	a = .5;
	b = .5 * df;
	/* rxb = (1 - x) ^ b   [ ~= 1 - b*x for tiny x --> see 'xeven' below]
	 *       where '(1 - x)' =: rxb {accurately!} above */
	rxb = pow(rxb, b);
	albeta = M_LN_SQRT_PI + lgammafn(b) - lgammafn(.5 + b);
	xodd = pbeta(x, a, b, /*lower*/TRUE, /*log_p*/FALSE);
	godd = 2. * rxb * exp(a * log(x) - albeta);
	tnc = b * x;
	xeven = (tnc < DBL_EPSILON) ? tnc : 1. - rxb;
	geven = tnc * rxb;
	tnc = p * xodd + q * xeven;

	/* repeat until convergence or iteration limit */
	for(it = 1; it <= itrmax; it++) {
	    a += 1.;
	    xodd  -= godd;
	    xeven -= geven;
	    godd  *= x * (a + b - 1.) / a;
	    geven *= x * (a + b - .5) / (a + .5);
	    p *= lambda / (2 * it);
	    q *= lambda / (2 * it + 1);
	    tnc += p * xodd + q * xeven;
	    s -= p;
	    /* R 2.4.0 added test for rounding error here. */
	    if(s < -1.e-10) { /* happens e.g. for (t,df,ncp)=(40,10,38.5), after 799 it.*/
		ML_WARNING(ME_PRECISION, "pnt");
		goto finis;
	    }
	    if(s <= 0 && it > 1) goto finis;
	    errbd = (double)(2. * s * (xodd - godd));
	    if(fabs(errbd) < errmax) goto finis;/*convergence*/
	}
	/* non-convergence:*/
	ML_WARNING(ME_NOCONV, "pnt");
    }
    else { /* x = t = 0 */
	tnc = 0.;
    }
 finis:
    tnc += pnorm(- del, 0., 1., /*lower*/TRUE, /*log_p*/FALSE);

    lower_tail = lower_tail != negdel; /* xor */
    if(tnc > 1 - 1e-10 && lower_tail)
	ML_WARNING(ME_PRECISION, "pnt{final}");

    return R_DT_val(fmin2((double)tnc, 1.) /* Precaution */);
}

// ---- R/nmath/qnt.c

inline double qnt(double p, double df, double ncp, int lower_tail, int log_p)
{
    const static double accu = 1e-13;
    const static double Eps = 1e-11; /* must be > accu */

    double ux, lx, nx, pp;

    if (ISNAN(p) || ISNAN(df) || ISNAN(ncp))
	return p + df + ncp;
    /* Was
     * df = floor(df + 0.5);
     * if (df < 1 || ncp < 0) ML_WARN_return_NAN;
     */
    if (df <= 0.0) ML_WARN_return_NAN;

    if(ncp == 0.0 && df >= 1.0) return qt(p, df, lower_tail, log_p);

    R_Q_P01_boundaries(p, ML_NEGINF, ML_POSINF);

    if (!R_FINITE(df)) // df = Inf ==> limit N(ncp,1)
	return qnorm(p, ncp, 1., lower_tail, log_p);

    p = R_DT_qIv(p);

    /* Invert pnt(.) :
     * 1. finding an upper and lower bound */
    if(p > 1 - DBL_EPSILON) return ML_POSINF;
    pp = fmin2(1 - DBL_EPSILON, p * (1 + Eps));
    for(ux = fmax2(1., ncp);
	ux < DBL_MAX && pnt(ux, df, ncp, TRUE, FALSE) < pp;
	ux *= 2);
    pp = p * (1 - Eps);
    for(lx = fmin2(-1., -ncp);
	lx > -DBL_MAX && pnt(lx, df, ncp, TRUE, FALSE) > pp;
	lx *= 2);

    /* 2. interval (lx,ux)  halving : */
    do {
	nx = 0.5 * (lx + ux); // could be zero
	if (pnt(nx, df, ncp, TRUE, FALSE) > p) ux = nx; else lx = nx;
    }
    while ((ux - lx) > accu * fmax2(fabs(lx), fabs(ux)));

    return 0.5 * (lx + ux);
}

// ---- R/nmath/dnt.c

inline double dnt(double x, double df, double ncp, int give_log)
{
    double u;
    if (ISNAN(x) || ISNAN(df))
	return x + df;

    /* If non-positive df then error */
    if (df <= 0.0) ML_WARN_return_NAN;

    if(ncp == 0.0) return dt(x, df, give_log);

    /* If x is infinite then return 0 */
    if(!R_FINITE(x))
	return R_D__0;

    /* If infinite df then the density is identical to a
       normal distribution with mean = ncp.  However, the formula
       loses a lot of accuracy around df=1e9
    */
    if(!R_FINITE(df) || df > 1e8)
	return dnorm(x, ncp, 1., give_log);

    /* Do calculations on log scale to stabilize */

    /* Consider two cases: x ~= 0 or not */
    if (fabs(x) > sqrt(df * DBL_EPSILON)) {
	u = log(df) - log(fabs(x)) +
	    log(fabs(pnt(x*sqrt((df+2)/df), df+2, ncp, 1, 0) -
		     pnt(x, df, ncp, 1, 0)));
	/* FIXME: the above still suffers from cancellation (but not horribly) */
    }
    else {  /* x ~= 0 : -> same value as for  x = 0 */
	u = lgammafn((df+1)/2) - lgammafn(df/2)
	    - (M_LN_SQRT_PI + .5*(log(df) + ncp*ncp));
    }

    return (give_log ? u : exp(u));
}

// ---- R/nmath/pnchisq.c

/*----------- DEBUGGING -------------
 *
 *	make CFLAGS='-DDEBUG_pnch ....'
(cd `R-devel RHOME`/src/nmath; gcc -I. -I../../src/include -I../../../R/src/include -I/usr/local/include -DHAVE_CONFIG_H -fopenmp -g -O0 -pedantic -Wall -DDEBUG_pnch -DDEBUG_q -c ../../../R/src/nmath/pnchisq.c -o pnchisq.o )

 * -- Feb.6, 2000 (R pre0.99); M.Maechler:  still have
 * bad precision & non-convergence in some cases (x ~= f, both LARGE)
 */

# define EXP expl
# define FABS fabsl
# define LOG logl

static const double _dbl_min_exp = R_M_LN2 * DBL_MIN_EXP;
/*= -708.3964 for IEEE double precision */


inline double pnchisq(double x, double df, double ncp, int lower_tail, int log_p)
{
    double ans;
    if (ISNAN(x) || ISNAN(df) || ISNAN(ncp))
	return x + df + ncp;
    if (!R_FINITE(df) || !R_FINITE(ncp))
	ML_WARN_return_NAN;

    if (df < 0. || ncp < 0.) ML_WARN_return_NAN;

    ans = pnchisq_raw(x, df, ncp, 1e-12, 8*DBL_EPSILON, 1000000, lower_tail, log_p);

    if (x <= 0. || x == ML_POSINF)
	return ans; // because it's perfect

    if(ncp >= 80) {
	if(lower_tail) {
	    ans = fmin2(ans, R_D__1);  /* e.g., pchisq(555, 1.01, ncp = 80) */
	} else { /* !lower_tail */
	    /* since we computed the other tail cancellation is likely */
	    // FIXME: There are cases where  ans == 0. if(!log_p) is perfect
	    if(ans < (log_p ? (-10. * R_M_LN10) : 1e-10)) ML_WARNING(ME_PRECISION, "pnchisq");
	    if(!log_p && ans < 0.) ans = 0.;  /* Precaution PR#7099 */
	}
    }
    /* MM: the following "hack" from c51179 (<--> PR#14216, by Jerry Lewis)
     * -- is "kind of ok" ... but potentially suboptimal: we do  log1p(- p(*, <other tail>, log=FALSE)),
     *    but that  p(*, log=FALSE) may already be an exp(.) or even expm1(..)
     *   <---> "in principle"  this check should happen there, not here  */
    if (!log_p || ans < -1e-8)
	return ans;
    else { // log_p (==> ans <= 0) &&  -1e-8 <= ans <= 0
	// prob. = exp(ans) is near one: we can do better using the other tail
	ans = pnchisq_raw(x, df, ncp, 1e-12, 8*DBL_EPSILON, 1000000, !lower_tail, FALSE);
	return log1p(-ans);
    }
}

inline double 
pnchisq_raw(double x, double f, double theta /* = ncp */,
	    double errmax, double reltol, int itrmax,
	    Rboolean lower_tail, Rboolean log_p)
{
    double lam, x2, f2, term, bound, f_x_2n, f_2n;
    double l_lam = -1., l_x = -1.; /* initialized for -Wall */
    int n;
    Rboolean lamSml, tSml, is_r, is_b;
    LDOUBLE ans, u, v, t, lt, lu =-1;

    if (x <= 0.) {
	if(x == 0. && f == 0.) { // chi^2_0(.) has point mass at zero
#define _L  (-0.5 * theta) // = -lambda
	    return lower_tail ? R_D_exp(_L) : (log_p ? R_Log1_Exp(_L) : -expm1(_L));
	}
	/* x < 0  or {x==0, f > 0} */
	return R_DT_0;
    }
    if(!R_FINITE(x))	return R_DT_1;

    /* This is principally for use from qnchisq */

    if(theta < 80) { /* use 110 for Inf, as ppois(110, 80/2, lower.tail=FALSE) is 2e-20 */
	LDOUBLE ans;
	int i;
	// Have  pgamma(x,s) < x^s / Gamma(s+1) (< and ~= for small x)
	// ==> pchisq(x, f) = pgamma(x, f/2, 2) = pgamma(x/2, f/2)
	//                  <  (x/2)^(f/2) / Gamma(f/2+1) < eps
	// <==>  f/2 * log(x/2) - log(Gamma(f/2+1)) < log(eps) ( ~= -708.3964 )
	// <==>        log(x/2) < 2/f*(log(Gamma(f/2+1)) + log(eps))
	// <==> log(x) < log(2) + 2/f*(log(Gamma(f/2+1)) + log(eps))
	if(lower_tail && f > 0. &&
	   log(x) < R_M_LN2 + 2/f*(lgamma(f/2. + 1) + _dbl_min_exp)) {
	    // all  pchisq(x, f+2*i, lower_tail, FALSE), i=0,...,110 would underflow to 0.
	    // ==> work in log scale
	    double lambda = 0.5 * theta; // < 40
	    double sum, sum2, pr = -lambda, log_lam = log(lambda);
	    sum = sum2 = ML_NEGINF;
	    /* we need to renormalize here: the result could be very close to 1 */
	    for(i = 0; i < 110;  pr += log_lam - log(++i)) {
		sum2 = logspace_add(sum2, pr);
		sum  = logspace_add(sum , pr + pchisq(x, f+2*i, lower_tail, TRUE));
		if (sum2 >= -1e-15) /*<=> EXP(sum2) >= 1-1e-15 */ break;
	    }
	    ans = sum - sum2;
	    return (double) (log_p ? ans : EXP(ans));
	}
	else {
	    LDOUBLE lambda = 0.5 * theta; // < 40
	    LDOUBLE sum = 0, sum2 = 0, pr = EXP(-lambda); // does this need a feature test?
	    /* we need to renormalize here: the result could be very close to 1 */
	    for(i = 0; i < 110;  pr *= lambda/++i) {
		// pr == exp(-lambda) lambda^i / i!  ==  dpois(i, lambda)
		sum2 += pr;
		// pchisq(*, i, *) is  strictly decreasing to 0 for lower_tail=TRUE
		//                 and strictly increasing to 1 for lower_tail=FALSE
		sum += pr * pchisq(x, f+2*i, lower_tail, FALSE);
		if (sum2 >= 1-1e-15) break;
	    }
	    ans = sum/sum2;
	    return (double) (log_p ? LOG(ans) : ans);
	}
    } // if(theta < 80)

    // else: theta == ncp >= 80 --------------------------------------------
    // Series expansion ------- FIXME: log_p=TRUE, lower_tail=FALSE only applied at end ==> underflow

    lam = .5 * theta; // = lambda = ncp/2
    lamSml = (-lam < _dbl_min_exp);
    if(lamSml) {
	// originally error: "non centrality parameter too large for current algorithm"
        u = 0;
        lu = -lam;/* == ln(u) */
        l_lam = log(lam);
    } else {
	u = exp(-lam);
    }

    /* evaluate the first term */
    v = u;
    x2 = .5 * x;
    f2 = .5 * f;
    f_x_2n = f - x;


    if(f2 * DBL_EPSILON > 0.125 && /* very large f and x ~= f: probably needs */
       FABS(t = x2 - f2) <         /* another algorithm anyway */
       sqrt(DBL_EPSILON) * f2) {
	/* evade cancellation error */
	/* t = exp((1 - t)*(2 - t/(f2 + 1))) / sqrt(2*R_M_PI*(f2 + 1));*/
        lt = (1 - t)*(2 - t/(f2 + 1)) - M_LN_SQRT_2PI - 0.5 * log(f2 + 1);
    }
    else {
	/* Usual case 2: careful not to overflow .. : */
	lt = f2*log(x2) -x2 - lgammafn(f2 + 1);
    }

    tSml = (lt < _dbl_min_exp);
    if(tSml) {
	if (x > f + theta +  5* sqrt( 2*(f + 2*theta))) {
	    /* x > E[X] + 5* sigma(X) */
	    return R_DT_1; /* FIXME: could be more accurate than 0. */
	} /* else */
	l_x = log(x);
	ans = term = 0.; t = 0;
    }
    else {
	t = EXP(lt);
	ans = term = (double) (v * t);
    }

    for (n = 1, f_2n = f + 2., f_x_2n += 2.; n <= itrmax ; n++, f_2n += 2, f_x_2n += 2) {
	/* f_2n    === f + 2*n
	 * f_x_2n  === f - x + 2*n   > 0  <==> (f+2n)  >   x */
	if (f_x_2n > 0) {
	    /* find the error bound and check for convergence */

	    bound = (double) (t * x / f_x_2n);
	    is_r = FALSE;
	    /* convergence only if BOTH absolute and relative error < 'bnd' */
	    if (((is_b = (bound <= errmax)) &&
                 (is_r = (term <= reltol * ans))))
            {
		break; /* out completely */
            }
	}

	/* evaluate the next term of the */
	/* expansion and then the partial sum */

        if(lamSml) {
            lu += l_lam - log(n); /* u = u* lam / n */
            if(lu >= _dbl_min_exp) {
		/* no underflow anymore ==> change regime */
                v = u = EXP(lu); /* the first non-0 'u' */
                lamSml = FALSE;
            }
        } else {
	    u *= lam / n;
	    v += u;
	}
	if(tSml) {
            lt += l_x - log(f_2n);/* t <- t * (x / f2n) */
            if(lt >= _dbl_min_exp) {
		/* no underflow anymore ==> change regime */
                t = EXP(lt); /* the first non-0 't' */
                tSml = FALSE;
            }
        } else {
	    t *= x / f_2n;
	}
        if(!lamSml && !tSml) {
	    term = (double) (v * t);
	    ans += term;
	}

    } /* for(n ...) */

    if (n > itrmax) {
	MATHLIB_WARNING4(_("pnchisq(x=%g, f=%g, theta=%g, ..): not converged in %d iter."),
			 x, f, theta, itrmax);
    }
    double dans = (double) ans;
    return R_DT_val(dans);
}

// ---- R/nmath/qnchisq.c

inline double qnchisq(double p, double df, double ncp, int lower_tail, int log_p)
{
    const static double accu = 1e-13;
    const static double racc = 4*DBL_EPSILON;
    /* these two are for the "search" loops, can have less accuracy: */
    const static double Eps = 1e-11; /* must be > accu */
    const static double rEps= 1e-10; /* relative tolerance ... */

    double ux, lx, ux0, nx, pp;

    if (ISNAN(p) || ISNAN(df) || ISNAN(ncp))
	return p + df + ncp;
    if (!R_FINITE(df)) ML_WARN_return_NAN;

    /* Was
     * df = floor(df + 0.5);
     * if (df < 1 || ncp < 0) ML_WARN_return_NAN;
     */
    if (df < 0 || ncp < 0) ML_WARN_return_NAN;

    R_Q_P01_boundaries(p, 0, ML_POSINF);

    pp = R_D_qIv(p); // exp(p) iff log_p
    if(pp > 1 - DBL_EPSILON)
	return lower_tail ? ML_POSINF : 0.0; // early under/over flow  iff log_p (FIXME)

    /* Invert pnchisq(.) :
     * 1. finding an upper and lower bound */
    {
       /* This is Pearson's (1959) approximation,
          which is usually good to 4 figs or so.  */
	double b, c, ff;
	b = (ncp*ncp)/(df + 3*ncp);
	c = (df + 3*ncp)/(df + 2*ncp);
	ff = (df + 2 * ncp)/(c*c);
	ux = b + c * qchisq(p, ff, lower_tail, log_p);
	if(ux <= 0.) ux = 1;
	ux0 = ux;
    }

    if(!lower_tail && ncp >= 80) {
	/* in this case, pnchisq() works via lower_tail = TRUE */
	if(pp < 1e-10) ML_WARNING(ME_PRECISION, "qnchisq");
	p = /* R_DT_qIv(p)*/ log_p ? -expm1(p) : (0.5 - (p) + 0.5);
	lower_tail = TRUE;
    } else {
	p = pp;
    }

    pp = fmin2(1 - DBL_EPSILON, p * (1 + Eps));
    if(lower_tail) {
        for(; ux < DBL_MAX &&
		pnchisq_raw(ux, df, ncp, Eps, rEps, 10000, TRUE, FALSE) < pp;
	    ux *= 2);
	pp = p * (1 - Eps);
        for(lx = fmin2(ux0, DBL_MAX);
	    lx > DBL_MIN &&
		pnchisq_raw(lx, df, ncp, Eps, rEps, 10000, TRUE, FALSE) > pp;
	    lx *= 0.5);
    }
    else {
        for(; ux < DBL_MAX &&
		pnchisq_raw(ux, df, ncp, Eps, rEps, 10000, FALSE, FALSE) > pp;
	    ux *= 2);
	pp = p * (1 - Eps);
        for(lx = fmin2(ux0, DBL_MAX);
	    lx > DBL_MIN &&
		pnchisq_raw(lx, df, ncp, Eps, rEps, 10000, FALSE, FALSE) < pp;
	    lx *= 0.5);
    }

    /* 2. interval (lx,ux)  halving : */
    if(lower_tail) {
	do {
	    nx = 0.5 * (lx + ux);
	    if (pnchisq_raw(nx, df, ncp, accu, racc, 100000, TRUE, FALSE) > p)
		ux = nx;
	    else
		lx = nx;
	}
	while ((ux - lx) / nx > accu);
    } else {
	do {
	    nx = 0.5 * (lx + ux);
	    if (pnchisq_raw(nx, df, ncp, accu, racc, 100000, FALSE, FALSE) < p)
		ux = nx;
	    else
		lx = nx;
	}
	while ((ux - lx) / nx > accu);
    }
    return 0.5 * (ux + lx);
}

// ---- R/nmath/dnchisq.c

inline double dnchisq(double x, double df, double ncp, int give_log)
{
    const static double eps = 5e-15;

    double i, ncp2, q, mid, dfmid, imax;
    LDOUBLE sum, term;

    if (ISNAN(x) || ISNAN(df) || ISNAN(ncp))
	return x + df + ncp;

    if (!R_FINITE(df) || !R_FINITE(ncp) || ncp < 0 || df < 0)
    	ML_WARN_return_NAN;

    if(x < 0) return R_D__0;
    if(x == 0 && df < 2.)
	return ML_POSINF;
    if(ncp == 0)
	return (df > 0) ? dchisq(x, df, give_log) : R_D__0;
    if(x == ML_POSINF) return R_D__0;

    ncp2 = 0.5 * ncp;

    /* find max element of sum */
    imax = ceil((-(2+df) +sqrt((2-df) * (2-df) + 4 * ncp * x))/4);
    if (imax < 0) imax = 0;
    if(R_FINITE(imax)) {
	dfmid = df + 2 * imax;
	mid = dpois_raw(imax, ncp2, FALSE) * dchisq(x, dfmid, FALSE);
    } else /* imax = Inf */
	mid = 0;

    if(mid == 0) {
	/* underflow to 0 -- maybe numerically correct; maybe can be more accurate,
	 * particularly when  give_log = TRUE */
	/* Use  central-chisq approximation formula when appropriate;
	 * ((FIXME: the optimal cutoff also depends on (x,df);  use always here? )) */
	if(give_log || ncp > 1000.) {
	    double nl = df + ncp, ic = nl/(nl + ncp);/* = "1/(1+b)" Abramowitz & St.*/
	    return dchisq(x*ic, nl*ic, give_log);
	} else
	    return R_D__0;
    }

    sum = mid;

    /* errorbound := term * q / (1-q)  now subsumed in while() / if() below: */

    /* upper tail */
    term = mid; df = dfmid; i = imax;
    double x2 = x * ncp2;
    do {
	i++;
	q = x2 / i / df;
	df += 2;
	term *= q;
	sum += term;
    } while (q >= 1 || term * q > (1-q)*eps || term > 1e-10*sum);
    /* lower tail */
    term = mid; df = dfmid; i = imax;
    while (i != 0) {
	df -= 2;
	q = i * df / x2;
	i--;
	term *= q;
	sum += term;
	if (q < 1 && term * q <= (1-q)*eps) break;
    }
    return R_D_val((double) sum);
}


#undef ADD1
#undef C10
#undef C7
#undef C8
#undef C9
#undef EPS1
#undef EPS2
#undef EPS_N
#undef EXP
#undef FABS
#undef FALSE
#undef ISNAN
#undef LDOUBLE
#undef LN_EPS
#undef LOG
#undef MATHLIB_WARNING
#undef MATHLIB_WARNING3
#undef MATHLIB_WARNING4
#undef MATHLIB_WARNING5
#undef MATHLIB_WARNING6
#undef MAXIT
#undef ML_NAN
#undef ML_NEGINF
#undef ML_POSINF
#undef ML_WARNING
#undef ML_WARN_return_NAN
#undef M_1_SQRT_2PI
#undef M_2PI
#undef M_LN_SQRT_2PI
#undef M_LN_SQRT_PI
#undef M_SQRT_2PI
#undef M_SQRT_2dPI
#undef M_SQRT_32
#undef M_SQRT_PI
#undef NEEDED_SCALE
#undef R_DT_0
#undef R_DT_1
#undef R_DT_CIv
#undef R_DT_Cexp
#undef R_DT_Clog
#undef R_DT_Cval
#undef R_DT_Log
#undef R_DT_exp
#undef R_DT_log
#undef R_DT_qIv
#undef R_DT_val
#undef R_D_Clog
#undef R_D_Cval
#undef R_D_LExp
#undef R_D_Lval
#undef R_D__0
#undef R_D__1
#undef R_D_exp
#undef R_D_fexp
#undef R_D_half
#undef R_D_log
#undef R_D_negInonint
#undef R_D_nonint_check
#undef R_D_qIv
#undef R_D_rtxp
#undef R_D_val
#undef R_FINITE
#undef R_Log1_Exp
#undef R_M_1_PI
#undef R_M_LN10
#undef R_M_LN2
#undef R_M_PI
#undef R_M_PI_2
#undef R_M_SQRT2
#undef R_P_bounds_01
#undef R_P_bounds_Inf_01
#undef R_Q_P01_boundaries
#undef R_Q_P01_check
#undef R_forceint
#undef R_ifDEBUG_printf
#undef Rboolean
#undef Rf_d1mach
#undef Rf_i1mach
#undef SET_0_noswap
#undef SET_0_swap
#undef SQR
#undef TRUE
#undef _
#undef _L
#undef d_2
#undef do_del
#undef give_log
#undef max
#undef max_it
#undef min
#undef n_terms_bgrat
#undef num_IT
#undef pMAX
#undef pMIN
#undef swap_tail
#undef x_LRG

}  // namespace greybox_nmath

#endif  // GREYBOX_NMATH_H
