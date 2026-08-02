"""
core/yield_curve.py

Yield curve analytics: bootstrapping spot (zero-coupon) rates from a par
curve, deriving forward rates from spot rates, and pricing a bond off a
full spot curve instead of a single blended YTM.

Written for CFA Level II "Term Structure and Interest Rate Dynamics"
review -- every formula below is spelled out algebraically before the
code, not just cited.

Conventions (consistent with core/bond_pricing.py):
    face_value        (FV)  : redemption / par value, e.g. 100
    par_rate           (c)  : the annualized coupon rate that makes a
                                bond of that maturity price at exactly
                                par (100), i.e. the "par curve"
    spot_rate           (z)  : annualized zero-coupon / spot rate for a
                                given maturity, compounded periods_per_year
                                times (same y/m convention as bond_price)
    forward_rate        (f)  : annualized rate for a future period,
                                implied by two points on the spot curve
    periods_per_year    (m)  : compounding/coupon frequency per year
    t                        : period index, t = 1, 2, ..., N (NOT years,
                                unless periods_per_year == 1)

All rates are periodic-compounded like the rest of this package: a spot
rate z for period t discounts a cash flow as CF / (1 + z/m)^t.
"""

import numpy as np

from core.bond_pricing import _cash_flows


def bootstrap_spot_rates(par_rates, periods_per_year=1):
    """
    Bootstrap the spot (zero-coupon) rate curve from a par curve.

    A "par rate" for maturity t is, by definition, the coupon rate that
    makes a bond of that maturity price at exactly 100 (par) TODAY. The
    par curve is directly observable (it's just current market yields on
    par-priced bonds); the spot curve is not directly observable and must
    be inferred -- that inference is "bootstrapping."

    CFA formula, one maturity at a time:
        Every par bond of maturity t must satisfy:
            100 = sum_{k=1}^{t} (c_t/m * 100) / (1 + z_k/m)^k
                    + 100 / (1 + z_t/m)^t

        i.e. its coupons and redemption, discounted at the CORRECT spot
        rate for each individual cash flow date, must sum to par. Every
        term in that sum uses a spot rate z_k for k < t that was already
        solved for in a previous (shorter-maturity) step -- that's the
        "boot-strapping" part: each new spot rate is solved for using only
        rates already bootstrapped, one maturity longer at a time.

    Isolating the one unknown (z_t) at each step:
        Let coupon = c_t/m * 100 (the bond's periodic cash flow) and
        PV_coupons = sum_{k=1}^{t-1} coupon / (1 + z_k/m)^k  (all already known).

        100 - PV_coupons = (100 + coupon) / (1 + z_t/m)^t

        (1 + z_t/m)^t = (100 + coupon) / (100 - PV_coupons)

        z_t = m * [ ((100 + coupon) / (100 - PV_coupons))^(1/t) - 1 ]

    Base case (t=1): PV_coupons = 0, so z_1 reduces to exactly c_1 -- the
    1-period spot rate always equals the 1-period par rate, since there is
    only one cash flow date and nothing to disentangle.

    Parameters
    ----------
    par_rates : array-like of float
        Annualized par rates for maturities t = 1, 2, ..., N periods,
        i.e. par_rates[0] is the 1-period par rate, par_rates[1] is the
        2-period par rate, etc. Assumes face value 100 and coupon
        frequency periods_per_year.
    periods_per_year : int
        Compounding/coupon frequency per year (m).

    Returns
    -------
    np.ndarray
        Annualized spot rates z_1, ..., z_N (same length as par_rates),
        quoted on the periods_per_year-compounded basis (1 + z_t/m)^t.
    """
    par_rates = np.asarray(par_rates, dtype=float)
    m = periods_per_year
    n = len(par_rates)

    discount_factors = np.zeros(n)  # df[k-1] = 1 / (1 + z_k/m)^k
    spot_rates = np.zeros(n)

    for t in range(1, n + 1):
        coupon = par_rates[t - 1] / m * 100.0
        pv_coupons = np.sum(coupon * discount_factors[: t - 1])

        df_t = (100.0 - pv_coupons) / (100.0 + coupon)
        discount_factors[t - 1] = df_t
        spot_rates[t - 1] = m * (df_t ** (-1.0 / t) - 1.0)

    return spot_rates


def forward_rate(spot_rates, period1, period2, periods_per_year=1):
    """
    Implied forward rate between two points on the spot curve.

    No-arbitrage linkage equation: investing directly for period2 periods
    must earn the same total return as investing for period1 periods and
    then rolling into a forward contract locked in today for the
    remaining (period2 - period1) periods -- otherwise an arbitrage
    exists between the two strategies.

    CFA formula:
        (1 + z_2/m)^t2 = (1 + z_1/m)^t1 * (1 + f_{1,2}/m)^(t2 - t1)

        f_{1,2} = m * [ ( (1+z_2/m)^t2 / (1+z_1/m)^t1 )^(1/(t2-t1)) - 1 ]

    Setting period1 = 0 (t1 = 0, so (1+z_1/m)^0 = 1) recovers the trivial
    identity that the spot rate itself IS the "forward rate" starting
    today: f_{0,t2} = z_2.

    Parameters
    ----------
    spot_rates : array-like of float
        Spot rates z_1, ..., z_N, where spot_rates[k-1] is the spot rate
        for period k (same layout as bootstrap_spot_rates' output).
    period1, period2 : int
        The two period indices (t1 < t2). period1 = 0 means "today."
    periods_per_year : int
        Compounding frequency per year (m).

    Returns
    -------
    float
        Annualized forward rate for the period spanning (period1, period2],
        quoted on the periods_per_year-compounded basis.
    """
    if period2 <= period1:
        raise ValueError("period2 must be greater than period1")

    spot_rates = np.asarray(spot_rates, dtype=float)
    m = periods_per_year

    growth_2 = (1 + spot_rates[period2 - 1] / m) ** period2
    growth_1 = 1.0 if period1 == 0 else (1 + spot_rates[period1 - 1] / m) ** period1

    periods_fwd = period2 - period1
    return float(m * ((growth_2 / growth_1) ** (1.0 / periods_fwd) - 1.0))


def forward_curve(spot_rates, periods_per_year=1):
    """
    The series of consecutive one-period forward rates implied by a spot
    curve: f_1 = z_1 (today -> period 1), f_2 = the 1-period forward rate
    starting at period 1 (period 1 -> period 2), f_3 = period 2 -> period 3,
    and so on.

    This is the curve CFA L2 problems usually mean by "the forward rate
    curve" -- e.g. "1y1y" is forward_curve(...)[1] on an annual (m=1) curve.

    Reconstitution check (useful sanity check / test): chaining these
    one-period forwards must reproduce the original spot rate exactly,
    since spot rates are just geometric averages of the one-period
    forwards that make them up:
        (1 + z_t/m)^t = product_{k=1}^{t} (1 + f_k/m)

    Parameters
    ----------
    spot_rates : array-like of float
        Spot rates z_1, ..., z_N.
    periods_per_year : int
        Compounding frequency per year (m).

    Returns
    -------
    np.ndarray
        One-period forward rates f_1, ..., f_N (f_1 == spot_rates[0]).
    """
    spot_rates = np.asarray(spot_rates, dtype=float)
    n = len(spot_rates)

    forwards = np.zeros(n)
    forwards[0] = spot_rates[0]
    for t in range(2, n + 1):
        forwards[t - 1] = forward_rate(spot_rates, t - 1, t, periods_per_year)

    return forwards


def bond_price_with_spot_rates(face_value, coupon_rate, years_to_maturity, spot_rates, periods_per_year=1):
    """
    Price a bond by discounting each cash flow at the spot rate matching
    ITS OWN maturity, rather than a single blended YTM.

    Why this differs from bond_price(): a single YTM is an internal rate
    of return -- one flat discount rate that happens to make the PV of
    all cash flows equal the price. It implicitly assumes every cash flow
    could be reinvested at that same flat rate, which is only true if the
    spot curve is flat. In reality the market prices a 1-year cash flow
    off the 1-year spot rate, a 2-year cash flow off the 2-year spot rate,
    etc. -- discounting the whole bond off a single YTM is an
    approximation of that "correct" no-arbitrage price. Pricing off the
    full spot curve is the arbitrage-free approach (and is exactly how
    the bootstrap in bootstrap_spot_rates() was derived in the first
    place: par bonds MUST match this price, by construction).

    CFA formula:
        P = sum_{t=1}^{N} CF_t / (1 + z_t/m)^t

    identical in form to bond_price()'s P = sum CF_t / (1+y/m)^t, except
    every cash flow gets its own maturity-matched rate z_t instead of a
    shared y.

    Parameters
    ----------
    face_value, coupon_rate, years_to_maturity, periods_per_year
        Same meaning as in bond_pricing.bond_price().
    spot_rates : array-like of float
        Spot rates z_1, ..., z_N for periods 1..N, where
        N = years_to_maturity * periods_per_year. Must have exactly one
        spot rate per coupon period of this bond (e.g. from
        bootstrap_spot_rates()).

    Returns
    -------
    float
        The bond's price.
    """
    spot_rates = np.asarray(spot_rates, dtype=float)
    m = periods_per_year
    t, cf = _cash_flows(face_value, coupon_rate, years_to_maturity, periods_per_year)

    if len(spot_rates) != len(t):
        raise ValueError(
            f"Expected {len(t)} spot rates (one per coupon period), got {len(spot_rates)}"
        )

    price = np.sum(cf / (1 + spot_rates / m) ** t)
    return float(price)
