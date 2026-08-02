"""
core/bond_pricing.py

Core fixed-income analytics for an option-free (bullet) bond:
    - bond_price          : present value of a bond's cash flows
    - macaulay_duration    : weighted-average time to receipt of cash flows
    - modified_duration    : first-order price sensitivity to yield
    - dv01                 : dollar (money) value of a 1 basis point yield move
    - convexity             : second-order price sensitivity to yield

Conventions used throughout (matches CFA curriculum notation):
    face_value       (FV)  : redemption / par value paid at maturity
    coupon_rate      (c)   : ANNUAL coupon rate, e.g. 0.06 for 6%
    yield_rate       (y)   : ANNUAL yield to maturity (BEY / nominal annual
                              rate compounded periods_per_year times), e.g. 0.06
    years_to_maturity (n)  : whole number of years to maturity
    periods_per_year (m)   : compounding/coupon frequency per year
                              (2 = semiannual, the standard US bond convention)

    N = n * m   -> total number of coupon periods
    i = y / m   -> periodic (per-coupon-period) yield
    coupon per period = FV * c / m
"""

import numpy as np


def _cash_flows(face_value, coupon_rate, years_to_maturity, periods_per_year):
    """
    Build the periodic cash flow schedule for a plain-vanilla bond.

    Returns
    -------
    t  : np.ndarray of period indices, t = 1, 2, ..., N
    cf : np.ndarray of cash flows received at each period t
         (coupon each period, plus face value returned at t = N)
    """
    n_periods = round(years_to_maturity * periods_per_year)
    coupon_per_period = face_value * coupon_rate / periods_per_year

    t = np.arange(1, n_periods + 1)
    cf = np.full(n_periods, coupon_per_period, dtype=float)
    cf[-1] += face_value  # redemption of principal at maturity
    return t, cf


def bond_price(face_value, coupon_rate, yield_rate, years_to_maturity, periods_per_year=2):
    """
    Present value of a bond's cash flows, discounted at the periodic yield.

    CFA formula:
        P = sum_{t=1}^{N} [ CF_t / (1 + y/m)^t ]

    where CF_t is the coupon in every period and CF_N additionally includes
    the face value redemption. This is simply "PV of an annuity (coupons)
    + PV of a lump sum (redemption)" collapsed into one summation.

    Returns
    -------
    float : the bond's price (quoted on the same basis as face_value, e.g.
            per 100 of par).
    """
    i = yield_rate / periods_per_year
    t, cf = _cash_flows(face_value, coupon_rate, years_to_maturity, periods_per_year)

    price = np.sum(cf / (1 + i) ** t)
    return float(price)


def macaulay_duration(face_value, coupon_rate, yield_rate, years_to_maturity, periods_per_year=2):
    """
    Macaulay Duration: the weighted-average time (in YEARS) until the
    bond's cash flows are received, where each cash flow's time-weight
    is its share of the bond's total present value.

    CFA formula (in periods, then converted to years):
        D_Mac (periods) = [ sum_{t=1}^{N} t * CF_t / (1+y/m)^t ] / P
        D_Mac (years)   = D_Mac (periods) / m

    Intuition: each term t * PV(CF_t) is "time weighted by the present
    value of the cash flow it represents." Dividing by price P normalizes
    those PV weights so they sum to 1, giving a proper weighted average of
    the payment times t. Dividing by m converts periods -> years.

    Returns
    -------
    float : Macaulay duration in years.
    """
    i = yield_rate / periods_per_year
    t, cf = _cash_flows(face_value, coupon_rate, years_to_maturity, periods_per_year)

    price = np.sum(cf / (1 + i) ** t)
    weighted_time_sum = np.sum(t * cf / (1 + i) ** t)

    mac_duration_periods = weighted_time_sum / price
    return float(mac_duration_periods / periods_per_year)


def modified_duration(face_value, coupon_rate, yield_rate, years_to_maturity, periods_per_year=2):
    """
    Modified Duration: the approximate PERCENTAGE price change of a bond
    for a 1.0 (100%) change in yield, i.e. the first derivative of price
    with respect to yield, scaled by price.

    CFA formula:
        D_Mod = D_Mac / (1 + y/m)

    Derivation sketch: P = sum CF_t (1+i)^-t, so
        dP/di = -sum t * CF_t * (1+i)^-(t+1) = -(1/(1+i)) * sum t*CF_t*(1+i)^-t
    Dividing by -P gives (D_Mac in periods)/(1+i); the periods-to-years
    conversion (divide by m) is already baked into D_Mac (years), so
    modified duration in years is simply D_Mac(years) / (1 + y/m).

    Usage (linear price-change approximation):
        %ΔP  ≈  -D_Mod * Δy

    Returns
    -------
    float : modified duration (years), the % price sensitivity per 1.0
            (100%) change in annual yield.
    """
    i = yield_rate / periods_per_year
    d_mac = macaulay_duration(face_value, coupon_rate, yield_rate, years_to_maturity, periods_per_year)
    return float(d_mac / (1 + i))


def dv01(face_value, coupon_rate, yield_rate, years_to_maturity, periods_per_year=2):
    """
    DV01 (Dollar Value of an 01 / PVBP - Price Value of a Basis Point):
    the approximate DOLLAR (money) price change of the bond for a
    1 basis point (0.0001) move in yield.

    CFA formula:
        DV01 = D_Mod * P * 0.0001

    This follows directly from the modified duration relationship
    %ΔP ≈ -D_Mod * Δy: setting Δy = 0.0001 (1 bp) and multiplying the
    percentage change by price P converts it into a dollar-price change.
    Reported here as a positive magnitude (the price change for a 1bp
    move is symmetric to a first-order approximation, with the loss
    occurring when yields RISE).

    Returns
    -------
    float : dollar price change per 1 basis point (same currency units
            as face_value/price, e.g. price points per 100 of par).
    """
    price = bond_price(face_value, coupon_rate, yield_rate, years_to_maturity, periods_per_year)
    d_mod = modified_duration(face_value, coupon_rate, yield_rate, years_to_maturity, periods_per_year)
    return float(d_mod * price * 0.0001)


def convexity(face_value, coupon_rate, yield_rate, years_to_maturity, periods_per_year=2):
    """
    Convexity: the second-order (curvature) measure of how a bond's price
    changes with yield. Duration alone is a linear (tangent-line)
    approximation; convexity corrects for the fact that the price-yield
    relationship is curved (convex), so duration under-predicts price
    gains for yield decreases and over-predicts price losses for yield
    increases.

    CFA formula (annualized):
        Convexity = [ sum_{t=1}^{N} t*(t+1)*CF_t / (1+y/m)^t ] / [ P * m^2 * (1+y/m)^2 ]

    Derivation sketch: differentiating P = sum CF_t(1+i)^-t twice w.r.t.
    the periodic rate i gives
        d2P/di2 = (1/(1+i)^2) * sum t*(t+1)*CF_t*(1+i)^-t
    Dividing by P gives convexity in "per period^2" units; since i = y/m,
    d2P/dy2 = d2P/di2 / m^2, which is where the m^2 term in the
    denominator comes from (annualizing the periodic curvature).

    Usage (full second-order price-change approximation, combined with
    modified duration):
        %ΔP  ≈  -D_Mod * Δy + 0.5 * Convexity * Δy^2

    Returns
    -------
    float : annualized convexity measure (years^2).
    """
    i = yield_rate / periods_per_year
    t, cf = _cash_flows(face_value, coupon_rate, years_to_maturity, periods_per_year)

    price = np.sum(cf / (1 + i) ** t)
    weighted_sum = np.sum(t * (t + 1) * cf / (1 + i) ** t)

    convexity_value = weighted_sum / (price * periods_per_year ** 2 * (1 + i) ** 2)
    return float(convexity_value)
