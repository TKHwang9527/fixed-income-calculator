"""
core/binomial_tree.py

Arbitrage-free binomial interest rate tree: calibrate a lognormal short-rate
lattice to a spot curve, then value option-free, callable, and putable
bonds on it via backward induction.

Written for CFA Level II "Valuation and Analysis of Bonds with Embedded
Options" review -- every formula is derived, not just cited.

Conventions (consistent with core/bond_pricing.py and core/yield_curve.py):
    face_value        (FV)  : redemption / par value, e.g. 100
    coupon_rate         (c)  : ANNUAL coupon rate, e.g. 0.07
    spot_rate            (z)  : annualized spot rate for period t, from
                                core.yield_curve.bootstrap_spot_rates()
    volatility          (σ)  : ANNUAL interest rate volatility assumption
                                used to size the up/down step
    periods_per_year    (m)  : compounding/coupon frequency per year
    t / k                    : period index, 1..N (years, if m == 1)

-----------------------------------------------------------------------
WHY A TREE AT ALL, AND WHY BACKWARD INDUCTION?
-----------------------------------------------------------------------
A single YTM (bond_pricing.bond_price) or even a full spot curve
(yield_curve.bond_price_with_spot_rates) can only value an OPTION-FREE
bond, because both assume the bond's cash flows are known in advance.
A callable or putable bond's cash flows are NOT fixed -- whether the
issuer calls or the holder puts depends on what interest rates actually
do in the future, which is exactly what an interest rate TREE models:
a branching set of possible future short rates, one node per possible
path. "Backward induction" means starting at the bond's maturity (where
its value is unambiguous: just the redemption amount) and working
backward one period at a time, at each node computing the bond's value
as a decision made with the benefit of already knowing every possible
FUTURE value one step ahead -- exactly like solving a decision tree from
the leaves inward. That's what lets an issuer/holder's optimal exercise
decision be folded in node by node, in the same recursion.

-----------------------------------------------------------------------
WHY RISK-NEUTRAL (0.5 / 0.5) PROBABILITIES?
-----------------------------------------------------------------------
The tree does NOT use real-world probabilities of rates rising or
falling, and deliberately does not need to. Under the risk-neutral /
no-arbitrage valuation framework, if a tree's short rates are calibrated
(see build_rate_tree below) so that they exactly reproduce today's
observed spot curve, then discounting expected future cash flows at
those SAME rates using flat 50/50 "pricing" probabilities is guaranteed
to recover arbitrage-free prices for ANY cash flow stream built on that
tree -- option-free or not. Real-world up/down probabilities would
additionally require estimating a market risk premium, which is exactly
what the calibration step avoids: all of the market's risk pricing is
already embedded in the CALIBRATED RATE LEVELS themselves, not in the
probabilities. This is why every node below uses a plain 0.5/0.5 average.
"""

import numpy as np

from core.bond_pricing import _cash_flows


def _price_zero_via_tree(rate_tree, n_periods, periods_per_year):
    """
    Backward-induct the price of a $100-face zero-coupon bond maturing at
    period n_periods through a (partial or full) rate tree.

    CFA formula, applied one period at a time from maturity back to today:
        V(k, i) = 0.5 * [ V(k+1, i+1) + V(k+1, i) ] / (1 + r(k, i)/m)

    where node (k, i) means "period k, i up-moves so far" and r(k, i) is
    the one-period rate prevailing from k to k+1 at that node. This is
    the risk-neutral valuation step described in the module docstring.
    """
    values = np.full(n_periods + 1, 100.0)
    for k in range(n_periods - 1, -1, -1):
        rates_k = rate_tree[k]
        values = 0.5 * (values[1:] + values[:-1]) / (1 + rates_k / periods_per_year)
    return values[0]


def _solve_lower_rate(price_fn, target, tol=1e-10, max_doublings=60):
    """
    Bisection root-finder for the single unknown "lower rate" at each
    calibration step (see build_rate_tree). price_fn(r0) -- the tree
    price of a zero-coupon bond as a function of the candidate lower
    rate -- is monotonically DECREASING in r0 (a higher discount rate can
    only lower a bond's price), so bisection is guaranteed to converge
    once price_fn(hi) < target < price_fn(lo).
    """
    lo, hi = 1e-8, 1.0
    doublings = 0
    while price_fn(hi) > target:
        hi *= 2
        doublings += 1
        if doublings > max_doublings:
            raise RuntimeError("Could not bracket the calibration root -- check spot_rates/volatility")

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if price_fn(mid) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break

    return 0.5 * (lo + hi)


def _solve_constant_spread(price_fn, target_price, tol=1e-8, max_expansions=60):
    """
    Bisection root-finder for a single constant spread added uniformly to
    a whole set of rates (every spot rate, for a Z-spread; every tree
    node, for an OAS) so that the resulting model price equals a target
    (market) price. Shared by calculate_zspread and calculate_oas below.

    price_fn(spread) is monotonically DECREASING in spread -- adding more
    spread to every discount rate can only lower a bond's PV, whether or
    not the bond has embedded options (capping/flooring a node's value at
    a fixed call/put price cannot reverse that direction, it can only
    mute it) -- so bisection is guaranteed to converge once the target is
    bracketed.

    Unlike _solve_lower_rate's calibration search (which only ever solves
    for a non-negative rate), the correct spread here can be NEGATIVE --
    e.g. a market price ABOVE the zero-spread model price implies a
    negative Z-spread/OAS -- so both directions are expanded outward from
    zero until the target is bracketed, rather than assuming lo = 0.
    """
    lo, hi = -0.01, 0.01
    expansions = 0
    while price_fn(lo) < target_price:
        lo -= 0.01
        expansions += 1
        if expansions > max_expansions:
            raise RuntimeError("Could not bracket the spread below -- check market_price/inputs")

    expansions = 0
    while price_fn(hi) > target_price:
        hi += 0.01
        expansions += 1
        if expansions > max_expansions:
            raise RuntimeError("Could not bracket the spread above -- check market_price/inputs")

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if price_fn(mid) > target_price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break

    return 0.5 * (lo + hi)


def build_rate_tree(spot_rates, volatility, periods_per_year=1):
    """
    Calibrate a recombining, lognormal binomial short-rate tree so that it
    exactly reproduces the given spot curve -- i.e. so that an arbitrage-
    free investor is indifferent between the tree and the spot curve for
    ANY option-free bond. This is what makes the tree "arbitrage-free."

    Lattice structure:
        At period k there are k+1 nodes (a "recombining" tree: an
        up-then-down path lands on the same rate as down-then-up). Node
        i at period k (i = 0 lowest ... k highest) has one-period rate

            r(k, i) = r(k, 0) * u^i,    u = e^(2 * σ * sqrt(Δt)),  Δt = 1/m

        The multiplicative up/down factor u is the standard lognormal
        step size for an annualized volatility σ: two adjacent nodes at
        the same period always differ by the constant ratio u, regardless
        of level, so rate volatility scales with the rate itself
        (a common, mean-reverting-friendly assumption in CFA curriculum
        binomial trees).

    Calibration (the actual "bootstrap"), one period at a time:
        r(k, 0) -- the only unknown at each step, since r(k, i) = r(k,0)*u^i
        -- is solved for so that a $100-face zero-coupon bond maturing at
        period k+1, backward-induced through the tree using ALREADY
        calibrated rates for periods 0..k-1 plus the candidate r(k, ·)
        for period k, prices to EXACTLY the spot-curve value:

            V_tree(0; matures at k+1) = 100 / (1 + z_{k+1}/m)^(k+1)

        Because V_tree(r(k,0)) is monotonic in r(k,0), this has a unique
        solution, found here by bisection (see _solve_lower_rate). This
        is done period-by-period, shortest maturity first, because each
        step's calibration depends only on shorter (already-solved)
        periods -- the defining trait of a bootstrap.

    Parameters
    ----------
    spot_rates : array-like of float
        Spot rates z_1, ..., z_N (e.g. from
        core.yield_curve.bootstrap_spot_rates), one per period.
    volatility : float
        Annualized interest rate volatility, σ (e.g. 0.10 for 10%).
    periods_per_year : int
        Compounding frequency per year (m).

    Returns
    -------
    list of np.ndarray
        rate_tree[k] holds the k+1 one-period rates prevailing from
        period k to k+1, ordered lowest to highest, for k = 0, ..., N-1.
    """
    spot_rates = np.asarray(spot_rates, dtype=float)
    n = len(spot_rates)
    m = periods_per_year
    dt = 1.0 / m
    u = np.exp(2 * volatility * np.sqrt(dt))

    tree = []
    for k in range(n):
        target = 100.0 / (1 + spot_rates[k] / m) ** (k + 1)

        def price_given_lower_rate(r0, k=k, tree=tree):
            candidate_period = r0 * u ** np.arange(k + 1)
            return _price_zero_via_tree(tree + [candidate_period], k + 1, m)

        r0 = _solve_lower_rate(price_given_lower_rate, target)
        tree.append(r0 * u ** np.arange(k + 1))

    return tree


def _backward_induct_bond(face_value, coupon_rate, rate_tree, periods_per_year, constrain=None):
    """
    Shared backward-induction engine for option-free, callable, and
    putable bonds.

    CFA formula at each node (k, i), working from maturity back to today:
        V(k, i) = [ 0.5*(V(k+1, i+1) + coupon) + 0.5*(V(k+1, i) + coupon) ]
                  / (1 + r(k, i)/m)

    i.e. the coupon paid at the CHILD node's time is added to each child's
    value before averaging and discounting -- so by the time we reach
    k = n-1, V(n) (= face value only) + coupon reproduces exactly the
    bond's final cash flow (coupon + redemption), matching
    bond_pricing._cash_flows' final-period convention.

    `constrain(k, node_values)` is an optional hook applied AFTER
    discounting to node_values at period k, letting a caller cap
    (callable) or floor (putable) the value at that node before it
    becomes next iteration's "child" value -- this is precisely how the
    issuer's/holder's optimal exercise decision gets folded into the
    recursion one period at a time.
    """
    n = len(rate_tree)
    m = periods_per_year
    coupon = face_value * coupon_rate / m

    values = np.full(n + 1, face_value)  # maturity: redemption only
    for k in range(n - 1, -1, -1):
        continuation = values + coupon
        node_values = 0.5 * (continuation[1:] + continuation[:-1]) / (1 + rate_tree[k] / m)
        if constrain is not None:
            node_values = constrain(k, node_values)
        values = node_values

    return float(values[0])


def price_option_free_bond(face_value, coupon_rate, rate_tree, periods_per_year=1):
    """
    Value a plain-vanilla (option-free) bond by backward induction through
    a calibrated rate tree.

    Because the tree was calibrated (build_rate_tree) to exactly reproduce
    the spot curve, this MUST equal
    yield_curve.bond_price_with_spot_rates() for the same bond -- that
    equality is the standard CFA L2 proof/check that a tree is correctly
    calibrated, and is the primary correctness test for this module.

    Returns
    -------
    float : the bond's price.
    """
    return _backward_induct_bond(face_value, coupon_rate, rate_tree, periods_per_year, constrain=None)


def price_callable_bond(face_value, coupon_rate, rate_tree, call_price=100.0, first_call_period=1, periods_per_year=1):
    """
    Value a callable bond: at each callable node, the ISSUER will exercise
    (call the bond back at call_price) whenever the tree-implied
    continuation value exceeds call_price, since the issuer can then
    refinance more cheaply. That caps the bondholder's upside:

        V_callable(k, i) = min( V_unconstrained(k, i), call_price )

    applied at every node from period `first_call_period` through period
    N-1 (bonds are conventionally not callable at issue, t=0, nor "called"
    at their own maturity, t=N -- redemption there is not a call).

    Because capping a node's value can only ever reduce (never raise) the
    values propagated backward through the rest of the tree,
    price_callable_bond(...) <= price_option_free_bond(...) always holds
    -- an embedded call option can only cost the holder value, never add it.

    Parameters
    ----------
    call_price : float
        The fixed price (per 100 face) at which the issuer may call the
        bond at any exercisable node.
    first_call_period : int
        First period (1-indexed) at which the bond becomes callable.

    Returns
    -------
    float : the callable bond's price.
    """
    n = len(rate_tree)

    def constrain(k, node_values):
        if first_call_period <= k <= n - 1:
            return np.minimum(node_values, call_price)
        return node_values

    return _backward_induct_bond(face_value, coupon_rate, rate_tree, periods_per_year, constrain)


def price_putable_bond(face_value, coupon_rate, rate_tree, put_price=100.0, first_put_period=1, periods_per_year=1):
    """
    Value a putable bond: at each putable node, the HOLDER will exercise
    (put the bond back at put_price) whenever the tree-implied
    continuation value falls below put_price, since the holder can then
    reinvest at par rather than hold a now-cheap bond. That floors the
    bondholder's downside:

        V_putable(k, i) = max( V_unconstrained(k, i), put_price )

    applied at every node from period `first_put_period` through period
    N-1, mirroring price_callable_bond's exercise window.

    Because flooring a node's value can only ever raise (never lower) the
    values propagated backward through the rest of the tree,
    price_putable_bond(...) >= price_option_free_bond(...) always holds
    -- an embedded put option can only add value for the holder, never cost it.

    Parameters
    ----------
    put_price : float
        The fixed price (per 100 face) at which the holder may put the
        bond back to the issuer at any exercisable node.
    first_put_period : int
        First period (1-indexed) at which the bond becomes putable.

    Returns
    -------
    float : the putable bond's price.
    """
    n = len(rate_tree)

    def constrain(k, node_values):
        if first_put_period <= k <= n - 1:
            return np.maximum(node_values, put_price)
        return node_values

    return _backward_induct_bond(face_value, coupon_rate, rate_tree, periods_per_year, constrain)


def call_option_value(straight_price, callable_price):
    """
    Value of the embedded call option to the ISSUER (equivalently, the
    cost of the call to the bondholder).

    CFA identity:  V_callable = V_straight - V_call
                => V_call     = V_straight - V_callable

    Always >= 0: a rational issuer never exercises a call that would make
    the bond MORE valuable to hold, so the option can only ever transfer
    value away from the holder, never toward them.
    """
    return float(straight_price - callable_price)


def put_option_value(straight_price, putable_price):
    """
    Value of the embedded put option to the HOLDER.

    CFA identity:  V_putable = V_straight + V_put
                => V_put     = V_putable - V_straight

    Always >= 0, by the mirror-image argument to call_option_value: the
    put can only ever transfer value toward the holder, never away.
    """
    return float(putable_price - straight_price)


# ---------------------------------------------------------------------
# Z-SPREAD, OAS, AND OPTION COST
# ---------------------------------------------------------------------
# WHY Z-SPREAD?
#
# A bond's spot curve (from yield_curve.bootstrap_spot_rates) is a
# RISK-FREE (or benchmark) term structure. A real bond's MARKET price
# almost never exactly equals bond_price_with_spot_rates() off that
# curve, because the market price also reflects credit risk, liquidity,
# and (for bonds with embedded options) optionality -- none of which the
# benchmark spot curve prices in. The Z-spread ("zero-volatility spread")
# is the single CONSTANT spread that, added to every point on the spot
# curve, reconciles the two: it is the extra compensation, spread evenly
# across every maturity, that the market is demanding over the benchmark
# curve to hold this specific bond. For an OPTION-FREE bond, that extra
# compensation is (to a first approximation) pure credit + liquidity
# spread, since there is no optionality to price.
#
# WHY OAS, AND WHY IT NEEDS THE TREE (NOT JUST THE SPOT CURVE)?
#
# For a bond WITH an embedded option (callable/putable), the Z-spread is
# contaminated: part of that "extra compensation" the market demands is
# really compensation for the option, not credit/liquidity risk, because
# Z-spread's calculation (discounting fixed, already-known cash flows off
# a flat-shifted spot curve) has no way to represent the issuer's/
# holder's exercise decision -- it implicitly prices the bond AS IF its
# cash flows were fixed, which they are not. The Option-Adjusted Spread
# fixes this by shifting every node of the CALIBRATED RATE TREE (not the
# static spot curve) by a constant spread and then re-running the SAME
# backward induction used by price_callable_bond/price_putable_bond --
# i.e. the issuer's/holder's optimal exercise decision at every node is
# re-evaluated at the shifted rates too. Solving for the constant spread
# that reconciles THAT tree-based, option-aware price to the market price
# yields a spread that has had the option's cost/benefit backed out,
# leaving (to a first approximation) just credit + liquidity spread --
# which is why OAS, unlike Z-spread, is directly comparable across bonds
# with different (or no) embedded options.
#
# WHY OPTION COST = Z-SPREAD - OAS?
#
# Z-spread prices the bond AS IF it had no optionality (fixed cash
# flows); OAS prices it WITH the optionality correctly folded in. The gap
# between the two is, by construction, exactly the piece of the spread
# attributable to the option:
#     Option Cost = Z-Spread - OAS
# For a CALLABLE bond the option hurts the holder, so the market charges
# a larger flat (Z-spread) compensation than the "true" credit spread
# would require once the call is properly accounted for: OAS < Z-Spread,
# so Option Cost > 0. For a PUTABLE bond the option benefits the holder,
# so the reverse holds: OAS > Z-Spread, so Option Cost < 0.
# ---------------------------------------------------------------------


def calculate_zspread(market_price, face_value, coupon_rate, years_to_maturity, spot_rates, periods_per_year=1):
    """
    Solve for the Z-spread of a STRAIGHT (option-free) bond given its
    observed market price.

    CFA formula, solved for the single unknown ZS:
        P_market = sum_{t=1}^{N} CF_t / (1 + (z_t + ZS)/m)^t

    This is identical in form to yield_curve.bond_price_with_spot_rates's
    P = sum CF_t / (1+z_t/m)^t, except every spot rate z_t is bumped by
    the SAME constant ZS before discounting, and ZS is the unknown being
    solved for (rather than P). Because raising ZS raises every discount
    factor's denominator, P_market(ZS) is monotonically decreasing in ZS,
    so bisection (via _solve_constant_spread) converges to a unique root.

    Parameters
    ----------
    market_price : float
        The bond's observed market price (per 100 face).
    face_value, coupon_rate, years_to_maturity, periods_per_year
        Same meaning as in bond_pricing.bond_price().
    spot_rates : array-like of float
        Spot rates z_1, ..., z_N for periods 1..N (e.g. from
        yield_curve.bootstrap_spot_rates()), one per coupon period.

    Returns
    -------
    float
        The Z-spread, annualized and expressed in the same (decimal)
        units as spot_rates, e.g. 0.0125 for a 125 bp Z-spread.
    """
    spot_rates = np.asarray(spot_rates, dtype=float)
    m = periods_per_year
    t, cf = _cash_flows(face_value, coupon_rate, years_to_maturity, periods_per_year)

    if len(spot_rates) != len(t):
        raise ValueError(
            f"Expected {len(t)} spot rates (one per coupon period), got {len(spot_rates)}"
        )

    def price_given_spread(spread):
        return np.sum(cf / (1 + (spot_rates + spread) / m) ** t)

    return _solve_constant_spread(price_given_spread, market_price)


def calculate_oas(market_price, face_value, coupon_rate, rate_tree, call_price=100.0, first_call_period=1, periods_per_year=1):
    """
    Solve for the Option-Adjusted Spread of a CALLABLE bond given its
    observed market price.

    CFA formula, solved for the single unknown OAS:
        P_market = TreePrice_callable( rate_tree shifted by OAS at every node )

    Concretely: rate_tree[k] + OAS replaces rate_tree[k] at every period
    k, and the callable bond is then repriced by the SAME backward
    induction as price_callable_bond -- issuer call decisions included --
    on that shifted tree. As with calculate_zspread, raising OAS can only
    lower every discounted node value (before AND after the call cap is
    applied), so the shifted-tree price is monotonically decreasing in
    OAS and bisection converges to a unique root.

    Parameters
    ----------
    market_price : float
        The callable bond's observed market price (per 100 face).
    face_value, coupon_rate, periods_per_year
        Same meaning as in price_callable_bond().
    rate_tree : list of np.ndarray
        A calibrated rate tree from build_rate_tree() (calibrated to the
        benchmark spot curve, WITHOUT any spread applied).
    call_price, first_call_period
        Same meaning as in price_callable_bond().

    Returns
    -------
    float
        The OAS, annualized and expressed in the same (decimal) units as
        the tree's rates, e.g. 0.0080 for an 80 bp OAS.
    """
    def price_given_spread(spread):
        shifted_tree = [level + spread for level in rate_tree]
        return price_callable_bond(face_value, coupon_rate, shifted_tree, call_price, first_call_period, periods_per_year)

    return _solve_constant_spread(price_given_spread, market_price)


def option_cost(zspread, oas):
    """
    Cost of the embedded option, isolated as the portion of the Z-spread
    NOT explained by credit/liquidity risk once optionality is correctly
    priced in (see the module-level discussion above).

    CFA formula:
        Option Cost = Z-Spread - OAS

    Positive for a callable bond (the call costs the holder value, so the
    market's flat Z-spread overstates the "true" credit spread); negative
    for a putable bond (the put benefits the holder, understating it).

    Returns
    -------
    float
        The option cost, in the same (decimal) spread units as zspread
        and oas, e.g. 0.0035 for a 35 bp option cost.
    """
    return float(zspread - oas)
