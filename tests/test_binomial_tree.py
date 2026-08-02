"""
tests/test_binomial_tree.py

Verifies core/binomial_tree.py: calibrating a rate tree to a spot curve,
valuing option-free bonds by backward induction, and valuing callable /
putable bonds and their embedded option values.

As with tests/test_yield_curve.py, most checks lean on the module's own
provable invariants rather than memorized textbook numbers:

    - A zero-coupon bond of any maturity, priced via the tree, must
      reproduce the spot-curve price used to calibrate it -- that's the
      literal definition of "arbitrage-free calibration."
    - An option-free COUPON bond priced via the tree must match
      yield_curve.bond_price_with_spot_rates() for the same bond exactly
      (no-arbitrage additivity of cash flows).
    - With volatility = 0 (a single deterministic path, no branching),
      the tree's one-period rates must exactly equal
      yield_curve.forward_curve() -- a flat/no-uncertainty tree IS the
      forward curve.
    - A call option can only ever cost the bondholder value (callable
      price <= straight price) and a put option can only ever benefit
      the holder (putable price >= straight price), for ANY inputs --
      these are structural inequalities, not numeric coincidences.
"""

import numpy as np
import pytest

from core.binomial_tree import (
    build_rate_tree,
    calculate_oas,
    calculate_zspread,
    call_option_value,
    option_cost,
    price_callable_bond,
    price_option_free_bond,
    price_putable_bond,
    put_option_value,
)
from core.yield_curve import bond_price_with_spot_rates, bootstrap_spot_rates, forward_curve

TOL = 1e-4

PAR_RATES = [0.03, 0.035, 0.04, 0.045, 0.05]
SPOT_RATES = bootstrap_spot_rates(PAR_RATES, periods_per_year=1)


class TestBuildRateTree:
    def test_tree_shape(self):
        tree = build_rate_tree(SPOT_RATES, volatility=0.15, periods_per_year=1)
        assert len(tree) == len(SPOT_RATES)
        for k, level in enumerate(tree):
            assert len(level) == k + 1

    def test_nodes_within_a_period_share_constant_ratio_u(self):
        """Every pair of adjacent nodes at the same period must differ by
        exactly u = e^(2*sigma*sqrt(dt)) -- the defining lognormal-lattice
        property that makes the tree recombine."""
        sigma = 0.15
        tree = build_rate_tree(SPOT_RATES, volatility=sigma, periods_per_year=1)
        u = np.exp(2 * sigma)

        for level in tree:
            if len(level) > 1:
                ratios = level[1:] / level[:-1]
                np.testing.assert_allclose(ratios, u, rtol=1e-6)

    def test_higher_volatility_widens_dispersion(self):
        low_vol_tree = build_rate_tree(SPOT_RATES, volatility=0.05, periods_per_year=1)
        high_vol_tree = build_rate_tree(SPOT_RATES, volatility=0.25, periods_per_year=1)

        low_spread = low_vol_tree[-1][-1] / low_vol_tree[-1][0]
        high_spread = high_vol_tree[-1][-1] / high_vol_tree[-1][0]
        assert high_spread > low_spread

    def test_zero_coupon_bonds_reprice_to_the_spot_curve(self):
        """The defining calibration property: a zero-coupon bond maturing
        at period k, backward-induced through the tree using only the
        first k levels, must reprice to exactly the spot-curve value."""
        tree = build_rate_tree(SPOT_RATES, volatility=0.15, periods_per_year=1)

        for k in range(1, len(SPOT_RATES) + 1):
            tree_price = price_option_free_bond(100, 0.0, tree[:k], periods_per_year=1)
            target_price = 100 / (1 + SPOT_RATES[k - 1]) ** k
            assert tree_price == pytest.approx(target_price, abs=TOL)

    def test_zero_volatility_tree_equals_forward_curve(self):
        """With sigma = 0 there is no branching (u = 1): the tree
        collapses to a single deterministic path. That path's rates must
        exactly equal the one-period forward rates implied by the spot
        curve -- a riskless term structure IS its own forward curve."""
        tree = build_rate_tree(SPOT_RATES, volatility=0.0, periods_per_year=1)
        tree_rates = np.array([level[0] for level in tree])
        forwards = forward_curve(SPOT_RATES, periods_per_year=1)

        np.testing.assert_allclose(tree_rates, forwards, atol=TOL)


class TestPriceOptionFreeBond:
    def test_matches_spot_curve_pricing(self):
        """The core no-arbitrage check: an option-free coupon bond valued
        by tree backward induction must match discounting the same bond's
        cash flows directly off the spot curve."""
        tree = build_rate_tree(SPOT_RATES, volatility=0.15, periods_per_year=1)

        price_tree = price_option_free_bond(100, 0.045, tree, periods_per_year=1)
        price_spot = bond_price_with_spot_rates(100, 0.045, 5, SPOT_RATES, periods_per_year=1)

        assert price_tree == pytest.approx(price_spot, abs=TOL)

    def test_matches_spot_curve_pricing_across_volatilities(self):
        """This equality must hold for ANY volatility assumption -- vol
        only matters once an embedded option makes exercise
        path-dependent; an option-free bond's value never does."""
        price_spot = bond_price_with_spot_rates(100, 0.04, 5, SPOT_RATES, periods_per_year=1)

        for vol in (0.0, 0.05, 0.20, 0.40):
            tree = build_rate_tree(SPOT_RATES, volatility=vol, periods_per_year=1)
            price_tree = price_option_free_bond(100, 0.04, tree, periods_per_year=1)
            assert price_tree == pytest.approx(price_spot, abs=TOL)


class TestPriceCallableBond:
    TREE = build_rate_tree(SPOT_RATES, volatility=0.15, periods_per_year=1)

    def test_callable_price_never_exceeds_straight_price(self):
        """Structural inequality: capping node values can only reduce (or
        leave unchanged) every value propagated backward through the
        tree, for ANY coupon/call price combination."""
        for coupon in (0.02, 0.04, 0.05, 0.07, 0.09):
            straight = price_option_free_bond(100, coupon, self.TREE, periods_per_year=1)
            callable_ = price_callable_bond(100, coupon, self.TREE, call_price=100.0, periods_per_year=1)
            assert callable_ <= straight + 1e-9

    def test_call_option_value_is_never_negative(self):
        for coupon in (0.02, 0.04, 0.05, 0.07, 0.09):
            straight = price_option_free_bond(100, coupon, self.TREE, periods_per_year=1)
            callable_ = price_callable_bond(100, coupon, self.TREE, call_price=100.0, periods_per_year=1)
            assert call_option_value(straight, callable_) >= -1e-9

    def test_call_price_never_binding_matches_straight_bond(self):
        """A call price so high it can never be advantageous to exercise
        must leave the bond's value unchanged -- the call option is
        worthless if it's never rational to use it."""
        straight = price_option_free_bond(100, 0.07, self.TREE, periods_per_year=1)
        callable_ = price_callable_bond(100, 0.07, self.TREE, call_price=1_000_000.0, periods_per_year=1)
        assert callable_ == pytest.approx(straight, abs=TOL)
        assert call_option_value(straight, callable_) == pytest.approx(0.0, abs=TOL)

    def test_higher_coupon_relative_to_curve_increases_call_option_value(self):
        """A bond trading well above the call price (high coupon relative
        to the curve) has more call risk than one near/below par -- its
        embedded call option should be worth more."""
        low_coupon_straight = price_option_free_bond(100, 0.03, self.TREE, periods_per_year=1)
        low_coupon_callable = price_callable_bond(100, 0.03, self.TREE, call_price=100.0, periods_per_year=1)

        high_coupon_straight = price_option_free_bond(100, 0.09, self.TREE, periods_per_year=1)
        high_coupon_callable = price_callable_bond(100, 0.09, self.TREE, call_price=100.0, periods_per_year=1)

        low_call_value = call_option_value(low_coupon_straight, low_coupon_callable)
        high_call_value = call_option_value(high_coupon_straight, high_coupon_callable)

        assert high_call_value > low_call_value


class TestPricePutableBond:
    TREE = build_rate_tree(SPOT_RATES, volatility=0.15, periods_per_year=1)

    def test_putable_price_never_below_straight_price(self):
        for coupon in (0.02, 0.04, 0.05, 0.07, 0.09):
            straight = price_option_free_bond(100, coupon, self.TREE, periods_per_year=1)
            putable = price_putable_bond(100, coupon, self.TREE, put_price=100.0, periods_per_year=1)
            assert putable >= straight - 1e-9

    def test_put_option_value_is_never_negative(self):
        for coupon in (0.02, 0.04, 0.05, 0.07, 0.09):
            straight = price_option_free_bond(100, coupon, self.TREE, periods_per_year=1)
            putable = price_putable_bond(100, coupon, self.TREE, put_price=100.0, periods_per_year=1)
            assert put_option_value(straight, putable) >= -1e-9

    def test_put_price_never_binding_matches_straight_bond(self):
        """A put price so low it can never be advantageous to exercise
        must leave the bond's value unchanged."""
        straight = price_option_free_bond(100, 0.03, self.TREE, periods_per_year=1)
        putable = price_putable_bond(100, 0.03, self.TREE, put_price=0.0, periods_per_year=1)
        assert putable == pytest.approx(straight, abs=TOL)
        assert put_option_value(straight, putable) == pytest.approx(0.0, abs=TOL)

    def test_lower_coupon_relative_to_curve_increases_put_option_value(self):
        """A bond trading well below the put price (low coupon relative
        to the curve) has more put value to the holder than one near/above
        par -- the downside floor is worth more the further it is
        currently below that floor."""
        high_coupon_straight = price_option_free_bond(100, 0.09, self.TREE, periods_per_year=1)
        high_coupon_putable = price_putable_bond(100, 0.09, self.TREE, put_price=100.0, periods_per_year=1)

        low_coupon_straight = price_option_free_bond(100, 0.02, self.TREE, periods_per_year=1)
        low_coupon_putable = price_putable_bond(100, 0.02, self.TREE, put_price=100.0, periods_per_year=1)

        high_put_value = put_option_value(high_coupon_straight, high_coupon_putable)
        low_put_value = put_option_value(low_coupon_straight, low_coupon_putable)

        assert low_put_value > high_put_value


class TestOptionValueIdentities:
    """Direct algebraic checks of the option-value formulas themselves."""

    def test_call_option_value_formula(self):
        assert call_option_value(101.5, 99.2) == pytest.approx(2.3, abs=1e-9)

    def test_put_option_value_formula(self):
        assert put_option_value(98.4, 100.9) == pytest.approx(2.5, abs=1e-9)


SPREAD_TOL = 1e-6


class TestCalculateZSpread:
    """As with build_rate_tree's calibration, these tests lean on the
    fact that calculate_zspread is the literal inverse of shifting the
    spot curve by a constant and discounting -- so feeding it a market
    price manufactured that same way must recover the shift exactly."""

    def test_market_price_at_zero_spread_gives_zero_zspread(self):
        """A bond priced with NO extra compensation over the spot curve
        (i.e. its market price is exactly the spot-curve price) must have
        a Z-spread of zero -- there's nothing left to explain."""
        market_price = bond_price_with_spot_rates(100, 0.045, 5, SPOT_RATES, periods_per_year=1)
        zspread = calculate_zspread(market_price, 100, 0.045, 5, SPOT_RATES, periods_per_year=1)
        assert zspread == pytest.approx(0.0, abs=SPREAD_TOL)

    def test_recovers_a_known_positive_spread(self):
        """Manufacture a market price by shifting every spot rate up by a
        known K, then check calculate_zspread recovers exactly K."""
        K = 0.015
        market_price = bond_price_with_spot_rates(100, 0.045, 5, SPOT_RATES + K, periods_per_year=1)
        zspread = calculate_zspread(market_price, 100, 0.045, 5, SPOT_RATES, periods_per_year=1)
        assert zspread == pytest.approx(K, abs=SPREAD_TOL)

    def test_recovers_a_known_negative_spread(self):
        """A market price ABOVE the zero-spread model price implies a
        negative Z-spread; the bisection search must handle that
        direction too, not just positive spreads."""
        K = -0.008
        market_price = bond_price_with_spot_rates(100, 0.045, 5, SPOT_RATES + K, periods_per_year=1)
        zspread = calculate_zspread(market_price, 100, 0.045, 5, SPOT_RATES, periods_per_year=1)
        assert zspread == pytest.approx(K, abs=SPREAD_TOL)

    def test_higher_market_price_implies_lower_zspread(self):
        """Monotonicity check: since price falls as spread rises, a
        cheaper market price must imply a WIDER Z-spread."""
        cheap_price = bond_price_with_spot_rates(100, 0.045, 5, SPOT_RATES, periods_per_year=1) - 5
        rich_price = bond_price_with_spot_rates(100, 0.045, 5, SPOT_RATES, periods_per_year=1) + 5

        cheap_zspread = calculate_zspread(cheap_price, 100, 0.045, 5, SPOT_RATES, periods_per_year=1)
        rich_zspread = calculate_zspread(rich_price, 100, 0.045, 5, SPOT_RATES, periods_per_year=1)

        assert cheap_zspread > rich_zspread

    def test_wrong_number_of_spot_rates_raises(self):
        with pytest.raises(ValueError):
            calculate_zspread(100.0, 100, 0.045, 5, SPOT_RATES[:-1], periods_per_year=1)


class TestCalculateOAS:
    """Mirrors TestCalculateZSpread's identity-recovery approach, but
    shifting the TREE (not the flat spot curve) and repricing through
    price_callable_bond -- including the call decision -- at every node."""

    TREE = build_rate_tree(SPOT_RATES, volatility=0.15, periods_per_year=1)

    def test_market_price_at_zero_spread_gives_zero_oas(self):
        market_price = price_callable_bond(100, 0.05, self.TREE, call_price=100.0, first_call_period=1, periods_per_year=1)
        oas = calculate_oas(market_price, 100, 0.05, self.TREE, call_price=100.0, first_call_period=1, periods_per_year=1)
        assert oas == pytest.approx(0.0, abs=SPREAD_TOL)

    def test_recovers_a_known_spread_when_call_never_binds(self):
        """With a call price so high it never binds, the tree behaves
        exactly like an option-free bond -- calculate_oas should still
        exactly recover a spread manufactured by shifting the tree."""
        K = 0.012
        shifted_tree = [level + K for level in self.TREE]
        market_price = price_callable_bond(100, 0.05, shifted_tree, call_price=1_000_000.0, first_call_period=1, periods_per_year=1)

        oas = calculate_oas(market_price, 100, 0.05, self.TREE, call_price=1_000_000.0, first_call_period=1, periods_per_year=1)
        assert oas == pytest.approx(K, abs=SPREAD_TOL)

    def test_recovers_a_known_spread_with_a_binding_call(self):
        """Same identity check, but now with a realistic, binding call --
        the issuer's exercise decision at the shifted rates must be
        re-solved consistently by the bisection search for OAS to still
        recover K exactly."""
        K = 0.009
        shifted_tree = [level + K for level in self.TREE]
        market_price = price_callable_bond(100, 0.09, shifted_tree, call_price=100.0, first_call_period=1, periods_per_year=1)

        oas = calculate_oas(market_price, 100, 0.09, self.TREE, call_price=100.0, first_call_period=1, periods_per_year=1)
        assert oas == pytest.approx(K, abs=SPREAD_TOL)

    def test_higher_market_price_implies_lower_oas(self):
        base_price = price_callable_bond(100, 0.05, self.TREE, call_price=100.0, periods_per_year=1)
        cheap_oas = calculate_oas(base_price - 5, 100, 0.05, self.TREE, call_price=100.0, periods_per_year=1)
        rich_oas = calculate_oas(base_price + 5, 100, 0.05, self.TREE, call_price=100.0, periods_per_year=1)
        assert cheap_oas > rich_oas


class TestOptionCost:
    TREE = build_rate_tree(SPOT_RATES, volatility=0.15, periods_per_year=1)

    def test_option_cost_formula(self):
        assert option_cost(0.0150, 0.0080) == pytest.approx(0.0070, abs=1e-9)

    def test_option_cost_positive_for_a_binding_callable_bond(self):
        """Z-spread prices the bond's stated cash flows AS IF fixed
        (ignoring the call); OAS prices them WITH the issuer's call
        decision folded in. For a high-coupon bond where the call is
        genuinely binding, matching the same market price with a
        option-blind Z-spread must overstate the "true" (OAS) credit
        spread -- i.e. Option Cost = Z-Spread - OAS > 0."""
        K = 0.01
        shifted_tree = [level + K for level in self.TREE]
        coupon = 0.09  # well above the curve -> call is likely to bind
        market_price = price_callable_bond(100, coupon, shifted_tree, call_price=100.0, first_call_period=1, periods_per_year=1)

        zspread = calculate_zspread(market_price, 100, coupon, 5, SPOT_RATES, periods_per_year=1)
        oas = calculate_oas(market_price, 100, coupon, self.TREE, call_price=100.0, first_call_period=1, periods_per_year=1)

        assert oas == pytest.approx(K, abs=SPREAD_TOL)
        assert option_cost(zspread, oas) > 0

    def test_option_cost_near_zero_when_call_never_binds(self):
        """With no realistic chance of exercise, the call adds no cost --
        Z-spread and OAS should coincide (option cost ~ 0)."""
        K = 0.01
        shifted_tree = [level + K for level in self.TREE]
        coupon = 0.045
        market_price = price_callable_bond(100, coupon, shifted_tree, call_price=1_000_000.0, first_call_period=1, periods_per_year=1)

        zspread = calculate_zspread(market_price, 100, coupon, 5, SPOT_RATES, periods_per_year=1)
        oas = calculate_oas(market_price, 100, coupon, self.TREE, call_price=1_000_000.0, first_call_period=1, periods_per_year=1)

        assert option_cost(zspread, oas) == pytest.approx(0.0, abs=1e-3)
