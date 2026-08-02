"""
tests/test_yield_curve.py

Verifies core/yield_curve.py: bootstrapping spot rates from a par curve,
deriving forward rates from spot rates, and pricing bonds off a spot
curve. Rather than leaning on memorized textbook numbers, most tests
exploit the fact that these formulas are self-verifying by construction:

    - A par bond, by definition, must reprice to exactly 100 when
      discounted on the very spot curve that was bootstrapped from it.
    - Chaining one-period forward rates must reproduce the original spot
      rate (spot rates ARE geometric averages of forwards).
    - A FLAT par curve must bootstrap to an identically flat spot curve,
      and spot-rate pricing on a flat curve must exactly match
      bond_pricing.bond_price() using a single YTM.

Numeric values for the upward-sloping curve case were computed directly
from this module and are pinned here as regression checks.
"""

import numpy as np
import pytest

from core.bond_pricing import bond_price
from core.yield_curve import (
    bond_price_with_spot_rates,
    bootstrap_spot_rates,
    forward_curve,
    forward_rate,
)

TOL = 1e-4


class TestBootstrapFlatCurve:
    """A flat par curve (every maturity has the same par rate) must
    bootstrap to an identically flat spot curve. Proof by induction: if
    z_1..z_{t-1} all equal the flat rate r, discounting a par bond with
    coupon r at those (also flat) rates collapses to the ordinary
    single-yield par-bond identity, which is satisfied by z_t = r too."""

    FLAT_RATE, YEARS = 0.05, 5

    def test_spot_rates_equal_par_rate(self):
        par_rates = [self.FLAT_RATE] * self.YEARS
        spot_rates = bootstrap_spot_rates(par_rates, periods_per_year=1)
        np.testing.assert_allclose(spot_rates, [self.FLAT_RATE] * self.YEARS, atol=TOL)

    def test_matches_single_ytm_pricing(self):
        """On a flat curve, spot-rate pricing must exactly match the
        existing single-YTM bond_price() -- they're mathematically
        identical when every spot rate equals the YTM."""
        par_rates = [self.FLAT_RATE] * self.YEARS
        spot_rates = bootstrap_spot_rates(par_rates, periods_per_year=1)

        price_spot = bond_price_with_spot_rates(100, self.FLAT_RATE, self.YEARS, spot_rates, periods_per_year=1)
        price_ytm = bond_price(100, self.FLAT_RATE, self.FLAT_RATE, self.YEARS, periods_per_year=1)

        assert price_spot == pytest.approx(price_ytm, abs=TOL)
        assert price_spot == pytest.approx(100.0, abs=TOL)


class TestBootstrapUpwardSlopingCurve:
    """5-year annual par curve rising from 3% to 5%. Values pinned below
    were computed from this module; the reprice check is the real
    correctness test and doesn't depend on those pinned numbers at all."""

    PAR_RATES = [0.03, 0.035, 0.04, 0.045, 0.05]
    EXPECTED_SPOTS = [0.030000, 0.035088, 0.040272, 0.045585, 0.051066]

    def test_spot_rates(self):
        spot_rates = bootstrap_spot_rates(self.PAR_RATES, periods_per_year=1)
        np.testing.assert_allclose(spot_rates, self.EXPECTED_SPOTS, atol=TOL)

    def test_first_spot_equals_first_par_rate(self):
        """The 1-period spot rate always equals the 1-period par rate --
        there's only one cash flow date, so there's nothing to bootstrap."""
        spot_rates = bootstrap_spot_rates(self.PAR_RATES, periods_per_year=1)
        assert spot_rates[0] == pytest.approx(self.PAR_RATES[0], abs=TOL)

    def test_reprices_every_par_bond_to_par(self):
        """The defining property of a bootstrapped spot curve: discounting
        each maturity's OWN par bond on the spot curve (using only the
        spot rates up to that maturity) must reproduce price = 100,
        for every maturity along the curve."""
        spot_rates = bootstrap_spot_rates(self.PAR_RATES, periods_per_year=1)

        for t in range(1, len(self.PAR_RATES) + 1):
            price = bond_price_with_spot_rates(
                100, self.PAR_RATES[t - 1], t, spot_rates[:t], periods_per_year=1
            )
            assert price == pytest.approx(100.0, abs=TOL)

    def test_upward_sloping_par_curve_implies_upward_sloping_spot_curve(self):
        spot_rates = bootstrap_spot_rates(self.PAR_RATES, periods_per_year=1)
        assert np.all(np.diff(spot_rates) > 0)
        assert np.all(spot_rates >= self.PAR_RATES)  # spot > par on an upward curve


class TestBootstrapSemiannual:
    """Same bootstrap logic, but semiannual coupon/compounding (m=2),
    to confirm the formula generalizes beyond annual-pay bonds."""

    PAR_RATES = [0.02, 0.025, 0.03, 0.035]  # 4 semiannual periods = 2 years

    def test_reprices_par_bonds_to_par(self):
        spot_rates = bootstrap_spot_rates(self.PAR_RATES, periods_per_year=2)
        for t in range(1, len(self.PAR_RATES) + 1):
            years = t / 2
            price = bond_price_with_spot_rates(
                100, self.PAR_RATES[t - 1], years, spot_rates[:t], periods_per_year=2
            )
            assert price == pytest.approx(100.0, abs=TOL)


class TestForwardRates:
    """Forward rates implied by the upward-sloping spot curve above."""

    PAR_RATES = [0.03, 0.035, 0.04, 0.045, 0.05]

    def test_forward_rate_from_today_equals_spot_rate(self):
        """f(0, t) is just the spot rate for t -- there's no "starting
        point" rate to divide out when period1 = 0 (today)."""
        spot_rates = bootstrap_spot_rates(self.PAR_RATES, periods_per_year=1)
        for t in range(1, len(self.PAR_RATES) + 1):
            assert forward_rate(spot_rates, 0, t, periods_per_year=1) == pytest.approx(
                spot_rates[t - 1], abs=TOL
            )

    def test_one_year_forward_two_years_out(self):
        """The "2y1y" forward: rate on a 1-period loan starting 2 periods
        from now, implied by the 2-period and 3-period spot rates."""
        spot_rates = bootstrap_spot_rates(self.PAR_RATES, periods_per_year=1)
        f_2_3 = forward_rate(spot_rates, 2, 3, periods_per_year=1)
        assert f_2_3 == pytest.approx(0.050718, abs=TOL)

    def test_forward_curve_matches_pairwise_forward_rate(self):
        spot_rates = bootstrap_spot_rates(self.PAR_RATES, periods_per_year=1)
        forwards = forward_curve(spot_rates, periods_per_year=1)

        assert forwards[0] == pytest.approx(spot_rates[0], abs=TOL)
        for t in range(2, len(spot_rates) + 1):
            expected = forward_rate(spot_rates, t - 1, t, periods_per_year=1)
            assert forwards[t - 1] == pytest.approx(expected, abs=TOL)

    def test_forwards_chain_back_to_spot_rate(self):
        """No-arbitrage reconstitution: (1+z_t)^t == product of (1+f_k)
        for the one-period forwards k=1..t. Spot rates are nothing more
        than the geometric average of the forward curve up to that point."""
        spot_rates = bootstrap_spot_rates(self.PAR_RATES, periods_per_year=1)
        forwards = forward_curve(spot_rates, periods_per_year=1)

        for t in range(1, len(spot_rates) + 1):
            chained_growth = np.prod(1 + forwards[:t])
            direct_growth = (1 + spot_rates[t - 1]) ** t
            assert chained_growth == pytest.approx(direct_growth, abs=TOL)

    def test_upward_sloping_curve_implies_rising_forwards(self):
        """On an upward-sloping spot curve, each successive one-period
        forward rate must exceed the last (the market expects short rates
        to keep rising, or equivalently demands rising compensation)."""
        spot_rates = bootstrap_spot_rates(self.PAR_RATES, periods_per_year=1)
        forwards = forward_curve(spot_rates, periods_per_year=1)
        assert np.all(np.diff(forwards) > 0)

    def test_period2_must_exceed_period1(self):
        spot_rates = bootstrap_spot_rates(self.PAR_RATES, periods_per_year=1)
        with pytest.raises(ValueError):
            forward_rate(spot_rates, 3, 3, periods_per_year=1)
        with pytest.raises(ValueError):
            forward_rate(spot_rates, 3, 1, periods_per_year=1)


class TestBondPriceWithSpotRates:
    """Spot-curve pricing vs. single-YTM pricing on a genuinely
    non-flat curve: a bond's cash flows should generally price
    differently off the full curve than off a single blended YTM,
    except in the coincidental flat-curve case covered above."""

    PAR_RATES = [0.03, 0.035, 0.04, 0.045, 0.05]

    def test_zero_coupon_bond_price_equals_discount_factor(self):
        """A pure zero-coupon bond's price is just 100 discounted at the
        single maturity-matched spot rate -- the simplest possible check
        of the discounting formula, with no coupons to sum over."""
        spot_rates = bootstrap_spot_rates(self.PAR_RATES, periods_per_year=1)
        price = bond_price_with_spot_rates(100, 0.0, 5, spot_rates, periods_per_year=1)
        expected = 100 / (1 + spot_rates[-1]) ** 5
        assert price == pytest.approx(expected, abs=TOL)

    def test_wrong_number_of_spot_rates_raises(self):
        spot_rates = bootstrap_spot_rates(self.PAR_RATES, periods_per_year=1)
        with pytest.raises(ValueError):
            bond_price_with_spot_rates(100, 0.04, 5, spot_rates[:3], periods_per_year=1)

    def test_coupon_bond_priced_off_curve_differs_from_flat_ytm(self):
        """Pricing a 7% coupon, 5-year bond off the upward-sloping spot
        curve should NOT equal pricing it at a single flat 5% YTM (the
        curve's longest-maturity par rate) -- discounting each cash flow
        at its own (higher, for later years) spot rate rather than a
        single blended 5% pulls the price away from the flat-YTM price.
        Note: a coupon exactly equal to a par rate on this curve would
        coincidentally reprice to 100 under both methods (see
        test_reprices_every_par_bond_to_par), so 7% is deliberately
        chosen to not match any par rate in PAR_RATES."""
        spot_rates = bootstrap_spot_rates(self.PAR_RATES, periods_per_year=1)
        price_curve = bond_price_with_spot_rates(100, 0.07, 5, spot_rates, periods_per_year=1)
        price_flat_ytm = bond_price(100, 0.07, 0.05, 5, periods_per_year=1)

        assert price_curve != pytest.approx(price_flat_ytm, abs=TOL)
