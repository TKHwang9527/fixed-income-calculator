"""
tests/test_bond_pricing.py

Verifies core/bond_pricing.py against standard CFA-style textbook bond
examples with known/hand-verifiable results.

Reference case (most important sanity check):
    A 3-year, 6% annual-pay coupon bond priced at a 6% yield is a PAR bond
    (price = 100) with a widely-cited Macaulay duration of 2.833 years and
    modified duration of 2.673 years -- this is the canonical CFA Level I
    duration example and is used below as the primary correctness check.
"""

import pytest

from core.bond_pricing import (
    bond_price,
    macaulay_duration,
    modified_duration,
    dv01,
    convexity,
)

TOL = 1e-4


class TestParBondAnnual:
    """3-year, 6% annual coupon bond, YTM = 6%, annual pay.
    Classic CFA example: coupon rate == yield => bond prices at par."""

    FACE, COUPON, YIELD, YEARS, FREQ = 100, 0.06, 0.06, 3, 1

    def test_price_equals_par(self):
        price = bond_price(self.FACE, self.COUPON, self.YIELD, self.YEARS, self.FREQ)
        assert price == pytest.approx(100.0, abs=TOL)

    def test_macaulay_duration(self):
        d = macaulay_duration(self.FACE, self.COUPON, self.YIELD, self.YEARS, self.FREQ)
        assert d == pytest.approx(2.833393, abs=TOL)

    def test_modified_duration(self):
        d = modified_duration(self.FACE, self.COUPON, self.YIELD, self.YEARS, self.FREQ)
        assert d == pytest.approx(2.673012, abs=TOL)

    def test_dv01(self):
        v = dv01(self.FACE, self.COUPON, self.YIELD, self.YEARS, self.FREQ)
        assert v == pytest.approx(0.02673012, abs=1e-6)

    def test_convexity(self):
        c = convexity(self.FACE, self.COUPON, self.YIELD, self.YEARS, self.FREQ)
        assert c == pytest.approx(9.891032, abs=TOL)


class TestZeroCouponAnnual:
    """5-year zero-coupon bond, YTM = 5%, annual compounding.
    For a zero-coupon bond, Macaulay duration must equal maturity exactly,
    and convexity has the closed form n(n+1)/(1+y)^2."""

    FACE, COUPON, YIELD, YEARS, FREQ = 100, 0.0, 0.05, 5, 1

    def test_price(self):
        price = bond_price(self.FACE, self.COUPON, self.YIELD, self.YEARS, self.FREQ)
        assert price == pytest.approx(78.352617, abs=TOL)

    def test_macaulay_duration_equals_maturity(self):
        d = macaulay_duration(self.FACE, self.COUPON, self.YIELD, self.YEARS, self.FREQ)
        assert d == pytest.approx(5.0, abs=TOL)

    def test_modified_duration(self):
        d = modified_duration(self.FACE, self.COUPON, self.YIELD, self.YEARS, self.FREQ)
        assert d == pytest.approx(4.761905, abs=TOL)

    def test_convexity_closed_form(self):
        c = convexity(self.FACE, self.COUPON, self.YIELD, self.YEARS, self.FREQ)
        expected = self.YEARS * (self.YEARS + 1) / (1 + self.YIELD) ** 2
        assert c == pytest.approx(expected, abs=TOL)
        assert c == pytest.approx(27.210884, abs=TOL)


class TestZeroCouponSemiannual:
    """Same 5-year zero-coupon bond but with semiannual compounding (m=2),
    to confirm duration is frequency-invariant (still = maturity in years)
    while convexity and modified duration change with compounding frequency."""

    FACE, COUPON, YIELD, YEARS, FREQ = 100, 0.0, 0.05, 5, 2

    def test_price(self):
        price = bond_price(self.FACE, self.COUPON, self.YIELD, self.YEARS, self.FREQ)
        assert price == pytest.approx(78.119840, abs=TOL)

    def test_macaulay_duration_equals_maturity(self):
        d = macaulay_duration(self.FACE, self.COUPON, self.YIELD, self.YEARS, self.FREQ)
        assert d == pytest.approx(5.0, abs=TOL)

    def test_modified_duration(self):
        d = modified_duration(self.FACE, self.COUPON, self.YIELD, self.YEARS, self.FREQ)
        assert d == pytest.approx(4.878049, abs=TOL)

    def test_convexity(self):
        c = convexity(self.FACE, self.COUPON, self.YIELD, self.YEARS, self.FREQ)
        assert c == pytest.approx(26.174896, abs=TOL)


class TestParBondSemiannual:
    """10-year, 8% semiannual-pay coupon bond, YTM = 8%.
    Standard US Treasury-style bond convention (m=2); coupon == yield
    so it must also price at par."""

    FACE, COUPON, YIELD, YEARS, FREQ = 100, 0.08, 0.08, 10, 2

    def test_price_equals_par(self):
        price = bond_price(self.FACE, self.COUPON, self.YIELD, self.YEARS, self.FREQ)
        assert price == pytest.approx(100.0, abs=TOL)

    def test_macaulay_duration(self):
        d = macaulay_duration(self.FACE, self.COUPON, self.YIELD, self.YEARS, self.FREQ)
        assert d == pytest.approx(7.066970, abs=TOL)

    def test_modified_duration(self):
        d = modified_duration(self.FACE, self.COUPON, self.YIELD, self.YEARS, self.FREQ)
        assert d == pytest.approx(6.795163, abs=TOL)

    def test_dv01(self):
        v = dv01(self.FACE, self.COUPON, self.YIELD, self.YEARS, self.FREQ)
        assert v == pytest.approx(0.06795163, abs=1e-6)

    def test_convexity(self):
        c = convexity(self.FACE, self.COUPON, self.YIELD, self.YEARS, self.FREQ)
        assert c == pytest.approx(60.170679, abs=TOL)


class TestPremiumBond:
    """3-year, 8% annual coupon bond, YTM = 6% -> prices above par (premium)."""

    FACE, COUPON, YIELD, YEARS, FREQ = 100, 0.08, 0.06, 3, 1

    def test_price_above_par(self):
        price = bond_price(self.FACE, self.COUPON, self.YIELD, self.YEARS, self.FREQ)
        assert price == pytest.approx(105.346024, abs=TOL)
        assert price > self.FACE

    def test_macaulay_duration(self):
        d = macaulay_duration(self.FACE, self.COUPON, self.YIELD, self.YEARS, self.FREQ)
        assert d == pytest.approx(2.789130, abs=TOL)

    def test_modified_duration(self):
        d = modified_duration(self.FACE, self.COUPON, self.YIELD, self.YEARS, self.FREQ)
        assert d == pytest.approx(2.631255, abs=TOL)

    def test_convexity(self):
        c = convexity(self.FACE, self.COUPON, self.YIELD, self.YEARS, self.FREQ)
        assert c == pytest.approx(9.681438, abs=TOL)


class TestDiscountBond:
    """3-year, 4% annual coupon bond, YTM = 6% -> prices below par (discount)."""

    FACE, COUPON, YIELD, YEARS, FREQ = 100, 0.04, 0.06, 3, 1

    def test_price_below_par(self):
        price = bond_price(self.FACE, self.COUPON, self.YIELD, self.YEARS, self.FREQ)
        assert price == pytest.approx(94.653976, abs=TOL)
        assert price < self.FACE

    def test_macaulay_duration(self):
        d = macaulay_duration(self.FACE, self.COUPON, self.YIELD, self.YEARS, self.FREQ)
        assert d == pytest.approx(2.882655, abs=TOL)

    def test_modified_duration(self):
        d = modified_duration(self.FACE, self.COUPON, self.YIELD, self.YEARS, self.FREQ)
        assert d == pytest.approx(2.719486, abs=TOL)

    def test_convexity(self):
        c = convexity(self.FACE, self.COUPON, self.YIELD, self.YEARS, self.FREQ)
        assert c == pytest.approx(10.124302, abs=TOL)


class TestDurationConvexityConsistency:
    """Cross-check: the second-order Taylor approximation
    %dP ~= -ModDur*dy + 0.5*Convexity*dy^2 should closely match the
    ACTUAL percentage price change from repricing the bond directly,
    for a reasonably small yield shock (100 bps)."""

    def test_taylor_approximation_matches_repricing(self):
        face, coupon, y0, years, freq = 100, 0.07, 0.07, 10, 2
        dy = 0.01  # 100 bp shock

        price0 = bond_price(face, coupon, y0, years, freq)
        price1 = bond_price(face, coupon, y0 + dy, years, freq)
        actual_pct_change = (price1 - price0) / price0

        mod_dur = modified_duration(face, coupon, y0, years, freq)
        conv = convexity(face, coupon, y0, years, freq)
        approx_pct_change = -mod_dur * dy + 0.5 * conv * dy ** 2

        assert approx_pct_change == pytest.approx(actual_pct_change, abs=5e-4)
