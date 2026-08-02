"""
app/streamlit_app.py

Streamlit web app for the fixed-income calculators in core/:
    - Bond Calculator : price, Macaulay/modified duration, DV01, convexity
    - Yield Curve      : bootstrap spot rates + forward rates from a par curve
    - Binomial Tree    : straight/callable/putable bond pricing and OAS off a
                          calibrated interest rate tree

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.bond_pricing import bond_price, convexity, dv01, macaulay_duration, modified_duration
from core.yield_curve import bootstrap_spot_rates, forward_curve
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

st.set_page_config(page_title="Fixed Income Calculator", layout="wide")
st.title("Fixed Income Calculator")

tab_bond, tab_curve, tab_tree = st.tabs(["Bond Calculator", "Yield Curve", "Binomial Tree"])


# ---------------------------------------------------------------------
# Tab 1: Bond Calculator
# ---------------------------------------------------------------------
with tab_bond:
    st.header("Bond Calculator")

    col1, col2 = st.columns(2)
    with col1:
        face_value = st.number_input("Face Value", value=100.0, min_value=0.0, key="bc_face")
        coupon_rate = st.number_input("Coupon Rate (%)", value=6.0, key="bc_coupon") / 100
        periods_per_year = st.number_input("Periods per Year", value=2, min_value=1, step=1, key="bc_ppy")
    with col2:
        yield_rate = st.number_input("Yield to Maturity (%)", value=6.0, key="bc_yield") / 100
        years_to_maturity = st.number_input("Years to Maturity", value=10.0, min_value=0.0, key="bc_years")

    args = (face_value, coupon_rate, yield_rate, years_to_maturity, int(periods_per_year))

    st.subheader("Results")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Price", f"{bond_price(*args):.4f}")
    r2.metric("Macaulay Duration", f"{macaulay_duration(*args):.4f}")
    r3.metric("Modified Duration", f"{modified_duration(*args):.4f}")
    r4.metric("DV01", f"{dv01(*args):.4f}")
    r5.metric("Convexity", f"{convexity(*args):.4f}")


# ---------------------------------------------------------------------
# Tab 2: Yield Curve
# ---------------------------------------------------------------------
with tab_curve:
    st.header("Yield Curve")

    yc_ppy = st.number_input("Periods per Year", value=1, min_value=1, step=1, key="yc_ppy")

    default_par_rates = "3.0, 3.3, 3.6, 3.8, 4.0"
    par_rates_input = st.text_input(
        "Par Rates (%, comma-separated, one per maturity period)",
        value=default_par_rates,
        key="yc_par_rates",
    )

    try:
        par_rates = [float(x.strip()) / 100 for x in par_rates_input.split(",") if x.strip() != ""]
    except ValueError:
        par_rates = None
        st.error("Could not parse par rates. Use a comma-separated list of numbers, e.g. 3.0, 3.3, 3.6")

    if par_rates:
        spot_rates = bootstrap_spot_rates(par_rates, int(yc_ppy))
        forwards = forward_curve(spot_rates, int(yc_ppy))

        periods = list(range(1, len(par_rates) + 1))
        df = pd.DataFrame(
            {
                "Period": periods,
                "Par Rate (%)": [r * 100 for r in par_rates],
                "Spot Rate (%)": spot_rates * 100,
                "Forward Rate (%)": forwards * 100,
            }
        ).set_index("Period")

        st.subheader("Rates Table")
        st.dataframe(df.style.format("{:.4f}"), use_container_width=True)

        st.subheader("Rates Chart")
        st.line_chart(df[["Par Rate (%)", "Spot Rate (%)", "Forward Rate (%)"]])


# ---------------------------------------------------------------------
# Tab 3: Binomial Tree
# ---------------------------------------------------------------------
with tab_tree:
    st.header("Binomial Interest Rate Tree")

    st.subheader("Bond Parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        bt_face_value = st.number_input("Face Value", value=100.0, min_value=0.0, key="bt_face")
        bt_coupon_rate = st.number_input("Coupon Rate (%)", value=6.0, key="bt_coupon") / 100
    with c2:
        bt_ppy = st.number_input("Periods per Year", value=1, min_value=1, step=1, key="bt_ppy")
        bt_volatility = st.number_input("Volatility (%)", value=10.0, min_value=0.0, key="bt_vol") / 100
    with c3:
        call_price = st.number_input("Call Price", value=100.0, key="bt_call_price")
        first_call_period = st.number_input("First Call Period", value=1, min_value=1, step=1, key="bt_call_period")

    c4, c5 = st.columns(2)
    with c4:
        put_price = st.number_input("Put Price", value=100.0, key="bt_put_price")
    with c5:
        first_put_period = st.number_input("First Put Period", value=1, min_value=1, step=1, key="bt_put_period")

    st.subheader("Par Rate Curve")
    default_bt_par_rates = "3.0, 3.3, 3.6, 3.8, 4.0"
    bt_par_rates_input = st.text_input(
        "Par Rates (%, comma-separated, one per period)",
        value=default_bt_par_rates,
        key="bt_par_rates",
    )

    market_price_input = st.text_input(
        "Market Price of Callable Bond (optional, for OAS)",
        value="",
        key="bt_market_price",
    )

    try:
        bt_par_rates = [float(x.strip()) / 100 for x in bt_par_rates_input.split(",") if x.strip() != ""]
    except ValueError:
        bt_par_rates = None
        st.error("Could not parse par rates. Use a comma-separated list of numbers, e.g. 3.0, 3.3, 3.6")

    if bt_par_rates:
        bt_spot_rates = bootstrap_spot_rates(bt_par_rates, int(bt_ppy))

        try:
            rate_tree = build_rate_tree(bt_spot_rates, bt_volatility, int(bt_ppy))
        except RuntimeError as exc:
            rate_tree = None
            st.error(f"Could not calibrate rate tree: {exc}")

        if rate_tree:
            straight_price = price_option_free_bond(bt_face_value, bt_coupon_rate, rate_tree, int(bt_ppy))
            callable_price = price_callable_bond(
                bt_face_value, bt_coupon_rate, rate_tree, call_price, int(first_call_period), int(bt_ppy)
            )
            putable_price = price_putable_bond(
                bt_face_value, bt_coupon_rate, rate_tree, put_price, int(first_put_period), int(bt_ppy)
            )

            st.subheader("Bond Prices")
            p1, p2, p3 = st.columns(3)
            p1.metric("Straight (Option-Free) Price", f"{straight_price:.4f}")
            p2.metric("Callable Price", f"{callable_price:.4f}", f"{-call_option_value(straight_price, callable_price):.4f} vs straight")
            p3.metric("Putable Price", f"{putable_price:.4f}", f"{put_option_value(straight_price, putable_price):.4f} vs straight")

            st.caption(
                f"Call option value: {call_option_value(straight_price, callable_price):.4f}   |   "
                f"Put option value: {put_option_value(straight_price, putable_price):.4f}"
            )

            if market_price_input.strip():
                try:
                    market_price = float(market_price_input)
                    zspread = calculate_zspread(
                        market_price, bt_face_value, bt_coupon_rate, len(bt_par_rates) / bt_ppy, bt_spot_rates, int(bt_ppy)
                    )
                    oas = calculate_oas(
                        market_price, bt_face_value, bt_coupon_rate, rate_tree, call_price, int(first_call_period), int(bt_ppy)
                    )
                    opt_cost = option_cost(zspread, oas)

                    st.subheader("Spread Analysis")
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Z-Spread (bp)", f"{zspread * 10000:.2f}")
                    s2.metric("OAS (bp)", f"{oas * 10000:.2f}")
                    s3.metric("Option Cost (bp)", f"{opt_cost * 10000:.2f}")
                except ValueError:
                    st.error("Market price must be a number.")

            st.subheader("Interest Rate Tree")
            display_rows = []
            for i in range(len(rate_tree)):
                row = []
                for k, level in enumerate(rate_tree):
                    row.append(f"{level[i] * 100:.3f}%" if i < len(level) else "")
                display_rows.append(row)
            tree_display = pd.DataFrame(
                display_rows[::-1],
                columns=[f"t={k}" for k in range(len(rate_tree))],
            )
            st.dataframe(tree_display, use_container_width=True)
