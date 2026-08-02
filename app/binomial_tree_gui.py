"""
app/binomial_tree_gui.py

Simple tkinter GUI for the binomial interest rate tree in
core/binomial_tree.py. The tree is calibrated to a par rate curve (one
input per year, matching Years to Maturity) via core/yield_curve.py's
bootstrap, then used to value a straight bond and its callable/putable
variants.
"""

import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
from core.yield_curve import bootstrap_spot_rates

MAX_YEARS = 20
CURVE_COLUMNS = 6

RESULT_FIELDS = [
    ("straight", "Straight Bond Price"),
    ("callable", "Callable Bond Price"),
    ("putable", "Putable Bond Price"),
    ("call_value", "Call Option Value"),
    ("put_value", "Put Option Value"),
    ("zspread", "Z-Spread (%)"),
    ("oas", "OAS (%)"),
    ("option_cost", "Option Cost (%)"),
]


class BinomialTreeApp:
    def __init__(self, root):
        self.root = root
        root.title("Binomial Tree Bond Calculator")
        root.resizable(True, True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(4, weight=1)  # tree frame grows first when the window is enlarged

        self.curve_entries = []
        self.curve_years = 0

        self._build_param_frame()
        self._build_curve_frame()
        self.set_years()  # populate curve fields for the default Years to Maturity

        ttk.Button(root, text="Calculate", command=self.calculate).grid(row=2, column=0, pady=(0, 10))

        self._build_results_frame()
        self._build_tree_frame()

        # Don't let the window shrink below the size needed for its initial
        # layout -- resizable(True, True) alone would allow shrinking until
        # widgets clip/overlap.
        root.update_idletasks()
        root.minsize(root.winfo_width(), root.winfo_height())

    def _build_param_frame(self):
        frame = ttk.LabelFrame(self.root, text="Bond & Tree Inputs", padding=10)
        frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.face_value = self._labeled_entry(frame, 0, "Face Value", "100")
        self.coupon_rate = self._labeled_entry(frame, 1, "Coupon Rate (%)", "7")

        ttk.Label(frame, text="Years to Maturity").grid(row=2, column=0, sticky="w", pady=4)
        self.years = ttk.Entry(frame, width=15)
        self.years.insert(0, "5")
        self.years.grid(row=2, column=1, pady=4, padx=(10, 0))
        ttk.Button(frame, text="Set Years", command=self.set_years).grid(row=2, column=2, padx=(10, 0))

        self.volatility = self._labeled_entry(frame, 3, "Volatility (%)", "15")
        self.call_price = self._labeled_entry(frame, 4, "Call Price (optional)", "")
        self.put_price = self._labeled_entry(frame, 5, "Put Price (optional)", "")
        self.market_price = self._labeled_entry(frame, 6, "Market Price (optional)", "")

    @staticmethod
    def _labeled_entry(frame, row, label, default):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
        entry = ttk.Entry(frame, width=15)
        entry.grid(row=row, column=1, pady=4, padx=(10, 0))
        if default:
            entry.insert(0, default)
        return entry

    def _build_curve_frame(self):
        self.curve_outer = ttk.LabelFrame(self.root, text="Par Rate Curve (%)", padding=10)
        self.curve_outer.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")

    def set_years(self):
        try:
            n = int(self.years.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Years to Maturity must be a whole number.")
            return

        if not (1 <= n <= MAX_YEARS):
            messagebox.showerror("Invalid Input", f"Years to Maturity must be between 1 and {MAX_YEARS}.")
            return

        for widget in self.curve_outer.winfo_children():
            widget.destroy()

        self.curve_entries = []
        for i in range(n):
            row, col = divmod(i, CURVE_COLUMNS)
            cell = ttk.Frame(self.curve_outer)
            cell.grid(row=row, column=col, padx=5, pady=4)
            ttk.Label(cell, text=f"{i + 1}Y").pack()
            entry = ttk.Entry(cell, width=7, justify="center")
            entry.insert(0, f"{3.0 + 0.3 * i:.2f}")
            entry.pack()
            self.curve_entries.append(entry)

        self.curve_years = n

    def _build_results_frame(self):
        frame = ttk.LabelFrame(self.root, text="Results", padding=10)
        frame.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="nsew")

        self.results = {}
        for row, (key, label) in enumerate(RESULT_FIELDS):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
            value = ttk.Label(frame, text="--", width=15, anchor="e")
            value.grid(row=row, column=1, sticky="e", pady=4, padx=(10, 0))
            self.results[key] = value

    def _build_tree_frame(self):
        self.tree_outer = ttk.LabelFrame(self.root, text="Interest Rate Tree", padding=10)
        self.tree_outer.grid(row=4, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.tree_inner = ttk.Frame(self.tree_outer)
        self.tree_inner.pack()

    def _render_tree(self, rate_tree):
        for widget in self.tree_inner.winfo_children():
            widget.destroy()

        for k in range(len(rate_tree)):
            ttk.Label(self.tree_inner, text=f"t={k}", font=("TkDefaultFont", 9, "bold")).grid(
                row=0, column=k, padx=6, pady=(0, 4)
            )

        for k, level in enumerate(rate_tree):
            for i, rate in enumerate(level):
                ttk.Label(self.tree_inner, text=f"{rate * 100:.3f}%").grid(row=i + 1, column=k, padx=6, pady=2)

    def calculate(self):
        try:
            face_value = float(self.face_value.get())
            coupon_rate = float(self.coupon_rate.get()) / 100
            years = int(self.years.get())
            volatility = float(self.volatility.get()) / 100
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for the bond and tree inputs.")
            return

        if years != self.curve_years:
            messagebox.showerror(
                "Curve Not Set",
                "Years to Maturity has changed. Click 'Set Years' to rebuild the par rate curve fields first.",
            )
            return

        try:
            par_rates = [float(e.get()) / 100 for e in self.curve_entries]
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers in every par rate field.")
            return

        call_price_str = self.call_price.get().strip()
        put_price_str = self.put_price.get().strip()
        market_price_str = self.market_price.get().strip()
        try:
            call_price = float(call_price_str) if call_price_str else None
            put_price = float(put_price_str) if put_price_str else None
            market_price = float(market_price_str) if market_price_str else None
        except ValueError:
            messagebox.showerror(
                "Invalid Input", "Call Price, Put Price, and Market Price must be numbers (or left blank)."
            )
            return

        spot_rates = bootstrap_spot_rates(par_rates, periods_per_year=1)
        rate_tree = build_rate_tree(spot_rates, volatility, periods_per_year=1)

        straight = price_option_free_bond(face_value, coupon_rate, rate_tree, periods_per_year=1)
        self.results["straight"].config(text=f"{straight:.4f}")

        if call_price is not None:
            callable_price = price_callable_bond(
                face_value, coupon_rate, rate_tree, call_price=call_price, periods_per_year=1
            )
            self.results["callable"].config(text=f"{callable_price:.4f}")
            self.results["call_value"].config(text=f"{call_option_value(straight, callable_price):.4f}")
        else:
            self.results["callable"].config(text="--")
            self.results["call_value"].config(text="--")

        if put_price is not None:
            putable_price = price_putable_bond(
                face_value, coupon_rate, rate_tree, put_price=put_price, periods_per_year=1
            )
            self.results["putable"].config(text=f"{putable_price:.4f}")
            self.results["put_value"].config(text=f"{put_option_value(straight, putable_price):.4f}")
        else:
            self.results["putable"].config(text="--")
            self.results["put_value"].config(text="--")

        zspread = None
        if market_price is not None:
            zspread = calculate_zspread(market_price, face_value, coupon_rate, years, spot_rates, periods_per_year=1)
            self.results["zspread"].config(text=f"{zspread * 100:.4f}")
        else:
            self.results["zspread"].config(text="--")

        if market_price is not None and call_price is not None:
            oas = calculate_oas(market_price, face_value, coupon_rate, rate_tree, call_price=call_price, periods_per_year=1)
            self.results["oas"].config(text=f"{oas * 100:.4f}")
            self.results["option_cost"].config(text=f"{option_cost(zspread, oas) * 100:.4f}")
        else:
            self.results["oas"].config(text="--")
            self.results["option_cost"].config(text="--")

        self._render_tree(rate_tree)


def main():
    root = tk.Tk()
    BinomialTreeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
