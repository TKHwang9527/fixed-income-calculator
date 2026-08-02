"""
app/yield_curve_gui.py

Simple tkinter GUI for the yield-curve bootstrapping in core/yield_curve.py.
Takes an annual (periods_per_year = 1) par curve for maturities 1Y-5Y and
shows the bootstrapped spot rates and implied one-period forward rates.
"""

import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.yield_curve import bootstrap_spot_rates, forward_curve

MATURITIES = ["1Y", "2Y", "3Y", "4Y", "5Y"]
DEFAULT_PAR_RATES = ["3.0", "3.5", "4.0", "4.5", "5.0"]


class YieldCurveApp:
    def __init__(self, root):
        self.root = root
        root.title("Yield Curve Bootstrapper")
        root.resizable(False, False)

        self.par_inputs = {}

        input_frame = ttk.LabelFrame(root, text="Par Rates (%)", padding=10)
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        for col, maturity in enumerate(MATURITIES):
            ttk.Label(input_frame, text=maturity).grid(row=0, column=col, padx=5)
            entry = ttk.Entry(input_frame, width=8, justify="center")
            entry.grid(row=1, column=col, padx=5, pady=4)
            entry.insert(0, DEFAULT_PAR_RATES[col])
            self.par_inputs[maturity] = entry

        ttk.Button(root, text="Calculate", command=self.calculate).grid(row=1, column=0, pady=(0, 10))

        result_frame = ttk.LabelFrame(root, text="Bootstrapped Curve", padding=10)
        result_frame.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="nsew")

        columns = ("maturity", "par", "spot", "forward")
        self.tree = ttk.Treeview(result_frame, columns=columns, show="headings", height=len(MATURITIES))
        self.tree.heading("maturity", text="Maturity")
        self.tree.heading("par", text="Par Rate")
        self.tree.heading("spot", text="Spot Rate")
        self.tree.heading("forward", text="Forward Rate")
        for col in columns:
            self.tree.column(col, width=100, anchor="center")
        self.tree.grid(row=0, column=0)

        for maturity in MATURITIES:
            self.tree.insert("", "end", iid=maturity, values=(maturity, "--", "--", "--"))

    def calculate(self):
        try:
            par_rates = [float(self.par_inputs[m].get()) / 100 for m in MATURITIES]
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers in all par rate fields.")
            return

        spot_rates = bootstrap_spot_rates(par_rates, periods_per_year=1)
        forwards = forward_curve(spot_rates, periods_per_year=1)

        for i, maturity in enumerate(MATURITIES):
            self.tree.item(
                maturity,
                values=(
                    maturity,
                    f"{par_rates[i] * 100:.4f}%",
                    f"{spot_rates[i] * 100:.4f}%",
                    f"{forwards[i] * 100:.4f}%",
                ),
            )


def main():
    root = tk.Tk()
    YieldCurveApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
