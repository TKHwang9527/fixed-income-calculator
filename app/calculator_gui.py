"""
app/calculator_gui.py

Simple tkinter GUI for the bond pricing/risk calculators in core/bond_pricing.py.
"""

import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.bond_pricing import bond_price, convexity, dv01, macaulay_duration, modified_duration

INPUT_FIELDS = [
    ("face_value", "Face Value"),
    ("coupon_rate", "Coupon Rate (%)"),
    ("years_to_maturity", "Years to Maturity"),
    ("yield_rate", "YTM (%)"),
    ("periods_per_year", "Periods per Year"),
]

RESULT_FIELDS = [
    ("price", "Bond Price"),
    ("mac_duration", "Macaulay Duration"),
    ("mod_duration", "Modified Duration"),
    ("dv01", "DV01"),
    ("convexity", "Convexity"),
]


class CalculatorApp:
    def __init__(self, root):
        self.root = root
        root.title("Bond Calculator")
        root.resizable(False, False)

        self.inputs = {}
        self.results = {}

        input_frame = ttk.LabelFrame(root, text="Bond Inputs", padding=10)
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        for row, (key, label) in enumerate(INPUT_FIELDS):
            ttk.Label(input_frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
            entry = ttk.Entry(input_frame, width=15)
            entry.grid(row=row, column=1, pady=4, padx=(10, 0))
            self.inputs[key] = entry

        self.inputs["face_value"].insert(0, "100")
        self.inputs["coupon_rate"].insert(0, "6")
        self.inputs["years_to_maturity"].insert(0, "3")
        self.inputs["yield_rate"].insert(0, "6")
        self.inputs["periods_per_year"].insert(0, "2")

        ttk.Button(root, text="Calculate", command=self.calculate).grid(row=1, column=0, pady=(0, 10))

        result_frame = ttk.LabelFrame(root, text="Results", padding=10)
        result_frame.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="nsew")

        for row, (key, label) in enumerate(RESULT_FIELDS):
            ttk.Label(result_frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
            value = ttk.Label(result_frame, text="--", width=15, anchor="e")
            value.grid(row=row, column=1, sticky="e", pady=4, padx=(10, 0))
            self.results[key] = value

    def calculate(self):
        try:
            face_value = float(self.inputs["face_value"].get())
            coupon_rate = float(self.inputs["coupon_rate"].get()) / 100
            years_to_maturity = float(self.inputs["years_to_maturity"].get())
            yield_rate = float(self.inputs["yield_rate"].get()) / 100
            periods_per_year = int(self.inputs["periods_per_year"].get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers in all fields.")
            return

        if periods_per_year <= 0:
            messagebox.showerror("Invalid Input", "Periods per Year must be a positive integer.")
            return

        args = (face_value, coupon_rate, yield_rate, years_to_maturity, periods_per_year)

        self.results["price"].config(text=f"{bond_price(*args):.4f}")
        self.results["mac_duration"].config(text=f"{macaulay_duration(*args):.4f}")
        self.results["mod_duration"].config(text=f"{modified_duration(*args):.4f}")
        self.results["dv01"].config(text=f"{dv01(*args):.4f}")
        self.results["convexity"].config(text=f"{convexity(*args):.4f}")


def main():
    root = tk.Tk()
    CalculatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
