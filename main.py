"""
AI-Powered Consulting Platform — Final Version
Sectors: Real Estate | Corporate Finance (REAL data via yfinance) | Insurance
Shared engine: Monte Carlo simulation + Risk-Adjusted NPV
No PDF — results shown directly in terminal + rich matplotlib dashboards.

Install first:
    pip install numpy matplotlib yfinance

Run:
    python main.py
"""

import hashlib
import numpy as np
import matplotlib.pyplot as plt

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

np.random.seed(42)


# ============================================================
# SHARED RISK ENGINE
# ============================================================

class SimulationResult:
    def __init__(self, npv_samples, label):
        self.npv_samples = npv_samples
        self.label = label

    def stats(self):
        p10, p50, p90 = np.percentile(self.npv_samples, [10, 50, 90])
        mean = np.mean(self.npv_samples)
        prob_negative = np.mean(self.npv_samples < 0) * 100
        return {"mean": mean, "p10": p10, "p50": p50, "p90": p90, "prob_negative": prob_negative}

    def print_summary(self):
        s = self.stats()
        print(f"\n{'='*60}")
        print(f"  RESULT: {self.label}")
        print(f"{'='*60}")
        print(f"  Mean NPV (expected value):   {s['mean']:>15,.0f}")
        print(f"  Pessimistic case (P10):      {s['p10']:>15,.0f}")
        print(f"  Median case (P50):           {s['p50']:>15,.0f}")
        print(f"  Optimistic case (P90):       {s['p90']:>15,.0f}")
        print(f"  Probability of loss:         {s['prob_negative']:>14.1f}%")
        verdict = "STRONG BUY" if s["prob_negative"] < 10 else \
                  "FAVORABLE" if s["prob_negative"] < 25 else \
                  "PROCEED WITH CAUTION" if s["prob_negative"] < 45 else "HIGH RISK"
        print(f"  Verdict:                     {verdict:>15s}")
        print(f"{'='*60}")


def monte_carlo_npv(cashflow_generator, discount_rate, years, n_sim, label):
    npvs = np.zeros(n_sim)
    discount_factors = np.array([(1 + discount_rate) ** -t for t in range(1, years + 1)])
    for i in range(n_sim):
        cashflows = np.array(cashflow_generator())
        npvs[i] = np.sum(cashflows * discount_factors)
    return SimulationResult(npvs, label)


def show_dashboard(results: dict, main_title):
    """Rich multi-panel matplotlib dashboard: histogram + ranking bar chart."""
    n = len(results)
    fig, axes = plt.subplots(1, n + 1, figsize=(6 * (n + 1), 5))
    if n == 1:
        axes = [axes[0], axes[1]] if isinstance(axes, np.ndarray) else axes

    colors = plt.cm.tab10(np.linspace(0, 1, n))
    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx] if n > 1 else axes[0]
        ax.hist(result.npv_samples, bins=50, color=colors[idx], edgecolor="white", alpha=0.85)
        s = result.stats()
        ax.axvline(s["p50"], color="black", linestyle="--", linewidth=1.5, label=f"Median: {s['p50']:,.0f}")
        ax.axvline(0, color="red", linestyle=":", linewidth=1, label="Break-even")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("NPV")
        ax.legend(fontsize=8)

    ranking_ax = axes[-1]
    names = list(results.keys())
    means = [results[n_].stats()["mean"] for n_ in names]
    risks = [results[n_].stats()["prob_negative"] for n_ in names]
    order = np.argsort(means)[::-1]
    names_sorted = [names[i] for i in order]
    means_sorted = [means[i] for i in order]
    risk_sorted = [risks[i] for i in order]
    bar_colors = ["#16A34A" if r < 20 else "#F59E0B" if r < 40 else "#DC2626" for r in risk_sorted]
    ranking_ax.barh(names_sorted, means_sorted, color=bar_colors)
    ranking_ax.set_title(f"{main_title}\nRanked by expected NPV (green=low risk, red=high risk)",
                          fontsize=11, fontweight="bold")
    ranking_ax.set_xlabel("Mean NPV")
    ranking_ax.invert_yaxis()

    plt.tight_layout()
    plt.show()


# ============================================================
# SECTOR 1: REAL ESTATE
# ============================================================

REAL_ESTATE_SEGMENTS = {
    "Residential": {"price_multiplier": 1.00, "rental_yield": 0.055, "growth_low": 0.01, "growth_mid": 0.035, "growth_high": 0.07, "volatility": 0.03},
    "Commercial": {"price_multiplier": 1.35, "rental_yield": 0.075, "growth_low": 0.00, "growth_mid": 0.03, "growth_high": 0.06, "volatility": 0.05},
    "Land": {"price_multiplier": 0.55, "rental_yield": 0.0, "growth_low": -0.02, "growth_mid": 0.05, "growth_high": 0.12, "volatility": 0.08},
    "Luxury": {"price_multiplier": 2.10, "rental_yield": 0.04, "growth_low": -0.01, "growth_mid": 0.045, "growth_high": 0.10, "volatility": 0.07},
}


def derive_city_profile(city, country):
    seed_str = f"{city.strip().lower()}|{country.strip().lower()}"
    h = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16)
    rng = np.random.default_rng(h % (2**32))
    return {
        "base_price_per_sqm": rng.uniform(400, 4500),
        "macro_growth_adj": rng.uniform(-0.01, 0.02),
        "volatility_mult": rng.uniform(0.8, 1.4),
    }


def real_estate_cashflow_factory(inputs):
    def generate():
        price_per_sqm = np.random.normal(inputs["price_per_sqm"], inputs["price_per_sqm"] * 0.10)
        purchase_price = price_per_sqm * inputs["size_sqm"]
        rental_yield = max(np.random.normal(inputs["rental_yield"], 0.01), 0)
        cashflows, value = [], purchase_price
        for year in range(inputs["years"]):
            growth = np.random.triangular(inputs["growth_low"], inputs["growth_mid"], inputs["growth_high"])
            growth += inputs["macro_adj"] + np.random.normal(0, inputs["volatility"])
            rental_income = value * rental_yield
            value *= (1 + growth)
            cf = rental_income
            if year == inputs["years"] - 1:
                cf += value
            if year == 0:
                cf -= purchase_price
            cashflows.append(cf)
        return cashflows
    return generate


def run_real_estate_module():
    print("\n" + "#" * 60)
    print("#  REAL ESTATE — MULTI-SEGMENT CITY ANALYSIS")
    print("#" * 60)
    country = input("Country: ").strip() or "Sample Country"
    city = input("City / village: ").strip() or "Sample City"
    try:
        years = int(input("Forecast horizon (years): ") or 5)
        size_sqm = float(input("Property size (sqm): ") or 150)
    except ValueError:
        years, size_sqm = 5, 150
    override = input("Known real price per sqm (Enter to auto-estimate): ").strip()
    profile = derive_city_profile(city, country)
    base_price = float(override) if override else profile["base_price_per_sqm"]
    source = "REAL (user-provided)" if override else "MODEL ESTIMATE (no live data feed connected)"

    print(f"\nCity: {city}, {country}  |  Base price/sqm: {base_price:,.0f}  |  Source: {source}")

    results = {}
    for seg_name, seg in REAL_ESTATE_SEGMENTS.items():
        seg_price = base_price * seg["price_multiplier"]
        inputs = {
            "price_per_sqm": seg_price, "size_sqm": size_sqm, "rental_yield": seg["rental_yield"],
            "growth_low": seg["growth_low"], "growth_mid": seg["growth_mid"], "growth_high": seg["growth_high"],
            "macro_adj": profile["macro_growth_adj"], "volatility": seg["volatility"] * profile["volatility_mult"],
            "years": years,
        }
        result = monte_carlo_npv(real_estate_cashflow_factory(inputs), 0.08, years, 3000, seg_name)
        result.print_summary()
        results[seg_name] = result

    best = max(results.items(), key=lambda kv: kv[1].stats()["mean"])
    print(f"\n>>> BEST OPTION: {best[0]} in {city}, {country} <<<\n")
    show_dashboard(results, f"Real Estate — {city}, {country}")


# ============================================================
# SECTOR 2: CORPORATE FINANCE — REAL data via yfinance
# ============================================================

def fetch_real_company_data(ticker):
    stock = yf.Ticker(ticker)
    fin = stock.financials
    if fin is None or fin.empty or "Total Revenue" not in fin.index:
        return None
    try:
        revenues = fin.loc["Total Revenue"].sort_index().values.astype(float)
        net_income = fin.loc["Net Income"].sort_index().values.astype(float)
        if len(revenues) < 2:
            return None
        growth_rates = np.diff(revenues) / revenues[:-1]
        margins = net_income / revenues
        return {
            "current_revenue": float(revenues[-1]),
            "revenue_growth": float(np.mean(growth_rates)),
            "growth_volatility": max(float(np.std(growth_rates)), 0.01),
            "net_margin": max(float(np.mean(margins)), 0.0),
        }
    except Exception:
        return None


def corporate_finance_cashflow_factory(inputs):
    def generate():
        growth_rate = np.random.normal(inputs["revenue_growth"], inputs["growth_volatility"])
        margin = max(np.random.normal(inputs["net_margin"], 0.02), 0)
        revenue = inputs["current_revenue"]
        cashflows = []
        for year in range(inputs["years"]):
            revenue *= (1 + growth_rate)
            earnings = revenue * margin
            cf = earnings - (inputs["investment_amount"] if year == 0 else 0)
            cashflows.append(cf)
        return cashflows
    return generate


def run_corporate_finance_module():
    print("\n" + "#" * 60)
    print("#  CORPORATE FINANCE — EARNINGS & INVESTMENT FORECAST")
    print("#" * 60)
    use_real = False
    company_label = "Sample Co."

    if YFINANCE_AVAILABLE:
        ticker = input("Stock ticker for REAL data (e.g. AAPL, MSFT, TSLA): ").strip().upper()
        if ticker:
            print("Fetching real financial data from Yahoo Finance...")
            real_data = fetch_real_company_data(ticker)
            if real_data:
                use_real = True
                company_label = ticker
                inputs_base = real_data
                print(f"  [REAL DATA] Revenue: {real_data['current_revenue']:,.0f}")
                print(f"  [REAL DATA] Avg historical growth: {real_data['revenue_growth']*100:.1f}%")
                print(f"  [REAL DATA] Avg net margin: {real_data['net_margin']*100:.1f}%")
            else:
                print("Could not fetch data for that ticker — falling back to manual input.")
    else:
        print("(yfinance not installed — run: pip install yfinance)")

    if not use_real:
        company_label = input("Company name: ") or "Sample Co."
        try:
            current_revenue = float(input("Current annual revenue: ") or 1_000_000)
            net_margin = float(input("Current net margin (e.g. 0.1): ") or 0.10)
            growth_rate = float(input("Expected annual revenue growth (e.g. 0.05): ") or 0.05)
        except ValueError:
            current_revenue, net_margin, growth_rate = 1_000_000, 0.10, 0.05
        inputs_base = {"current_revenue": current_revenue, "net_margin": net_margin,
                        "revenue_growth": growth_rate, "growth_volatility": 0.03}

    try:
        investment_amount = float(input("Planned investment amount: ") or 100_000)
    except ValueError:
        investment_amount = 100_000

    inputs = {**inputs_base, "investment_amount": investment_amount, "years": 5}
    result = monte_carlo_npv(corporate_finance_cashflow_factory(inputs), 0.10, inputs["years"], 5000,
                              f"Corporate Finance: {company_label}")
    result.print_summary()
    show_dashboard({company_label: result}, f"Investment case — {company_label}")


# ============================================================
# SECTOR 3: INSURANCE
# ============================================================

def insurance_cashflow_factory(inputs):
    def generate():
        adoption_rate = np.random.triangular(0.3, inputs["target_adoption"], 0.95)
        efficiency_gain = np.random.normal(inputs["efficiency_gain"], 0.05)
        fraud_reduction = np.random.beta(2, 5) * inputs["max_fraud_reduction"]
        cashflows = []
        for year in range(inputs["years"]):
            ramp = min(1.0, adoption_rate * (year + 1) / inputs["years"])
            labor_savings = inputs["annual_claims"] * inputs["cost_per_claim"] * efficiency_gain * ramp
            fraud_savings = inputs["annual_claims"] * inputs["avg_claim_value"] * fraud_reduction * ramp * 0.01
            cf = labor_savings + fraud_savings
            cf -= inputs["implementation_cost"] if year == 0 else inputs["maintenance_cost"]
            cashflows.append(cf)
        return cashflows
    return generate


def run_insurance_module():
    print("\n" + "#" * 60)
    print("#  INSURANCE — AI CLAIMS TRIAGE BUSINESS CASE")
    print("#" * 60)
    try:
        annual_claims = float(input("Annual number of claims: ") or 50000)
        cost_per_claim = float(input("Current avg processing cost per claim: ") or 40)
        avg_claim_value = float(input("Average claim value: ") or 2000)
        implementation_cost = float(input("Implementation cost: ") or 300000)
    except ValueError:
        annual_claims, cost_per_claim, avg_claim_value, implementation_cost = 50000, 40, 2000, 300000

    inputs = {
        "annual_claims": annual_claims, "cost_per_claim": cost_per_claim, "avg_claim_value": avg_claim_value,
        "implementation_cost": implementation_cost, "maintenance_cost": implementation_cost * 0.1,
        "target_adoption": 0.6, "efficiency_gain": 0.25, "max_fraud_reduction": 0.15, "years": 5,
    }
    result = monte_carlo_npv(insurance_cashflow_factory(inputs), 0.09, inputs["years"], 5000,
                              "Insurance: AI Claims Triage")
    result.print_summary()
    show_dashboard({"AI Claims Triage": result}, "Insurance — AI Claims Triage")


# ============================================================
# MAIN MENU
# ============================================================

def main():
    print("=" * 60)
    print("   AI CONSULTING PLATFORM — RISK-ADJUSTED DECISION ENGINE")
    print("=" * 60)
    print("  1. Real Estate — multi-segment city analysis")
    print("  2. Corporate Finance — REAL data via stock ticker")
    print("  3. Insurance — AI claims triage business case")
    print("  0. Exit")

    while True:
        choice = input("\nSelect a sector (0-3): ").strip()
        if choice == "1":
            run_real_estate_module()
        elif choice == "2":
            run_corporate_finance_module()
        elif choice == "3":
            run_insurance_module()
        elif choice == "0":
            print("Exiting platform.")
            break
        else:
            print("Invalid choice, try again.")


if __name__ == "__main__":
    main()