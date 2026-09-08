"""
AI-Powered Consulting Platform — Streamlit Web App
Sectors: Real Estate | Corporate Finance (REAL data via yfinance) | Insurance
Shared engine: Monte Carlo simulation + Risk-Adjusted NPV

Install first:
    pip install streamlit numpy matplotlib yfinance

Run:
    streamlit run app.py
"""

import hashlib
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

np.random.seed(42)

st.set_page_config(page_title="AI Consulting Platform", layout="wide")

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

    def verdict(self):
        p = self.stats()["prob_negative"]
        if p < 10:
            return "STRONG BUY", "green"
        elif p < 25:
            return "FAVORABLE", "green"
        elif p < 45:
            return "PROCEED WITH CAUTION", "orange"
        return "HIGH RISK", "red"


def monte_carlo_npv(cashflow_generator, discount_rate, years, n_sim, label):
    npvs = np.zeros(n_sim)
    discount_factors = np.array([(1 + discount_rate) ** -t for t in range(1, years + 1)])
    for i in range(n_sim):
        cashflows = np.array(cashflow_generator())
        npvs[i] = np.sum(cashflows * discount_factors)
    return SimulationResult(npvs, label)


def render_result_metrics(result: SimulationResult):
    s = result.stats()
    verdict, color = result.verdict()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mean NPV", f"{s['mean']:,.0f}")
    c2.metric("Pessimistic (P10)", f"{s['p10']:,.0f}")
    c3.metric("Median (P50)", f"{s['p50']:,.0f}")
    c4.metric("Optimistic (P90)", f"{s['p90']:,.0f}")
    c5.metric("Prob. of loss", f"{s['prob_negative']:.1f}%")
    st.markdown(f"### Verdict: :{color}[{verdict}]")


def render_histogram(result: SimulationResult):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(result.npv_samples, bins=50, color="#3B82F6", edgecolor="white", alpha=0.85)
    s = result.stats()
    ax.axvline(s["p50"], color="black", linestyle="--", linewidth=1.5, label=f"Median: {s['p50']:,.0f}")
    ax.axvline(0, color="red", linestyle=":", linewidth=1, label="Break-even")
    ax.set_title(f"NPV distribution — {result.label}")
    ax.set_xlabel("Net Present Value")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)


def render_ranking_chart(results: dict, title):
    names = list(results.keys())
    means = [results[n].stats()["mean"] for n in names]
    risks = [results[n].stats()["prob_negative"] for n in names]
    order = np.argsort(means)[::-1]
    names_sorted = [names[i] for i in order]
    means_sorted = [means[i] for i in order]
    risk_sorted = [risks[i] for i in order]
    bar_colors = ["#16A34A" if r < 20 else "#F59E0B" if r < 40 else "#DC2626" for r in risk_sorted]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names_sorted, means_sorted, color=bar_colors)
    ax.set_title(f"{title} — ranked by expected NPV")
    ax.set_xlabel("Mean NPV")
    ax.invert_yaxis()
    st.pyplot(fig)


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


def render_real_estate():
    st.header("Real estate — multi-segment city analysis")
    col1, col2 = st.columns(2)
    country = col1.text_input("Country", "Lebanon")
    city = col2.text_input("City / village", "Baabda")
    col3, col4 = st.columns(2)
    years = col3.number_input("Forecast horizon (years)", min_value=1, max_value=20, value=5)
    size_sqm = col4.number_input("Property size (sqm)", min_value=10.0, value=150.0)
    override = st.text_input("Known real price per sqm (leave blank to auto-estimate)", "")

    if st.button("Run real estate analysis", type="primary"):
        profile = derive_city_profile(city, country)
        base_price = float(override) if override else profile["base_price_per_sqm"]
        source = "REAL (user-provided)" if override else "MODEL ESTIMATE — no live data feed connected"
        st.info(f"City: **{city}, {country}**  |  Base price/sqm: **{base_price:,.0f}**  |  Source: **{source}**")

        results = {}
        for seg_name, seg in REAL_ESTATE_SEGMENTS.items():
            seg_price = base_price * seg["price_multiplier"]
            inputs = {
                "price_per_sqm": seg_price, "size_sqm": size_sqm, "rental_yield": seg["rental_yield"],
                "growth_low": seg["growth_low"], "growth_mid": seg["growth_mid"], "growth_high": seg["growth_high"],
                "macro_adj": profile["macro_growth_adj"], "volatility": seg["volatility"] * profile["volatility_mult"],
                "years": years,
            }
            results[seg_name] = monte_carlo_npv(real_estate_cashflow_factory(inputs), 0.08, years, 3000, seg_name)

        best_name = max(results.items(), key=lambda kv: kv[1].stats()["mean"])[0]
        st.success(f"Best expected option: **{best_name}**")

        render_ranking_chart(results, f"Real estate — {city}, {country}")

        tabs = st.tabs(list(results.keys()))
        for tab, (name, result) in zip(tabs, results.items()):
            with tab:
                render_result_metrics(result)
                render_histogram(result)


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


def render_corporate_finance():
    st.header("Corporate finance — earnings & investment forecast")

    use_real = False
    inputs_base = None
    company_label = "Sample Co."

    if YFINANCE_AVAILABLE:
        ticker = st.text_input("Stock ticker for REAL data (e.g. AAPL, MSFT, TSLA)", "")
        if ticker and st.button("Fetch real financial data"):
            with st.spinner("Fetching from Yahoo Finance..."):
                real_data = fetch_real_company_data(ticker.upper())
            if real_data:
                st.session_state["real_data"] = real_data
                st.session_state["company_label"] = ticker.upper()
                st.success(f"Loaded real data for {ticker.upper()}")
                st.write(f"Revenue: {real_data['current_revenue']:,.0f} | "
                         f"Avg growth: {real_data['revenue_growth']*100:.1f}% | "
                         f"Avg margin: {real_data['net_margin']*100:.1f}%")
            else:
                st.error("Could not fetch data for that ticker.")
    else:
        st.warning("yfinance not installed — run: pip install yfinance")

    if "real_data" in st.session_state:
        use_real = True
        inputs_base = st.session_state["real_data"]
        company_label = st.session_state["company_label"]

    st.divider()
    st.subheader("Or enter manually")
    if not use_real:
        company_label = st.text_input("Company name", "Sample Co.")
        current_revenue = st.number_input("Current annual revenue", min_value=0.0, value=1_000_000.0)
        net_margin = st.number_input("Current net margin (e.g. 0.1 for 10%)", min_value=0.0, max_value=1.0, value=0.10)
        growth_rate = st.number_input("Expected annual revenue growth", min_value=-1.0, max_value=2.0, value=0.05)
        inputs_base = {"current_revenue": current_revenue, "net_margin": net_margin,
                        "revenue_growth": growth_rate, "growth_volatility": 0.03}

    investment_amount = st.number_input("Planned investment amount", min_value=0.0, value=100_000.0)

    if st.button("Run corporate finance analysis", type="primary"):
        inputs = {**inputs_base, "investment_amount": investment_amount, "years": 5}
        result = monte_carlo_npv(corporate_finance_cashflow_factory(inputs), 0.10, 5, 5000,
                                  f"Corporate Finance: {company_label}")
        render_result_metrics(result)
        render_histogram(result)


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


def render_insurance():
    st.header("Insurance — AI claims triage business case")
    col1, col2 = st.columns(2)
    annual_claims = col1.number_input("Annual number of claims", min_value=0.0, value=50000.0)
    cost_per_claim = col2.number_input("Current avg processing cost per claim", min_value=0.0, value=40.0)
    col3, col4 = st.columns(2)
    avg_claim_value = col3.number_input("Average claim value", min_value=0.0, value=2000.0)
    implementation_cost = col4.number_input("Implementation cost", min_value=0.0, value=300000.0)

    if st.button("Run insurance analysis", type="primary"):
        inputs = {
            "annual_claims": annual_claims, "cost_per_claim": cost_per_claim, "avg_claim_value": avg_claim_value,
            "implementation_cost": implementation_cost, "maintenance_cost": implementation_cost * 0.1,
            "target_adoption": 0.6, "efficiency_gain": 0.25, "max_fraud_reduction": 0.15, "years": 5,
        }
        result = monte_carlo_npv(insurance_cashflow_factory(inputs), 0.09, 5, 5000, "Insurance: AI Claims Triage")
        render_result_metrics(result)
        render_histogram(result)


# ============================================================
# MAIN APP LAYOUT
# ============================================================

def main():
    st.sidebar.title("AI Consulting Platform")
    st.sidebar.caption("Risk-adjusted decision engine")
    sector = st.sidebar.radio("Choose a sector", ["Real Estate", "Corporate Finance", "Insurance"])

    st.title("AI Consulting Platform")
    st.caption("Monte Carlo simulation + risk-adjusted NPV across three sectors")

    if sector == "Real Estate":
        render_real_estate()
    elif sector == "Corporate Finance":
        render_corporate_finance()
    elif sector == "Insurance":
        render_insurance()


if __name__ == "__main__":
    main()