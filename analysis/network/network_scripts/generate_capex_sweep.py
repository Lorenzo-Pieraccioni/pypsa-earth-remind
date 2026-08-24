"""
Generate CN2060 nuclear CAPEX sweep configs.
One config per CAPEX value. Everything frozen except nuclear investment.
- Cap removed (NUCAP dropped from opts): nuclear is free.
- BIOCAP kept: biomass stays capped, frozen like all else.
- Scenario 95% only (Co2L0.05).
- Override in EUR/MW (bypasses currency conversion, so values are EUR directly).
Baseline 2211 EUR/kW = 2520 USD/kW (Columbia CGEP 2023) at 0.8772.
"""
import shutil, os

BASE = "config.CN2060_admin1_test.yaml"
CAPEX_EUR_PER_KW = [2211, 4000, 6000, 8000, 10000]  # baseline + sweep

with open(BASE) as f:
    base_text = f.read()

for capex in CAPEX_EUR_PER_KW:
    run_name = f"CN2060_nuc{capex}"
    invest_eur_per_mw = capex * 1000  # EUR/kW -> EUR/MW

    text = base_text

    # 1. run name
    import re
    text = re.sub(r"name:\s*CN2060_admin1_test", f"name: {run_name}", text)

    # 2. opts: drop NUCAP, keep BIOCAP, only 95%
    text = re.sub(
        r"opts:\s*\[.*?\]",
        "opts: [BIOCAP-Co2L0.05-3h]",
        text,
    )

    # 3. nuclear CAPEX override (EUR/MW). Append a costs override block.
    #    If a 'costs:' section exists, we add investment under it; else create.
    override_block = (
        "\ncosts:\n"
        "  year: 2060\n"
        "  investment:\n"
        f"    nuclear: {invest_eur_per_mw}  # {capex} EUR/kW sweep point\n"
    )
    # remove any existing simple 'costs:\n  year: 2060' to avoid duplication
    text = re.sub(r"\ncosts:\s*\n\s*year:\s*2060\s*\n", "\n", text)
    text = text + override_block

    out = f"config.{run_name}.yaml"
    with open(out, "w") as f:
        f.write(text)
    print(f"written {out}: run={run_name}, nuclear={capex} EUR/kW ({invest_eur_per_mw} EUR/MW), opts=BIOCAP-Co2L0.05-3h")

print("\nDone. 5 configs generated.")
