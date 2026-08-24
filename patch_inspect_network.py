"""
patch_inspect_network.py
Applica 5 modifiche a inspect_network.py:
  1. Aggiunge PYPSA_TO_DISPLAY_CAP per aggregare capacità
  2. Aggiorna load_irena con mapping display names
  3. Carica sempre IRENA per capacità (non come fallback Ember)
  4. Aggiunge sezione 6 al report testuale (confronto vs IRENA e Ember)
  5. Aggiorna Plot 1 per usare IRENA invece di Ember per le capacità
"""

TARGET = "analysis/network/inspect_network.py"

with open(TARGET, "r") as f:
    src = f.read()

changes = [

    # ── 1. PYPSA_TO_DISPLAY_CAP ───────────────────────────────────────────────
    (
        'def get_color(carrier):',
        '''PYPSA_TO_DISPLAY_CAP = {
    "solar":      "Solar",
    "onwind":     "Onshore Wind",
    "offwind-ac": "Offshore Wind",
    "offwind-dc": "Offshore Wind",
    "hydro":      "Hydro",
    "ror":        "Hydro",
    "PHS":        "PHS",
    "nuclear":    "Nuclear",
    "coal":       "Coal",
    "lignite":    "Coal",
    "CCGT":       "Gas",
    "OCGT":       "Gas",
    "oil":        "Oil",
}

def get_color(carrier):'''
    ),

    # ── 2. load_irena: IRENA_TO_EMBER → IRENA_TO_DISPLAY ─────────────────────
    (
        """    IRENA_TO_EMBER = {
        'Solar photovoltaic':   'Solar',
        'Onshore wind energy':  'Wind',
        'Offshore wind energy': 'Wind',
        'Renewable hydropower': 'Hydro',
        'Mixed hydropower':     'Hydro',
        'Nuclear energy':       'Nuclear',
        'Coal':                 'Coal',
        'Natural gas':          'Gas',
        'Oil':                  'Other Fossil',
    }
    result = {}
    for tech, ember_name in IRENA_TO_EMBER.items():
        val = df[df['Technology'] == tech]['Value_MW'].sum()
        if val > 0:
            result[ember_name] = result.get(ember_name, 0) + val / 1e3
    return pd.Series(result)""",
        """    IRENA_TO_DISPLAY = {
        'Solar photovoltaic':   'Solar',
        'Onshore wind energy':  'Onshore Wind',
        'Offshore wind energy': 'Offshore Wind',
        'Renewable hydropower': 'Hydro',
        'Mixed hydropower':     'Hydro',
        'Pumped Storage':       'PHS',
        'Nuclear energy':       'Nuclear',
        'Coal':                 'Coal',
        'Natural gas':          'Gas',
        'Oil':                  'Oil',
    }
    result = {}
    for tech, display_name in IRENA_TO_DISPLAY.items():
        val = df[df['Technology'] == tech]['Value_MW'].sum()
        if val > 0:
            result[display_name] = result.get(display_name, 0) + val / 1e3
    return pd.Series(result)"""
    ),

    # ── 3. Loading logic: IRENA sempre per capacità ───────────────────────────
    (
        """    cap_ember, gen_ember = None, None
    if ember_year:
        cap_ember, gen_ember = load_ember(EMBER_FILE, ember_year)
        if cap_ember is not None and cap_ember.empty:
            cap_ember = load_irena(IRENA_FILE, ember_year)
            if cap_ember is not None:
                log(f"  Capacity loaded from IRENA for year {ember_year}")
        if gen_ember is not None:
            log(f"  Ember data loaded for year {ember_year}")""",
        """    irena_cap = None
    gen_ember = None
    if ember_year:
        _, gen_ember = load_ember(EMBER_FILE, ember_year)
        irena_cap = load_irena(IRENA_FILE, ember_year)
        if irena_cap is not None:
            log(f"  Capacity loaded from IRENA for year {ember_year}")
        if gen_ember is not None:
            log(f"  Ember data loaded for year {ember_year}")"""
    ),

    # ── 4. Sezione 6: confronto numerico ─────────────────────────────────────
    (
        '    report_path = os.path.join(BASE_OUTPUT_DIR, f"{network_name}_inspect.txt")',
        '''    # 6. Comparison vs reference data
    log("")
    log("=" * 60)
    log("6. COMPARISON VS REFERENCE DATA")
    log("=" * 60)
    if irena_cap is not None:
        model_cap_display = aggregate_to_ember(all_cap, PYPSA_TO_DISPLAY_CAP)
        log(f"  Capacity (GW) — model vs IRENA {ember_year}")
        log(f"  {'Carrier':<20} {'Model':>10} {'IRENA':>10} {'Error':>8}")
        log("  " + "-" * 52)
        for c in sorted(set(model_cap_display.index) | set(irena_cap.index)):
            m = model_cap_display.get(c, 0.0)
            r = irena_cap.get(c, 0.0)
            if r > 0:
                err = (m - r) / r * 100
                log(f"  {c:<20} {m:>10.1f} {r:>10.1f} {err:>+8.1f}%")
            elif m > 0:
                log(f"  {c:<20} {m:>10.1f} {'n/a':>10} {'n/a':>8}")
    if gen_ember is not None:
        model_gen_agg2 = aggregate_to_ember(all_gen, PYPSA_TO_EMBER)
        log("")
        log(f"  Generation (TWh) — model vs Ember {ember_year}")
        log(f"  {'Carrier':<20} {'Model':>10} {'Ember':>10} {'Error':>8}")
        log("  " + "-" * 52)
        for c in sorted(set(model_gen_agg2.index) | set(gen_ember.index)):
            m = model_gen_agg2.get(c, 0.0)
            r = gen_ember.get(c, 0.0)
            if r > 0:
                err = (m - r) / r * 100
                log(f"  {c:<20} {m:>10.1f} {r:>10.1f} {err:>+8.1f}%")
            elif m > 0:
                log(f"  {c:<20} {m:>10.1f} {'n/a':>10} {'n/a':>8}")

    report_path = os.path.join(BASE_OUTPUT_DIR, f"{network_name}_inspect.txt")'''
    ),

    # ── 5. Plot 1: usa IRENA invece di cap_ember ──────────────────────────────
    (
        """    model_cap_agg = aggregate_to_ember(all_cap, PYPSA_TO_EMBER)
    model_gen_agg = aggregate_to_ember(all_gen, PYPSA_TO_EMBER)

    # ── PLOT 1: Capacità modello vs Ember ─────────────────────────────────────
    carriers_cap = sorted(set(model_cap_agg.index) | (set(cap_ember.index) if cap_ember is not None else set()))
    x = np.arange(len(carriers_cap))
    width = 0.35
    mv = [model_cap_agg.get(c, 0) for c in carriers_cap]
    ev = [cap_ember.get(c, 0) if cap_ember is not None else 0 for c in carriers_cap]""",
        """    model_cap_agg = aggregate_to_ember(all_cap, PYPSA_TO_DISPLAY_CAP)
    model_gen_agg = aggregate_to_ember(all_gen, PYPSA_TO_EMBER)

    # ── PLOT 1: Capacità modello vs IRENA ─────────────────────────────────────
    irena_label = f"IRENA {ember_year}" if ember_year else "IRENA (n/a)"
    carriers_cap = sorted(set(model_cap_agg.index) | (set(irena_cap.index) if irena_cap is not None else set()))
    x = np.arange(len(carriers_cap))
    width = 0.35
    mv = [model_cap_agg.get(c, 0) for c in carriers_cap]
    ev = [irena_cap.get(c, 0) if irena_cap is not None else 0 for c in carriers_cap]"""
    ),

    # ── 5b. Plot 1: label IRENA nella legenda ─────────────────────────────────
    (
        '               edgecolor="black", linewidth=0.8, label=ember_label, hatch="///")',
        '               edgecolor="black", linewidth=0.8, label=irena_label, hatch="///")'
    ),

    # ── 5c. Plot 1: titolo con IRENA ──────────────────────────────────────────
    (
        '    ax_top.set_title(f"Installed capacity: model vs {ember_label}\\n{network_name}",',
        '    ax_top.set_title(f"Installed capacity: model vs {irena_label}\\n{network_name}",'
    ),
]

for old, new in changes:
    if old not in src:
        print(f"[ERROR] Pattern not found:\n{old[:80]}...")
        raise SystemExit(1)
    src = src.replace(old, new, 1)
    print(f"[OK] Applied: {old[:60].strip()[:60]}...")

with open(TARGET, "w") as f:
    f.write(src)

print(f"\nDone. {len(changes)} changes applied to {TARGET}.")
