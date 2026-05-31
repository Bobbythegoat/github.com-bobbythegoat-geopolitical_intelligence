"""
services/commodities.py
------------------------
AI-relevant commodity & natural-resource intelligence reference.

This module encodes the reference sheet "AI Commodities & Natural Resource Trade
Metrics" (created 2026-05-30) into a structured, queryable catalog so the system
can tell the user, for any tracked commodity:

  • WHAT to track          (the metrics that actually move supply/price)
  • WHERE to find it        (the authoritative / true-source data provider)
  • HOW to track it         (the exact query: HS/HTS code, flow, field)
  • WHY it matters for AI    (datacenter power, chips, batteries, grid)
  • A plain-English explanation the user can read
  • PROVENANCE / reliability so every number is traceable to its true source

Design rules (from the reference sheet):
  - Import/export tracking depends on the EXACT HS/HTS/Schedule B code.
    Codes here are STARTER codes — always confirm the product form first.
  - No portfolio advice, no buy/sell signals — this is a sourcing/observability
    layer, not a recommender.
  - Trade data gets revised: record source + download date every time.

Live prices are best-effort convenience reads (CME/COMEX futures via Yahoo
Finance). They are clearly labelled with their true authoritative source and a
"verify against official settlements" note — they are NOT a substitute for the
official source.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Official source map (reference sheet — "Official Source Map")
# tier: official_gov | multilateral | exchange | price_reporting
# ---------------------------------------------------------------------------

SOURCE_MAP: Dict[str, Dict] = {
    "un_comtrade": {
        "name": "UN Comtrade",
        "tier": "multilateral",
        "what_it_gives": "Monthly/annual global import-export data by reporter, partner, trade flow, HS code, value and quantity.",
        "use_for": "Global trade-flow checks across countries; mirror-data discrepancy checks.",
        "url": "https://comtradeplus.un.org/TradeFlow",
    },
    "usitc_dataweb": {
        "name": "USITC DataWeb",
        "tier": "official_gov",
        "what_it_gives": "Official U.S. import/export statistics, duties, trade-partner detail, HTS categories and customs-district fields.",
        "use_for": "U.S. trade flow by commodity, country and customs district.",
        "url": "https://dataweb.usitc.gov",
    },
    "census_foreign_trade": {
        "name": "U.S. Census Foreign Trade / Schedule B",
        "tier": "official_gov",
        "what_it_gives": "Export codes, trade data, port/customs fields, Schedule B lookup.",
        "use_for": "Confirming U.S. export classifications and port-level trade.",
        "url": "https://www.census.gov/foreign-trade",
    },
    "usgs_mcs": {
        "name": "USGS Mineral Commodity Summaries",
        "tier": "official_gov",
        "what_it_gives": "Annual production, reserves, import reliance, country supply, tariffs and market notes for 90+ minerals.",
        "use_for": "Baseline supply, reserves and U.S. import dependence.",
        "url": "https://www.usgs.gov/centers/national-minerals-information-center/mineral-commodity-summaries",
    },
    "iea_critical_minerals": {
        "name": "IEA Critical Minerals Data Explorer",
        "tier": "multilateral",
        "what_it_gives": "Critical-mineral demand/supply projections and scenario data.",
        "use_for": "Forward supply-demand pressure for copper, lithium, nickel, cobalt, graphite, rare earths, gallium, germanium.",
        "url": "https://www.iea.org/data-and-statistics/data-tools/critical-minerals-data-explorer",
    },
    "eia_natural_gas": {
        "name": "EIA Natural Gas Data",
        "tier": "official_gov",
        "what_it_gives": "U.S. natural-gas imports, exports, LNG exports, storage, prices, pipeline movements and country-level flows.",
        "use_for": "Natural gas and LNG import/export tracking; Henry Hub; storage.",
        "url": "https://www.eia.gov/naturalgas/",
    },
    "doe_fecm_lng": {
        "name": "DOE/FECM LNG Monthly Reports",
        "tier": "official_gov",
        "what_it_gives": "Natural-gas import/export authorizations and LNG shipment detail.",
        "use_for": "LNG cargo/destination checks and permit context.",
        "url": "https://www.energy.gov/fecm/regulation",
    },
    "lme": {
        "name": "LME Market Data",
        "tier": "exchange",
        "what_it_gives": "Official prices, warehouse stocks, warrant data, cancelled warrants and stock reports for base metals.",
        "use_for": "Metal price and visible inventory stress.",
        "url": "https://www.lme.com/en/market-data",
    },
    "cme": {
        "name": "CME Group",
        "tier": "exchange",
        "what_it_gives": "Futures settlements, volume, open interest, term structure and contract specs.",
        "use_for": "Copper, aluminum, natural gas, uranium-related market pricing where available.",
        "url": "https://www.cmegroup.com/markets.html",
    },
    "worldbank_pinksheet": {
        "name": "World Bank Pink Sheet",
        "tier": "multilateral",
        "what_it_gives": "Monthly global commodity price series and historical data.",
        "use_for": "Clean monthly price history for broad commodity tracking.",
        "url": "https://www.worldbank.org/en/research/commodity-markets",
    },
    "usitc_tariff": {
        "name": "USITC Tariff Database / HTS Search",
        "tier": "official_gov",
        "what_it_gives": "Exact HTS codes, duty columns, Section 301 / antidumping context.",
        "use_for": "Confirming the exact HTS code and duty before pulling trade data.",
        "url": "https://hts.usitc.gov/",
    },
    "bis": {
        "name": "U.S. BIS (Bureau of Industry and Security)",
        "tier": "official_gov",
        "what_it_gives": "Export controls, Entity List, dual-use licensing for minerals/semiconductor inputs.",
        "use_for": "Export-control and entity-restriction checks.",
        "url": "https://www.bis.gov",
    },
    "ofac": {
        "name": "OFAC Sanctions Lists",
        "tier": "official_gov",
        "what_it_gives": "Sanctioned entities, owners, vessels and countries.",
        "use_for": "Checking whether a supplier/mine/shipper/buyer is sanctioned.",
        "url": "https://ofac.treasury.gov/sanctions-programs-and-country-information",
    },
    "price_reporting": {
        "name": "Fastmarkets / Platts / CRU / Argus / Asian Metal",
        "tier": "price_reporting",
        "what_it_gives": "Physical premiums and assessed prices for metals, lithium products, rare earths, graphite.",
        "use_for": "Regional physical premiums above exchange reference price (subscription).",
        "url": "https://www.fastmarkets.com",
    },
}


# ---------------------------------------------------------------------------
# Metric catalog by category (reference sheet sections 1–5)
# Each metric: what to track, where (source keys), how to track.
# ---------------------------------------------------------------------------

METRIC_CATEGORIES: List[Dict] = [
    {
        "category": "Trade Flow",
        "intro": "Physical and dollar flows of a commodity between countries. The single most important rule: every query is only as good as the exact HS/HTS/Schedule B code.",
        "metrics": [
            {"metric": "Import volume", "what": "Physical quantity imported by commodity and country of origin.", "where": ["un_comtrade", "usitc_dataweb", "census_foreign_trade"], "how": "Query by HS/HTS code, reporter/importer, partner/exporter, period and quantity unit."},
            {"metric": "Export volume", "what": "Physical quantity exported by commodity and destination country.", "where": ["un_comtrade", "usitc_dataweb", "census_foreign_trade"], "how": "Use export flow, Schedule B/HS code, destination partner, quantity and value."},
            {"metric": "Net imports", "what": "Imports minus exports for the same commodity and period.", "where": ["un_comtrade", "usitc_dataweb"], "how": "Pull imports and exports on the same code and period; compare quantity, not just value."},
            {"metric": "Unit import price", "what": "Import value divided by import quantity.", "where": ["un_comtrade", "usitc_dataweb"], "how": "Use value and quantity from the same query; watch for unit changes or mixed product categories."},
            {"metric": "Supplier concentration", "what": "Share of imports from the top 1/3/5 countries.", "where": ["un_comtrade", "usitc_dataweb"], "how": "Rank partner countries by import quantity/value for each commodity code."},
            {"metric": "Country-of-origin exposure", "what": "Whether physical origin is concentrated in one risky supplier.", "where": ["usitc_dataweb", "census_foreign_trade", "un_comtrade"], "how": "Filter imports by origin/partner country and compare share over time."},
            {"metric": "Port/customs district concentration", "what": "Whether imports enter through a narrow set of ports/districts.", "where": ["usitc_dataweb", "census_foreign_trade"], "how": "Pull import quantity/value by customs district or port for the same HTS code."},
            {"metric": "Re-export / transshipment signal", "what": "Goods routed through intermediaries rather than direct origin.", "where": ["un_comtrade"], "how": "Compare exporter-reported exports vs importer-reported imports for the same HS code and period (mirror data)."},
        ],
    },
    {
        "category": "Physical Supply",
        "intro": "Where the material actually comes from and who controls conversion. The common mistake is confusing mining country with refining/processing country — the bottleneck is usually in processing.",
        "metrics": [
            {"metric": "Mine production by country", "what": "Annual production of raw mineral supply.", "where": ["usgs_mcs", "iea_critical_minerals"], "how": "Find the commodity chapter and record production by country each year."},
            {"metric": "Reserves by country", "what": "Estimated economically recoverable reserves.", "where": ["usgs_mcs", "iea_critical_minerals"], "how": "Use the reserves table; compare reserve location with processing location."},
            {"metric": "Refining/processing country share", "what": "Who controls conversion from ore/concentrate into usable material.", "where": ["iea_critical_minerals", "usgs_mcs"], "how": "Separate mining supply from processing/refining capacity — this is where the real chokepoint hides."},
            {"metric": "U.S. net import reliance", "what": "Share of U.S. apparent consumption met by imports.", "where": ["usgs_mcs"], "how": "Use the import-reliance line in each USGS commodity summary."},
            {"metric": "Secondary/recycled supply", "what": "How much supply comes from scrap or recycling.", "where": ["usgs_mcs"], "how": "Track recycled supply share and scrap import/export flows when listed."},
            {"metric": "Project pipeline", "what": "New mines, refineries, smelters, LNG terminals, processing plants.", "where": ["iea_critical_minerals", "usgs_mcs"], "how": "Record project name, location, capacity, expected start date, permitting status and delays."},
            {"metric": "Concentrate vs refined product flow", "what": "Whether a country imports raw feedstock or refined material.", "where": ["un_comtrade", "usitc_dataweb"], "how": "Track the ore/concentrate code separately from the unwrought/refined code."},
        ],
    },
    {
        "category": "Market Stress",
        "intro": "Price and inventory signals that reveal whether near-term supply is tight or loose.",
        "metrics": [
            {"metric": "Spot/reference price", "what": "Daily or monthly price of the commodity.", "where": ["lme", "cme", "worldbank_pinksheet", "eia_natural_gas"], "how": "Record price by commodity and date; use the same source consistently."},
            {"metric": "Futures curve", "what": "Nearby price vs later-dated contracts.", "where": ["cme", "lme"], "how": "Compare front-month to 3-, 6- and 12-month contracts."},
            {"metric": "Backwardation/contango", "what": "Whether near-term supply is tight or loose.", "where": ["cme", "lme"], "how": "Nearby above later = backwardation (tight); nearby below later = contango (loose)."},
            {"metric": "Exchange inventories", "what": "Visible stocks held in exchange warehouses.", "where": ["lme", "cme"], "how": "Track total stocks, cancelled warrants and month-over-month stock changes."},
            {"metric": "Cancelled warrants / warehouse queues", "what": "Metal already requested for withdrawal from LME warehouses.", "where": ["lme"], "how": "Monitor cancelled warrants and queue data for copper, aluminum, nickel, zinc, tin."},
            {"metric": "Open interest & volume", "what": "Outstanding contracts and futures activity.", "where": ["cme", "lme"], "how": "Watch volume/OI spikes around policy news, export controls or inventory drawdowns."},
            {"metric": "Physical premium", "what": "Price paid above exchange reference for physical delivery.", "where": ["price_reporting"], "how": "Track regional premiums for copper cathode, aluminum, lithium products, rare earths, graphite."},
        ],
    },
    {
        "category": "Energy / LNG",
        "intro": "Natural gas, LNG and power-relevant fuels — the energy that actually runs AI datacenters.",
        "metrics": [
            {"metric": "LNG export volume", "what": "Monthly U.S. LNG exports in volume terms.", "where": ["eia_natural_gas", "doe_fecm_lng"], "how": "Track monthly LNG exports and compare destination countries."},
            {"metric": "Natural gas imports/exports", "what": "Pipeline and LNG movements by country.", "where": ["eia_natural_gas"], "how": "Pull monthly import/export volumes by country and mode."},
            {"metric": "Natural gas storage", "what": "Working gas in underground storage.", "where": ["eia_natural_gas"], "how": "Track current storage vs same week last year and the five-year range."},
            {"metric": "Henry Hub price", "what": "U.S. natural-gas benchmark price.", "where": ["eia_natural_gas", "cme"], "how": "Track spot/futures prices and term structure."},
            {"metric": "LNG export capacity", "what": "Operating and under-construction LNG export capacity.", "where": ["eia_natural_gas", "doe_fecm_lng"], "how": "Record terminal capacity, start date, permits and expansion phases."},
            {"metric": "Uranium supply/import reliance", "what": "Uranium production, imports and dependency for nuclear fuel.", "where": ["eia_natural_gas", "usgs_mcs"], "how": "Track uranium purchases, origin countries and contracted vs spot exposure."},
        ],
    },
    {
        "category": "Policy / Chokepoint",
        "intro": "The political layer: export controls, tariffs, sanctions and permitting that can reprice supply overnight.",
        "metrics": [
            {"metric": "Export controls", "what": "Restrictions on exporting minerals, metals, graphite, semiconductor inputs or processing tech.", "where": ["bis"], "how": "Search the commodity name plus 'export control', 'license', 'restriction' or 'dual-use'."},
            {"metric": "Tariffs and duties", "what": "Import duties, Section 301, antidumping/countervailing duties and exemptions.", "where": ["usitc_tariff"], "how": "Check the exact HTS code and duty column; track changes by effective date."},
            {"metric": "Sanctions / entity restrictions", "what": "Whether suppliers, mines, shippers or buyers are restricted.", "where": ["ofac", "bis"], "how": "Search company names, owners, project names and countries."},
            {"metric": "Permitting status", "what": "Whether mines, refineries, pipelines, LNG terminals or power plants are delayed.", "where": ["doe_fecm_lng", "usgs_mcs"], "how": "Record approval stage, expected decision date, litigation and revised start date."},
            {"metric": "Strategic stockpiles", "what": "Government purchases, releases or reserve-building.", "where": ["usgs_mcs"], "how": "Search the commodity name plus 'stockpile', 'strategic reserve' or 'procurement'."},
            {"metric": "Trade mirror discrepancy", "what": "Mismatch between exporter- and importer-reported trade.", "where": ["un_comtrade"], "how": "Compare reporter A export-to-B vs reporter B import-from-A for the same HS code and period."},
        ],
    },
]


# ---------------------------------------------------------------------------
# Commodity catalog (reference sheet section 6 + AI relevance)
# HS codes are STARTER codes — confirm exact form before querying.
# live_ticker: best-effort CME/COMEX futures symbol via Yahoo Finance, or None.
# ---------------------------------------------------------------------------

COMMODITIES: List[Dict] = [
    {
        "key": "copper",
        "name": "Copper",
        "group": "metal",
        "why_ai": "The metal of electrification: grid buildout, datacenter power distribution, busbars and cabling. AI power demand is structurally copper-intensive.",
        "explanation": "Track mined concentrate separately from refined cathode — the refining bottleneck (heavily concentrated in a few countries) matters more than mining. Watch LME cancelled warrants and the futures curve for tightness.",
        "hs_codes": [
            {"form": "Ore/concentrates", "code": "HS 2603", "confirm_at": "usitc_tariff", "note": "Use separately from refined copper."},
            {"form": "Refined/unwrought copper", "code": "HS 7403", "confirm_at": "usitc_tariff", "note": "Cathode/refined supply flows."},
            {"form": "Copper wire", "code": "HS 7408", "confirm_at": "usitc_tariff", "note": "Grid / datacenter electrical buildout."},
        ],
        "primary_sources": ["usgs_mcs", "iea_critical_minerals", "lme", "un_comtrade"],
        "live_ticker": {"symbol": "HG=F", "source": "cme", "unit": "USD/lb (COMEX front-month)", "verify_at": "lme"},
    },
    {
        "key": "aluminum",
        "name": "Aluminum",
        "group": "metal",
        "why_ai": "Datacenter construction, heat sinks, transmission lines and structural framing. Energy-intensive to smelt, so power policy feeds straight into supply.",
        "explanation": "Track primary (unwrought) aluminum separately from fabricated products, and watch alumina (the feedstock). Smelting is power-hungry, so energy prices and curtailments drive supply.",
        "hs_codes": [
            {"form": "Unwrought aluminum", "code": "HS 7601", "confirm_at": "usitc_tariff", "note": "Primary imports, separate from fabricated."},
            {"form": "Alumina / aluminum oxide", "code": "HS 2818", "confirm_at": "un_comtrade", "note": "Feedstock for aluminum production."},
        ],
        "primary_sources": ["usgs_mcs", "lme", "un_comtrade"],
        "live_ticker": {"symbol": "ALI=F", "source": "cme", "unit": "USD/tonne (CME front-month)", "verify_at": "lme"},
    },
    {
        "key": "tin",
        "name": "Tin",
        "group": "metal",
        "why_ai": "Solder is the connective tissue of all electronics — chips, boards, servers. Small market, so supply shocks bite hard.",
        "explanation": "Tin matters for solder/electronics. A thin, concentrated supply chain makes it sensitive to export controls and mine disruptions.",
        "hs_codes": [
            {"form": "Tin ores", "code": "HS 2609", "confirm_at": "un_comtrade", "note": "Mined feedstock."},
            {"form": "Unwrought tin", "code": "HS 8001", "confirm_at": "usitc_tariff", "note": "Refined tin."},
        ],
        "primary_sources": ["usgs_mcs", "lme", "un_comtrade"],
        "live_ticker": None,
    },
    {
        "key": "natural_graphite",
        "name": "Natural graphite",
        "group": "mineral",
        "why_ai": "Battery anodes (energy storage for datacenters / grid) and industrial uses. Processing is highly concentrated and subject to export controls.",
        "explanation": "Separate natural from artificial graphite — they trade under different codes. Processing/spherical-graphite capacity is the chokepoint, not raw mining.",
        "hs_codes": [
            {"form": "Natural graphite", "code": "HS 2504", "confirm_at": "un_comtrade", "note": "Separate from artificial graphite."},
        ],
        "primary_sources": ["usgs_mcs", "iea_critical_minerals", "un_comtrade"],
        "live_ticker": None,
    },
    {
        "key": "artificial_graphite",
        "name": "Artificial graphite",
        "group": "mineral",
        "why_ai": "Battery and high-temperature industrial applications; substitute/complement to natural graphite in anodes.",
        "explanation": "Tracked separately from natural graphite. Useful when natural-graphite supply is constrained by export policy.",
        "hs_codes": [
            {"form": "Artificial graphite", "code": "HS 3801", "confirm_at": "un_comtrade", "note": "Battery and industrial applications."},
        ],
        "primary_sources": ["un_comtrade", "usitc_tariff"],
        "live_ticker": None,
    },
    {
        "key": "lithium",
        "name": "Lithium (carbonate / hydroxide)",
        "group": "mineral",
        "why_ai": "Battery cathodes for grid-scale and backup storage supporting AI power loads. Hydroxide vs carbonate suit different cell chemistries.",
        "explanation": "Carbonate and hydroxide are tracked under different codes and serve different battery chemistries. Watch the physical premium and refining-country share, not just spodumene mining.",
        "hs_codes": [
            {"form": "Lithium carbonate", "code": "HS 2836.91", "confirm_at": "usitc_tariff", "note": "Often tracked separately from hydroxide."},
            {"form": "Lithium hydroxide", "code": "HS 2825.20", "confirm_at": "usitc_tariff", "note": "Important for high-nickel battery chemistries."},
        ],
        "primary_sources": ["usgs_mcs", "iea_critical_minerals", "price_reporting"],
        "live_ticker": None,
    },
    {
        "key": "nickel",
        "name": "Nickel",
        "group": "metal",
        "why_ai": "High-energy-density battery cathodes and stainless steel for infrastructure. Class 1 (battery-grade) vs Class 2 is a key distinction.",
        "explanation": "Separate mined feedstock from refined nickel. Battery-grade (Class 1) supply is tighter than headline production suggests.",
        "hs_codes": [
            {"form": "Nickel ore", "code": "HS 2604", "confirm_at": "un_comtrade", "note": "Mined feedstock."},
            {"form": "Unwrought nickel", "code": "HS 7502", "confirm_at": "usitc_tariff", "note": "Refined nickel."},
        ],
        "primary_sources": ["usgs_mcs", "iea_critical_minerals", "lme"],
        "live_ticker": None,
    },
    {
        "key": "cobalt",
        "name": "Cobalt",
        "group": "metal",
        "why_ai": "Battery cathode stabilizer; supply is geographically concentrated (DRC mining, China refining), a classic chokepoint.",
        "explanation": "The exact code depends on product form (ore vs oxide vs metal). Refining concentration matters more than mining location.",
        "hs_codes": [
            {"form": "Cobalt ore", "code": "HS 2605", "confirm_at": "un_comtrade", "note": "Mined feedstock."},
            {"form": "Cobalt materials", "code": "HS 8105 / 2822 area", "confirm_at": "usitc_tariff", "note": "Exact code depends on product form."},
        ],
        "primary_sources": ["usgs_mcs", "iea_critical_minerals"],
        "live_ticker": None,
    },
    {
        "key": "rare_earths",
        "name": "Rare earth elements",
        "group": "mineral",
        "why_ai": "Permanent magnets for motors, cooling fans, hard drives and robotics; defense overlap. Processing is dominated by a single country.",
        "explanation": "Processing country matters far more than mining country. Track permanent-magnet imports (HS 8505), not just the raw oxides — that is where the real dependence shows up.",
        "hs_codes": [
            {"form": "RE metals / scandium / yttrium", "code": "HS 2805.30", "confirm_at": "usitc_tariff", "note": "Processing country matters more than mining."},
            {"form": "RE compounds", "code": "HS 2846.90 area", "confirm_at": "usitc_tariff", "note": "Use the exact chemical/form code."},
            {"form": "Permanent magnets", "code": "HS 8505.11 / 8505.19", "confirm_at": "usitc_tariff", "note": "Track magnet imports, not just raw materials."},
        ],
        "primary_sources": ["usgs_mcs", "iea_critical_minerals", "bis"],
        "live_ticker": None,
    },
    {
        "key": "gallium_germanium",
        "name": "Gallium & Germanium",
        "group": "mineral",
        "why_ai": "Compound semiconductors, high-speed/RF chips, fiber optics and advanced photonics. Subject to active export controls.",
        "explanation": "Codes vary by purity and product form — do NOT rely on one blanket code. These are prime export-control targets, so pair trade data with BIS/MOFCOM notices.",
        "hs_codes": [
            {"form": "Gallium / germanium", "code": "HTS 8112.92 / 8112.99 area", "confirm_at": "usitc_tariff", "note": "Codes vary by purity/product form; verify exact form."},
        ],
        "primary_sources": ["usgs_mcs", "bis", "un_comtrade"],
        "live_ticker": None,
    },
    {
        "key": "lng",
        "name": "LNG (liquefied natural gas)",
        "group": "energy",
        "why_ai": "Exported gas that sets global power-fuel pricing; AI datacenter electricity demand is increasingly gas-backed.",
        "explanation": "Use EIA/DOE for cleaner U.S. LNG detail than generic trade data. Track export volume, destination mix and export-capacity build-out (permits + start dates).",
        "hs_codes": [
            {"form": "LNG", "code": "HS 2711.11", "confirm_at": "eia_natural_gas", "note": "Use EIA/DOE for cleaner U.S. LNG detail."},
        ],
        "primary_sources": ["eia_natural_gas", "doe_fecm_lng", "un_comtrade"],
        "live_ticker": None,
    },
    {
        "key": "natural_gas",
        "name": "Natural gas (Henry Hub)",
        "group": "energy",
        "why_ai": "The marginal fuel powering AI datacenters in much of the U.S.; Henry Hub is the benchmark.",
        "explanation": "Track pipeline gas separately from LNG. Watch storage vs the five-year range and the futures term structure for power-cost pressure.",
        "hs_codes": [
            {"form": "Natural gas, gaseous", "code": "HS 2711.21", "confirm_at": "eia_natural_gas", "note": "Track pipeline gas separately from LNG."},
        ],
        "primary_sources": ["eia_natural_gas", "cme"],
        "live_ticker": {"symbol": "NG=F", "source": "cme", "unit": "USD/MMBtu (Henry Hub front-month)", "verify_at": "eia_natural_gas"},
    },
    {
        "key": "uranium",
        "name": "Uranium",
        "group": "energy",
        "why_ai": "Nuclear fuel for the baseload power increasingly contracted to run AI datacenters (SMRs, restarts, PPAs).",
        "explanation": "Exact code depends on product/isotope/form. Use EIA uranium marketing reports for purchases, origin countries and contracted-vs-spot exposure.",
        "hs_codes": [
            {"form": "Uranium ores", "code": "HS 2612", "confirm_at": "usitc_tariff", "note": "Mined feedstock."},
            {"form": "Uranium compounds", "code": "HS 2844 area", "confirm_at": "usitc_tariff", "note": "Depends on product and isotope/form."},
        ],
        "primary_sources": ["eia_natural_gas", "usgs_mcs"],
        "live_ticker": None,
    },
]

COMMODITY_INDEX = {c["key"]: c for c in COMMODITIES}


# ---------------------------------------------------------------------------
# Provenance / reliability (reference sheet section 7 — minimum fields)
# ---------------------------------------------------------------------------

PROVENANCE_FIELDS: List[Dict] = [
    {"field": "Commodity name + exact HS/HTS/Schedule B code", "why": "Prevents mixing ore, concentrate, refined metal, compounds and finished goods."},
    {"field": "Reporter / importer country", "why": "Shows who is dependent on supply."},
    {"field": "Partner / exporter / origin country", "why": "Shows who controls the flow."},
    {"field": "Trade flow (import / export / re-export)", "why": "Prevents double-counting and misreading transshipment."},
    {"field": "Quantity, unit and value", "why": "Quantity shows physical flow; value captures price/mix changes."},
    {"field": "Period and frequency", "why": "Monthly data catches shocks faster; annual data is cleaner but slower."},
    {"field": "Source and download date", "why": "Trade data gets revised — you need to know which version you used."},
]

RELIABILITY_NOTE = (
    "Reliability rule: import/export tracking depends on the exact "
    "HS/HTS/Schedule B code, and trade data is revised over time. Always confirm "
    "the product form against an official classification tool (USITC HTS / UN "
    "Comtrade) before relying on a query, and record the source + download date. "
    "Live prices shown here are best-effort exchange futures via Yahoo Finance and "
    "must be verified against the official exchange settlement (LME/CME) or EIA."
)


# ---------------------------------------------------------------------------
# Catalog assembly helpers
# ---------------------------------------------------------------------------

def _expand_sources(keys: List[str]) -> List[Dict]:
    out = []
    for k in keys:
        s = SOURCE_MAP.get(k)
        if s:
            out.append({"key": k, **s})
    return out


def get_commodity(key: str) -> Optional[Dict]:
    """Return the full, source-expanded detail for one commodity (or None)."""
    c = COMMODITY_INDEX.get(key)
    if not c:
        return None
    detail = dict(c)
    detail["sources"] = _expand_sources(c.get("primary_sources", []))
    return detail


def list_commodities() -> List[Dict]:
    """Lightweight list view for the catalog index."""
    return [
        {
            "key": c["key"],
            "name": c["name"],
            "group": c["group"],
            "why_ai": c["why_ai"],
            "hs_codes": [h["code"] for h in c["hs_codes"]],
            "primary_sources": [SOURCE_MAP[k]["name"] for k in c["primary_sources"] if k in SOURCE_MAP],
            "has_live_price": c.get("live_ticker") is not None,
        }
        for c in COMMODITIES
    ]


def get_reference() -> Dict:
    """Full reference payload: sources, metric catalog and provenance."""
    return {
        "source_map": [{"key": k, **v} for k, v in SOURCE_MAP.items()],
        "metric_categories": METRIC_CATEGORIES,
        "provenance_fields": PROVENANCE_FIELDS,
        "reliability_note": RELIABILITY_NOTE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Best-effort live prices (clearly provenance-tagged)
# ---------------------------------------------------------------------------

_PRICE_CACHE: Dict[str, tuple] = {}   # key -> (timestamp, payload)
_PRICE_TTL = 15 * 60                   # 15 minutes


def get_live_prices() -> List[Dict]:
    """Best-effort live reference prices for commodities that have a futures
    proxy. Each row is explicitly labelled with its TRUE authoritative source
    and a verify-against-official note. Degrades gracefully if yfinance fails.
    """
    import time as _t
    now = _t.time()

    results: List[Dict] = []
    for c in COMMODITIES:
        lt = c.get("live_ticker")
        if not lt:
            continue

        sym = lt["symbol"]
        cached = _PRICE_CACHE.get(sym)
        if cached and (now - cached[0]) < _PRICE_TTL:
            results.append(cached[1])
            continue

        row = {
            "key":          c["key"],
            "name":         c["name"],
            "symbol":       sym,
            "unit":         lt["unit"],
            "price":        None,
            "change_pct":   None,
            "as_of":        None,
            # Provenance: this is convenience data, NOT the authoritative source.
            "data_source":      "Yahoo Finance (delayed futures)",
            "true_source":      SOURCE_MAP[lt["source"]]["name"],
            "true_source_url":  SOURCE_MAP[lt["source"]]["url"],
            "verify_against":   SOURCE_MAP[lt["verify_at"]]["name"],
            "verify_url":       SOURCE_MAP[lt["verify_at"]]["url"],
            "note":             "Delayed/indicative. Verify against the official exchange settlement before use.",
        }

        try:
            import yfinance as yf
            hist = yf.Ticker(sym).history(period="5d", interval="1d")
            if hist is not None and not hist.empty and len(hist) >= 1:
                closes = hist["Close"].tolist()
                last = float(closes[-1])
                prev = float(closes[-2]) if len(closes) >= 2 else last
                row["price"] = round(last, 4)
                row["change_pct"] = round((last - prev) / prev * 100, 2) if prev else None
                row["as_of"] = str(hist.index[-1])[:10]
        except Exception:
            row["note"] = "Live price unavailable right now — use the official source link."

        _PRICE_CACHE[sym] = (now, row)
        results.append(row)

    return results
