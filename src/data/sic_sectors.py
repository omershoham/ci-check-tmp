"""SIC code -> (sector, industry_group): the peer sets a broad universe is ranked in.

WHY THIS EXISTS. A percentile rank is only meaningful against a peer set. With the
curated 105 the peer set was the four hand-written sub-sectors; on a universe seeded
from SEC registrants the only free, point-in-time classification available is the SIC
code carried on every filing (`sub.txt`), so it is what the peer sets are built from.

Two levels, because they answer different questions:

    sector          11 GICS-like buckets - the VALUATION peer set. A bank at 11x
                    earnings is not "cheap" next to software at 60x.
    industry_group  ~26 finer buckets - the RANKING peer set, and what `subsector`
                    becomes in broad mode.

THE FAILURE THIS PREVENTS is a 2-digit mapping. SIC's major groups predate software,
biotech and REITs, so the obvious `sic // 100` lookup is wrong for the largest
industries in the market: major group 28 "Chemicals" contains every pharmaceutical
company (2833-2836), 35 "Industrial machinery" contains every computer maker
(3570-3579) and the semiconductor-equipment makers (3559), 38 "Instruments" holds
medical devices (3841-3851) next to test equipment, and 67 "Holding offices" holds
REITs (6798) next to the closed-end funds (6726) the universe filter throws away.
Mapping by major group would file Pfizer under Materials and Dell under Industrials,
and every sector-relative rank downstream would be noise dressed as structure.

The mapping is DATA - an exact-code table checked first, then ordered ranges - so
adding a refinement is a line, not a branch.

Where SIC and GICS genuinely disagree, GICS wins - these are VALUATION peer sets, and
a screen that ranks Walmart's multiple against Home Depot's is answering the wrong
question. Two cases, handled at the two levels they belong to:
  * 5331 variety stores (WMT, TGT, COST, DG) -> Consumer Staples, per GICS's 2023
    reclassification. A whole SIC code moves, so it is a code-level entry.
  * 7370 computer services covers BOTH infrastructure software and the ad-funded
    platforms GICS moved to Communication Services in 2018. No SIC code separates
    them, so those few names are corrected by NAME in SECTOR_OVERRIDES - the one place
    this module is allowed to know a ticker.
"""
from __future__ import annotations

import pandas as pd

ENERGY = "Energy"
MATERIALS = "Materials"
INDUSTRIALS = "Industrials"
DISCRETIONARY = "Consumer Discretionary"
STAPLES = "Consumer Staples"
HEALTH = "Health Care"
FINANCIALS = "Financials"
TECH = "Information Technology"
COMMS = "Communication Services"
UTILITIES = "Utilities"
REAL_ESTATE = "Real Estate"
UNKNOWN = "Unknown"

SECTORS = (TECH, HEALTH, FINANCIALS, REAL_ESTATE, UTILITIES, ENERGY, MATERIALS,
           INDUSTRIALS, DISCRETIONARY, STAPLES, COMMS)

# (sector, industry_group) pairs. Named once so a code maps to both levels at once and
# the two can never disagree.
_ENERGY = (ENERGY, "Energy")
_MATERIALS = (MATERIALS, "Materials")
_CAPGOODS = (INDUSTRIALS, "Capital Goods")
_COMM_PRO = (INDUSTRIALS, "Commercial & Professional Services")
_TRANSPORT = (INDUSTRIALS, "Transportation")
_AUTOS = (DISCRETIONARY, "Automobiles & Components")
_DURABLES = (DISCRETIONARY, "Consumer Durables & Apparel")
_CONS_SVCS = (DISCRETIONARY, "Consumer Services")
_CD_RETAIL = (DISCRETIONARY, "Consumer Discretionary Distribution & Retail")
_CS_RETAIL = (STAPLES, "Consumer Staples Distribution & Retail")
_FOOD = (STAPLES, "Food, Beverage & Tobacco")
_HOUSEHOLD = (STAPLES, "Household & Personal Products")
_HC_EQUIP = (HEALTH, "Health Care Equipment & Services")
_PHARMA = (HEALTH, "Pharmaceuticals, Biotechnology & Life Sciences")
_BANKS = (FINANCIALS, "Banks")
_FIN_SVCS = (FINANCIALS, "Financial Services")
_INSURANCE = (FINANCIALS, "Insurance")
_SOFTWARE = (TECH, "Software & Services")
_HARDWARE = (TECH, "Technology Hardware & Equipment")
_SEMIS = (TECH, "Semiconductors & Semiconductor Equipment")
_TELECOM = (COMMS, "Telecommunication Services")
_MEDIA = (COMMS, "Media & Entertainment")
_UTILITIES = (UTILITIES, "Utilities")
_REITS = (REAL_ESTATE, "Equity REITs")
_RE_MGMT = (REAL_ESTATE, "Real Estate Management & Development")
_UNKNOWN = (UNKNOWN, "Unknown")

INDUSTRY_GROUPS = tuple(dict.fromkeys(g for _, g in (
    _ENERGY, _MATERIALS, _CAPGOODS, _COMM_PRO, _TRANSPORT, _AUTOS, _DURABLES,
    _CONS_SVCS, _CD_RETAIL, _CS_RETAIL, _FOOD, _HOUSEHOLD, _HC_EQUIP, _PHARMA,
    _BANKS, _FIN_SVCS, _INSURANCE, _SOFTWARE, _HARDWARE, _SEMIS, _TELECOM,
    _MEDIA, _UTILITIES, _REITS, _RE_MGMT)))

# Exact 4-digit codes, checked BEFORE the ranges. Each one is a case where the
# surrounding major group means something else entirely.
_EXACT: dict[int, tuple[str, str]] = {
    1311: _ENERGY,          # crude petroleum & natural gas
    1531: _DURABLES,        # operative builders - homebuilding is Consumer Discretionary
    2842: _HOUSEHOLD,       # specialty cleaning preparations
    2844: _HOUSEHOLD,       # perfumes & cosmetics
    3011: _AUTOS,           # tires
    3559: _SEMIS,           # special industry machinery = semiconductor equipment
    3652: _MEDIA,           # prerecorded records & tapes
    3674: _SEMIS,           # semiconductors & related devices
    3711: _AUTOS, 3713: _AUTOS, 3714: _AUTOS, 3715: _AUTOS, 3716: _AUTOS,
    3751: _AUTOS,           # motorcycles & bicycles
    3792: _DURABLES,        # travel trailers & campers
    3812: _HARDWARE,        # search, detection, navigation & guidance electronics
    3826: _HARDWARE,        # laboratory analytical instruments
    3827: _HARDWARE,        # optical instruments & lenses
    4911: _UTILITIES,       # electric services
    4931: _UTILITIES,       # electric & other services combined
    5045: _HARDWARE,        # computers & peripherals wholesale
    5047: _HC_EQUIP,        # medical & hospital equipment wholesale
    5065: _HARDWARE,        # electronic parts & equipment wholesale
    5122: _HC_EQUIP,        # drugs & druggists' sundries wholesale
    5171: _ENERGY, 5172: _ENERGY,   # petroleum wholesale
    5331: _CS_RETAIL,       # variety stores - WMT/TGT/COST/DG, Staples since GICS 2023
    5411: _CS_RETAIL,       # grocery stores
    6324: _HC_EQUIP,        # hospital & medical service plans - managed care, not insurance
    6726: _FIN_SVCS,        # investment offices/CEFs/BDCs - EXCLUDED by universe.broad
    6770: _FIN_SVCS,        # blank checks (SPACs) - likewise
    6792: _ENERGY,          # oil royalty traders
    6795: _MATERIALS,       # mineral royalty traders
    6798: _REITS,           # REITs - kept, unlike their 6726 neighbours
    7375: _MEDIA,           # information retrieval services
    8711: _CAPGOODS,        # engineering services (construction & engineering)
}

# Ordered ranges, inclusive; FIRST match wins, so a refinement is placed above the
# major group it carves out of.
_RANGES: tuple[tuple[int, int, tuple[str, str]], ...] = (
    (100, 299, _FOOD),          # agricultural production
    (700, 799, _FOOD),          # agricultural services
    (800, 899, _MATERIALS),     # forestry
    (900, 999, _FOOD),          # fishing
    (1000, 1099, _MATERIALS),   # metal mining
    (1200, 1299, _ENERGY),      # coal
    (1300, 1399, _ENERGY),      # oil & gas extraction and services
    (1400, 1499, _MATERIALS),   # nonmetallic minerals
    (1500, 1799, _CAPGOODS),    # construction (homebuilders carved out above)
    (2000, 2199, _FOOD),        # food & tobacco
    (2200, 2399, _DURABLES),    # textiles & apparel
    (2400, 2499, _MATERIALS),   # lumber & wood
    (2500, 2599, _DURABLES),    # furniture
    (2600, 2699, _MATERIALS),   # paper
    (2700, 2799, _MEDIA),       # printing & publishing
    (2833, 2836, _PHARMA),      # medicinals, pharma preparations, diagnostics, biologicals
    (2800, 2899, _MATERIALS),   # rest of chemicals
    (2900, 2999, _ENERGY),      # petroleum refining
    (3000, 3099, _MATERIALS),   # rubber & plastics
    (3100, 3199, _DURABLES),    # leather
    (3200, 3399, _MATERIALS),   # stone/clay/glass, primary metals
    (3400, 3499, _CAPGOODS),    # fabricated metal
    (3570, 3579, _HARDWARE),    # computer & office equipment
    (3500, 3599, _CAPGOODS),    # rest of industrial machinery
    (3600, 3629, _CAPGOODS),    # electrical distribution & industrial apparatus
    (3630, 3639, _DURABLES),    # household appliances
    (3640, 3659, _CAPGOODS),    # lighting & wiring
    (3660, 3699, _HARDWARE),    # communications equipment & electronic components
    (3700, 3799, _CAPGOODS),    # aerospace, rail, ships (autos carved out above)
    (3800, 3839, _HARDWARE),    # measuring & controlling instruments
    (3840, 3859, _HC_EQUIP),    # 3841-3851 surgical, orthopedic, dental, x-ray, ophthalmic
    (3860, 3869, _HARDWARE),    # photographic equipment
    (3870, 3999, _DURABLES),    # watches, jewelry, toys, sporting goods
    (4000, 4599, _TRANSPORT),   # rail, trucking, water, air
    (4600, 4699, _ENERGY),      # pipelines
    (4700, 4799, _TRANSPORT),   # transportation services
    (4800, 4829, _TELECOM),
    (4830, 4849, _MEDIA),       # broadcasting & cable
    (4850, 4899, _TELECOM),
    (4900, 4949, _UTILITIES),
    (4950, 4959, _COMM_PRO),    # sanitary services - waste management
    (4960, 4999, _UTILITIES),
    (5000, 5099, _CAPGOODS),    # wholesale durables (trading companies & distributors)
    (5140, 5149, _CS_RETAIL),   # groceries wholesale
    (5100, 5199, _CD_RETAIL),   # rest of wholesale nondurables
    (5812, 5813, _CONS_SVCS),   # eating & drinking places
    (5910, 5919, _CS_RETAIL),   # drug stores
    (5200, 5999, _CD_RETAIL),   # rest of retail
    (6000, 6099, _BANKS),
    (6100, 6299, _FIN_SVCS),    # nondepository credit, brokers
    (6300, 6499, _INSURANCE),
    (6500, 6599, _RE_MGMT),
    (6700, 6799, _FIN_SVCS),    # holding & other investment offices
    (7000, 7099, _CONS_SVCS),   # hotels
    (7200, 7299, _CONS_SVCS),   # personal services
    (7310, 7319, _MEDIA),       # advertising
    (7370, 7379, _SOFTWARE),    # computer programming, software, data processing
    (7300, 7399, _COMM_PRO),    # rest of business services
    (7500, 7699, _CONS_SVCS),   # auto & misc repair
    (7800, 7899, _MEDIA),       # motion pictures
    (7900, 7999, _CONS_SVCS),   # amusement & recreation
    (8000, 8099, _HC_EQUIP),    # health services
    (8100, 8199, _COMM_PRO),    # legal services
    (8200, 8399, _CONS_SVCS),   # education & social services
    (8600, 8699, _CONS_SVCS),   # membership organizations
    (8731, 8734, _PHARMA),      # commercial biological research & testing labs
    (8700, 8799, _COMM_PRO),    # engineering, accounting, management (8711 above)
    (8900, 8999, _COMM_PRO),
    (9000, 9999, _UNKNOWN),     # public administration & non-classifiable (99xx)
)


# Corrections SIC cannot express, applied AFTER the code lookup. Kept deliberately
# tiny: every entry is a company whose SIC is shared with peers that belong somewhere
# else, so no code-level rule can fix it. GICS reclassified these out of Information
# Technology in September 2018.
SECTOR_OVERRIDES: dict[str, tuple[str, str]] = {
    "GOOGL": _MEDIA, "GOOG": _MEDIA,    # SIC 7370, but ad-funded search
    "META": _MEDIA,                     # SIC 7370, but ad-funded social
}


def _code(sic) -> int | None:
    """SIC arrives as a string from sub.txt, a float from parquet, blank when absent."""
    if sic is None or (isinstance(sic, float) and sic != sic):
        return None
    try:
        return int(float(str(sic).strip()))
    except (TypeError, ValueError):
        return None


def classify(sic) -> tuple[str, str]:
    """(sector, industry_group) for one SIC code; ('Unknown', 'Unknown') when unmapped."""
    code = _code(sic)
    if code is None:
        return _UNKNOWN
    hit = _EXACT.get(code)
    if hit:
        return hit
    for lo, hi, pair in _RANGES:
        if lo <= code <= hi:
            return pair
    return _UNKNOWN


def sector(sic) -> str:
    return classify(sic)[0]


def industry_group(sic) -> str:
    return classify(sic)[1]


def classify_frame(sic: pd.Series, symbols: pd.Series | None = None) -> pd.DataFrame:
    """Vectorised `classify` over a SIC column: one lookup per DISTINCT code, not per
    row. The broad universe classifies ~5,000 registrants that share ~400 codes.

    Pass `symbols` to apply SECTOR_OVERRIDES on top — the only way a caller gets the
    name-level corrections, so a frame classified without symbols is SIC-pure."""
    uniq = pd.Index(sic.dropna().unique())
    table = {v: classify(v) for v in uniq}
    out = pd.DataFrame({
        "sector": sic.map(lambda v: table.get(v, _UNKNOWN)[0]).fillna(UNKNOWN),
        "industry_group": sic.map(lambda v: table.get(v, _UNKNOWN)[1]).fillna(UNKNOWN),
    }, index=sic.index)
    if symbols is not None:
        hit = symbols.map(SECTOR_OVERRIDES)
        out.loc[hit.notna(), "sector"] = hit[hit.notna()].map(lambda p: p[0])
        out.loc[hit.notna(), "industry_group"] = hit[hit.notna()].map(lambda p: p[1])
    return out
