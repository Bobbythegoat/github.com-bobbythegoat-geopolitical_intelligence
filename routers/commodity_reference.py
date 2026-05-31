"""
routers/commodity_reference.py
-------------------------------
AI commodity & natural-resource REFERENCE endpoints.

Complements routers/commodities.py (the live signal/tracker engine): this router
surfaces the sourced reference catalog — what to track, where the TRUE source
is, how to query it (HS/HTS codes), why it matters for AI, plain-English
explanations, provenance rules — plus best-effort live reference prices with
explicit source attribution.

Mounted under /commodity-reference to avoid colliding with the /commodities
signal router's /{symbol} catch-all route.
"""

from fastapi import APIRouter, HTTPException

from services import commodities as cx

router = APIRouter(prefix="/commodity-reference", tags=["Commodities"])


@router.get("/", summary="List AI-relevant commodities (reference)")
def list_commodities():
    """Catalog index: each commodity with its AI relevance, starter HS codes and
    authoritative sources."""
    items = cx.list_commodities()
    return {"items": items, "count": len(items)}


@router.get("/reference", summary="Sources, metric catalog and provenance rules")
def get_reference():
    """The full reference layer: official source map, metric catalog grouped by
    category, the minimum provenance fields to record, and the reliability note."""
    return cx.get_reference()


@router.get("/sources", summary="Official source map")
def get_sources():
    """The authoritative ('true source') data providers, with what each gives you
    and the link to go straight to it."""
    return {"sources": [{"key": k, **v} for k, v in cx.SOURCE_MAP.items()]}


@router.get("/prices", summary="Best-effort live reference prices (provenance-tagged)")
def get_prices():
    """Delayed/indicative futures prices for commodities that have an exchange
    proxy. Every row names its true authoritative source and what to verify
    against — these are convenience reads, not the official record."""
    prices = cx.get_live_prices()
    return {"prices": prices, "count": len(prices), "reliability_note": cx.RELIABILITY_NOTE}


@router.get("/{key}", summary="Commodity reference detail")
def get_commodity(key: str):
    """Full detail for one commodity: AI relevance, plain-English explanation,
    every starter HS/HTS code (with where to confirm it), and the authoritative
    sources to pull data from."""
    detail = cx.get_commodity(key.lower())
    if not detail:
        raise HTTPException(status_code=404, detail=f"Unknown commodity '{key}'.")
    return detail
