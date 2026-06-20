#!/usr/bin/env python3
"""
Ejecuta el resolver del bracket manualmente (idempotente).

Usage:
    python scripts/resolve_bracket.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.session import SessionLocal
from app.services.bracket_resolver import resolve_bracket


def main() -> None:
    print("Resolviendo bracket desde resultados en DB...")
    with SessionLocal() as db:
        res = resolve_bracket(db)
    print(f"  Grupos resueltos   : {res.groups_resolved}/12")
    print(f"  Terceros asignados : {res.thirds_assigned}")
    print(f"  Slots de grupo     : {res.slots_filled}")
    print(f"  Propagación KO     : {res.knockout_propagated}")


if __name__ == "__main__":
    main()
