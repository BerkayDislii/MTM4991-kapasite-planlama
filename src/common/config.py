"""Proje kökü ve YAML yapılandırma okuma yardımcıları."""
from pathlib import Path
from typing import Any

import yaml

# src/common/config.py -> proje kökü iki üst dizin
PROJE_KOKU: Path = Path(__file__).resolve().parents[2]


def yaml_oku(ad: str) -> dict[str, Any]:
    """configs/<ad>.yaml dosyasını sözlük olarak okur."""
    with open(PROJE_KOKU / "configs" / f"{ad}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def yol(goreli: str | Path) -> Path:
    """Proje köküne göreli yolu mutlak yola çevirir."""
    return PROJE_KOKU / goreli
