"""Guía de publicación (Fase 3, ítems 9–11): mejor hora, franja nocturna y reintentos.

Todo lo de este módulo es lógica pura, sin acceso a disco ni a red, para
que se pueda probar sin backend:

* La "mejor hora" por plataforma es una GUÍA GENERAL estática, sacada de
  las recomendaciones públicas habituales de cada plataforma. NO son datos
  de la audiencia del usuario: la app no mide audiencia y no debe
  inventarla. La interfaz lo dice así ("guía orientativa").
* La franja nocturna ("procesar de noche") es una ventana horaria
  configurable; fuera de ella la cola de procesamiento espera.
* La política de reintento de publicaciones: un fallo transitorio (red,
  5xx, timeout) se reencola con backoff; un fallo permanente (falta de
  credenciales, cuenta desconectada, archivo perdido) es terminal.
"""

from __future__ import annotations

import re
from datetime import datetime, time, timedelta
from typing import Any

# --- Mejor hora por plataforma (guía general, orientativa) -------------------
# Horas locales (0–23) por tipo de día. Fuente: recomendaciones públicas
# habituales de cada plataforma (franjas de mayor actividad general).
# No son datos de la audiencia del usuario.
BEST_TIMES_GUIDE: dict[str, dict[str, list[int]]] = {
    "youtube": {
        "weekday": [19, 20, 21],
        "weekend": [10, 11, 19],
    },
    "tiktok": {
        "weekday": [12, 18, 19, 21],
        "weekend": [11, 18, 20],
    },
    "instagram": {
        "weekday": [11, 12, 19, 20],
        "weekend": [10, 11, 19],
    },
}

BEST_TIMES_NOTE = (
    "Guía orientativa basada en las franjas de mayor actividad general de "
    "cada plataforma. No son datos de tu audiencia: kellyClipAi no mide "
    "audiencia."
)

PLATFORM_COLORS: dict[str, str] = {
    "youtube": "#c00",
    "tiktok": "#111111",
    "instagram": "#c13584",
}

PLATFORM_LABELS: dict[str, str] = {
    "youtube": "YouTube",
    "tiktok": "TikTok",
    "instagram": "Instagram",
}


def best_time_suggestions(platform: str) -> dict[str, Any]:
    """Guía de mejor hora para una plataforma (o todas si no se reconoce)."""
    guide = BEST_TIMES_GUIDE.get(platform)
    if guide is None:
        return {
            "platforms": BEST_TIMES_GUIDE,
            "note": BEST_TIMES_NOTE,
        }
    return {
        "platform": platform,
        "label": PLATFORM_LABELS.get(platform, platform),
        "weekday_hours": guide["weekday"],
        "weekend_hours": guide["weekend"],
        "note": BEST_TIMES_NOTE,
    }


def next_best_slots(
    platform: str,
    *,
    count: int = 3,
    now: datetime | None = None,
) -> list[str]:
    """Próximas franjas sugeridas como ISO locales (hora local del equipo).

    Busca hacia adelante hasta 8 días: para cada día toma las horas de la
    guía según sea laborable o fin de semana y se queda con las futuras.
    """
    guide = BEST_TIMES_GUIDE.get(platform)
    if guide is None:
        return []
    current = now or datetime.now()  # noqa: DTZ005 - hora local del equipo, a propósito
    slots: list[str] = []
    for day_offset in range(8):
        day = current + timedelta(days=day_offset)
        hours = guide["weekend"] if day.weekday() >= 5 else guide["weekday"]
        for hour in hours:
            slot = day.replace(hour=hour, minute=0, second=0, microsecond=0)
            if slot > current:
                slots.append(slot.isoformat(timespec="minutes"))
            if len(slots) >= count:
                return slots
    return slots


def group_jobs_by_date(jobs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Agrupa trabajos programados por fecha local (YYYY-MM-DD)."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        raw = str(job.get("scheduled_at") or "")
        try:
            day = datetime.fromisoformat(raw.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            day = raw[:10] or "sin-fecha"
        grouped.setdefault(day, []).append(job)
    return grouped


# --- Franja nocturna ("procesar de noche") ------------------------------------
_HHMM_PATTERN = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


def parse_hhmm(value: str) -> time:
    """Valida "HH:MM" (24 h). ValueError si no tiene ese formato."""
    match = _HHMM_PATTERN.match(value.strip())
    if match is None:
        raise ValueError(f"La hora debe tener formato HH:MM (24 h): {value!r}.")
    return time(int(match.group(1)), int(match.group(2)))


def within_window(moment: time, start: time, end: time) -> bool:
    """True si `moment` cae dentro de [start, end), admitiendo ventanas
    que cruzan la medianoche (p. ej. 22:00–06:00). Si start == end, la
    ventana cubre todo el día."""
    if start == end:
        return True
    if start < end:
        return start <= moment < end
    return moment >= start or moment < end


def night_processing_allowed(
    *,
    enabled: bool,
    start: str,
    end: str,
    now: datetime | None = None,
) -> bool:
    """True si la cola puede procesar ahora según la franja nocturna.

    Apagada (por defecto) → siempre permitido. Encendida → solo dentro de
    la franja. Un valor mal formado no bloquea la cola: se registra como
    permitido (fail-open) y la validación del endpoint impide guardarlo.
    """
    if not enabled:
        return True
    try:
        start_time = parse_hhmm(start)
        end_time = parse_hhmm(end)
    except ValueError:
        return True
    current = now or datetime.now()  # noqa: DTZ005 - hora local del equipo, a propósito
    return within_window(current.time(), start_time, end_time)


# --- Reintentos de publicación (por job, con backoff) --------------------------
# El backoff del bucle del worker (scheduler.py) protege al supervisor;
# este protege a cada publicación: 1 intento inicial + 2 reintentos.
SCHEDULE_MAX_ATTEMPTS = 3
SCHEDULE_RETRY_DELAYS_MINUTES = (5, 15, 45)

_PERMANENT_ERROR_PATTERN = re.compile(
    r"credencial|no hay una cuenta|cuenta .* desconectada|sesión .* venc"
    r"|no admite|no (se encontró|está disponible)|ya no está disponible"
    r"|no se pudo .*autoriz|permiso denegado|cuenta .* suspendida",
    re.IGNORECASE,
)


def is_transient_publish_error(error: str | None) -> bool:
    """True si el fallo de publicación merece reintento.

    Ante la duda se reintenta: un 429, un 5xx, un timeout o un corte de
    red se resuelven solos. Solo los fallos permanentes (falta de
    credenciales, cuenta desconectada, archivo perdido, plataforma que
    rechaza el contenido) son terminales.
    """
    if not error:
        return True
    return _PERMANENT_ERROR_PATTERN.search(error) is None


def retry_delay_for_attempt(attempts: int) -> timedelta:
    """Espera antes del siguiente intento (attempts = intentos ya hechos)."""
    index = min(max(attempts - 1, 0), len(SCHEDULE_RETRY_DELAYS_MINUTES) - 1)
    return timedelta(minutes=SCHEDULE_RETRY_DELAYS_MINUTES[index])
