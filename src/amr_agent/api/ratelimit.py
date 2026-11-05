"""Límite de peticiones en memoria para la demo pública.

Por qué: `/ask` gasta ~3.000 tokens de Groq y la cuota gratuita es de 200.000 tokens al día
(compartida con las evals). Sin límite, un solo visitante con un bucle agotaría la demo.

Dos límites:
- por IP (p. ej. 5 por minuto): que nadie acapare la demo;
- global (p. ej. 50 al día): un techo que protege la cuota aunque lleguen muchas IPs.

Ventana deslizante: se guardan las marcas de tiempo de las últimas peticiones de cada clave.
Vive en memoria porque Render Free ejecuta una sola instancia; al reiniciar se pone a cero,
lo cual es aceptable para una demo. Con varias instancias haría falta Redis o la base de datos.
"""

import time
from collections import deque
from collections.abc import Callable


class SlidingWindowLimiter:
    def __init__(
        self, limit: int, window_s: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}

    def retry_after(self, key: str) -> float:
        """Segundos que hay que esperar (0 si la petición cabe). No registra nada."""
        if self.limit <= 0:  # 0 = sin límite
            return 0.0
        now = self._clock()
        hits = self._hits.get(key)
        if hits is None:
            return 0.0
        while hits and hits[0] <= now - self.window_s:  # descarta las que ya caducaron
            hits.popleft()
        if len(hits) < self.limit:
            return 0.0
        return hits[0] + self.window_s - now

    def record(self, key: str) -> None:
        if self.limit <= 0:
            return
        if len(self._hits) > 10_000:  # evita que crezca sin fin con muchas IPs distintas
            self._hits = {k: v for k, v in self._hits.items() if v}
        self._hits.setdefault(key, deque()).append(self._clock())


class AskQuota:
    """Los dos límites de `/ask` juntos. `acquire` solo registra si caben ambos."""

    def __init__(
        self, per_ip_per_minute: int, per_day: int, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.per_ip = SlidingWindowLimiter(per_ip_per_minute, 60, clock)
        self.global_ = SlidingWindowLimiter(per_day, 24 * 3600, clock)

    def acquire(self, ip: str) -> tuple[str, float] | None:
        """None si se permite; si no, (motivo, segundos de espera)."""
        if wait := self.per_ip.retry_after(ip):
            return "Too many questions from your address, please wait a minute.", wait
        if wait := self.global_.retry_after("all"):
            return "The demo's daily question budget is used up, please come back tomorrow.", wait
        self.per_ip.record(ip)
        self.global_.record("all")
        return None
