from amr_agent.api.ratelimit import AskQuota, SlidingWindowLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_window_slides():
    clock = FakeClock()
    limiter = SlidingWindowLimiter(limit=2, window_s=60, clock=clock)
    for _ in range(2):
        assert limiter.retry_after("ip") == 0
        limiter.record("ip")
    assert limiter.retry_after("ip") == 60  # la primera caduca dentro de 60 s
    clock.now += 30
    assert limiter.retry_after("ip") == 30
    clock.now += 30
    assert limiter.retry_after("ip") == 0  # la primera ya salió de la ventana


def test_keys_are_independent():
    limiter = SlidingWindowLimiter(limit=1, window_s=60, clock=FakeClock())
    limiter.record("a")
    assert limiter.retry_after("a") > 0
    assert limiter.retry_after("b") == 0


def test_zero_means_unlimited():
    limiter = SlidingWindowLimiter(limit=0, window_s=60, clock=FakeClock())
    for _ in range(100):
        limiter.record("ip")
    assert limiter.retry_after("ip") == 0


def test_global_budget_is_shared_across_ips():
    quota = AskQuota(per_ip_per_minute=10, per_day=2, clock=FakeClock())
    assert quota.acquire("1.1.1.1") is None
    assert quota.acquire("2.2.2.2") is None
    reason, wait = quota.acquire("3.3.3.3")
    assert "daily" in reason and wait > 0


def test_rejected_requests_do_not_consume_budget():
    clock = FakeClock()
    quota = AskQuota(per_ip_per_minute=1, per_day=2, clock=clock)
    assert quota.acquire("a") is None
    assert quota.acquire("a") is not None  # bloqueada por IP: no gasta del global
    assert quota.acquire("b") is None
