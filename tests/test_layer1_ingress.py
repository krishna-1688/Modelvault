from modelvault.layer1_ingress.velocity_governor import VelocityGovernor


def test_allows_requests_within_capacity():
    gov = VelocityGovernor(requests_per_window=5, window_seconds=10)
    now = 1000.0
    for _ in range(5):
        assert gov.allow("client-a", now=now) is True


def test_blocks_requests_over_capacity():
    gov = VelocityGovernor(requests_per_window=3, window_seconds=10)
    now = 1000.0
    for _ in range(3):
        assert gov.allow("client-a", now=now) is True
    assert gov.allow("client-a", now=now) is False


def test_refills_over_time():
    gov = VelocityGovernor(requests_per_window=2, window_seconds=10)
    now = 1000.0
    assert gov.allow("client-a", now=now) is True
    assert gov.allow("client-a", now=now) is True
    assert gov.allow("client-a", now=now) is False
    # After a full window, tokens should be replenished.
    assert gov.allow("client-a", now=now + 10) is True


def test_clients_are_independent():
    gov = VelocityGovernor(requests_per_window=1, window_seconds=10)
    now = 1000.0
    assert gov.allow("client-a", now=now) is True
    assert gov.allow("client-b", now=now) is True
    assert gov.allow("client-a", now=now) is False
