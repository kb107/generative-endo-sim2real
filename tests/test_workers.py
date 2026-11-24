from generative_sim2real.data import workers


def test_worker_cleanup_closes_resource_sharer_first(monkeypatch) -> None:
    calls = []

    def fake_finalize(obj, callback, *, exitpriority):
        calls.append((obj, callback, exitpriority))

    monkeypatch.setattr(workers.multiprocessing_util, "Finalize", fake_finalize)
    workers.configure_worker_cleanup(0)

    assert calls == [(None, workers.resource_sharer.stop, 10)]
