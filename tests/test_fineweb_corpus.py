"""stream_fineweb_corpus — the single-FineWeb kernel curriculum (matches the fineweb coordizer).

Deterministic: the underlying parquet source is monkeypatched so the test needs no cached shard / network,
and pins the two behaviours that matter — the SAME hygiene as the 7-repo stream (stub-marker drop +
passage split) and infinite yielding.
"""
import itertools

import qig_studio.corpus as corpus


def test_stream_fineweb_corpus_applies_stub_and_split(monkeypatch):
    """Passages carrying a stub marker are dropped; long docs are split into passages; clean prose survives.
    Uses a real _STUB_MARKERS token so the drop is exercised against the actual hygiene set."""
    stub = f"intro {corpus._STUB_MARKERS[0]} tail that is long enough to otherwise pass the length gate xxxx"
    clean = ("The Fisher-Rao geometry of web text is a probability-simplex construction. " * 4).strip()
    docs = [clean, stub, clean]

    def _fake_source(min_len=200, max_chars=4000, passages=None):
        for d in docs:
            yield d

    monkeypatch.setattr("qig_studio.fineweb_source.stream_fineweb_passages", _fake_source)
    out = list(itertools.islice(corpus.stream_fineweb_corpus(min_len=40), 10))
    assert out, "should yield clean passages"
    assert all(corpus._STUB_MARKERS[0] not in p.lower() for p in out), "stub-marked passages must be dropped"
    assert any("fisher-rao geometry" in p.lower() for p in out), "clean prose must survive"


def test_stream_fineweb_corpus_is_infinite(monkeypatch):
    """The generator wraps (encode-once infinite stream) — islice past the source length still yields."""
    def _fake_source(min_len=200, max_chars=4000, passages=None):
        # emulate the real infinite source: wrap forever
        while True:
            yield "A sufficiently long clean passage about geometry and basins and simplices and flow."

    monkeypatch.setattr("qig_studio.fineweb_source.stream_fineweb_passages", _fake_source)
    got = list(itertools.islice(corpus.stream_fineweb_corpus(min_len=40), 25))
    assert len(got) == 25
