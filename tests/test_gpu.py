import threading
import time

from filesearcher.gpu import GpuArbiter


def test_query_goes_before_next_background_job():
    arb = GpuArbiter()
    order = []
    holding = threading.Event()
    release = threading.Event()

    def first_bg():
        with arb.background():
            holding.set()
            release.wait(5)
            order.append("bg1")

    def second_bg():
        with arb.background():
            order.append("bg2")

    def query():
        with arb.priority():
            order.append("query")

    t1 = threading.Thread(target=first_bg)
    t1.start()
    holding.wait(5)
    t2 = threading.Thread(target=second_bg)
    t2.start()
    time.sleep(0.05)
    t3 = threading.Thread(target=query)
    t3.start()
    time.sleep(0.05)
    release.set()
    for t in (t1, t2, t3):
        t.join(5)
    assert order == ["bg1", "query", "bg2"]
