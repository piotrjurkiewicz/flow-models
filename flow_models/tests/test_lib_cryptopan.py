import ipaddress

from flow_models.lib.cryptopan import CryptoPan

KEY = b"boojahyoo3vaeToong0Eijee7Ahz3yee"

def test_good_encryption():
    print("correctness test")
    c = CryptoPan(KEY)
    assert c.anonymize(int(ipaddress.ip_address("192.0.2.1"))) == int(ipaddress.ip_address("206.2.124.120"))
    print("OK")

def _test_perf():
    import time
    nb_tests = 100000
    addr = int(ipaddress.ip_address("192.0.2.1"))
    c = CryptoPan(KEY)
    print("starting performance check")
    stime = time.time()
    for i in range(0, nb_tests):
        c.anonymize(addr)
    dtime = time.time() - stime
    print("%d anonymizations in %s s" % (nb_tests, dtime))
    print("rate: %f anonymizations /sec " % (nb_tests / dtime))
    print("OK")


if __name__ == '__main__':
    test_good_encryption()
    _test_perf()
