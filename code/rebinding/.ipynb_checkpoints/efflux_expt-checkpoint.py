def washout_zero_bound(t, y, kon, koff, m, mu):
    c, s = y
    dcdt = -mu * (c - s)
    dsdt = -koff * s + kon * (c - s) * (m - s)
    return [dcdt, dsdt]


def washout_zero_bound_comp(t, y, c0, kon, koff, m, mu, k):
    c, c1, s, s1 = y
    dcdt = -mu * (c - s)
    dc1dt = mu * (k * c0 - (c1 - s1))
    dsdt = -koff * s + kon * (c - s) * (m - s - s1)
    ds1dt = -koff * s1 + kon * (c1 - s1) * (m - s - s1)
    return [dcdt, dc1dt, dsdt, ds1dt]


def washout_once(t, y, kon, koff, m, mu, k, Vc, V):
    c, cm, s = y
    dcdt = -mu * (c - s - cm * k)
    dcmdt = -dcdt * Vc / V
    dsdt = -koff * s + kon * (c - s) * (m - s)
    return [dcdt, dcmdt, dsdt]


def washout_once_comp(t, y, kon, koff, m, mu, k, Vc, V):
    c, cm, s, c1, cm1, s1 = y
    dcdt = -mu * (c - s - k * cm)
    dcmdt = -dcdt * Vc / V
    dsdt = -koff * s + kon * (c - s) * (m - s - s1)
    dc1dt = -mu * (c1 - s1 - k * cm1)
    dcm1dt = -dc1dt * Vc / V
    ds1dt = -koff * s1 + kon * (c1 - s1) * (m - s - s1)
    return [dcdt, dcmdt, dsdt, dc1dt, dcm1dt, ds1dt]
