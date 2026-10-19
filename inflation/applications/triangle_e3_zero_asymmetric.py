"""Construct extra general-classical inner data in the E3=0 triangle slice.

Sources: da Silva--Pozas-Kerstjens--Parisio, arXiv:2503.16654v2,
Eq. (5) and Appendix B2a, Eq. (B3). This script uses actual classical
models and local binary channels; no fitting or inflation is performed.

Named cases:
  asymmetric-322-band: fix y=3/5 in B3, vary x over its valid interval,
      then apply a common affine output channel to set E3=0. A continuous
      model family fills a conservative band, for |E1|<=1/2.
  asymmetric-eq5-curve: use Eq. (5), a common biased local channel, and
      depolarization. This gives curves reaching beyond |E1|=1/2.

The models establish C membership. No exclusion from C_sym is inferred.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import product
from pathlib import Path

import sympy as sp


EVENTS = tuple(product((-1, 1), repeat=3))
XSTAR = (21-math.sqrt(381))/10


def probability_table(q, r, s, A, B, C):
    """A[beta,gamma], B[gamma,alpha], C[alpha,beta] specify P(+1)."""
    probabilities = {abc: 0 for abc in EVENTS}
    for alpha, beta, gamma in product(range(len(q)), range(len(r)), range(len(s))):
        plus = (A[beta][gamma], B[gamma][alpha], C[alpha][beta])
        weight = q[alpha]*r[beta]*s[gamma]
        for abc in EVENTS:
            probabilities[abc] += weight*math.prod(
                p if a == 1 else 1-p for a, p in zip(abc, plus))
    return probabilities


def b3_model(x):
    z = (5*x*x-21*x+3)/(9*(1-2*x))
    t = 5*x*(x+2)/(4*(1-2*x))
    q = (1-2*x, x, x)
    r = s = (sp.Rational(3, 5), sp.Rational(2, 5))
    A = ((z, 1), (1, t))
    B = ((1, 0, 1), (0, 0, 1))
    C = ((1, 0), (1, 1), (0, 0))
    return q, r, s, A, B, C


def eq5_model():
    q = r = s = (sp.Rational(1, 3),)*3
    A = ((0, 1, 0), (1, 1, 0), (1, 1, 1))
    B = ((0, 1, 1), (0, 0, 1), (1, 1, 1))
    C = ((1, 1, 1), (1, 1, 0), (1, 0, 0))
    return q, r, s, A, B, C


def moments(probabilities):
    return [sum(p*math.prod(abc[i] for i in subset)
                for abc, p in probabilities.items())
            for subset in ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))]


def symbolic_checks():
    x, r = sp.symbols("x r", real=True)
    table = probability_table(*b3_model(x))
    e1 = (1-2*x)/5
    e2 = (1-52*x)/25
    e3 = (40*x*x+18*x-19)/25
    expected = [e1]*3+[e2]*3+[e3]
    assert all(sp.cancel(a-b) == 0 for a, b in zip(moments(table), expected))
    polynomial = r**3+3*e1*r*r+3*e2*r+e3
    # A positive root is unique: polynomial coefficients have one sign
    # change, since e1>0, e3<0 and e2 can have either sign.
    at_channel_bound = sp.factor(polynomial.subs(r, 1-2*e1))
    assert sp.expand(at_channel_bound+8*(4*x**3+47*x*x+30*x+4)/125) == 0
    # Therefore the positive root exceeds 1-2*e1. For target bias b<=1/2,
    # eta=b/(e1+r), offset=r*eta obey eta+offset<=1.
    zero_covariance = sp.factor(e2-e1**2)
    assert sp.expand(zero_covariance+4*x*(x+12)/25) == 0

    table5 = probability_table(*eq5_model())
    m5 = moments(table5)
    assert m5 == [sp.Rational(1, 3)]*3+[-sp.Rational(5, 27)]*3+[-sp.Rational(5, 9)]
    eta = 3/(3+sp.sqrt(5))
    offset = 1-eta
    E1, E2, E3 = m5[0], m5[3], m5[6]
    final = [sp.simplify(eta*E1+offset),
             sp.simplify(eta**2*E2+2*eta*offset*E1+offset**2),
             sp.simplify(eta**3*E3+3*eta**2*offset*E2
                         +3*eta*offset**2*E1+offset**3)]
    assert sp.simplify(final[0]-(sp.sqrt(5)-1)/2) == 0
    assert sp.simplify(final[1]-(3*sp.sqrt(5)-5)/6) == 0
    assert final[2] == 0
    assert sp.simplify(final[1]/final[0]**2-sp.sqrt(5)/3) == 0
    return {
        "published_B3_full_table_moments_checked_symbolically": True,
        "Eq5_full_table_moments_checked_symbolically": True,
        "valid_B3_parameter_interval": "0<=x<=(21-sqrt(381))/10<3/20, y=3/5",
        "B3_response_probabilities": {
            "z": "(5*x^2-21*x+3)/(9*(1-2*x))",
            "t": "5*x*(x+2)/(4*(1-2*x))",
            "validity": "On this interval z>=0; z<=1 since 5*x^2-3*x-6<0. "
                        "Also t>=0; t<=1 since 5*x^2+18*x-4<0 for x<=3/20."
        },
        "positive_root_uniqueness": "Descartes' rule: e1>0, e3<0; one coefficient sign change",
        "channel_bound_identity": str(at_channel_bound),
        "band_membership": "R(x)=(e2+2*r*e1+r^2)/(e1+r)^2 is continuous. "
                           "R(0)=1 and R(xstar)=Rstar; for each |b|<=1/2, "
                           "eta=b/(e1+r) and offset=r*eta produce E1=b, E2=R(x)*b^2, E3=0. "
                           "Continuity proves every point between Rstar*b^2 and b^2.",
        "Eq5_transformed_endpoint": [str(value) for value in final],
        "Eq5_depolarized_curve": "E2=sqrt(5)/3*E1^2, |E1|<=(sqrt(5)-1)/2",
        "scope": "Constructive C inner data; C_sym membership unresolved; no completeness claim",
    }


def bisect(function, lower, upper):
    first, last = function(lower), function(upper)
    if abs(first) < 1e-13:
        return lower
    if abs(last) < 1e-13:
        return upper
    assert first*last <= 0, (first, last)
    for _ in range(70):
        midpoint = (lower+upper)/2
        value = function(midpoint)
        if first*value <= 0:
            upper = midpoint
        else:
            lower, first = midpoint, value
    return (lower+upper)/2


def base_moments(x):
    return (1-2*x)/5, (1-52*x)/25, (40*x*x+18*x-19)/25


def root_and_ratio(x):
    e1, e2, e3 = base_moments(x)
    r = bisect(lambda r: r**3+3*e1*r*r+3*e2*r+e3, 0, 3)
    return r, (e2+2*r*e1+r*r)/(e1+r)**2


def noisy_table(probabilities, eta, offset):
    assert abs(eta)+abs(offset) <= 1+1e-12
    return {abc: sum(float(p)*math.prod((1+a*(eta*s+offset))/2
                                      for a, s in zip(abc, original))
                     for original, p in probabilities.items())
            for abc in EVENTS}


def validate(probabilities, e1, e2):
    assert abs(sum(probabilities.values())-1) < 2e-12
    assert min(probabilities.values()) >= -2e-12
    error = max(abs(p-(1+e1*(a+b+c)+e2*(a*b+b*c+c*a))/8)
                for (a, b, c), p in probabilities.items())
    assert error < 2e-12, error
    return error


def write_csv(output, name, rows):
    with (output/name).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("bias", "pair"))
        writer.writerows((f"{a:.12f}", f"{b:.12f}") for a, b in rows)


def generate(output, samples):
    checks = symbolic_checks()
    _, rstar = root_and_ratio(XSTAR)
    errors = []
    for i in range(41):
        bias = -.5+i/40
        for j in range(7):
            ratio = rstar+(1-rstar)*j/6
            x = bisect(lambda x: root_and_ratio(x)[1]-ratio, 0, XSTAR)
            params = b3_model(x)
            for distribution in params[:3]:
                assert min(distribution) >= -2e-12
                assert abs(sum(distribution)-1) < 2e-12
            for response in params[3:]:
                assert min(min(row) for row in response) >= -2e-12
                assert max(max(row) for row in response) <= 1+2e-12
            original = probability_table(*params)
            r, _ = root_and_ratio(x)
            eta = bias/(base_moments(x)[0]+r)
            result = noisy_table(original, eta, r*eta)
            errors.append(validate(result, bias, ratio*bias*bias))
    bmax = (math.sqrt(5)-1)/2
    eta0 = 3/(3+math.sqrt(5))
    p5 = probability_table(*eq5_model())
    for i in range(101):
        bias = bmax*(-1+2*i/100)
        depolarize = bias/bmax
        result = noisy_table(p5, eta0*depolarize, (1-eta0)*depolarize)
        errors.append(validate(result, bias, math.sqrt(5)/3*bias*bias))
    output.mkdir(parents=True, exist_ok=True)
    biases = [-.5+i/(samples-1) for i in range(samples)]
    upper = [(b, b*b) for b in biases]
    lower = [(b, rstar*b*b) for b in biases]
    write_csv(output, "322-band.csv", upper+list(reversed(lower)))
    write_csv(output, "322-upper.csv", upper)
    write_csv(output, "322-lower.csv", lower)
    curve = [(-bmax+2*bmax*i/(samples-1),
              math.sqrt(5)/3*(-bmax+2*bmax*i/(samples-1))**2)
             for i in range(samples)]
    write_csv(output, "eq5-curve.csv", curve)
    checks.update({
        "cases": ["asymmetric-322-band", "asymmetric-eq5-curve"],
        "B3_xstar": XSTAR,
        "B3_band_Rstar": rstar,
        "Eq5_final_maximum_bias": bmax,
        "full_tables_checked": {"322_band": 287, "eq5_curve": 101},
        "maximum_full_table_error": max(errors),
        "rendering_samples_per_curve": samples,
        "graphics": "CSV polygons approximate the analytic construction; no convex-hull inference",
    })
    (output/"checks.json").write_text(json.dumps(checks, indent=2)+"\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=801)
    args = parser.parse_args()
    if args.samples < 3 or args.samples % 2 != 1:
        parser.error("--samples must be odd and >=3")
    generate(args.output, args.samples)
