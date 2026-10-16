"""Render a constructive classical inner region of the E3=0 triangle slice.

Evaluate the identical-source/response ternary model in da Silva et al.,
arXiv:2503.16654v2, App. B1 and C1, and independent local depolarization
from Gisin et al., arXiv:1906.06495v2, Supplementary Note 2.
No optimization or inflation solver is used.

For y0 <= y <= 1/2, choose
    x = (1-y + sqrt(1-y*y/3-1/(3*y)))/2,
where y0 is the root of 4*y0**4-8*y0+3 in (0,1).
This gives E3=0 and a continuous, increasing base bias from 0 to 1/2.
For any fixed target bias b>0, varying the base bias from b to 1/2 and
setting eta=b/base_bias connects the upper model curve to E2=2*b*b.
The intermediate value theorem supplies the entire intervening band.
The negative-bias half follows by flipping every output. At b=0 use
local noise on the unbiased ternary model.

CSV polygons approximate the analytic region for drawing only. Membership
comes from the continuous model construction, not convex interpolation.
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


def response(i, j):
    return 1 if (i, j) in ((0, 1), (1, 0), (1, 1)) else -1


def symbolic_checks():
    x, y, u = sp.symbols("x y u", real=True)
    q = (x, y, 1-x-y)
    probabilities = {abc: sp.S.Zero for abc in EVENTS}
    for i, j, k in product(range(3), repeat=3):
        abc = (response(j, k), response(k, i), response(i, j))
        probabilities[abc] += q[i]*q[j]*q[k]
    moments = [sp.expand(sum(prob*math.prod(abc[i] for i in subset)
                             for abc, prob in probabilities.items()))
               for subset in ((0,), (0, 1), (0, 1, 2))]
    expected = [
        -1+4*x*y+2*y**2,
        1-8*x*y+4*x**2*y-4*y**2+12*x*y**2+4*y**3,
        -1+12*x*y-12*x**2*y+6*y**2-12*x*y**2-4*y**3,
    ]
    assert all(sp.expand(a-b) == 0 for a, b in zip(moments, expected))
    for (a, b, c), probability in probabilities.items():
        table_entry = (1+moments[0]*(a+b+c)
                       +moments[1]*(a*b+b*c+c*a)+moments[2]*a*b*c)/8
        assert sp.expand(probability-table_entry) == 0
    radical_squared = 1-y**2/3-1/(3*y)
    zero_equation = sp.expand(moments[2].subs(x, (1-y+u)/2))
    remainder = sp.rem(zero_equation, u**2-radical_squared, u)
    assert sp.simplify(remainder) == 0
    assert [m.subs({x: sp.Rational(1, 2), y: sp.Rational(1, 2)})
            for m in moments] == [sp.Rational(1, 2), sp.Rational(1, 2), 0]
    return {
        "full_table_and_published_correlators_checked_symbolically": True,
        "E3_identically_zero_on_chosen_algebraic_branch": True,
        "endpoint_at_x_y_half": ["1/2", "1/2", "0"],
        "source_probability_domain_argument": (
            "S^2=1-y^2/3-1/(3y)>0 on [y0,1/2]; "
            "(1-y)^2-S^2=(2y-1)(2y^2-2y-1)/(3y)>=0, "
            "so x and 1-x-y are nonnegative."
        ),
        "base_bias_monotonicity_argument": (
            "E1=2y-1+2y*S; (S^2)'=(1-2y^3)/(3y^2)>0; "
            "therefore E1'(y)=2+2S+2y*S'>0."
        ),
        "band_membership_argument": (
            "For target bias b in (0,1/2], vary base bias from b to 1/2 "
            "and choose eta=b/base_bias. The final E2 is continuous and "
            "connects the constructed upper value to 2*b^2. "
            "Flip all outputs for b<0; depolarize the unbiased model for b=0."
        ),
    }


def bisect(function, lower, upper, iterations=64):
    first, last = function(lower), function(upper)
    if abs(first) < 1e-13:
        return lower
    if abs(last) < 1e-13:
        return upper
    assert first*last <= 0, (lower, upper, first, last)
    for _ in range(iterations):
        midpoint = (lower+upper)/2
        value = function(midpoint)
        if first*value <= 0:
            upper = midpoint
        else:
            lower, first = midpoint, value
    return (lower+upper)/2


Y0 = bisect(lambda y: 4*y**4-8*y+3, .38, .39)


def model(y):
    x = (1-y+math.sqrt(1-y*y/3-1/(3*y)))/2
    q = (x, y, 1-x-y)
    e1 = -1+4*x*y+2*y*y
    e2 = 1-8*x*y+4*x*x*y-4*y*y+12*x*y*y+4*y**3
    e3 = -1+12*x*y-12*x*x*y+6*y*y-12*x*y*y-4*y**3
    return q, e1, e2, e3


def y_for_bias(bias):
    return bisect(lambda y: model(y)[1]-bias, Y0, .5)


def upper_pair(bias):
    return model(y_for_bias(abs(bias)))[2]


def witness(bias, pair):
    if abs(bias) < 1e-14:
        y = Y0
        eta = math.sqrt(max(0, pair)/model(y)[2])
    else:
        start = y_for_bias(abs(bias))
        def residual(y):
            _, base_bias, base_pair, _ = model(y)
            return (bias/base_bias)**2*base_pair-pair
        y = bisect(residual, start, .5)
        eta = bias/model(y)[1]
    q, _, _, _ = model(y)
    return q, eta


def full_table(q, eta):
    probabilities = {abc: 0. for abc in EVENTS}
    for i, j, k in product(range(3), repeat=3):
        original = (response(j, k), response(k, i), response(i, j))
        weight = q[i]*q[j]*q[k]
        for abc in EVENTS:
            probabilities[abc] += weight*math.prod(
                (1+a*eta*s)/2 for a, s in zip(abc, original))
    return probabilities


def generate(output, samples):
    checks = symbolic_checks()
    maximum_error = 0.
    maximum_e3 = 0.
    checked_tables = 0
    for i in range(41):
        bias = -.5+i/40
        low, high = 2*bias*bias, upper_pair(bias)
        assert high >= low-2e-12
        for j in range(5):
            pair = low+(high-low)*j/4
            q, eta = witness(bias, pair)
            assert min(q) >= -2e-12 and max(q) <= 1+2e-12
            assert abs(sum(q)-1) < 2e-12 and abs(eta) <= 1+2e-12
            probabilities = full_table(q, eta)
            assert abs(sum(probabilities.values())-1) < 2e-12
            for (a, b, c), probability in probabilities.items():
                target = (1+bias*(a+b+c)+pair*(a*b+b*c+c*a))/8
                maximum_error = max(maximum_error, abs(probability-target))
                assert probability >= -2e-12
                assert abs(probability-target) < 2e-12
            e3 = sum(a*b*c*p for (a, b, c), p in probabilities.items())
            maximum_e3 = max(maximum_e3, abs(e3))
            checked_tables += 1
    upper = []
    for i in range(samples):
        bias = -.5+i/(samples-1)
        q, base_bias, pair, e3 = model(y_for_bias(abs(bias)))
        assert abs(abs(bias)-base_bias) < 2e-12 and abs(e3) < 2e-12
        upper.append((bias, pair))
    lower = [(bias, 2*bias*bias) for bias, _ in reversed(upper)]
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("upper.csv", upper), ("band.csv", upper+lower)):
        with (output/filename).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("bias", "pair"))
            writer.writerows((f"{a:.12f}", f"{b:.12f}") for a, b in rows)
    checks.update({
        "case": "ternary-e3-zero-band",
        "y0": Y0,
        "unbiased_positive_endpoint": upper_pair(0),
        "upper_samples": samples,
        "full_tables_checked_in_band": checked_tables,
        "maximum_full_table_error": maximum_error,
        "maximum_absolute_E3_in_checked_tables": maximum_e3,
        "region": "Analytic band between the model curve and E2=2*E1^2, |E1|<=1/2",
        "symmetry": "Identical independent ternary sources and identical noisy responses",
        "scope": "Constructive inner region only; no optimality or completeness claim",
        "graphics": "CSV polygon edges are drawing approximations to the analytic curves",
    })
    (output/"checks.json").write_text(json.dumps(checks, indent=2)+"\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=801)
    arguments = parser.parse_args()
    if arguments.samples < 3 or arguments.samples % 2 != 1:
        parser.error("--samples must be an odd integer >=3, to include zero bias")
    generate(arguments.output, arguments.samples)
