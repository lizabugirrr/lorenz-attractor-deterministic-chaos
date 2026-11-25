"""
Символьна перевірка аналітичних результатів, які використовуються в
презентації (sympy). Результат: results/analytic.txt
"""
import os
import sympy as sp

x, y, z, s, r, b, lam = sp.symbols("x y z sigma r beta lambda", real=True)
f = sp.Matrix([s * (y - x), x * (r - z) - y, x * y - b * z])
X = sp.Matrix([x, y, z])
J = f.jacobian(X)
out = []
P = lambda *a: out.append(" ".join(str(t) for t in a))

P("=== Якобіан ===")
P(sp.pretty(J))
P("div f =", sp.simplify(J.trace()))

P("\n=== Рівноваги ===")
sols = sp.solve(list(f), [x, y, z], dict=True)
for so in sols:
    P(so)

P("\n=== Початок координат ===")
J0 = J.subs({x: 0, y: 0, z: 0})
P("char poly:", sp.factor(J0.charpoly(lam).as_expr()))
ev = sp.solve(sp.det(J0 - lam * sp.eye(3)), lam)
P("eigenvalues:", ev)

P("\n=== C± ===")
c = sp.sqrt(b * (r - 1))
Jc = J.subs({x: c, y: c, z: r - 1})
cp = sp.expand(sp.det(lam * sp.eye(3) - Jc))
P("char poly:", sp.collect(cp, lam))
_, a2, a1, a0 = [sp.factor(cc) for cc in sp.Poly(cp, lam).all_coeffs()]
P("a2 =", a2, "; a1 =", sp.factor(a1), "; a0 =", sp.factor(a0))
hurwitz = sp.factor(a2 * a1 - a0)
P("Routh–Hurwitz a2*a1 - a0 =", hurwitz)
rH = sp.solve(sp.Eq(a2 * a1 - a0, 0), r)[0]
P("r_H =", sp.simplify(rH), "=", rH.subs({s: 10, b: sp.Rational(8, 3)}),
  "≈", sp.N(rH.subs({s: 10, b: sp.Rational(8, 3)}), 8))
# на r_H пара чисто уявних коренів ±i*omega, omega^2 = a1
om2 = sp.simplify(a1.subs(r, rH))
P("omega_H^2 = a1(r_H) =", sp.factor(om2), "≈", sp.N(om2.subs({s: 10, b: sp.Rational(8, 3)})),
  "; omega_H ≈", sp.N(sp.sqrt(om2.subs({s: 10, b: sp.Rational(8, 3)}))))
# коли власні значення C± стають комплексними: дискримінант кубічного = 0
disc = sp.discriminant(cp.subs({s: 10, b: sp.Rational(8, 3)}), lam)
roots = [sp.N(t) for t in sp.Poly(disc, r).nroots() if abs(sp.im(t)) < 1e-9 and sp.re(t) > 1]
P("discriminant roots (r>1):", roots)

P("\n=== Глобальна стійкість при r<1: V = x^2/sigma + y^2 + z^2 ===")
V = x ** 2 / s + y ** 2 + z ** 2
dV = sp.expand(sp.Matrix([V]).jacobian(X) * f)[0]
P("dV/dt =", sp.factor(dV))
target = -2 * ((x - (r + 1) / 2 * y) ** 2 + (1 - ((r + 1) / 2) ** 2) * y ** 2 + b * z ** 2)
P("dV/dt - target =", sp.simplify(dV - sp.expand(target)), "(0 => тотожність доведена)")

P("\n=== Поглинаюча множина: V = r x^2 + sigma y^2 + sigma (z-2r)^2 ===")
V2 = r * x ** 2 + s * y ** 2 + s * (z - 2 * r) ** 2
dV2 = sp.expand(sp.Matrix([V2]).jacobian(X) * f)[0]
P("dV/dt =", sp.factor(dV2))
target2 = -2 * s * (r * x ** 2 + y ** 2 + b * (z - r) ** 2 - b * r ** 2)
P("dV/dt - target =", sp.simplify(dV2 - sp.expand(target2)))

P("\n=== Пекора–Керролл: синхронізація (y,z)-підсистеми ===")
ey, ez, xd = sp.symbols("e_y e_z x", real=True)
dey = -ey - xd * ez
dez = xd * ey - b * ez
P("d/dt (e_y^2+e_z^2)/2 =", sp.simplify(ey * dey + ez * dez))

P("\n=== Конвекція Релея–Бенара: Ra_c(a) = pi^4 (1+a^2)^3 / a^2 ===")
a = sp.symbols("a", positive=True)
Rac = sp.pi ** 4 * (1 + a ** 2) ** 3 / a ** 2
amin = sp.solve(sp.diff(Rac, a), a)
P("argmin a =", amin, "; Ra_c min =", sp.simplify(Rac.subs(a, amin[0])), "≈", sp.N(Rac.subs(a, amin[0])))
P("beta = 4/(1+a^2) at a^2=1/2:", sp.Rational(4) / (1 + sp.Rational(1, 2)))

os.makedirs("results", exist_ok=True)
with open("results/analytic.txt", "w") as fh:
    fh.write("\n".join(out) + "\n")
print("\n".join(out))
