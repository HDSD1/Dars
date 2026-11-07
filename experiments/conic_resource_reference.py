"""Independent SOCP reference for fixed-association, fixed-offload weighted delay.

The cone model uses u=f/F, v=b/W, p>=a/u, q>=c/v, z>=max(1-x,p+q).
Reciprocals are exact SOC epigraphs: ||(p-u,2*sqrt(a))|| <= p+u.
The computation-energy constraint is ||sqrt(e/E)*u|| <= 1.
Only this resource subproblem is convex; no joint trajectory optimality claim.
"""
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent/'solver_deps'))
import clarabel
import numpy as np
from scipy import sparse


def solve_group(a, c, local, energy, budget, u_cf, v_cf):
    started = time.perf_counter()
    a, c, local, energy = [np.asarray(v, dtype=float) for v in (a,c,local,energy)]
    n = len(a)
    if not n:
        return dict(valid=True, trivial=True, objective=0., lower_bound=0.,
                    primal_dual_gap=0., cpu_share=[], bandwidth_share=[], seconds=0.)
    if budget <= 0 or np.any(a <= 0) or np.any(c <= 0):
        return dict(valid=False, trivial=False, status='invalid_offload_budget_or_coefficients')
    nv = 5*n
    rows, rhs, cones = [], [], []
    def linear(coeff, b):
        row = np.zeros(nv)
        for j, value in coeff.items():
            row[j] = value
        rows.append(row); rhs.append(float(b))
    linear({i:1 for i in range(n)}, 1)
    linear({n+i:1 for i in range(n)}, 1)
    for i in range(n):
        linear({4*n+i:-1}, -local[i])
        linear({2*n+i:1,3*n+i:1,4*n+i:-1}, 0)
    cones.append(clarabel.NonnegativeConeT(2+2*n))
    dimensions = [('nonnegative',2+2*n)]
    for i in range(n):
        for resource, latency, coeff in ((i,2*n+i,a[i]),(n+i,3*n+i,c[i])):
            linear({resource:-1,latency:-1}, 0)
            linear({resource:1,latency:-1}, 0)
            linear({}, 2*np.sqrt(coeff))
            cones.append(clarabel.SecondOrderConeT(3))
            dimensions.append(('soc',3))
    linear({}, 1)
    for i in range(n):
        linear({i:-np.sqrt(energy[i]/budget)}, 0)
    cones.append(clarabel.SecondOrderConeT(n+1))
    dimensions.append(('soc',n+1))
    A = sparse.csc_matrix(np.asarray(rows)); b = np.asarray(rhs)
    obj = np.zeros(nv); obj[4*n:] = 1
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 250
    settings.tol_gap_abs = 1e-10
    settings.tol_gap_rel = 1e-10
    settings.tol_feas = 1e-10
    solver = clarabel.DefaultSolver(sparse.csc_matrix((nv,nv)),obj,A,b,cones,settings)
    solution = solver.solve()
    raw = np.asarray(solution.x)
    if raw.shape != (nv,) or not np.all(np.isfinite(raw)):
        return dict(valid=False,trivial=False,status=str(solution.status))
    u, v = raw[:n].copy(), raw[n:2*n].copy()
    if np.any(u <= 0) or np.any(v <= 0):
        return dict(valid=False,trivial=False,status=str(solution.status),reason='nonpositive shares')
    # Remove numerical resource-cap excess, then evaluate the physical objective.
    u /= max(1.,float(u.sum()))
    v /= max(1.,float(v.sum()))
    used = float(np.dot(energy,u*u))
    if used > budget:
        u *= np.sqrt(budget/used)*(1-1e-12)
    delays = np.maximum(local,a/u+c/v)
    primal = float(delays.sum())
    uc, vc = np.asarray(u_cf).copy(), np.asarray(v_cf).copy()
    uc /= max(1.,float(uc.sum())); vc /= max(1.,float(vc.sum()))
    cf_energy = float(np.dot(energy,uc*uc))
    if cf_energy > budget:
        uc *= np.sqrt(budget/cf_energy)*(1-1e-12)
    cf_j = float(np.maximum(local,a/uc+c/vc).sum())
    # Project dual variables into the self-dual cones. Account for nonzero
    # stationarity using valid bounds at an optimum no worse than the CF point.
    # At such an optimum: u,v<=1 and 0<=p,q,z<=CF objective.
    dual = np.asarray(solution.z).copy()
    offset = 0
    for kind, size in dimensions:
        block = dual[offset:offset+size]
        if kind == 'nonnegative':
            block[:] = np.maximum(block,0)
        else:
            head, tailnorm = float(block[0]), float(np.linalg.norm(block[1:]))
            if tailnorm <= -head:
                block[:] = 0
            elif tailnorm > head:
                newhead = (tailnorm+head)/2
                block[1:] *= newhead/tailnorm
                block[0] = newhead
            # A tiny inward margin makes the cone projection conservative.
            if np.any(block):
                block[0] += 1e-12*(1+np.linalg.norm(block))
        offset += size
    stationarity = obj + A.T@dual
    upper = np.r_[np.ones(2*n),np.full(3*n,cf_j)]
    lower = float(-b@dual + np.minimum(stationarity*upper,0).sum())
    gap = primal-lower
    resource_residual = max(0.,float(u.sum()-1),float(v.sum()-1))
    energy_excess = max(0.,float(np.dot(energy,u*u)-budget))
    valid = (str(solution.status)=='Solved' and resource_residual<=1e-9
             and energy_excess<=1e-6 and gap>=-1e-8
             and gap<=1e-6*(1+abs(primal)) and primal<=cf_j+1e-6*(1+abs(cf_j)))
    return dict(valid=bool(valid),trivial=False,status=str(solution.status),
                objective=primal,lower_bound=lower,primal_dual_gap=gap,
                relative_certificate_gap=gap/(1+abs(primal)),
                cpu_share=u.tolist(),bandwidth_share=v.tolist(),delay_ratio=delays.tolist(),
                resource_capacity_excess=resource_residual,energy_excess_j=energy_excess,
                dual_stationarity_inf=float(np.max(np.abs(stationarity))),
                solver_primal_residual=float(solution.r_prim),solver_dual_residual=float(solution.r_dual),
                solver_objective=float(solution.obj_val),solver_dual_objective=float(solution.obj_val_dual),
                iterations=int(solution.iterations),solver_seconds=float(solution.solve_time),
                seconds=time.perf_counter()-started)


def self_test():
    tests = [([.2],[.1],[.5],[1.],10.,[1.],[1.],.5),
             ([.2],[.1],[.5],[1.],.01,[.1],[1.],2.1),
             ([.2,.2],[.1,.1],[.5,.5],[1.,1.],10.,[.5,.5],[.5,.5],1.2)]
    results=[]
    for a,c,local,e,b,u,v,expected in tests:
        r=solve_group(a,c,local,e,b,u,v)
        assert r['valid'], r
        assert abs(r['objective']-expected)<1e-6, r
        assert r['lower_bound']<=expected+1e-8, r
        results.append(dict(expected=expected,actual=r['objective'],lower=r['lower_bound']))
    return results


if __name__=='__main__':
    import json
    print(json.dumps(dict(clarabel_version=clarabel.__version__,analytic_checks=self_test())))
