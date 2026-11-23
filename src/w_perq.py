"""Per-question behaviour of a simulator's worth (Appendix, "Single questions").  -> results/w_perq.json"""
import os as _os; _REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import pickle, numpy as np, json
c = pickle.load(open(_REPO + '/results/y_cov_cache.pkl','rb')); h = c['h']; V = c['V']; names = c['names']
out = {}
for m in ['Claude-Fable-5.2','GPT-6-astra','GPT-5.6-luna','Qwen3.5-9B']:
    e2 = np.array([((c['C'][m][k]-h[k])**2).sum() for k in range(len(h))]); r = V.mean()/e2.mean()
    # per question: the number of respondents whose expected squared error equals the simulator's squared error on that question
    rq = V/np.maximum(e2,1e-9); srt = np.sort(e2)[::-1]
    out[m] = {'worth': r, 'share of questions on which its squared error is below the expected squared error of r respondents': float(np.mean(e2 < V/r)), 'median per-question equivalent': float(np.median(rq)),
              'quartiles of the per-question equivalent': [float(x) for x in np.percentile(rq,[10,25,50,75,90])], 'share of the total squared error from the worst tenth of questions': float(srt[:len(srt)//10].sum()/srt.sum()),
              'share of questions better than 25 respondents in expectation': float(np.mean(e2 < V/25)), 'share worse than 5 respondents': float(np.mean(e2 > V/5))}
    v = out[m]; print(f"{m:18s} worth {r:.1f} | questions better than its own average worth: {100*v['share of questions on which its squared error is below the expected squared error of r respondents']:.0f}% | per-question equivalent: 10/25/50/75/90th pct " + '/'.join(f'{x:.0f}' for x in v['quartiles of the per-question equivalent']) + f" | worst tenth of questions carries {100*v['share of the total squared error from the worst tenth of questions']:.0f}% of the squared error | worse than 5 respondents on {100*v['share worse than 5 respondents']:.0f}%")
json.dump(out, open(_REPO + '/results/w_perq.json','w'), indent=1)
# coverage of an interval built as if the simulator were r respondents (normal approximation per option): share of option shares within +-1.96 sqrt(c(1-c)/r)
for m in ['Claude-Fable-5.2']:
    r = V.mean()/np.mean([((c['C'][m][k]-h[k])**2).sum() for k in range(len(h))]); hit = []; 
    for k in range(len(h)):
        p = np.clip(c['C'][m][k], 1e-3, 1); hit += list(np.abs(p - h[k]) <= 1.96*np.sqrt(p*(1-p)/r))
    print('coverage of a nominal 95% interval that treats', m, 'as', round(r,1), 'respondents: %.0f%% of option shares'%(100*np.mean(hit)))
# equal-strength bound
for rho in (0.68, 0.81): print('k equally strong sources with error correlation', rho, ': limit gain', round(1/rho-1,2), '| k=2:', round(2/(1+rho)-1,2), '| k=3:', round(3/(1+2*rho)-1,2))
