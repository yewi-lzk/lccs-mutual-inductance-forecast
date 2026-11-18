"""Verify the V19 tables from the released CSVs. Requires only Python 3.9+."""
import csv
from collections import Counter, defaultdict
from math import isclose
from pathlib import Path
from statistics import mean

DATA = Path(__file__).resolve().parents[1] / 'data'
METHODS = {'CurrentHold', 'DenoisedHold', 'Linear', 'Ridge_Residual', 'TCN_Residual', 'GRU'}
GROUPS = [(10, 'ID'), (10, 'OOD'), (30, 'ID'), (30, 'OOD')]


def rows(relative):
    with (DATA / relative).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def check(condition, label):
    if not condition:
        raise ValueError(label)


def equal(actual, expected, label):
    check(isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=3e-8),
          f'{label}: {actual} differs from {expected}')


def percentile(samples, probability):
    ordered = sorted(samples)
    index = (len(ordered)-1)*probability
    lower = int(index)
    return ordered[lower] + (index-lower)*(ordered[min(lower+1,len(ordered)-1)]-ordered[lower])


def verify_splits():
    splits = rows('tables/trajectory_splits.csv')
    counts = Counter(r['group'] for r in splits)
    check(counts == {'training': 48, 'validation': 12, 'bridge_reference': 24,
                     'circuit_check': 2, 'main_test': 120, 'bridge': 24}, 'table 1 trajectory groups')
    by_id = {r['case_id']: r for r in splits}
    check(len(by_id)==len(splits), 'duplicated trajectory ID')
    for r in splits:
        if r['group'] in ('bridge', 'circuit_check'):
            original = by_id[r['same_motion_as']]
            expected_group = 'bridge_reference' if r['group']=='bridge' else 'validation'
            check(original['group']==expected_group and
                  all(original[k]==r[k] for k in ('motion_seed','domain','family')), 'paired motion identity')
    main = {r['case_id']:r for r in splits if r['group']=='main_test'}
    counts = Counter((r['domain'], r['family']) for r in main.values())
    check(len(counts)==12 and set(counts.values())=={10}, 'main test domain and family counts')
    training = {r['motion_seed'] for r in splits if r['group'] in ('training','validation')}
    check(not training.intersection(r['motion_seed'] for r in main.values()), 'train/test seed overlap')
    print('Table 1: 48 training, 12 validation, 120 main test, 24 paired bridge and 2 reused circuit-check trajectories.')
    return main


def verify_current(main):
    records = rows('checks/trajectory_nls.csv')
    check(len(records)==120 and {r['case_id'] for r in records}==set(main), 'NLS trajectory identities')
    for r in rows('tables/nls_current.csv'):
        group = [x for x in records if x['domain']==r['domain']]
        check(len(group)==60 and all(x['origin_count']=='366' and x['noise_repeats']=='3' for x in group),
              'table 2 current observations')
        estimate = mean(float(x[r['parameter']+'_MAPE_pct']) for x in group)
        equal(estimate, r['common_current_MAPE_pct'], 'table 2 '+r['domain']+' '+r['parameter'])
    print('Table 2: six same-time M, L1 and L2 errors match their trajectory aggregates.')


def verify_forecasts(main):
    lookup = {}
    by_method = defaultdict(dict)
    for r in rows('checks/trajectory_mape.csv'):
        h, cid, m = int(r['horizon_ms']), r['case_id'], r['method']
        check(cid in main and r['domain']==main[cid]['domain'] and r['family']==main[cid]['family'],
              'trajectory metadata')
        key = h, cid, m
        check(key not in lookup, 'duplicate trajectory/horizon/method')
        lookup[key] = float(r['M_MAPE_pct'])
        by_method[h,r['domain'],m][cid] = lookup[key]
    check(len(lookup)==1440 and all(len(by_method[h,d,m])==60 for h,d in GROUPS for m in METHODS),
          'complete trajectory, horizon and method coverage')
    aggregates = {}
    for r in rows('tables/forecast_by_horizon.csv'):
        h,d,m = int(r['horizon_ms']),r['domain'],r['method']
        if h not in (10,30):
            continue
        result = mean(by_method[h,d,m].values())
        equal(result,r['M_MAPE_pct'],'table 3 '+str((h,d,m)))
        equal(100-result,r['M_accuracy_pct'],'forecast accuracy')
        aggregates[h,d,m] = result
    check(len(aggregates)==24,'table 3 row coverage')
    pairs = rows('tables/gru_paired_gains.csv')
    check(len(pairs)==20,'table 4 row coverage')
    for r in pairs:
        h,d,m=int(r['horizon_ms']),r['domain'],r['baseline']
        base,pred=aggregates[h,d,m],aggregates[h,d,'GRU']
        equal(100*(1-pred/base),r['relative_error_reduction_pct'],'table 4 relative MAPE reduction')
        wins=sum(v<by_method[h,d,m][cid] for cid,v in by_method[h,d,'GRU'].items())
        check(wins==int(r['GRU_lower_on_n']), 'paired trajectory win count')
    for r in rows('tables/forecast_by_motion_family.csv'):
        group=[v for cid,v in by_method[int(r['horizon_ms']),r['domain'],r['method']].items()
               if main[cid]['family']==r['family']]
        equal(mean(group),r['M_MAPE_pct'],'motion-family MAPE')
    abs_errors=rows('checks/mae_30ms.csv')
    check(len(abs_errors)==120 and {r['case_id'] for r in abs_errors}==set(main),'physical-error cohorts')
    reference=rows('checks/reference_hold_and_mae.csv')
    for d in ('ID','OOD'):
        group=[r for r in abs_errors if main[r['case_id']]['domain']==d]
        for m in ('GRU','CurrentHold'):
            result=mean(float(v[m+'_MAE_uH']) for v in group)
            target,=[r['mean'] for r in reference if r['domain']==d and r['method']==m
                     and r['horizon_ms']=='30' and r['metric']=='MAE_uH']
            equal(result,target,'30 ms physical error')
    ood = [r for r in abs_errors if main[r['case_id']]['domain']=='OOD']
    print('Tables 3 and 4: 24 MAPE cells and 20 relative error reductions match.')
    print('30 ms OOD: {:.2f}% lower MAPE than current hold; MAE {:.3f} -> {:.3f} uH.'.format(
        100*(1-aggregates[30,'OOD','GRU']/aggregates[30,'OOD','CurrentHold']),
        mean(float(r['CurrentHold_MAE_uH']) for r in ood),mean(float(r['GRU_MAE_uH']) for r in ood)))
    return lookup


def verify_timing():
    times=rows('checks/gru_timing_samples.csv')
    check(len(times)==750 and {r['method'] for r in times}=={'GRU'},'four-step GRU timing cohort')
    summary=rows('tables/gru_timing.csv')
    check(len(summary)==4,'timing components')
    for r in summary:
        values=[float(t[r['component']]) for t in times]
        for q,key in [(0.5,'P50_ms'),(0.95,'P95_ms'),(0.99,'P99_ms')]:
            equal(percentile(values,q),r[key],'table 5 '+r['component']+' '+key)
    print('Table 5: 12 percentiles match 750 measured four-step GRU calls.')


def verify_curves(lookup):
    curves=defaultdict(list)
    for r in rows('figures/fig07_gru_forecast.csv'):
        horizon=int(r['horizon_ms'])
        equal(float(r['target_s'])-float(r['origin_s']),horizon/1000,'figure 7 time alignment')
        curves[r['case_id'],horizon].append(float(r['mean_individual_APE_pct']))
    check(len(curves)==4,'figure 7 trajectory/horizon pairs')
    for (cid,h), errors in curves.items():
        check(len(errors)=={10:366,30:364}[h],'figure 7 valid origins')
        equal(mean(errors),lookup[h,cid,'GRU'],'figure 7 time-series versus trajectory MAPE')
    print('Figure 7: 1,460 aligned origins and targets reproduce all four trajectory errors.')


if __name__=='__main__':
    main_test=verify_splits()
    verify_current(main_test)
    verify_curves(verify_forecasts(main_test))
    verify_timing()
