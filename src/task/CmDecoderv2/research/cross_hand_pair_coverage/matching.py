"""精确效果筛选和配对数量诊断，不消费 decoder 输出。"""
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from src.task.CmDecoderv2.research.cross_hand_cm_swap.diagnostics import LIMITS

POLICIES = ('sync','within_5','within_15','within_30','any_time','nonoverlap','far_20')


def allowed(policy, delta):
    gap = np.abs(delta)
    if policy == 'sync':
        return gap == 0
    if policy.startswith('within_'):
        return gap <= int(policy.split('_')[1])
    if policy == 'any_time':
        return np.ones_like(gap,dtype=bool)
    return gap >= (5 if policy == 'nonoverlap' else 20)


def exact_effect_pairs(actual, reference, starts, target_valid, source_valid, device, budget=lambda:None):
    """Jensen 下界只排除不可能配对；所有保留边再计算4096点平均距离。"""
    a = torch.as_tensor(actual,device=device,dtype=torch.float64)
    b = torch.as_tensor(reference,device=device,dtype=torch.float64)
    means_a,means_b = a.mean(1),b.mean(1)
    rms_a,rms_b = a.square().sum(-1).mean(-1).sqrt(),b.square().sum(-1).mean(-1).sqrt()
    rows = torch.as_tensor(starts,device=device)
    possible = torch.as_tensor(target_valid[:,None]&source_valid[None,:],device=device)
    for k in range(4):
        lower = torch.linalg.vector_norm(means_a[rows+k,None]-means_b[None,rows+k],dim=-1)
        limit = torch.maximum(torch.full_like(lower,LIMITS['effect_absolute_mm']),
            LIMITS['effect_relative']*torch.maximum(rms_a[rows+k,None],rms_b[None,rows+k]))
        possible &= lower <= limit+1e-7
    ti,si = torch.where(possible)
    target,source,ratios,epes = [],[],[],[]
    for j in range(0,len(ti),64):
        budget()
        tr,sr = ti[j:j+64],si[j:j+64]
        errors,limits = [],[]
        for k in range(4):
            errors.append(torch.linalg.vector_norm(a[rows[tr]+k]-b[rows[sr]+k],dim=-1).mean(-1))
            limits.append(torch.maximum(torch.full_like(errors[-1],LIMITS['effect_absolute_mm']),
                LIMITS['effect_relative']*torch.maximum(rms_a[rows[tr]+k],rms_b[rows[sr]+k])))
        errors,limits = torch.stack(errors,-1),torch.stack(limits,-1)
        keep = (errors<=limits).all(-1)
        target.extend(tr[keep].cpu().numpy().tolist())
        source.extend(sr[keep].cpu().numpy().tolist())
        ratios.extend((errors/limits)[keep].cpu().numpy().tolist())
        epes.extend(errors[keep].cpu().numpy().tolist())
    target,source = np.asarray(target,np.int64),np.asarray(source,np.int64)
    ratios,epes = np.asarray(ratios).reshape(-1,4),np.asarray(epes).reshape(-1,4)
    effect = np.zeros((len(starts),len(starts)),bool)
    effect[target,source] = True
    return effect,dict(target=target,source=source,ratio=ratios,epe_mm=epes)


def contact_filter(edges, actual_mask, source_mask, points, starts):
    tr,sr = edges['target'],edges['source']
    na,nb = actual_mask.sum(-1),source_mask.sum(-1)
    ca = actual_mask @ points / np.maximum(na[:,None],1)
    cb = source_mask @ points / np.maximum(nb[:,None],1)
    iou,gap = np.zeros((len(tr),4)),np.zeros((len(tr),4))
    for j in range(0,len(tr),128):
        for k in range(4):
            t,s = starts[tr[j:j+128]]+k,starts[sr[j:j+128]]+k
            a,b = actual_mask[t],source_mask[s]
            iou[j:j+128,k] = (a&b).sum(-1)/np.maximum((a|b).sum(-1),1)
            d = np.linalg.norm(ca[t]-cb[s],axis=-1)*1000
            d[(na[t]==0)|(nb[s]==0)] = 1e6
            gap[j:j+128,k] = d
    keep = (iou>=LIMITS['contact_iou']).all(-1)&(gap<=LIMITS['contact_centroid_mm']).all(-1)
    return dict(target=tr[keep],source=sr[keep],ratio=edges['ratio'][keep],
                epe_mm=edges['epe_mm'][keep],contact_iou=iou[keep],centroid_mm=gap[keep])


def selection_stats(edges, starts, moving, policy):
    use = allowed(policy,starts[edges['source']]-starts[edges['target']])
    use &= moving[edges['target']]
    tr,sr = edges['target'][use],edges['source'][use]
    ratio = edges['ratio'][use].mean(-1)
    usable_indices = np.flatnonzero(use)
    delta = starts[sr]-starts[tr]
    order = np.lexsort((starts[sr],np.abs(delta),ratio,tr))
    first = order[np.r_[True,np.diff(tr[order])!=0]] if len(order) else order
    selected = usable_indices[first]
    graph = csr_matrix((np.ones(len(tr)),(tr,sr)),shape=(len(starts),len(starts)))
    capacity = int((maximum_bipartite_matching(graph,perm_type='column')>=0).sum())
    # Greedy lower bound: reserve all K+1 geometry timestamps on both streams.
    target_used,source_used = set(),set()
    independent = []
    for j in np.lexsort((starts[sr],starts[tr],np.abs(delta),ratio)):
        tf = set(range(int(starts[tr[j]]),int(starts[tr[j]])+5))
        sf = set(range(int(starts[sr[j]]),int(starts[sr[j]])+5))
        if not (tf&target_used or sf&source_used):
            independent.append(int(usable_indices[j]))
            target_used.update(tf)
            source_used.update(sf)
    return dict(windows=len(first),edges=len(tr),unique_selected_sources=len(np.unique(sr[first])),
                one_to_one_capacity=capacity,nonoverlap_greedy=len(independent),
                selected_edge_indices=selected.tolist(),nonoverlap_edge_indices=independent)
