#!/usr/bin/env python3
"""Rebuild publication figures from frozen JSON; never infer unmeasured sweeps."""
from pathlib import Path
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from make_tables import DATA, capacity_cells, empirical_fit_estimates, language_blocks, split_results

OUT=Path(__file__).resolve().parent/'figures'
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})

def finish(fig,name):
    fig.tight_layout()
    fig.savefig(OUT/(name+'.pdf'),metadata={'CreationDate':None})
    plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=DATA['kv_selective']['methods']
    fig,ax=plt.subplots(figsize=(5.5,2.45))
    baseline=d['FULL']
    for name,marker,fill in [('CDMA-MERGE-EASY','o','black'),('CDMA-MERGE-RANDOM','s','white'),
                             ('MEAN-MERGE-EASY','^','white'),('EVICT-EASY','D','black')]:
        p=d[name]
        # There is one compressed point. Lines only guide the eye to the shared baseline.
        ax.plot([baseline['effective_context_x'],p['effective_context_x']],
                [baseline['perplexity'],p['perplexity']],color='0.6',lw=.7,ls=':')
        ax.plot(p['effective_context_x'],p['perplexity'],marker=marker,color='black',
                markerfacecolor=fill,ls='none',label=name)
    ax.plot(baseline['effective_context_x'],baseline['perplexity'],'kx',label='FULL')
    ax.set(yscale='log',xlabel=r'Effective context ratio $K_{\rm eff}$ (not stream count)',ylabel='Perplexity')
    ax.legend(fontsize=7,loc='center left',bbox_to_anchor=(1,0.5))
    finish(fig,'kv_merge')
    payload=DATA.get('prototype_selective_heldout')
    splits=split_results(payload,'surprise') if payload is not None else {'train':DATA['prototype_selective']}
    fig,axs=plt.subplots(1,2,figsize=(6.4,2.65))
    for split,block in splits.items():
        ks=np.array([int(k) for k in block['by_K']])
        for label,key,marker in [('Easy','easy','o'),('Middle','mid','s'),('Hard','hard','^')]:
            # Open/dashed=train, filled/solid=held-out, distinguishable in B/W.
            style={'marker':marker,'ls':'-' if split=='heldout' else '--',
                   'color':'black','markerfacecolor':'black' if split=='heldout' else 'white',
                   'label':label+(' / held-out' if split=='heldout' else ' / train'),'ms':4}
            axs[0].plot(ks,[block['by_K'][str(k)]['acc_'+key] for k in ks],**style)
            ys=[block['retention_vs_K1'][str(k)][key] for k in ks]
            if any(y is not None for y in ys):
                axs[1].plot(ks,[np.nan if y is None else y for y in ys],**style)
    axs[0].set(xlabel='Streams K',ylabel='Absolute target-token accuracy')
    axs[1].set(xlabel='Streams K',ylabel='Retention relative to K=1')
    axs[1].set_title('Hard retention undefined',fontsize=8,pad=8)
    for ax in axs:ax.set_xticks(ks);ax.legend(fontsize=6,loc='best',ncol=2)
    finish(fig,'surprise')
    fit,ref=empirical_fit_estimates();cells=capacity_cells()
    fig,axs=plt.subplots(1,2,figsize=(6.5,2.45))
    for vocab,marker,ls in [(v,'o' if i==0 else 's','-' if i==0 else '--') for i,v in enumerate(sorted({p['V'] for p in cells}))]:
        ps=sorted([p for p in cells if p['V']==vocab],key=lambda p:p['d'])
        xs=np.array([p['d'] for p in ps]);ys=[p['k_cap'] for p in ps]
        axs[0].plot(xs,ys,marker=marker,ls='none',color='black',label=f'Observed, V={vocab}')
        axs[0].plot(xs,ref*xs/math.log2(vocab),ls=ls,color='0.4',label=f'Reference {ref:.2f}d/log₂V, V={vocab}')
    # Real width/capacity are recorded, vocabulary size is not: do not extrapolate the exploratory fit.
    ll=DATA['validate_llm']
    blocks=language_blocks(ll,DATA['validate_llm_full'],DATA.get('validate_llm_full_heldout'),DATA.get('prototype_selective_heldout'))
    caps=[b['k_cap_rel80'] for _,b in blocks if 'k_cap_rel80' in b]
    lo,hi=min(caps),max(caps)
    axs[0].errorbar(ll['d'],(lo+hi)/2,yerr=(hi-lo)/2,fmt='k*',ms=9,capsize=4,
                   label='Qwen, sample-dependent crossing')
    axs[0].set(xlabel='Residual width d',ylabel=r'$K_{\rm cap}$')
    axs[0].legend(fontsize=6.5)
    r=DATA['rank_sweep'];rank=[p['rank'] for p in r['points']]
    axs[1].plot(rank,[p['k_cap'] for p in r['points']],'ko-')
    axs[1].axhline(r['full']['k_cap'],color='0.4',ls='--',label='Full fine-tuning (toy)')
    axs[1].set(xscale='log',xlabel='Adapter rank (toy)',ylabel=r'$K_{\rm cap}$')
    axs[1].set_xticks(rank,labels=rank);axs[1].legend(fontsize=7)
    finish(fig,'capacity')
    selections=[('vision_kcap','image','Image/k-means, small V'),('vision_kcap_hardvocab','image','Image/k-means, larger V'),
                ('vqvae_kcap','image_vqvae','Image/VQ-VAE'),('vqvae_kcap','text_bpe','Text/BPE, short sweep'),
                ('code_kcap','text_bpe','Text/BPE, extended sweep'),('code_kcap','code','Code/BPE, extended sweep')]
    fig,ax=plt.subplots(figsize=(6.2,2.5))
    for i,(source,key,label) in enumerate(selections):
        p=DATA[source][key];cap=p['k_cap_signal80'];limit=max(map(int,p['trained']))
        ax.barh(i,cap,facecolor='white',edgecolor='black',hatch='///' if cap==limit else None)
        ax.text(cap+.5,i,('≥' if cap==limit else '')+str(cap),va='center',fontsize=8)
    ax.set_yticks(range(len(selections)),labels=[x[2] for x in selections]);ax.invert_yaxis()
    ax.set(xlabel=r'Signal-relative $K_{\rm cap}$; hatching = censored at sweep limit',xlim=(0,max(DATA['code_kcap']['meta']['eval_ks'])*1.18))
    finish(fig,'modalities')
    w=DATA['watermark_provenance'];b=str(min(map(int,w)));aa=w[b];xs=[float(a) for a in aa]
    fig,ax=plt.subplots(figsize=(5,2.3))
    for key,label,marker,ls in [('bitacc_residual','Residual','o','-'),('bitacc_logits','Logits','s','--'),('bitacc_control','Control','^',':')]:
        ax.plot(xs,[p[key] for p in aa.values()],marker=marker,ls=ls,color='black',label=label)
    ax.set(xscale='log',xlabel=r'Injection amplitude $\alpha$',ylabel='Bit accuracy',title=f'Payload B={b}')
    ax.set_xticks(xs,labels=[str(a) for a in aa]);ax.legend(fontsize=8)
    finish(fig,'watermark')
    print('Generated five PDF figures from frozen JSON results.')

if __name__=='__main__':main()
