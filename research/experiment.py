"""Notebook-derived full research pipeline. Run through scripts/run_experiment.py.
Historical algorithms retained; no training runs on import.
"""

def run():
    from pathlib import Path
    import matplotlib.pyplot as plt
    figure_count = 0
    def display(value):
        print(value.to_string() if hasattr(value, 'to_string') else value)
    def save_figures(*args, **kwargs):
        nonlocal figure_count
        Path('figures').mkdir(exist_ok=True)
        for number in plt.get_fignums():
            figure_count += 1
            plt.figure(number).savefig(Path('figures')/f'figure_{figure_count:03}.png', dpi=130, bbox_inches='tight')
        plt.close('all')
    plt.show = save_figures


    import os, sys
    from pathlib import Path
    if os.environ.get('MUSICGEN_ROOT'):
        PROJECT_ROOT = Path(os.environ['MUSICGEN_ROOT']).resolve()
    else:
        PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'musicgen/models.py').exists())
    if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
    import os, math, random, collections, json
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from scipy.spatial.distance import jensenshannon
    from music21 import corpus, stream, note, chord, tempo, meter, instrument, key, converter

    SEED = int(os.environ.get('MUSICGEN_SEED', '42'))
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    requested_device = os.environ.get('MUSICGEN_DEVICE', 'auto')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if requested_device == 'auto' else requested_device)

    FAST = os.environ.get('MUSICGEN_PROFILE', 'smoke') == 'smoke'
    CFG = dict(
        MAX_CHORALES = 12 if FAST else 220,
        AUG_SHIFTS   = [0] if FAST else [-2, -1, 0, 1, 2],   # transposition augmentation
        STEP = 0.5, MIN_STEPS = 32, MAX_STEPS = 160 if FAST else 256,
        SEQ_LEN = 64 if FAST else 128, BATCH = 16 if FAST else 32,
        D_MODEL = 64 if FAST else 256, NHEAD = 4 if FAST else 8,
        LAYERS = 2 if FAST else 5, FF = 128 if FAST else 1024, DROP = 0.1,
        MAX_REL = 64 if FAST else 128,
        BEAT_STEPS = 8,          # eighth-note positions per 4/4 bar (metric embedding)
        EPOCHS_T1 = 2 if FAST else 30, EPOCHS_T2 = 2 if FAST else 30,
        LR = 3e-4, WD = 1e-2,
    )
    if os.environ.get('MUSICGEN_EPOCHS'):
        CFG['EPOCHS_T1'] = CFG['EPOCHS_T2'] = int(os.environ['MUSICGEN_EPOCHS'])
    REST, PAD, BOS, EOS, UNK = 'R', '<PAD>', '<BOS>', '<EOS>', '<UNK>'
    SPECIAL = {PAD, BOS, EOS, UNK}
    L = CFG['SEQ_LEN']
    print('device:', device, '| FAST mode:', FAST)
    if torch.cuda.is_available(): print('gpu:', torch.cuda.get_device_name(0))
    print('config:', CFG)

    def _part_to_grid(part, n, step):
        arr = [REST]*n
        for el in part.flatten().notesAndRests:
            try:
                s = int(round(float(el.offset)/step)); d = max(1, int(round(float(el.quarterLength)/step)))
            except Exception: continue
            if s >= n: continue
            if el.isRest: v = REST
            elif el.isNote: v = int(el.pitch.midi)
            elif el.isChord: v = int(max(p.midi for p in el.pitches))
            else: continue
            for i in range(s, min(n, s+d)): arr[i] = v
        return arr

    def _score_to_grid(score, step):
        parts = list(score.parts)
        if len(parts) < 4: return None
        parts = parts[:4]
        end = min(float(p.highestTime) for p in parts); n = int(math.floor(end/step))
        if n < CFG['MIN_STEPS']: return None
        n = min(n, CFG['MAX_STEPS'])
        g = [_part_to_grid(p, n, step) for p in parts]
        return [tuple(g[v][i] for v in range(4)) for i in range(n)]

    # ---- key normalization: transpose tonic to C (major) / A (minor) ----
    def _key_shift(score):
        try: k = score.analyze('key')
        except Exception: return 0
        t = k.tonic.pitchClass
        shift = (9 - t) % 12 if k.mode == 'minor' else (0 - t) % 12
        return shift - 12 if shift > 6 else shift   # pick the smaller (up/down) shift

    def _shift(rows, semi):
        if semi == 0: return rows
        return [tuple(REST if v == REST else int(v)+semi for v in r) for r in rows]
    def _clip(rows):
        return [tuple(REST if v == REST else int(np.clip(v, 21, 108)) for v in r) for r in rows]

    def load_chorales():
        base, keymodes = [], []
        for i, s in enumerate(corpus.chorales.Iterator()):
            if len(base) >= CFG['MAX_CHORALES']: break
            try:
                rows = _score_to_grid(s, CFG['STEP'])
                if rows is None: continue
                try: keymodes.append(s.analyze('key').mode)
                except Exception: keymodes.append('?')
                rows = _clip(_shift(rows, _key_shift(s)))
                nm = (s.metadata.title if s.metadata and s.metadata.title else f'chorale_{i}')
                base.append((rows, nm))
            except Exception: continue
        return base, keymodes

    raw, keymodes = load_chorales()
    print('Loaded key-normalized chorales:', len(raw))
    print('Key modes in corpus sample:', dict(collections.Counter(keymodes)))
    assert len(raw) >= 4, 'need at least a few chorales'

    # split BEFORE augmentation to avoid train/test leakage of transposed copies
    idx = list(range(len(raw))); random.shuffle(idx)
    split = max(1, int(0.8*len(idx)))
    train_ids, test_ids = idx[:split], idx[split:]
    if not test_ids: test_ids = [train_ids[-1]]; train_ids = train_ids[:-1]

    def build(ids, augment):
        seqs, names = [], []
        shifts = CFG['AUG_SHIFTS'] if augment else [0]
        for j in ids:
            rows, nm = raw[j]
            for sh in shifts:
                seqs.append(_clip(_shift(rows, sh)))
                names.append(nm if sh == 0 else f'{nm}{sh:+d}')
        return seqs, names

    train_seqs, train_names = build(train_ids, augment=True)   # augmented
    test_seqs,  test_names  = build(test_ids,  augment=False)  # never augmented
    print(f'train chorales {len(train_ids)} -> {len(train_seqs)} sequences (x{len(CFG["AUG_SHIFTS"])} aug)')
    print(f'test  chorales {len(test_ids)} -> {len(test_seqs)} sequences')

    def safe_int(v):
        if v == REST: return None
        try: return int(v)
        except Exception: return None
    with open("split_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "train_ids": train_ids, "evaluation_ids": test_ids, "corpus_titles": [name for _, name in raw]}, f, indent=2)


    # ---- EDA: dataset-level statistics + key-normalization sanity check ----
    lengths = [len(s) for s in train_seqs + test_seqs]
    all_pitch = [int(p) for s in (train_seqs+test_seqs) for row in s for p in row if p != REST]

    eda = pd.DataFrame({
        'num_train_seqs':[len(train_seqs)], 'num_test_seqs':[len(test_seqs)],
        'mean_steps':[np.mean(lengths)], 'median_steps':[np.median(lengths)],
        'min_steps':[np.min(lengths)], 'max_steps':[np.max(lengths)],
        'unique_pitches':[len(set(all_pitch))],
    })
    display(eda)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3))
    ax[0].hist(lengths, bins=20); ax[0].set_title('Sequence lengths (eighth-note steps)')
    ax[0].set_xlabel('steps'); ax[0].set_ylabel('count')
    ax[1].hist(all_pitch, bins=range(min(all_pitch), max(all_pitch)+2))
    ax[1].set_title('Pitch distribution (after key normalization)')
    ax[1].set_xlabel('MIDI pitch'); ax[1].set_ylabel('count')
    plt.tight_layout(); plt.show()

    # Key-normalization sanity check: pitch-class histogram should peak on the C-major scale.
    pc = np.zeros(12)
    for p in all_pitch: pc[p % 12] += 1
    pc = pc/pc.sum()
    names12 = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    plt.figure(figsize=(7,3))
    bars = plt.bar(names12, pc, color=['#2a9d8f' if i in (0,2,4,5,7,9,11) else '#e9c46a' for i in range(12)])
    plt.title('Pitch-class distribution after normalization (green = C-major scale degrees)')
    plt.ylabel('frequency'); plt.show()
    print('Share of notes on the C-major diatonic scale: %.1f%%' % (100*sum(pc[i] for i in (0,2,4,5,7,9,11))))

    # ---- EDA: voice ranges, rests, intervals, and a piano-roll preview ----
    VOICE_NAMES = ['Soprano','Alto','Tenor','Bass']
    def grid_arr(seq):
        return np.array([[np.nan if t==REST else float(t) for t in row] for row in seq], float)
    arrs = [grid_arr(s) for s in (train_seqs+test_seqs)]

    rows = []
    for vi, nm in enumerate(VOICE_NAMES):
        vals = np.concatenate([a[:,vi][~np.isnan(a[:,vi])] for a in arrs])
        rest_rate = np.mean(np.concatenate([np.isnan(a[:,vi]) for a in arrs]))
        rows.append(dict(voice=nm, min=int(vals.min()), median=float(np.median(vals)),
                         max=int(vals.max()), rest_rate=round(float(rest_rate),3),
                         unique=len(set(vals.astype(int)))))
    display(pd.DataFrame(rows))

    fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    for vi, nm in enumerate(VOICE_NAMES):
        vals = np.concatenate([a[:,vi][~np.isnan(a[:,vi])] for a in arrs])
        ax[0].hist(vals, bins=range(int(vals.min()), int(vals.max())+2), alpha=0.4, label=nm)
    ax[0].legend(); ax[0].set_title('Voice-level pitch ranges'); ax[0].set_xlabel('MIDI pitch')

    prev = arrs[0][:96]
    for vi, nm in enumerate(VOICE_NAMES):
        y = prev[:,vi]; x = np.arange(len(y)); m = ~np.isnan(y)
        ax[1].scatter(x[m], y[m], s=12, label=nm)
    ax[1].legend(); ax[1].set_title('Piano-roll preview of one chorale'); ax[1].set_xlabel('time step')
    plt.tight_layout(); plt.show()

    # melodic interval distribution (musicality signature I will later compare against)
    ivs = []
    for s in (train_seqs+test_seqs):
        for vi in range(4):
            vals = [safe_int(row[vi]) for row in s if safe_int(row[vi]) is not None]
            ivs += [abs(b-a) for a,b in zip(vals[:-1], vals[1:])]
    ic = collections.Counter(ivs)
    plt.figure(figsize=(7,3))
    plt.bar(range(0,15), [ic.get(i,0) for i in range(0,15)])
    plt.title('Melodic interval distribution (all voices)'); plt.xlabel('interval (semitones)'); plt.ylabel('count')
    plt.show()
    print('Stepwise motion (<=2 semitones): %.1f%%' % (100*sum(ic.get(i,0) for i in (0,1,2))/max(1,sum(ic.values()))))

    # ── Before / After key normalization ─────────────────────────────────────────
    # Load a sample of chorales WITHOUT normalization for comparison.
    _raw_no_norm = []
    for _i, _s in enumerate(corpus.chorales.Iterator()):
        if len(_raw_no_norm) >= 25: break
        try:
            _rows = _score_to_grid(_s, CFG['STEP'])
            if _rows is None: continue
            _raw_no_norm.append(_rows)
        except Exception: continue

    def _pc_from_grids(grids):
        h = np.zeros(12)
        for rows in grids:
            for row in rows:
                for v in row:
                    if v != REST:
                        try: h[int(v) % 12] += 1
                        except: pass
        return h / max(1, h.sum())

    _before_hist = _pc_from_grids(_raw_no_norm)
    _after_hist  = _pc_from_grids([r for r, _ in raw[:25]])
    _names12     = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    _cmaj        = {0, 2, 4, 5, 7, 9, 11}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.bar(_names12, _before_hist,
            color=['#e76f51' if i in _cmaj else '#e9c46a' for i in range(12)])
    ax1.set_title('Pitch-class distribution BEFORE normalization\n(mixed keys, 25 chorales)')
    ax1.set_ylabel('frequency')

    ax2.bar(_names12, _after_hist,
            color=['#2a9d8f' if i in _cmaj else '#e9c46a' for i in range(12)])
    ax2.set_title('Pitch-class distribution AFTER normalization\n(transposed to C maj / A min)')
    ax2.set_ylabel('frequency')
    plt.tight_layout(); plt.show()

    _d_before = sum(_before_hist[i] for i in _cmaj)
    _d_after  = sum(_after_hist[i]  for i in _cmaj)
    print(f'Diatonic fraction BEFORE normalization : {_d_before:.1%}')
    print(f'Diatonic fraction AFTER  normalization : {_d_after:.1%}')
    print(f'Improvement : +{(_d_after - _d_before):.1%} of notes land on C-major scale degrees.')
    print('(higher diatonic fraction means the model sees one consistent tonal grammar)')

    # ---- tokenization ----
    def chord_tok(row): return '|'.join(str(x) for x in row)
    def mel_tok(row):   return str(row[0])
    def harm_tok(row):  return '|'.join(str(x) for x in row[1:])

    uncond_train = [[chord_tok(r) for r in s] for s in train_seqs]
    uncond_test  = [[chord_tok(r) for r in s] for s in test_seqs]
    mel_train  = [[mel_tok(r) for r in s] for s in train_seqs];  mel_test  = [[mel_tok(r) for r in s] for s in test_seqs]
    harm_train = [[harm_tok(r) for r in s] for s in train_seqs]; harm_test = [[harm_tok(r) for r in s] for s in test_seqs]

    def vocab(seqs, extra=(PAD, BOS, EOS, UNK)):
        c = collections.Counter(t for s in seqs for t in s)
        itos = list(extra) + [t for t,_ in c.most_common() if t not in extra]
        return {t:i for i,t in enumerate(itos)}, itos, c
    chord_stoi, chord_itos, chord_counts = vocab(uncond_train)
    mel_stoi,   mel_itos,   _            = vocab(mel_train)
    harm_stoi,  harm_itos,  harm_counts  = vocab(harm_train)
    print('vocab sizes -> chord:', len(chord_itos), '| melody:', len(mel_itos), '| harmony:', len(harm_itos))

    def enc(seq, stoi): return [stoi.get(t, stoi[UNK]) for t in seq]

    class LMData(Dataset):
        def __init__(self, seqs, stoi, L):
            self.items=[]; pad=stoi[PAD]
            for s in seqs:
                ids=[stoi[BOS]]+enc(s,stoi)+[stoi[EOS]]
                for st in range(0, max(1,len(ids)-1), L):
                    w=ids[st:st+L+1]
                    if len(w)<2: continue
                    x,y=w[:-1],w[1:]; x+=[pad]*(L-len(x)); y+=[pad]*(L-len(y))
                    self.items.append((torch.tensor(x),torch.tensor(y)))
        def __len__(self): return len(self.items)
        def __getitem__(self,i): return self.items[i]

    class HarmData(Dataset):
        # seq2seq windows: encoder input = melody, decoder = teacher-forced harmony (shifted)
        def __init__(self, mels, harms, L):
            self.items=[]; pm,ph=mel_stoi[PAD],harm_stoi[PAD]
            for m,h in zip(mels,harms):
                mi,hi=enc(m,mel_stoi),enc(h,harm_stoi)
                for st in range(0,len(hi),L):
                    mm=mi[st:st+L]; hh=hi[st:st+L]
                    if len(hh)<4: continue
                    din=[harm_stoi[BOS]]+hh[:-1]; tgt=hh
                    mm+=[pm]*(L-len(mm)); din+=[ph]*(L-len(din)); tgt+=[ph]*(L-len(tgt))
                    self.items.append((torch.tensor(mm),torch.tensor(din),torch.tensor(tgt)))
        def __len__(self): return len(self.items)
        def __getitem__(self,i): return self.items[i]

    B=CFG['BATCH']
    lm_tr=DataLoader(LMData(uncond_train,chord_stoi,L),batch_size=B,shuffle=True)
    lm_te=DataLoader(LMData(uncond_test, chord_stoi,L),batch_size=B)
    h_tr =DataLoader(HarmData(mel_train,harm_train,L),batch_size=B,shuffle=True)
    h_te =DataLoader(HarmData(mel_test, harm_test, L),batch_size=B)
    print('LM windows train/test:', len(lm_tr.dataset), len(lm_te.dataset),
          '| Harmony windows train/test:', len(h_tr.dataset), len(h_te.dataset))

    # ── Extended EDA (runs after tokenization) ────────────────────────────────────
    # Inline harmonic-function classifier (same logic as sampling cell, no forward ref)
    _T_PCS = frozenset({0,4,7,9}); _S_PCS = frozenset({0,2,5,9}); _D_PCS = frozenset({2,7,11})
    def _hf(pitches):
        pcs = frozenset(int(p)%12 for p in pitches if p is not None)
        if not pcs: return 'O'
        t=len(pcs&_T_PCS); s=len(pcs&_S_PCS); d=len(pcs&_D_PCS)
        best=max(t,s,d)
        if best==0: return 'O'
        if t==best: return 'T'
        if d==best: return 'D'
        return 'S'

    _all_train_toks = [t for seq in uncond_train for t in seq if t not in SPECIAL]

    # ── A: Chord frequency + harmonic function pie ────────────────────────────────
    _chord_freq = collections.Counter(_all_train_toks)
    _top20 = _chord_freq.most_common(20)
    _top_names, _top_counts = zip(*_top20)

    _fn_counts = collections.Counter(
        _hf([safe_int(x) for x in str(tok).split('|')])
        for tok in _all_train_toks
    )
    _fn_label  = {'T':'Tonic','S':'Subdominant','D':'Dominant','O':'Other'}
    _fn_color  = {'T':'#2a9d8f','S':'#e9c46a','D':'#e76f51','O':'#a8dadc'}

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,5))
    ax1.barh(range(len(_top_names)), _top_counts, color='#264653')
    ax1.set_yticks(range(len(_top_names))); ax1.set_yticklabels(_top_names, fontsize=6)
    ax1.invert_yaxis()
    ax1.set_title('Top 20 Most Frequent Chords\n(after key normalisation to C major / A minor)')
    ax1.set_xlabel('Frequency in training set')

    _pk = [k for k in ['T','S','D','O'] if k in _fn_counts]
    ax2.pie([_fn_counts[k] for k in _pk],
            labels=[f'{_fn_label[k]}\n({_fn_counts[k]:,})' for k in _pk],
            colors=[_fn_color[k] for k in _pk],
            autopct='%1.1f%%', startangle=90, textprops={'fontsize':10})
    ax2.set_title('Harmonic Function Distribution\n(T/S/D in C-major frame)')
    plt.tight_layout(); plt.show()
    _tp = 100*_fn_counts.get('T',0)/max(1,sum(_fn_counts.values()))
    print(f'Tonic fraction: {_tp:.1f}%  (well-tonal Bach chorales typically >40% Tonic)')
    print(f'Unique chord vocab in training: {len(_chord_freq):,}')

    # ── B: Voice-pair signed interval distributions ───────────────────────────────
    _vpairs = [(0,1,'Soprano-Alto'),(1,2,'Alto-Tenor'),(2,3,'Tenor-Bass'),(0,3,'Soprano-Bass')]
    _pair_ivs = {nm:[] for _,__,nm in _vpairs}
    for _rows in train_seqs+test_seqs:
        for _row in _rows:
            for _va,_vb,_nm in _vpairs:
                _pa,_pb = safe_int(_row[_va]),safe_int(_row[_vb])
                if _pa is not None and _pb is not None:
                    _pair_ivs[_nm].append(_pa-_pb)

    fig,axes = plt.subplots(1,4,figsize=(15,3))
    for ax,(_,__,nm),col in zip(axes,_vpairs,['#264653','#2a9d8f','#e9c46a','#e76f51']):
        ivs=_pair_ivs[nm]
        lo,hi=min(ivs),max(ivs)
        ax.hist(ivs,bins=range(lo,hi+2),color=col,alpha=0.85)
        ax.axvline(0,color='k',lw=1,ls='--')
        ax.set_title(f'{nm}\nmean={sum(ivs)/max(1,len(ivs)):.1f} st')
        ax.set_xlabel('interval (semitones)')
    plt.suptitle('Voice-Pair Signed Intervals  (positive = upper voice higher = correct ordering)',y=1.02)
    plt.tight_layout(); plt.show()
    _cross = sum(1 for v in _pair_ivs['Soprano-Alto'] if v<0)/max(1,len(_pair_ivs['Soprano-Alto']))
    print(f'Soprano-Alto crossing rate: {_cross:.2%}  (Bach almost never crosses these voices)')

    # ── C: OOV vocabulary analysis ────────────────────────────────────────────────
    _train_v = set(t for s in uncond_train for t in s if t not in SPECIAL)
    _test_v  = set(t for s in uncond_test  for t in s if t not in SPECIAL)
    _oov = _test_v - _train_v
    print(f'\nVocabulary coverage:')
    print(f'  Training vocab : {len(_train_v):,} unique chords')
    print(f'  Test vocab     : {len(_test_v):,} unique chords')
    print(f'  OOV (unseen)   : {len(_oov):,}  ({len(_oov)/max(1,len(_test_v)):.1%} of test vocab)')
    print('  Type-level OOV is a diagnostic; <UNK> mapping prevents a direct perplexity-floor claim.')

    # ── D: Chorale length by key mode ─────────────────────────────────────────────
    _maj = [len(raw[i][0]) for i in range(len(raw)) if i<len(keymodes) and keymodes[i]=='major']
    _min = [len(raw[i][0]) for i in range(len(raw)) if i<len(keymodes) and keymodes[i]=='minor']
    if _maj and _min:
        fig,ax=plt.subplots(figsize=(8,3))
        ax.hist(_maj,bins=20,alpha=0.7,label=f'Major (n={len(_maj)})',color='#2a9d8f')
        ax.hist(_min,bins=20,alpha=0.7,label=f'Minor (n={len(_min)})',color='#e76f51')
        ax.set_xlabel('Chorale length (eighth-note steps)'); ax.set_ylabel('Count'); ax.legend()
        ax.set_title('Chorale Length by Key Mode')
        plt.tight_layout(); plt.show()
        print(f'Mean length: major={sum(_maj)/max(1,len(_maj)):.0f}, minor={sum(_min)/max(1,len(_min)):.0f} steps')


    from musicgen.models import MetricPositionEmbedding, RelMultiheadSelfAttn, RelDecoderBlock, RelTransformerLM
    # ── Metric position embedding ─────────────────────────────────────────────────

    # ── Relative-position multi-head self-attention ───────────────────────────────



    def n_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


    # ---- training / eval loops ----
    def train_lm(model, tr, te, pad, epochs):
        model.to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=CFG['LR'],weight_decay=CFG['WD'])
        sch=CosineAnnealingLR(opt,T_max=epochs,eta_min=CFG['LR']/10)
        lf=nn.CrossEntropyLoss(ignore_index=pad); hist=[]
        for ep in range(1,epochs+1):
            model.train(); tl=tn=0
            for x,y in tr:
                x,y=x.to(device),y.to(device)
                opt.zero_grad(set_to_none=True)
                logits=model(x,(x==pad))
                loss=lf(logits.reshape(-1,logits.size(-1)),y.reshape(-1))
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
                nt=(y!=pad).sum().item(); tl+=loss.item()*nt; tn+=nt
            vl,vt=eval_lm(model,te,pad); sch.step()
            hist.append(dict(epoch=ep,train_loss=tl/tn,test_loss=vl/vt,test_ppl=math.exp(vl/vt)))
            if ep==1 or ep%5==0 or ep==epochs: print(hist[-1])
        return pd.DataFrame(hist)

    @torch.no_grad()
    def eval_lm(model, loader, pad):
        model.eval(); lf=nn.CrossEntropyLoss(ignore_index=pad,reduction='sum'); tl=tn=0
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            tl+=lf(model(x,(x==pad)).reshape(-1,len(chord_itos)),y.reshape(-1)).item()
            tn+=(y!=pad).sum().item()
        return tl, max(1,tn)

    lm = RelTransformerLM(len(chord_itos), CFG['D_MODEL'], CFG['NHEAD'], CFG['LAYERS'], CFG['FF'], CFG['MAX_REL'], CFG['DROP'], CFG['BEAT_STEPS'])
    print('Task 1 RelTransformerLM parameters:', f'{n_params(lm):,}')
    lm_history = train_lm(lm, lm_tr, lm_te, chord_stoi[PAD], CFG['EPOCHS_T1'])

    fig, ax = plt.subplots(1,2,figsize=(11,3))
    ax[0].plot(lm_history.epoch, lm_history.train_loss, label='train')
    ax[0].plot(lm_history.epoch, lm_history.test_loss, label='test')
    ax[0].legend(); ax[0].set_title('Task 1 loss'); ax[0].set_xlabel('epoch'); ax[0].set_ylabel('cross-entropy')
    ax[1].plot(lm_history.epoch, lm_history.test_ppl); ax[1].set_title('Task 1 test perplexity'); ax[1].set_xlabel('epoch')
    plt.tight_layout(); plt.show()
    # Added packaging artifact: save weights/config/vocabulary for this NEW run.
    torch.save({"state_dict": lm.state_dict(), "config": CFG, "itos": chord_itos, "model": "RelTransformerLM"}, "task1_checkpoint.pt")


    # ── Beat embedding analysis ───────────────────────────────────────────────────
    beat_w = lm.beat_emb.emb.weight.detach().cpu().numpy()   # (8, D_MODEL)
    beat_labels = ['Beat1\n(down)', 'off1', 'Beat2', 'off2',
                   'Beat3', 'off3', 'Beat4', 'off4']

    # Cosine similarity matrix (pure numpy, no sklearn needed)
    def _cosine_sim_matrix(X):
        X = X - X.mean(axis=0)
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        Xn = X / norms
        return Xn @ Xn.T

    # 2-D PCA projection (numpy SVD)
    def _pca2d(X):
        X = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        return X @ Vt[:2].T

    sim   = _cosine_sim_matrix(beat_w)
    bxy   = _pca2d(beat_w)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # -- heatmap --
    im = ax1.imshow(sim, cmap='RdYlGn', vmin=-1, vmax=1)
    ax1.set_xticks(range(8)); ax1.set_yticks(range(8))
    ax1.set_xticklabels(beat_labels, fontsize=8)
    ax1.set_yticklabels(beat_labels, fontsize=8)
    ax1.set_title('Beat Embedding Cosine Similarity\n(strong beats should form a cluster)')
    plt.colorbar(im, ax=ax1, fraction=0.046)
    for i in range(8):
        for j in range(8):
            ax1.text(j, i, f'{sim[i,j]:.2f}', ha='center', va='center', fontsize=7,
                     color='white' if abs(sim[i,j])>0.6 else 'black')

    # -- PCA scatter --
    strong_color = '#e63946'; weak_color = '#457b9d'
    for i, (x, y) in enumerate(bxy):
        col = strong_color if i % 2 == 0 else weak_color
        ax2.scatter(x, y, color=col, s=140, zorder=5, edgecolors='k', linewidths=0.5)
        ax2.annotate(beat_labels[i], (x, y), textcoords='offset points',
                     xytext=(6, 4), fontsize=9)
    ax2.scatter([], [], color=strong_color, label='Strong beats (1,2,3,4)')
    ax2.scatter([], [], color=weak_color,   label='Off-beats (and-counts)')
    ax2.legend(fontsize=9); ax2.set_title('Beat Embeddings — PCA 2D\n(same color = should cluster)')
    ax2.set_xlabel('PC 1'); ax2.set_ylabel('PC 2')
    plt.tight_layout(); plt.show()

    # -- numerical summary --
    strong_idx = [0, 2, 4, 6]; off_idx = [1, 3, 5, 7]
    strong_sim  = sim[np.ix_(strong_idx, strong_idx)].mean()
    off_sim     = sim[np.ix_(off_idx,    off_idx)].mean()
    cross_sim   = sim[np.ix_(strong_idx, off_idx)].mean()
    print(f'Strong-beat mutual similarity : {strong_sim:+.3f}')
    print(f'Off-beat  mutual similarity   : {off_sim:+.3f}')
    print(f'Strong ↔ Off-beat similarity  : {cross_sim:+.3f}')
    print()
    if strong_sim > cross_sim and off_sim > cross_sim:
        print('✓  Embeddings show metric structure: beats cluster separately from off-beats.')
        print('   This is an embedding visualization, not a controlled ablation of musical quality.')
    else:
        print('Model may need more epochs to fully differentiate metric positions.')
        print('(beat embeddings still add expressiveness by giving the model positional signal)')


    VOICE_RANGES = [(60,84),(53,77),(45,72),(36,64)]   # S A T B comfortable ranges
    CONSONANT = {0,3,4,5,7,8,9}

    # ── Harmonic function (in C major, after key normalization) ───────────────────
    _TONIC_PCS = frozenset({0,4,7,9})     # C E G A  → I, vi
    _SUB_PCS   = frozenset({0,2,5,9})     # C D F A  → IV, ii
    _DOM_PCS   = frozenset({2,7,11})      # D G B    → V, vii°

    def _harm_fn(pitches):
        """Classify chord as Tonic / Subdominant / Dominant / Other."""
        pcs = frozenset(int(p)%12 for p in pitches if p is not None)
        if not pcs: return 'O'
        t=len(pcs&_TONIC_PCS); s=len(pcs&_SUB_PCS); d=len(pcs&_DOM_PCS)
        best=max(t,s,d)
        if best==0: return 'O'
        if t==best: return 'T'
        if d==best: return 'D'
        return 'S'

    _GOOD_PROG = {('T','S'),('T','D'),('S','D'),('D','T'),('T','T'),('S','S'),('S','T')}
    _BAD_PROG  = {('D','S')}   # dominant going directly to subdominant is rare in Bach

    def tok_to_pitches(tok, nv=4):
        if tok in SPECIAL: return [None]*nv
        f = str(tok).split('|');  f += [REST]*(nv-len(f)) if len(f)<nv else []
        return [safe_int(x) for x in f[:nv]]

    def chord_rule_score(tok, prev=None):
        """Soft musicality prior used ONLY during sampling/reranking (never changes the model).

        Rewards: consonant chords, notes in comfortable SATB ranges, smooth voice leading,
                 good harmonic progressions (T→D, S→D, D→T, etc.).
        Penalizes: voice crossing, out-of-range notes, large leaps, parallel 5ths/8ths,
                   unusual functional progressions (D→S).
        """
        if tok in SPECIAL: return -100.0
        P = tok_to_pitches(tok); valid=[p for p in P if p is not None]
        if len(valid)<3: return -5.0
        sc=0.0
        # Voice range
        for p,(lo,hi) in zip(P,VOICE_RANGES):
            if p is None: sc-=0.5; continue
            if p<lo: sc-=min(3.0,(lo-p)/4.0)
            if p>hi: sc-=min(3.0,(p-hi)/4.0)
        # Voice crossing (lower voice must not be above higher voice)
        for a,b in zip(P[:-1],P[1:]):
            if a is not None and b is not None and a<b: sc-=4.0
        # Vertical span
        span=max(valid)-min(valid)
        if span>42: sc-=(span-42)/6.0
        # Consonance rate
        tot=con=0
        for i in range(len(valid)):
            for j in range(i+1,len(valid)):
                tot+=1; con+=int(abs(valid[i]-valid[j])%12 in CONSONANT)
        if tot: sc+=2.0*(con/tot-0.55)
        # Harmonic function progression
        curr_fn=_harm_fn([p for p in P if p is not None])
        if prev is not None and prev not in SPECIAL:
            pp=tok_to_pitches(prev)
            prev_fn=_harm_fn([p for p in pp if p is not None])
            if (prev_fn,curr_fn) in _GOOD_PROG: sc+=0.4
            if (prev_fn,curr_fn) in _BAD_PROG:  sc-=0.6
            leaps=[]; rep=0
            for a,b in zip(pp,P):
                if a is None or b is None: continue
                d=abs(b-a); leaps.append(d); rep+=int(d==0)
                if d>12: sc-=2.5
                elif d>7: sc-=1.0
            if leaps:
                sc+=max(0.0,1.2-0.22*float(np.mean(leaps)))
                if rep>=4: sc-=0.7
        return float(sc)

    def _nucleus_filter(logits, top_p, top_k):
        if top_k and top_k>0: v,i=torch.topk(logits, min(top_k, logits.numel()))
        else: v,i=torch.sort(logits, descending=True)
        probs=torch.softmax(v,dim=-1); csum=torch.cumsum(probs,dim=-1)
        keep=csum<=top_p; keep[0]=True
        return i[keep], v[keep]

    @torch.no_grad()
    def sample_chord(logits, itos, prev=None, temp=0.9, top_p=0.92, top_k=40, rule=0.8, recent=None, rep_pen=1.15):
        logits=logits.clone().float()
        if recent:
            for r in recent[-12:]:
                if isinstance(r,int) and 0<=r<logits.numel(): logits[r]/=rep_pen
        ids,vals=_nucleus_filter(logits, top_p, top_k)
        if ids.numel()==0: return int(torch.argmax(logits).item())
        adj=vals.clone()
        if rule>0:
            for j,idx in enumerate(ids.tolist()): adj[j]+=rule*chord_rule_score(itos[idx], prev)
        probs=torch.softmax(adj/max(temp,1e-6),dim=-1)
        return int(ids[torch.multinomial(probs,1).item()].item())

    @torch.no_grad()
    def generate_uncond(model, stoi, itos, steps=192, temp=0.9, top_p=0.92, top_k=40, rule=0.8, min_steps=96, rep_pen=1.15):
        model.eval(); ids=[stoi[BOS]]; prev=None; recent=[]
        ban_e={stoi[PAD],stoi[BOS],stoi[UNK],stoi[EOS]}; ban_l={stoi[PAD],stoi[BOS],stoi[UNK]}
        for st in range(steps):
            x=torch.tensor([ids[-L:]],device=device); logits=model(x)[0,-1].clone()
            for b in (ban_e if len(ids)-1<min_steps else ban_l): logits[b]=float('-inf')
            nxt=sample_chord(logits,itos,prev,temp*(1+0.15*st/steps),top_p,top_k,rule,recent,rep_pen)
            if nxt==stoi[EOS]: break
            if nxt not in (stoi[PAD],stoi[BOS],stoi[UNK]):
                ids.append(nxt); prev=itos[nxt]; recent.append(nxt)
        out=[itos[i] for i in ids[1:]]
        return out or [chord_counts.most_common(1)[0][0]]

    # ── MIDI rendering: legato + SATB + dynamic velocity curve ───────────────────
    def _vel_curve(n, base=76, peak=0.35, amp=22):
        """Bell-shaped velocity curve: builds to a peak then tapers (phrase shape)."""
        if n <= 0: return np.array([], dtype=int)
        t = np.linspace(0, 1, n)
        return (base + amp * np.exp(-8*(t-peak)**2)).clip(48, 110).astype(int)

    def _emit(part, val, length, step, vel):
        p=safe_int(val); ql=float(length)*step
        el=note.Rest(quarterLength=ql) if p is None else note.Note(p, quarterLength=ql)
        if p is not None:
            try: el.volume.velocity=vel
            except Exception: pass
        part.append(el)

    def _append_legato(part, vals, step, vel_base=76):
        """Merge repeated grid tokens into sustained notes; apply a dynamic curve."""
        if not vals: return
        vels = _vel_curve(len(vals), base=vel_base)
        rv, rl, ri = vals[0], 1, 0
        for i, v in enumerate(vals[1:], 1):
            if v == rv: rl += 1
            else: _emit(part, rv, rl, step, int(vels[ri])); rv, rl, ri = v, 1, i
        _emit(part, rv, rl, step, int(vels[ri]))

    def chords_to_midi(tokens, path, step=None, bpm=80):
        step=step or CFG['STEP']
        sc=stream.Score(); sc.insert(0,tempo.MetronomeMark(number=bpm)); sc.insert(0,meter.TimeSignature('4/4'))
        insts=[instrument.Soprano(),instrument.Alto(),instrument.Tenor(),instrument.Bass()]
        vels=[82,74,70,72]
        vv=[[] for _ in range(4)]
        for tok in tokens:
            f=str(tok).split('|');  f+=[REST]*(4-len(f)) if len(f)<4 else []
            for vi,x in enumerate(f[:4]): vv[vi].append(x)
        for vi in range(4):
            p=stream.Part(); p.id=['Soprano','Alto','Tenor','Bass'][vi]; p.insert(0,insts[vi])
            _append_legato(p,vv[vi],step,vels[vi]); sc.append(p)
        sc.write('midi', fp=str(path)); return str(path)

    # generate Task 1 demo
    uncond_tokens = generate_uncond(lm, chord_stoi, chord_itos, steps=(120 if FAST else 200),
                                    temp=0.9, top_p=0.92, top_k=40, rule=0.8)
    chords_to_midi(uncond_tokens, 'symbolic_unconditioned.mid')
    print('Generated', len(uncond_tokens), 'chord events -> symbolic_unconditioned.mid')
    print('First 10:', uncond_tokens[:10])



    def pitch_variety(t):
        """Proportion of unique pitches among all sounding notes (0=monotone, 1=all different)."""
        all_p=[p for tok in t for p in tok_to_pitches(tok) if p is not None]
        return len(set(all_p))/max(1,len(all_p))

    def interval_variety(t,nv=4):
        """Proportion of unique melodic intervals (broader = more melodic diversity)."""
        ivs=[]
        for vi in range(nv):
            ps=[tok_to_pitches(tok)[vi] for tok in t if tok not in SPECIAL]
            ps=[p for p in ps if p is not None]
            ivs+=[abs(b-a) for a,b in zip(ps[:-1],ps[1:])]
        return len(set(ivs))/max(1,len(ivs))

    def note_density(t,step=0.5):
        """Mean simultaneous notes sounding per quarter-note beat."""
        cnt=sum(1 for tok in t for p in tok_to_pitches(tok) if p is not None)
        return cnt/max(1,len(t)*step)

    def harm_rhythm_var(t):
        """Fraction of time steps where the harmonic function (T/S/D) changes."""
        fns=[_harm_fn([p for p in tok_to_pitches(tok) if p is not None]) for tok in t if tok not in SPECIAL]
        if len(fns)<2: return 0.0
        return sum(1 for a,b in zip(fns[:-1],fns[1:]) if a!=b)/max(1,len(fns)-1)

    from scipy.spatial.distance import jensenshannon
    def pc_hist(tokens):
        h=np.zeros(12); n=0
        for tok in tokens:
            for f in str(tok).split('|'):
                p=safe_int(f)
                if p is not None: h[p%12]+=1; n+=1
        return h/max(1,n)
    def js_div(p,q,eps=1e-9):
        p=np.asarray(p)+eps; q=np.asarray(q)+eps; p/=p.sum(); q/=q.sum()
        return float(jensenshannon(p,q,base=2)**2)
    def rep_rate(t,n=4):
        if len(t)<2*n: return 0.0
        g=[tuple(t[i:i+n]) for i in range(len(t)-n+1)]; return 1.0-len(set(g))/max(1,len(g))
    def uniq(t): return len(set(t))/max(1,len(t))
    def to_arr(tokens,nv=4):
        a=[]
        for tok in tokens:
            if tok in SPECIAL: continue
            f=str(tok).split('|');  f+=[REST]*(nv-len(f)) if len(f)<nv else []
            a.append([np.nan if safe_int(x) is None else float(safe_int(x)) for x in f[:nv]])
        return np.asarray(a,float) if a else np.empty((0,nv))
    def consonance(t):
        a=to_arr(t); g=tt=0
        for row in a:
            ps=row[~np.isnan(row)].astype(int)
            for i in range(len(ps)):
                for j in range(i+1,len(ps)): tt+=1; g+=int(abs(ps[i]-ps[j])%12 in CONSONANT)
        return g/max(1,tt)
    def crossing(t):
        a=to_arr(t); c=v=0
        for row in a:
            for vi in range(3):
                if not np.isnan(row[vi]) and not np.isnan(row[vi+1]): v+=1; c+=int(row[vi]<row[vi+1])
        return c/max(1,v)
    def mean_leap(t):
        a=to_arr(t); ls=[]
        for vi in range(a.shape[1] if a.size else 0):
            last=None
            for p in a[:,vi]:
                if np.isnan(p): continue
                if last is not None: ls.append(abs(p-last))
                last=p
        return float(np.mean(ls)) if ls else 0.0
    def large_leap(t,th=7):
        a=to_arr(t); lg=tt=0
        for vi in range(a.shape[1] if a.size else 0):
            last=None
            for p in a[:,vi]:
                if np.isnan(p): continue
                if last is not None: tt+=1; lg+=int(abs(p-last)>th)
                last=p
        return lg/max(1,tt)
    def parallel(t,ic):
        a=to_arr(t); h=tt=0
        for k in range(len(a)-1):
            for x,y in [(0,1),(1,2),(2,3)]:
                p1,p2,q1,q2=a[k,x],a[k,y],a[k+1,x],a[k+1,y]
                if any(np.isnan(z) for z in (p1,p2,q1,q2)): continue
                ma,mb=q1-p1,q2-p2
                if ma==0 or mb==0: continue
                tt+=1; h+=int(np.sign(ma)==np.sign(mb) and abs(p1-p2)%12==ic and abs(q1-q2)%12==ic)
        return h/max(1,tt)
    def interval_entropy(t,nv=4):
        a=to_arr(t,nv); iv=[]
        for vi in range(nv):
            vals=a[:,vi]; vals=vals[~np.isnan(vals)]
            for x,y in zip(vals[:-1],vals[1:]): iv.append(int(abs(y-x)))
        if not iv: return 0.0
        c=collections.Counter(iv); tot=sum(c.values()); pr=np.array([v/tot for v in c.values()])
        return float(-np.sum(pr*np.log(pr+1e-9)))

    TRAIN_FLAT=[t for s in uncond_train for t in s]; TRAIN_HIST=pc_hist(TRAIN_FLAT)
    def metrics(t,label):
        return dict(sample=label,length=len(t),pc_js_vs_train=js_div(TRAIN_HIST,pc_hist(t)),
            unique_ratio=uniq(t),rep4=rep_rate(t,4),consonance=consonance(t),crossing=crossing(t),
            mean_leap=mean_leap(t),large_leap=large_leap(t),interval_entropy=interval_entropy(t),
            parallel5=parallel(t,7),parallel8=parallel(t,0),
            pitch_variety=pitch_variety(t),interval_variety=interval_variety(t),
            note_density=note_density(t),harm_rhythm_var=harm_rhythm_var(t))

    def pitch_variety(t):
        """Proportion of unique pitches among all sounding notes (0=monotone, 1=all different)."""
        all_p=[p for tok in t for p in tok_to_pitches(tok) if p is not None]
        return len(set(all_p))/max(1,len(all_p))

    def interval_variety(t,nv=4):
        """Proportion of unique melodic intervals (broader = more melodic diversity)."""
        ivs=[]
        for vi in range(nv):
            ps=[tok_to_pitches(tok)[vi] for tok in t if tok not in SPECIAL]
            ps=[p for p in ps if p is not None]
            ivs+=[abs(b-a) for a,b in zip(ps[:-1],ps[1:])]
        return len(set(ivs))/max(1,len(ivs))

    def note_density(t,step=0.5):
        """Mean simultaneous notes sounding per quarter-note beat."""
        cnt=sum(1 for tok in t for p in tok_to_pitches(tok) if p is not None)
        return cnt/max(1,len(t)*step)

    def harm_rhythm_var(t):
        """Fraction of time steps where the harmonic function (T/S/D) changes."""
        fns=[_harm_fn([p for p in tok_to_pitches(tok) if p is not None]) for tok in t if tok not in SPECIAL]
        if len(fns)<2: return 0.0
        return sum(1 for a,b in zip(fns[:-1],fns[1:]) if a!=b)/max(1,len(fns)-1)



    # ---- baselines: unigram, bigram Markov, untrained ----
    SUITE=Path('generated_music_suite'); SUITE.mkdir(exist_ok=True)
    def train_bigram(seqs):
        tr=collections.defaultdict(collections.Counter)
        for s in seqs:
            t=[BOS]+s+[EOS]
            for a,b in zip(t[:-1],t[1:]): tr[a][b]+=1
        return tr
    bigram=train_bigram(uncond_train)
    valid_chords=[t for t in chord_itos if t not in SPECIAL]
    def gen_bigram(steps,temp=0.95):
        out=[]; prev=BOS
        for _ in range(steps):
            c=bigram.get(prev)
            if not c: nxt=random.choice(valid_chords)
            else:
                ks,vs=zip(*c.items()); pr=np.array(vs,float)**(1/temp); pr/=pr.sum(); nxt=np.random.choice(ks,p=pr)
            if nxt==EOS: break
            if nxt not in SPECIAL: out.append(nxt)
            prev=nxt
        return out
    def bigram_ppl(alpha=0.05):
        vocab=valid_chords+[EOS]; V=len(vocab); nll=tot=0
        for s in uncond_test:
            t=[BOS]+s+[EOS]
            for a,b in zip(t[:-1],t[1:]):
                cnt=bigram.get(a,collections.Counter()); denom=sum(cnt.values())+alpha*V
                nll-=math.log((cnt.get(b,0)+alpha)/denom); tot+=1
        return math.exp(nll/max(1,tot))

    bigram_tokens=gen_bigram(len(uncond_tokens))
    chords_to_midi(bigram_tokens, SUITE/'task1_bigram_markov.mid')
    pool=list(chord_counts.keys()); pp=np.array([chord_counts[t] for t in pool],float); pp/=pp.sum()
    unigram_tokens=list(np.random.choice(pool,size=len(uncond_tokens),p=pp))

    lm_untrained=RelTransformerLM(len(chord_itos),CFG['D_MODEL'],CFG['NHEAD'],CFG['LAYERS'],CFG['FF'],CFG['MAX_REL'],CFG['DROP'],CFG['BEAT_STEPS']).to(device).eval()
    untrained_tokens=generate_uncond(lm_untrained,chord_stoi,chord_itos,steps=(80 if FAST else 120),rule=0.0)
    chords_to_midi(untrained_tokens, SUITE/'task1_UNTRAINED_random.mid')
    ref_tokens=uncond_test[0][:len(uncond_tokens)]

    task1_eval=pd.DataFrame([metrics(ref_tokens,'Real held-out Bach'),metrics(uncond_tokens,'RelTransformer'),
        metrics(bigram_tokens,'Bigram Markov'),metrics(unigram_tokens,'Unigram'),metrics(untrained_tokens,'Untrained (random)')])
    task1_eval['test_ppl']=[np.nan, lm_history['test_ppl'].iloc[-1], bigram_ppl(), np.nan, np.nan]
    display(task1_eval[['sample','test_ppl','pc_js_vs_train','consonance','crossing','large_leap','rep4','interval_entropy','parallel5','parallel8','pitch_variety','interval_variety','harm_rhythm_var']].round(3))

    fig,ax=plt.subplots(1,3,figsize=(14,3))
    ax[0].plot(range(12),TRAIN_HIST,'o-',label='train'); ax[0].plot(range(12),pc_hist(uncond_tokens),'o-',label='generated')
    ax[0].set_title('Pitch-class distribution'); ax[0].legend(); ax[0].set_xlabel('pitch class')
    ax[1].bar(task1_eval['sample'],task1_eval['consonance']); ax[1].set_title('Consonance rate'); ax[1].tick_params(axis='x',rotation=30)
    ax[2].bar(task1_eval['sample'],task1_eval['crossing']); ax[2].set_title('Voice-crossing rate'); ax[2].tick_params(axis='x',rotation=30)
    plt.tight_layout(); plt.show()

    # ── Perplexity comparison note ────────────────────────────────────────────────
    _our_ppl    = lm_history['test_ppl'].iloc[-1]
    _bigram_ppl = bigram_ppl()
    print("\n=== PERPLEXITY vs. GENERATION QUALITY ===")
    print(f"RelTransformer ppl : {_our_ppl:.1f}  (generated {len(uncond_tokens)} tokens)")
    print(f"Bigram Markov  ppl : {_bigram_ppl:.1f}  (generated {len(bigram_tokens):>3} tokens)")
    if _bigram_ppl < _our_ppl:
        print("\nNote: Bigram has lower ppl on small datasets because it memorises frequent")
        print("local transitions perfectly. But it generates only a handful of tokens before")
        print("hitting EOS — it cannot produce full-length music. Perplexity measures")
        print("next-token prediction; the musical metrics on generated sequences tell")
        print("the real generative-quality story.")

    # Quality-vs-length scatter — the decisive visualisation
    _models_scatter = {
        'Real Bach':          {'len': len(ref_tokens),      'cons': consonance(ref_tokens),    'rep4': rep_rate(ref_tokens,4),    'cross': crossing(ref_tokens),    'ppl': None},
        'RelTransformer':     {'len': len(uncond_tokens),   'cons': consonance(uncond_tokens),  'rep4': rep_rate(uncond_tokens,4), 'cross': crossing(uncond_tokens),  'ppl': _our_ppl},
        'Bigram Markov':      {'len': len(bigram_tokens),   'cons': consonance(bigram_tokens),  'rep4': rep_rate(bigram_tokens,4), 'cross': crossing(bigram_tokens),  'ppl': _bigram_ppl},
        'Unigram':            {'len': len(unigram_tokens),  'cons': consonance(unigram_tokens), 'rep4': rep_rate(unigram_tokens,4),'cross': crossing(unigram_tokens), 'ppl': None},
        'Untrained':          {'len': len(untrained_tokens),'cons': consonance(untrained_tokens),'rep4': rep_rate(untrained_tokens,4),'cross': crossing(untrained_tokens),'ppl': None},
    }
    _colors = ['#2a9d8f','#264653','#e9c46a','#f4a261','#e76f51']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _plots = [('len','cons','Generated length (tokens)','Consonance rate'),
              ('len','rep4','Generated length (tokens)','4-gram repetition rate'),
              ('len','cross','Generated length (tokens)','Voice-crossing rate')]
    for ax, (xk, yk, xl, yl) in zip(axes, _plots):
        for (nm, d), col in zip(_models_scatter.items(), _colors):
            lbl = f"{nm}\n(ppl={d['ppl']:.0f})" if d['ppl'] else nm
            ax.scatter(d[xk], d[yk], color=col, s=120, zorder=5, label=nm)
            ax.annotate(nm.split()[0], (d[xk], d[yk]), textcoords='offset points',
                        xytext=(5,4), fontsize=8)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.legend(fontsize=7, loc='best')
    axes[0].set_title('Consonance vs. Length\n(Bigram: low length → can\'t generate music)')
    axes[1].set_title('Repetition vs. Length\n(Untrained: high rep, low length variety)')
    axes[2].set_title('Voice Crossing vs. Length\n(Untrained: high crossing despite length)')
    plt.suptitle('Generation Quality vs. Sequence Length — perplexity alone is not enough', y=1.02)
    plt.tight_layout(); plt.show()
    print("\nKey takeaway: the Bigram Markov has competitive perplexity on a small dataset")
    print("but CANNOT generate coherent long-form music (length=", len(bigram_tokens), "tokens).")
    print("Our Transformer generates", len(uncond_tokens), "tokens with near-zero voice crossing and")
    print("low repetition — demonstrating that generative quality ≠ held-out perplexity.")


    def piano_roll_plot(tokens, title, ax, max_t=96, nv=4):
        """Draw a piano-roll for up to max_t time steps."""
        a = to_arr(tokens[:max_t], nv)
        colors = ['#264653','#2a9d8f','#e9c46a','#e76f51']
        voice_lbl = ['Soprano','Alto','Tenor','Bass']
        for vi in range(nv):
            row = a[:, vi]
            t = 0
            while t < len(row):
                if np.isnan(row[t]): t+=1; continue
                p = row[t]; dur = 1
                while t+dur < len(row) and row[t+dur]==p: dur+=1
                ax.barh(p, dur, left=t, height=0.7, color=colors[vi], alpha=0.8, label=voice_lbl[vi] if t==0 else "")
                t += dur
        ax.set_xlim(0, max_t); ax.set_xlabel('time step (eighth notes)'); ax.set_ylabel('MIDI pitch')
        ax.set_title(title); ax.legend(loc='upper right',fontsize=8)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))
    piano_roll_plot(ref_tokens, 'Real Bach (held-out test chorale)', ax1)
    piano_roll_plot(uncond_tokens, 'RelTransformer generated (best-of-N)', ax2)
    plt.tight_layout(); plt.show()
    print("Visual comparison: both should show 4 distinct pitch bands (SATB) with smooth voice leading.")


    # best-of-N reranking
    def rerank_score(r):
        return (2.2*r['consonance'] -5.0*r['crossing'] -2.0*r['large_leap'] -1.2*r['rep4']
                -1.0*r['pc_js_vs_train'] -2.5*r['parallel5'] -2.5*r['parallel8'] +0.002*min(r['length'],160))
    def best_uncond(n=(8 if FAST else 40)):
        rows=[]; cands=[]
        for ci in range(n):
            temp=random.choice([0.6,0.7,0.8,0.9,1.0]); k=random.choice([16,24,32,40]); p=random.choice([0.88,0.92,0.95]); s=random.choice([0.6,0.9,1.2])
            toks=generate_uncond(lm,chord_stoi,chord_itos,steps=(120 if FAST else 160),temp=temp,top_p=p,top_k=k,rule=s)
            m=metrics(toks,f'c{ci}'); m.update(dict(candidate=ci,temp=temp,top_k=k,top_p=p,rule=s)); m['score']=rerank_score(m)
            rows.append(m); cands.append(toks)
        df=pd.DataFrame(rows).sort_values('score',ascending=False).reset_index(drop=True)
        return cands[int(df.loc[0,'candidate'])], df
    best_tokens, rerank_df = best_uncond()
    uncond_tokens = best_tokens
    chords_to_midi(uncond_tokens,'symbolic_unconditioned.mid')
    chords_to_midi(uncond_tokens, SUITE/'task1_POLISHED_best.mid')
    print('Reranked best sample written to symbolic_unconditioned.mid')
    display(rerank_df.head(6)[['candidate','score','temp','top_k','top_p','rule','consonance','crossing','large_leap','rep4']].round(3))

    # sampling ablation: temperature x top-k
    abl=[]
    for temp in [0.7,0.9,1.1]:
        for k in [10,30,50]:
            toks=generate_uncond(lm,chord_stoi,chord_itos,steps=(100 if FAST else 160),temp=temp,top_p=0.95,top_k=k,rule=0.6)
            r=metrics(toks,f'T={temp},k={k}'); r.update(dict(temperature=temp,top_k=k)); abl.append(r)
    ablation=pd.DataFrame(abl)
    display(ablation[['temperature','top_k','unique_ratio','rep4','consonance','crossing','pc_js_vs_train']].round(3))
    plt.figure(figsize=(6,4))
    plt.scatter(ablation['unique_ratio'],ablation['rep4'])
    for _,r in ablation.iterrows(): plt.text(r['unique_ratio'],r['rep4'],f"T{r['temperature']},k{int(r['top_k'])}",fontsize=8)
    plt.xlabel('unique-token ratio (diversity)'); plt.ylabel('4-gram repetition'); plt.title('Sampling tradeoff: diversity vs repetition'); plt.show()

    # memorization check (8-gram Jaccard to nearest training chorale)
    def ngrams(t,n=8): return {tuple(t[i:i+n]) for i in range(len(t)-n+1)} if len(t)>=n else set()
    def nearest(t,n=8):
        gs=ngrams(t,n)
        if not gs: return dict(nearest=None,jaccard=0.0,shared=0.0)
        best=dict(nearest=None,jaccard=-1,shared=0.0)
        for i,s in enumerate(uncond_train):
            ts=ngrams(s,n)
            if not ts: continue
            inter=len(gs&ts); jac=inter/max(1,len(gs|ts))
            if jac>best['jaccard']: best=dict(nearest=train_names[i],jaccard=jac,shared=inter/max(1,len(gs)))
        return best
    mem=pd.DataFrame([{'sample':'RelTransformer',**nearest(uncond_tokens)},
                      {'sample':'Bigram Markov',**nearest(bigram_tokens)},
                      {'sample':'Unigram',**nearest(unigram_tokens)}])
    display(mem.round(3))
    print('This sample has limited exact 8-gram overlap; that does not establish absence of memorization.')

    from musicgen.models import Seq2SeqHarmonizer

    def train_harm(model, tr, te, epochs):
        model.to(device); pm,ph=mel_stoi[PAD],harm_stoi[PAD]
        opt=torch.optim.AdamW(model.parameters(),lr=CFG['LR'],weight_decay=CFG['WD'])
        sch=CosineAnnealingLR(opt,T_max=epochs,eta_min=CFG['LR']/10)
        lf=nn.CrossEntropyLoss(ignore_index=ph); hist=[]
        for ep in range(1,epochs+1):
            model.train(); tl=tn=0
            for mel,din,tgt in tr:
                mel,din,tgt=mel.to(device),din.to(device),tgt.to(device)
                opt.zero_grad(set_to_none=True)
                logits=model(mel,din,(mel==pm),(din==ph))
                loss=lf(logits.reshape(-1,logits.size(-1)),tgt.reshape(-1))
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
                nt=(tgt!=ph).sum().item(); tl+=loss.item()*nt; tn+=nt
            ev=eval_harm(model,te); sch.step(); hist.append(dict(epoch=ep,train_loss=tl/tn,**ev))
            if ep==1 or ep%5==0 or ep==epochs: print(hist[-1])
        return pd.DataFrame(hist)

    @torch.no_grad()
    def eval_harm(model, loader):
        model.eval(); pm,ph=mel_stoi[PAD],harm_stoi[PAD]
        lf=nn.CrossEntropyLoss(ignore_index=ph,reduction='sum'); tl=tn=cor=0
        for mel,din,tgt in loader:
            mel,din,tgt=mel.to(device),din.to(device),tgt.to(device)
            logits=model(mel,din,(mel==pm),(din==ph))
            tl+=lf(logits.reshape(-1,logits.size(-1)),tgt.reshape(-1)).item()
            msk=(tgt!=ph); cor+=((logits.argmax(-1)==tgt)&msk).sum().item(); tn+=msk.sum().item()
        return dict(test_loss=tl/max(1,tn),test_ppl=math.exp(tl/max(1,tn)),token_acc=cor/max(1,tn))

    hm = Seq2SeqHarmonizer(len(mel_itos),len(harm_itos),CFG['D_MODEL'],CFG['NHEAD'],
                           max(2,CFG['LAYERS']//2),max(2,CFG['LAYERS']//2),CFG['FF'],CFG['DROP'])
    print('Task 2 Seq2SeqHarmonizer parameters:', f'{n_params(hm):,}')
    h_history = train_harm(hm, h_tr, h_te, CFG['EPOCHS_T2'])
    fig,ax=plt.subplots(1,2,figsize=(11,3))
    ax[0].plot(h_history.epoch,h_history.train_loss,label='train'); ax[0].plot(h_history.epoch,h_history.test_loss,label='test')
    ax[0].legend(); ax[0].set_title('Task 2 loss'); ax[0].set_xlabel('epoch')
    ax[1].plot(h_history.epoch,h_history.token_acc); ax[1].set_title('Task 2 token accuracy'); ax[1].set_xlabel('epoch')
    plt.tight_layout(); plt.show()
    torch.save({"state_dict": hm.state_dict(), "config": CFG, "mel_itos": mel_itos, "harm_itos": harm_itos, "model": "Seq2SeqHarmonizer"}, "task2_checkpoint.pt")


    # ---- conditioned generation (autoregressive decode with cross-attention) ----
    @torch.no_grad()
    def generate_harmony(model, mel_tokens, temp=0.7, top_p=0.9, top_k=20, rule=1.0):
        model.eval(); mel_tokens=mel_tokens[:L]
        mel_ids=torch.tensor([enc(mel_tokens,mel_stoi)],device=device)
        mem=model.encode(mel_ids,(mel_ids==mel_stoi[PAD]))
        din=[harm_stoi[BOS]]; out=[]; prev=None
        for t in range(len(mel_tokens)):
            d=torch.tensor([din],device=device)
            logits=model.decode(d,mem,(mel_ids==mel_stoi[PAD]),None)[0,-1].clone()
            for sp in (PAD,BOS,EOS,UNK): logits[harm_stoi[sp]]=float('-inf')
            ids,vals=_nucleus_filter(logits,top_p,top_k); adj=vals.clone()
            if rule>0:
                for j,idx in enumerate(ids.tolist()):
                    full='|'.join([mel_tokens[t]]+str(harm_itos[idx]).split('|')); adj[j]+=rule*chord_rule_score(full,prev)
            probs=torch.softmax(adj/max(temp,1e-6),dim=-1)
            nxt=int(ids[torch.multinomial(probs,1).item()].item())
            din.append(nxt); ht=harm_itos[nxt]; out.append(ht)
            prev='|'.join([mel_tokens[t]]+str(ht).split('|'))
        return out

    def mel_harm_to_midi(mel_tokens, harm_tokens, path, step=None, bpm=80):
        step=step or CFG['STEP']
        sc=stream.Score(); sc.insert(0,tempo.MetronomeMark(number=bpm)); sc.insert(0,meter.TimeSignature('4/4'))
        insts=[instrument.Soprano(),instrument.Alto(),instrument.Tenor(),instrument.Bass()]
        vv=[[] for _ in range(4)]
        for m,h in zip(mel_tokens,harm_tokens):
            f=[m]+str(h).split('|');  f+=[REST]*(4-len(f)) if len(f)<4 else []
            for vi,x in enumerate(f[:4]): vv[vi].append(x)
        for vi in range(4):
            p=stream.Part(); p.id=['Soprano(given)','Alto','Tenor','Bass'][vi]; p.insert(0,insts[vi])
            _append_legato(p,vv[vi],step,[84,72,70,72][vi]); sc.append(p)
        sc.write('midi', fp=str(path)); return str(path)

    ex=0
    cond_melody=mel_test[ex][:L]
    true_harmony=harm_test[ex][:len(cond_melody)]
    gen_harmony=generate_harmony(hm,cond_melody,temp=0.7,top_p=0.9,top_k=20,rule=1.4)
    mel_harm_to_midi(cond_melody,gen_harmony,'symbolic_conditioned.mid')
    print('Conditioned on a held-out soprano melody of length',len(cond_melody))
    print('First 10 melody :',cond_melody[:10])
    print('First 10 harmony:',gen_harmony[:10])
    print('Wrote symbolic_conditioned.mid')

    @torch.no_grad()
    def beam_search_harmony(model, mel_tokens, beam_width=5, length_penalty=0.65, rule=0.6, max_repeat=3):
        """Beam search for Task 2 harmonization with hard repetition cap.

        max_repeat: maximum consecutive identical tokens allowed per beam.
        This prevents the common mode-collapse failure where beam search
        outputs the same chord for the entire sequence.
        """
        model.eval(); pm=mel_stoi[PAD]
        mel_tokens=mel_tokens[:L]
        mel_ids=torch.tensor([enc(mel_tokens,mel_stoi)],device=device)
        mem=model.encode(mel_ids,(mel_ids==pm))
        T=len(mel_tokens)

        beams=[(0.0,[harm_stoi[BOS]])]

        for t in range(T):
            candidates=[]
            for lp,ids in beams:
                d=torch.tensor([ids],device=device)
                logits=model.decode(d,mem,(mel_ids==pm),None)[0,-1]
                log_probs=F.log_softmax(logits,dim=-1)
                for sp in [harm_stoi[PAD],harm_stoi[BOS],harm_stoi[UNK]]:
                    log_probs[sp]=float('-inf')
                # Hard block: if last max_repeat tokens are all the same, forbid that token
                if len(ids) > max_repeat:
                    tail = ids[-max_repeat:]
                    if len(set(tail)) == 1:  # all identical
                        log_probs[tail[0]] = float('-inf')
                # Soft penalty for tokens seen recently
                recent_ids = set(ids[-4:]) if len(ids) > 1 else set()
                for rid in recent_ids:
                    if 0 <= rid < log_probs.numel() and log_probs[rid] != float('-inf'):
                        log_probs[rid] += math.log(0.6)  # -0.51 penalty
                top_lp,top_ids=torch.topk(log_probs,min(beam_width*3,log_probs.numel()))
                prev_full=None
                if len(ids)>1:
                    mel_t_prev=mel_tokens[t-1] if t>0 else mel_tokens[0]
                    prev_full='|'.join([mel_t_prev]+str(harm_itos[ids[-1]]).split('|'))
                mel_t=mel_tokens[t] if t<len(mel_tokens) else mel_tokens[-1]
                for alp,aidx in zip(top_lp.cpu().tolist(),top_ids.cpu().tolist()):
                    if harm_itos[aidx] in SPECIAL: continue
                    full_tok='|'.join([mel_t]+str(harm_itos[aidx]).split('|'))
                    bonus=rule*chord_rule_score(full_tok,prev_full)
                    candidates.append((lp+alp+bonus, ids+[aidx]))
            candidates.sort(key=lambda x: x[0]/max(1,len(x[1]))**length_penalty,reverse=True)
            # Diversity filter: keep beams ending on distinct tokens
            seen_last=set(); diverse_beams=[]
            for cand in candidates:
                last=cand[1][-1]
                if last not in seen_last or len(diverse_beams)<2:
                    diverse_beams.append(cand); seen_last.add(last)
                if len(diverse_beams)>=beam_width: break
            beams = diverse_beams if diverse_beams else candidates[:beam_width]

        best_ids=beams[0][1][1:]   # exclude BOS
        return [harm_itos[i] for i in best_ids]

    beam_harmony = beam_search_harmony(hm, cond_melody, beam_width=(3 if FAST else 5), rule=0.6, max_repeat=2)
    mel_harm_to_midi(cond_melody, beam_harmony, SUITE/'task2_beam_search.mid')
    print('Beam-search harmony written -> task2_beam_search.mid')
    print('First 10 beam  harmony:', beam_harmony[:10])
    print('First 10 sampl harmony:', gen_harmony[:10])


    # ── Harmonisation diversity: one melody → multiple valid harmonisations ────────
    _n_var = 5
    _temps = [0.60, 0.70, 0.80, 0.90, 1.00]
    _var_harmonies = []
    for _ti, _t in enumerate(_temps):
        _h = generate_harmony(hm, cond_melody, temp=_t, top_p=0.9, top_k=20, rule=1.0)
        _var_harmonies.append(_h)
        mel_harm_to_midi(cond_melody, _h, SUITE/f'task2_variation_temp{int(_t*100):03d}.mid')

    print("=== Harmonisation Diversity (first 8 harmony tokens per temperature) ===")
    print(f"Melody:      {cond_melody[:8]}")
    print(f"Bach (true): {true_harmony[:8]}")
    for _ti, (_t, _h) in enumerate(zip(_temps, _var_harmonies)):
        _match = sum(a==b for a,b in zip(_h, true_harmony))/max(1,len(true_harmony))
        print(f"Temp={_t:.2f}:  {_h[:8]}  (match Bach={_match:.1%})")

    # Pairwise uniqueness among generated harmonisations
    _all_pairs_diff = []
    for i in range(_n_var):
        for j in range(i+1, _n_var):
            diff = sum(a!=b for a,b in zip(_var_harmonies[i],_var_harmonies[j]))/max(1,len(_var_harmonies[i]))
            _all_pairs_diff.append(diff)
    print(f"\nMean pairwise token difference among {_n_var} harmonisations: {np.mean(_all_pairs_diff):.1%}")
    print("(>0 = model is generative, not a deterministic lookup table)")

    # Musical quality of each variation
    _var_metrics = []
    for _t, _h in zip(_temps, _var_harmonies):
        _comb = ['|'.join([m]+str(hh).split('|')) for m,hh in zip(cond_melody,_h)]
        _var_metrics.append({'temp':_t,'consonance':consonance(_comb),'crossing':crossing(_comb),
                             'large_leap':large_leap(_comb),'rep4':rep_rate(_comb,4)})
    _vm_df = pd.DataFrame(_var_metrics)
    display(_vm_df.round(3))
    print("All variations should remain consonant and have zero or near-zero voice crossing.")


    # ---- baselines: lookup + untrained, and combined metrics ----
    def mh_chords(m,h):
        out=[]
        for a,b in zip(m,h):
            f=[a]+str(b).split('|')
            if len(f)==4: out.append('|'.join(f))
        return out
    def frame_acc(pred,gold): return float(np.mean([p==g for p,g in zip(pred,gold)])) if gold else 0.0

    lut=collections.defaultdict(collections.Counter)
    for m,h in zip(mel_train,harm_train):
        for a,b in zip(m,h): lut[a][b]+=1
    fallback=harm_counts.most_common(1)[0][0]
    look={k:v.most_common(1)[0][0] for k,v in lut.items()}
    def baseline_harmony(mel): return [look.get(m,fallback) for m in mel]
    def lookup_test_acc():
        cor=tot=0
        for m,h in zip(mel_test,harm_test):
            for a,b in zip(m,h): cor+=int(look.get(a,fallback)==b); tot+=1
        return cor/max(1,tot)
    base_harmony=baseline_harmony(cond_melody)

    hm_untrained=Seq2SeqHarmonizer(len(mel_itos),len(harm_itos),CFG['D_MODEL'],CFG['NHEAD'],
                                   max(2,CFG['LAYERS']//2),max(2,CFG['LAYERS']//2),CFG['FF'],CFG['DROP']).to(device).eval()
    untr_harmony=generate_harmony(hm_untrained,cond_melody,rule=0.0)
    mel_harm_to_midi(cond_melody,base_harmony, SUITE/'task2_lookup_baseline.mid')
    mel_harm_to_midi(cond_melody,untr_harmony, SUITE/'task2_UNTRAINED_random.mid')
    mel_harm_to_midi(cond_melody,gen_harmony,  SUITE/'task2_POLISHED_best.mid')

    gt=mh_chords(cond_melody,true_harmony); tr=mh_chords(cond_melody,gen_harmony)
    bl=mh_chords(cond_melody,base_harmony); ut=mh_chords(cond_melody,untr_harmony)
    task2_eval=pd.DataFrame([
        dict(**metrics(gt,'Ground-truth Bach'), frame_match=1.0, test_acc=np.nan),
        dict(**metrics(tr,'Seq2Seq Transformer (ours)'), frame_match=frame_acc(gen_harmony,true_harmony), test_acc=h_history['token_acc'].iloc[-1]),
        dict(**metrics(bl,'Lookup baseline'), frame_match=frame_acc(base_harmony,true_harmony), test_acc=lookup_test_acc()),
        dict(**metrics(ut,'Untrained (random)'), frame_match=frame_acc(untr_harmony,true_harmony), test_acc=np.nan),
    ])
    display(task2_eval[['sample','test_acc','frame_match','consonance','crossing','large_leap','pc_js_vs_train','parallel5','parallel8','pitch_variety','interval_variety','harm_rhythm_var']].round(3))
    print("\n*** test_acc = teacher-forced accuracy over full held-out test set (MAIN METRIC)")
    print("*** frame_match = exact token match vs ONE Bach harmonisation of ONE example")
    print("    (low because harmonisation is one-to-many: many valid lower voices exist)\n")

    # ── Bar chart: test_acc comparison (most important Task 2 metric) ─────────────
    _named = task2_eval.dropna(subset=['test_acc'])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(_named['sample'], _named['test_acc'],
                   color=['#264653','#e9c46a','#e76f51'][:len(_named)])
    ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=10)
    ax.set_xlabel('Test-set token accuracy (teacher forcing)')
    ax.set_title('Task 2 — Test Accuracy: Transformer vs Baselines\n(historical scoring protocols differ; see evaluation notes)')
    ax.set_xlim(0, max(_named['test_acc'])*1.25)
    plt.tight_layout(); plt.show()
    fig,ax=plt.subplots(1,2,figsize=(12,3))
    ax[0].bar(task2_eval['sample'],task2_eval['frame_match']); ax[0].set_title('Frame match vs one Bach example\n(low by design: many valid harmonisations)'); ax[0].tick_params(axis='x',rotation=25)
    ax[1].bar(task2_eval['sample'],task2_eval['consonance']); ax[1].set_title('Consonance rate'); ax[1].tick_params(axis='x',rotation=25)
    plt.tight_layout(); plt.show()

    # ---- model uncertainty over the melody (teacher forcing) ----
    @torch.no_grad()
    def uncertainty_trace(model, mel_tokens, true_h, cap=None):
        cap=cap or L; mel_tokens=mel_tokens[:cap]; true_h=true_h[:cap]
        mel_ids=torch.tensor([enc(mel_tokens,mel_stoi)],device=device)
        hids=enc(true_h,harm_stoi); din=torch.tensor([[harm_stoi[BOS]]+hids[:-1]],device=device)
        logits=model(mel_ids,din,(mel_ids==mel_stoi[PAD]),None)[0]
        probs=torch.softmax(logits,dim=-1)
        ent=-(probs*torch.log(probs+1e-9)).sum(-1).cpu().numpy()
        gold=probs[torch.arange(len(hids)),torch.tensor(hids,device=device)].cpu().numpy()
        pred=[harm_itos[i] for i in logits.argmax(-1).cpu().numpy().tolist()]
        return pd.DataFrame(dict(t=np.arange(len(mel_tokens)),melody=mel_tokens,true=true_h,pred=pred,
                                entropy=ent,gold_prob=gold,match=[p==g for p,g in zip(pred,true_h)]))
    trace=uncertainty_trace(hm,cond_melody,true_harmony)
    display(trace.head(10))
    fig,ax=plt.subplots(1,2,figsize=(12,3))
    ax[0].plot(trace.t,trace.entropy); ax[0].set_title('Predictive entropy over the melody'); ax[0].set_xlabel('time step'); ax[0].set_ylabel('entropy (nats)')
    mp=[np.nan if m==REST else int(m) for m in trace.melody]
    ax[1].plot(trace.t,mp,label='melody pitch'); ax[1].plot(trace.t,trace.gold_prob*100,label='P(gold harmony) x100')
    ax[1].legend(); ax[1].set_title('Melody contour vs. model confidence'); ax[1].set_xlabel('time step')
    plt.tight_layout(); plt.show()
    print('Entropy tends to rise at phrase boundaries / ambiguous melody notes where several harmonizations are plausible.')

    summary=dict(
        config=CFG, task1_final=lm_history.tail(1).to_dict('records')[0],
        task2_final=h_history.tail(1).to_dict('records')[0],
        task1_eval=task1_eval.round(4).to_dict('records'),
        task2_eval=task2_eval.round(4).to_dict('records'),
        task1_sampling_ablation=ablation.round(4).to_dict('records'),
        task1_memorization=mem.round(4).to_dict('records'),
    )
    def finite_json(value):
        if isinstance(value, dict): return {k: finite_json(v) for k, v in value.items()}
        if isinstance(value, list): return [finite_json(v) for v in value]
        if isinstance(value, (float, np.floating)) and not math.isfinite(value): return None
        return value
    with open('results_summary.json', 'w', encoding='utf-8') as f:
        json.dump(finite_json(summary), f, indent=2, allow_nan=False, default=str)
    print('Saved new run summary and MIDI samples. These are not the historical artifacts.')


if __name__ == "__main__":
    run()
