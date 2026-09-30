"""Synthetic dataset generator: 3 noisy views of the same businesses + hard negatives."""
import random, csv, os
FIRST=["Apex","Blue","Crest","Delta","Evergreen","Falcon","Global","Helix","Indus","Jade","Kaveri","Lotus","Metro","Nova","Orion","Pioneer","Quantum","Royal","Sigma","Titan","Urban","Vertex","Zenith","Aster","Bharat","Coral","Dynamo","Eco","Fusion","Galaxy","Horizon","Infinity","Jupiter","Kinetic","Lumina"]
SECOND=["Technologies","Solutions","Systems","Industries","Enterprises","Logistics","Foods","Textiles","Pharma","Motors","Software","Consulting","Engineering","Retail","Energy","Labs","Trading","Builders","Networks","Materials"]
SUF=["Pvt Ltd","Private Limited","Ltd","Limited","Inc","LLC","Corp","& Co","Company"]
STREETS=["MG Road","Anna Salai","Park Street","Main Street","Church Road","Station Road","Lake View Avenue","Industrial Estate Road","Gandhi Nagar","Residency Road","Brigade Road","High Street","Market Lane","Ring Road"]
CITIES={"India":["Bangalore","Chennai","Coimbatore","Mumbai","Delhi","Hyderabad","Pune"],"USA":["Austin","Boston","Seattle","Denver"],"United Kingdom":["London","Leeds","Bristol"],"Singapore":["Singapore"]}
ABBR=[("Road","Rd"),("Street","St"),("Avenue","Ave"),("Technologies","Tech"),("Solutions","Sol"),("Private Limited","Pvt Ltd"),("Pvt Ltd","Private Limited"),("Limited","Ltd"),("MG","M.G."),("&","and")]
def typo(s,r):
    if len(s)<5 or r.random()>.5: return s
    i=r.randrange(1,len(s)-1); m=r.choice(["del","swap","dup"])
    return s[:i]+s[i+1:] if m=="del" else (s[:i]+s[i+1]+s[i]+s[i+2:] if m=="swap" else s[:i]+s[i]+s[i:])
def noisy_name(n,r,lvl):
    for a,b in ABBR:
        if a in n and r.random()<.4*lvl: n=n.replace(a,b)
    w=n.split()
    if lvl>=1.5 and len(w)>2 and r.random()<.6: w=w[:-1] if r.random()<.5 else w[:2]
    n=" ".join(w)
    if r.random()<.3*lvl: n=typo(n,r)
    if r.random()<.2: n=n.upper()
    return n
def noisy_addr(a,city,r,lvl):
    for x,y in ABBR:
        if x in a and r.random()<.5*lvl: a=a.replace(x,y)
    if lvl>=1.5 and r.random()<.4: a=" ".join(a.split()[1:]) or a
    if r.random()<.5: a=f"{a}, {city}"
    if r.random()<.25*lvl: a=typo(a,r)
    return a
FIRST += ["Amber","Birch","Cedar","Drift","Ember","Flint","Garnet","Harbor","Ivory","Juniper","Karma","Lunar","Maple","Nimbus","Onyx","Prime","Quartz","Ridge","Solar","Terra","Unity","Vivid","Willow","Yonder","Zephyr","Atlas","Beacon","Cobalt","Dawn","Echo","Forge"]
MID = ["Smart","Digital","Prime","Micro","Allied","United","National","Modern","Precision","Advanced","Green","Rapid","Bright","Elite","Universal","Dynamic","Creative","Integrated","Applied","Pacific"]
SECOND += ["Healthcare","Chemicals","Steel","Media","Security","Aerospace","Marine","Agro","Telecom","Mobility"]
def make_all(out="data", seed=7, n1=25000, n2=15000, n3=15000, frac=(.70, .15, .15)):
    """25k/15k/15k universe, split by Source-1 entity into train/validation/test (links & distractors stay in the same split)."""
    r = random.Random(seed); E = []
    for _ in range(n1):
        c = r.choices(list(CITIES), [.6, .2, .12, .08])[0]; mid = r.choice(MID) + " " if r.random() < .6 else ""
        E.append((f"{r.choice(FIRST)} {mid}{r.choice(SECOND)} {r.choice(SUF)}", f"{r.randint(1,220)} {r.choice(STREETS)}", r.choice(CITIES[c]), c))
    order = list(range(n1)); r.shuffle(order); a, b = int(n1*frac[0]), int(n1*(frac[0]+frac[1]))
    sp = {i: ("train" if p < a else "validation" if p < b else "test") for p, i in enumerate(order)}
    m = {2: set(r.sample(range(n1), int(n2*.73))), 3: set(r.sample(range(n1), int(n3*.6)))}; N = {2: n2, 3: n3}; L = {2: 1, 3: 1.6}
    R = {2: [], 3: []}
    for k in (2, 3):
        for i in m[k]:
            nm, ad, city, c = E[i]; R[k].append((i, noisy_name(nm, r, L[k]), noisy_addr(ad, city, r, L[k]), c, True))
        for _ in range(N[k] - len(m[k])):          # hard negatives: siblings / same name elsewhere / random
            i = r.randrange(n1); nm, ad, city, c = E[i]; t = r.random(); w = nm.split()
            if t < .5: nn, ci = f"{w[0]} {r.choice(SECOND)} {r.choice(SUF)}", city
            elif t < .8: nn, ci = nm, r.choice(CITIES[c])
            else: nn, ci = f"{r.choice(FIRST)} {r.choice(SECOND)} {r.choice(SUF)}", r.choice(CITIES[c])
            R[k].append((i, noisy_name(nn, r, L[k]), noisy_addr(f"{r.randint(1,220)} {r.choice(STREETS)}", ci, r, L[k]), c, False))
    gt = {i: [] for i in range(n1)}; files = {s: {1: [], 2: [], 3: []} for s in ("train", "validation", "test")}
    for i, (nm, ad, city, c) in enumerate(E): files[sp[i]][1].append((f"S1-{i+1:05d}", nm, f"{ad}, {city}", c))
    for k in (2, 3):
        r.shuffle(R[k])
        for n, (i, nm, ad, c, ok) in enumerate(R[k], 1):
            rid = f"S{k}-{n:05d}"; files[sp[i]][k].append((rid, nm, ad, c))
            if ok: gt[i].append(rid)
    for s in files:
        d = os.path.join(out, s); os.makedirs(d, exist_ok=True)
        for k in (1, 2, 3):
            with open(f"{d}/{s}_source{k}.tsv", "w", newline="") as f:
                w = csv.writer(f, delimiter="\t"); w.writerow(["entity_id", "name", "address", "country"]); w.writerows(files[s][k])
        with open(f"{d}/{s}_ground_truth.tsv", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t"); w.writerow(["source1_entity_id", "matched_entity_ids"])
            for row in files[s][1]: w.writerow([row[0], ",".join(gt[int(row[0][3:]) - 1])])
    print({s: {k: len(v) for k, v in files[s].items()} for s in files})
if __name__ == "__main__": make_all()
