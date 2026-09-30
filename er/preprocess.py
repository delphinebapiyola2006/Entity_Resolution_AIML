import re, unicodedata
LEGAL = {"pvt","private","ltd","limited","inc","incorporated","llc","llp","corp","corporation","co","company","plc","gmbh","sa","the","and","of"}
SYN = {"technology":"tech","technologies":"tech","systems":"system","solutions":"solution","services":"service","sol":"solution",
       "industries":"industry","enterprises":"enterprise","international":"intl","labs":"lab","softwares":"soft","software":"soft"}
ADDR = {"rd":"road","st":"street","ave":"avenue","blvd":"boulevard","ln":"lane","hwy":"highway","dr":"drive","blk":"block",
        "fl":"floor","flr":"floor","bldg":"building","apt":"apartment","ste":"suite","opp":"opposite","nr":"near","sq":"square","pl":"place","ct":"court"}
CTRY = {"in":"india","ind":"india","bharat":"india","us":"usa","united states":"usa","u s a":"usa","uk":"united kingdom","gb":"united kingdom","uae":"united arab emirates","sg":"singapore"}

def _base(s):
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii","ignore").decode().lower()
    s = s.replace("&"," and ")
    s = re.sub(r"\b([a-z])\.\s*(?=[a-z]\b)", r"\1", s)
    s = re.sub(r"[^a-z0-9 ]"," ", s)
    s = re.sub(r"\b([a-z]) (?=[a-z]\b)", r"\1", s)
    return re.sub(r"\s+"," ", s).strip()

def norm_name(s):
    t = [SYN.get(w,w) for w in _base(s).split()]
    core = [w for w in t if w not in LEGAL]
    return " ".join(core or t)

def norm_addr(s):
    return " ".join(ADDR.get(w,w) for w in _base(s).split())

def norm_country(s):
    b = _base(s)
    return CTRY.get(b,b)
