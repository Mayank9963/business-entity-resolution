"""Name / address normalisation.

Everything here is language-agnostic string processing plus small hand-written
abbreviation tables, so it applies equally to US, India and unseen countries (France).
Run as a script to normalise every source file of both splits into parquet.
"""
import re
import sys
import time
from functools import lru_cache
from multiprocessing import Pool

import polars as pl
from unidecode import unidecode

import translit
from config import N_WORKERS, norm_path
from io_utils import read_source

INDIC = re.compile(r"[ऀ-෿]")
NON_ALNUM = re.compile(r"[^a-z0-9]+")
DIGIT = re.compile(r"\d+")
ALPHA_DIGIT = re.compile(r"^(?=.*[a-z])(?=.*\d)[a-z0-9]+$")
# alternative-name markers: "X dba Y", "X formerly known as Y", "X f/k/a Y" -> main = Y, alt = X
ALT = re.compile(r"\b(?:doing business as|trading as|formerly known as|previously known as|"
                 r"now known as|also known as|formerly|d\s*[./]?\s*b\s*[./]?\s*a|"
                 r"f\s*[./]?\s*k\s*[./]?\s*a|a\s*[./]?\s*k\s*[./]?\s*a|t\s*/\s*a|nee)\b\.?")
DOMAIN = re.compile(r"(?:https?://)?(?:www\.)?\b([a-z0-9\-]+)\.(?:com|net|org|co\.in|in|co|fr|biz|info|us|io)\b")
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "8": "b", "@": "a"})

# ---------------------------------------------------------------- names
NAME_CANON = {
    "pvt": "private", "pvte": "private", "prv": "private", "priv": "private",
    "ltd": "limited", "ltda": "limited", "lim": "limited", "limted": "limited",
    "inc": "incorporated", "incorp": "incorporated", "corp": "corporation",
    "co": "company", "cos": "company", "comp": "company", "llc": "llc", "llp": "llp",
    "intl": "international", "natl": "national", "mfg": "manufacturing",
    "svc": "services", "svcs": "services", "srvcs": "services", "mgmt": "management",
    "assoc": "associates", "bros": "brothers", "ent": "enterprises",
    "ents": "enterprises", "tech": "technologies", "techs": "technologies",
    "technology": "technologies", "grp": "group", "hldgs": "holdings", "ctr": "center",
    "centre": "center", "dept": "department", "univ": "university",
    "ste": "societe", "soc": "societe", "st": "saint", "lnc": "incorporated",
    "lndia": "india", "et": "and", "frs": "freres",
}
# honorifics (neutral on train: same match rate as average). NOTE: words such as
# "eastgate" / "holdings" are NOT noise - they mark decoy records (see token_rates.py)
NAME_NOISE = {"smt", "shri", "sri", "mr", "mrs", "ms", "dr", "known"}
# fragments of invented brand names ("belohalonex", "korquo", "arconyx"), learned on train
MORPHEMES = {
    "aria", "avi", "belo", "brix", "calo", "delta", "drex", "ecto", "evo", "faye", "flux",
    "gild", "halo", "iri", "jax", "kelo", "kor", "lum", "lyra", "mira", "nex", "novi",
    "nyla", "onyx", "orbi", "pyra", "quo", "riza", "syn", "tavo", "umbra", "vera", "vio",
    "xylo", "yuma", "zeph", "zeta", "veo", "arc", "solio", "sol", "dova",
}
LEGAL = {
    "private", "limited", "incorporated", "corporation", "company", "llc", "llp",
    "plc", "lp", "pllc", "pc", "opc", "sarl", "sas", "sasu", "eurl", "sa", "sci",
    "snc", "scop", "selarl", "gmbh", "ag", "bv", "nv", "pte", "pty",
}
NAME_STOP = {"the", "and", "of", "a", "an", "de", "la", "le", "les", "du", "des",
             "et", "l", "d", "en", "for", "m", "s", "et"}

# ---------------------------------------------------------------- addresses
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl",
    "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in",
    "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me",
    "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc", "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj",
    "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka",
    "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od",
    "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "ts", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk",
    "west bengal": "wb", "delhi": "dl", "nct of delhi": "dl", "new delhi": "dl",
    "jammu and kashmir": "jk", "jammu & kashmir": "jk", "puducherry": "py",
    "pondicherry": "py", "chandigarh": "ch", "ladakh": "la", "lakshadweep": "ld",
    "andaman and nicobar islands": "an", "dadra and nagar haveli": "dn",
    "daman and diu": "dd",
}
FR_REGIONS = {
    "hauts de france": "hdf", "nouvelle aquitaine": "naq", "pays de la loire": "pdl",
    "ile de france": "idf", "auvergne rhone alpes": "ara", "bretagne": "bre",
    "normandie": "nor", "grand est": "ges", "occitanie": "occ", "corse": "cor",
    "provence alpes cote d azur": "pac", "bourgogne franche comte": "bfc",
    "centre val de loire": "cvl",
}
ADDR_CANON = {
    "street": "st", "str": "st", "strt": "st", "road": "rd", "avenue": "ave",
    "av": "ave", "avn": "ave", "drive": "dr", "drv": "dr", "boulevard": "blvd",
    "bd": "blvd", "bld": "blvd", "boul": "blvd", "lane": "ln", "court": "ct",
    "place": "pl", "circle": "cir", "highway": "hwy", "parkway": "pkwy",
    "terrace": "ter", "trail": "trl", "square": "sq", "expressway": "expy",
    "freeway": "fwy", "turnpike": "tpke", "crossing": "xing", "point": "pt",
    "center": "ctr", "centre": "ctr", "mount": "mt", "fort": "ft", "heights": "hts",
    "north": "n", "south": "s", "east": "e", "west": "w", "so": "s",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "r": "rue", "all": "allee", "ch": "chemin", "chem": "chemin", "imp": "impasse",
    "rte": "route", "qu": "quai", "fbg": "faubourg", "sq": "sq",
    "ngr": "nagar", "col": "colony", "sec": "sector", "sect": "sector",
    "opp": "opposite", "nr": "near", "bldg": "building", "mkt": "market",
    "apts": "apartment", "apt": "apartment", "appt": "apartment", "flr": "floor",
    "fl": "floor", "ste": "suite", "dist": "district", "distt": "district",
    "bombay": "mumbai", "calcutta": "kolkata", "madras": "chennai",
    "bangalore": "bengaluru", "gurgaon": "gurugram",
}
ADDR_STOP = {
    "no", "door", "h", "house", "plot", "the", "of", "de", "la", "le", "les", "du",
    "des", "d", "l", "et", "po", "box", "near", "opposite", "behind", "unit",
    "suite", "apartment", "floor", "building", "and", "at", "via", "bis", "ter",
    "a", "b", "c", "none", "nan", "india", "usa", "us", "france", "greater",
}
STATE_PHRASES = sorted({**US_STATES, **IN_STATES, **FR_REGIONS}.items(),
                       key=lambda kv: -len(kv[0]))
STATE_LOOKUP = dict(STATE_PHRASES)
STATE_CODES = set(US_STATES.values()) | set(IN_STATES.values())
ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th|er|eme|e)\b")

_NAME_MAP, _SEG_MAP = {}, {}


def _init_maps():
    global _NAME_MAP, _SEG_MAP
    _NAME_MAP, _SEG_MAP = translit.load()


def to_ascii(s: str) -> str:
    return s if s.isascii() else unidecode(s)


_SKEL_TABLE = str.maketrans("cqzwgv", "kksvjb")


@lru_cache(maxsize=2_000_000)
def skeleton(tok: str) -> str:
    """Crude phonetic key: merge similar consonants, drop vowels/h/y, collapse repeats.
    'shakti' / 'shkti' -> 'shkt' ; 'latex' / 'loatx' -> 'ltks'."""
    t = tok.replace("ph", "f").replace("ck", "k").replace("x", "ks").translate(_SKEL_TABLE)
    out, prev = [], ""
    for ch in t:
        if ch in "aeiouyh":
            continue
        if ch != prev:
            out.append(ch)
        prev = ch
    return "".join(out)


def is_synthetic(tok: str) -> bool:
    """True if tok is a concatenation of >= 2 invented-name fragments."""
    n = len(tok)
    if n < 5 or not tok.isalpha():
        return False
    best = [0] + [-1] * n  # best[i] = pieces needed to cover tok[:i]
    for i in range(1, n + 1):
        for j in range(max(0, i - 5), i - 1):
            if best[j] >= 0 and tok[j:i] in MORPHEMES:
                best[i] = best[j] + 1
                break
    return best[n] >= 2


def _translit_name(s: str) -> str:
    if not INDIC.search(s):
        return s
    toks = []
    for t in s.split():
        k = translit.PUNCT.sub("", t)
        toks.append(_NAME_MAP.get(k, t) if INDIC.search(t) else t)
    return " ".join(toks)


def _name_tokens(s: str):
    s = DOMAIN.sub(lambda m: m.group(1).replace("-", " "), s)
    s = s.replace("&", " and ").replace("+", " and ").replace("'", "")
    toks = []
    for t in NON_ALNUM.sub(" ", s).split():
        if ALPHA_DIGIT.match(t) and sum(c.isalpha() for c in t) > sum(c.isdigit() for c in t):
            t = t.translate(LEET)  # 'onc0logy' -> 'oncology'
        toks.append(NAME_CANON.get(t, t))
    # merge runs of single letters: "l l c" -> llc, "s a s" -> sas
    out, run = [], []
    for t in toks + [""]:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            m = "".join(run)
            out.append(NAME_CANON.get(m, m) if len(run) > 1 else m)
            run = []
        if t:
            out.append(t)
    return out


def _core(toks):
    return [t for t in toks if t not in LEGAL and t not in NAME_STOP and t not in NAME_NOISE
            and not t.isdigit()]


def norm_name(raw: str):
    s = to_ascii(_translit_name(raw)).lower().strip()
    main, alt = s, ""
    parts = [x for x in ALT.split(s) if x.strip(" :-,")]
    altflag = int(len(parts) > 1)
    if altflag:
        main, alt = parts[-1], " ".join(parts[:-1])
    toks = [t for t in _name_tokens(main) if t not in NAME_NOISE]
    legal = sorted({t for t in toks if t in LEGAL})
    synth = sum(is_synthetic(t) for t in toks)
    core = [t for t in _core(toks) if not is_synthetic(t)]
    if not core and not synth:
        core = [t for t in toks if t not in NAME_STOP] or toks
    alt_toks = _name_tokens(alt)
    synth += sum(is_synthetic(t) for t in alt_toks)
    alt_core = [t for t in _core(alt_toks) if not is_synthetic(t)]
    return (" ".join(toks), " ".join(core), " ".join(sorted(core)), " ".join(legal),
            " ".join(alt_core), synth, altflag)


STATE_SEG = re.compile(r"^(" + "|".join(re.escape(k) for k, _ in STATE_PHRASES)
                       + "|" + "|".join(sorted(STATE_CODES)) + r")(?: \d{5,6})?$")


def norm_addr(raw: str):
    state, kept = "", []
    for seg in raw.split(","):
        seg = seg.strip().strip('"').strip()
        if INDIC.search(seg):
            seg = _SEG_MAP.get(seg, seg)
        seg = NON_ALNUM.sub(" ", to_ascii(seg).lower().replace("&", " and ")).strip()
        m = STATE_SEG.match(seg)
        if m:  # whole segment is a state / region (optionally followed by a postcode)
            state = STATE_LOOKUP.get(m.group(1), m.group(1))
            seg = seg[m.end(1):]
        kept.append(seg)
    s = ORDINAL.sub(lambda m: m.group(1), " ".join(kept))
    toks, nums = [], []
    for t in s.split():
        if t.isdigit():
            nums.append(t.lstrip("0") or "0")
            continue
        if any(c.isdigit() for c in t):
            nums.extend(d.lstrip("0") or "0" for d in DIGIT.findall(t))
            t = DIGIT.sub("", t)
            if len(t) < 2:
                continue
        t = ADDR_CANON.get(t, t)
        if t in ADDR_STOP:
            continue
        toks.append(t)
    post = next((n for n in nums if len(n) in (5, 6)), "")
    return (" ".join(toks), " ".join(sorted(set(toks))), " ".join(sorted(set(nums))),
            post, state)


def _norm_rows(rows):
    out = []
    for name, addr in rows:
        out.append(norm_name(name) + norm_addr(addr))
    return out


COLS = ["name_n", "name_core", "name_sorted", "legal", "name_alt", "synth", "altflag",
        "addr_n", "addr_set", "nums", "postcode", "state"]


def normalize_df(df: pl.DataFrame, workers: int = N_WORKERS, chunk: int = 50_000) -> pl.DataFrame:
    rows = list(zip(df["business_name"].to_list(), df["business_address"].to_list()))
    chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    with Pool(workers, initializer=_init_maps) as pool:
        res = [r for part in pool.imap(_norm_rows, chunks) for r in part]
    normed = pl.DataFrame(res, schema=COLS, orient="row")
    return pl.concat([df.select("entity_id", "country"), normed], how="horizontal_extend")


def run(targets=("train", "test"), chunk_rows: int = 2_000_000):
    """targets: split names ("train") or single files ("train:3"). Rows are processed
    in chunks so peak memory stays bounded on a 16 GB machine."""
    jobs = []
    for t in targets:
        split, _, src = t.partition(":")
        jobs += [(split, int(src))] if src else [(split, s) for s in (1, 2, 3)]
    for split, s in jobs:
        t0 = time.time()
        df = read_source(split, s)
        path = norm_path(split, s)
        parts = []
        for i, start in enumerate(range(0, df.height, chunk_rows)):
            part = path.with_name(f"{path.stem}.part{i}.parquet")
            normalize_df(df.slice(start, chunk_rows)).write_parquet(part)
            parts.append(part)
        pl.concat([pl.read_parquet(p) for p in parts]).write_parquet(path)
        for p in parts:
            p.unlink()
        print(f"[normalize] {split} s{s}: {df.height:,} rows in {time.time() - t0:.0f}s",
              flush=True)


if __name__ == "__main__":
    run(sys.argv[1:] or ("train", "test"))
