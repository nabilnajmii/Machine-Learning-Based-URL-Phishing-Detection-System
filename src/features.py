# src/features.py
# ------------------------------------------------------------
# Enhanced lexical feature extractor for phishing URL detection
# (No web requests; safe + fast)
#This module takes a raw URL string and converts it into a set of numeric lexical features such as length,
# number of digits, entropy, presence of HTTPS, suspicious characters, brand mismatch, etc.
#These features are used as input to the machine learning model to classify phishing vs legitimate URLs.”
# ------------------------------------------------------------

# src/features.py
# ------------------------------------------------------------
# Enhanced lexical feature extractor for phishing URL detection
# (No web requests; safe + fast)
# ------------------------------------------------------------

import re
import math
from urllib.parse import urlparse, parse_qs
import tldextract

URL_SHORTENERS = {
    "bit.ly","goo.gl","t.co","ow.ly","tinyurl.com","is.gd","buff.ly","cutt.ly",
    "rebrand.ly","adf.ly","lnkd.in","shorte.st","trib.al","soo.gd","s.id"
}

SUSPICIOUS_KEYWORDS = {
    "login","secure","verify","update","confirm","account","bank","pay",
    "signin","password","reset","unlock","credential","invoice"
}

SUSPICIOUS_TLDS = {"xyz","top","gq","tk","ml","cf"}

def _ensure_scheme(url: str) -> str:
    if not re.match(r'^[a-zA-Z]+://', url):
        return 'http://' + url
    return url

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(s)
    return -sum((c/length) * math.log2(c/length) for c in freq.values())

def is_ipv4(host: str) -> int:
    return int(re.fullmatch(r'(?:\d{1,3}\.){3}\d{1,3}', host) is not None)

def count_digits(s: str) -> int:
    return sum(ch.isdigit() for ch in s)

def count_specials(s: str) -> int:
    return sum(ch in "@-_.$%!*" for ch in s)

def num_subdomains(ext) -> int:
    return 0 if not ext.subdomain else len(ext.subdomain.split('.'))

def has_https(scheme: str) -> int:
    return int(scheme.lower() == "https")

def extract_features(url: str) -> dict:
    raw_url = url.strip()
    url = _ensure_scheme(raw_url)
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path_plus_query = parsed.path + (("?" + parsed.query) if parsed.query else "")
    ext = tldextract.extract(url)

    domain_full = ".".join([p for p in [ext.domain, ext.suffix] if p])
    tld = (ext.suffix or "").lower()

    query_params = parse_qs(parsed.query)
    num_params = len(query_params)

    is_shortener = int(domain_full in URL_SHORTENERS)
    keyword_hits = sum(kw in url.lower() for kw in SUSPICIOUS_KEYWORDS)
    has_suspicious_tld = int(tld in SUSPICIOUS_TLDS)

    url_trim = url[:2048]
    entropy_url = shannon_entropy(url_trim)
    entropy_path = shannon_entropy(path_plus_query[:1024])
    entropy_host = shannon_entropy(host)

    # --- New feature 1: number of suspicious characters in URL ---
    suspicious_chars = "@%$?&=/#"
    num_suspicious_chars = sum(url.count(ch) for ch in suspicious_chars)

    # --- New feature 2: brand/domain mismatch ---
    # If the URL contains known brand keywords, but the registered domain
    # does NOT contain that brand, flag as a mismatch.
    url_lower = url.lower()
    domain_part = (ext.domain or "").lower()

    known_brands = [
        "paypal", "google", "facebook", "apple", "microsoft",
        "bank", "maybank", "cimb", "spotify", "instagram", "twitter",
    ]

    brands_in_url = [b for b in known_brands if b in url_lower]
    brands_in_domain = [b for b in known_brands if b in domain_part]
    if brands_in_url and not brands_in_domain:
        brand_mismatch = 1
    else:
        brand_mismatch = 0

    feats = {
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(path_plus_query),

        "num_dots_host": host.count("."),
        "num_subdomains": num_subdomains(ext),
        "num_params": num_params,
        "num_digits_url": count_digits(url),
        "num_specials_url": count_specials(url),

        "has_https": has_https(parsed.scheme),
        "has_at": int("@" in url),
        "has_hyphen_host": int("-" in host),
        "is_ip_host": is_ipv4(host),
        "is_shortener": is_shortener,
        "has_suspicious_tld": has_suspicious_tld,

        "keyword_hits": keyword_hits,

        "entropy_url": entropy_url,
        "entropy_path": entropy_path,
        "entropy_host": entropy_host,

        "subdomain_len": len(ext.subdomain),
        "domain_len": len(ext.domain),
        "suffix_len": len(ext.suffix),

        # New features:
        "num_suspicious_chars": num_suspicious_chars,
        "brand_mismatch": brand_mismatch,
    }
    return feats

def feature_order():
    return [
        "url_length",
        "host_length",
        "path_length",
        "num_dots_host",
        "num_subdomains",
        "num_params",
        "num_digits_url",
        "num_specials_url",
        "has_https",
        "has_at",
        "has_hyphen_host",
        "is_ip_host",
        "is_shortener",
        "has_suspicious_tld",
        "keyword_hits",
        "entropy_url",
        "entropy_path",
        "entropy_host",
        "subdomain_len",
        "domain_len",
        "suffix_len",
        "num_suspicious_chars",
        "brand_mismatch",
    ]

def to_vector(features: dict):
    return [features[k] for k in feature_order()]

