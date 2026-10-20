import os, time, json, base64, mimetypes
from email.message import EmailMessage
import re, html as _html
import pandas as pd
import io, difflib
import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from ddgs import DDGS
from google import genai
from google.genai import types
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

load_dotenv()
SCOPES = ["https://www.googleapis.com/auth/gmail.compose"]
COLS = ["company", "founder_name", "email", "website", "notes"]
MODEL = "gemini-3.8-flash"  # if you get a "model not found" error, copy the current Flash name from AI Studio

st.set_page_config(page_title="Job Outreach", layout="wide")
st.title("Job outreach drafter")

def clean(v):
    return "" if pd.isna(v) else str(v).strip()

# ---------- INPUT (flexible schema mapping) ----------
SYN = {
    "company": ["company", "company name", "organisation", "organization", "startup",
                "startup name", "firm", "employer", "business", "name of company"],
    "founder_name": ["founder", "founder name", "name", "contact name", "contact",
                     "person", "ceo", "full name", "owner", "hiring manager"],
    "email": ["email", "e mail", "email id", "emailid", "mail", "contact email",
              "email address", "mail id"],
    "website": ["website", "url", "link", "domain", "site", "web",
                "company website", "homepage"],
    "notes": ["notes", "note", "description", "about", "remarks", "comments",
              "funding", "details"],
}
ALL_SYN = {s for v in SYN.values() for s in v}
MAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL_RE = re.compile(r"^(https?://|www\.)|^[\w-]+(\.[\w-]+)*\.(com|io|ai|co|in|app|dev|org|net|xyz)(/|$)", re.I)

def norm(s):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(s).lower()).split())

def is_hdr(v):
    n = norm(v)
    return bool(n) and (n in ALL_SYN or bool(difflib.get_close_matches(n, ALL_SYN, n=1, cutoff=0.85)))

def read_tables(f):
    data = f.getvalue()
    if f.name.lower().endswith(".csv"):
        for enc in ("utf-8-sig", "latin-1"):
            try:
                return {"csv": pd.read_csv(io.BytesIO(data), header=None, dtype=str,
                                           encoding=enc, sep=None, engine="python")}
            except UnicodeDecodeError:
                continue
    return pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, dtype=str)

def apply_header(raw):
    raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if raw.empty:
        return raw
    hdr = None
    for i in range(min(10, len(raw))):          # header may not be on row 1
        cells = [str(v) for v in raw.iloc[i].dropna()]
        if any(is_hdr(v) for v in cells) and not any("@" in v for v in cells):
            hdr = i
            break
    if hdr is None:                              # headerless file
        names, body = [f"col_{j}" for j in range(raw.shape[1])], raw
    else:
        names = []
        for j, v in enumerate(raw.iloc[hdr]):
            n = str(v).strip() if pd.notna(v) else f"col_{j}"
            while n in names:
                n += "_"
            names.append(n)
        body = raw.iloc[hdr + 1:]
    body = body.copy()
    body.columns = names
    return body.reset_index(drop=True)

def rule_map(df):
    cols, m, used = list(df.columns), {}, set()
    for f, syns in SYN.items():                  # layer 1: names + fuzzy
        for c in cols:
            n = norm(c)
            if c not in used and (n in syns or difflib.get_close_matches(n, syns, n=1, cutoff=0.85)):
                m[f] = c; used.add(c); break
    def share(c, test):                          # layer 2: look at the values
        v = df[c].dropna().astype(str).head(30)
        return (sum(bool(test(x)) for x in v) / len(v)) if len(v) else 0
    if "email" not in m:
        for c in cols:
            if c not in used and share(c, MAIL_RE.search) >= 0.5:
                m["email"] = c; used.add(c); break
    if "website" not in m:
        for c in cols:
            if c not in used and share(c, lambda x: URL_RE.match(x.strip()) and "@" not in x) >= 0.5:
                m["website"] = c; used.add(c); break
    return m

def llm_map(df):                                 # layer 3: Gemini fallback
    prompt = f"""Map spreadsheet columns to these fields: company, founder_name, email, website, notes.
Columns and 3 sample rows (CSV):
{df.head(3).to_csv(index=False)}
Return ONLY JSON like {{"company": "<exact column name or null>", "founder_name": null, "email": null, "website": null, "notes": null}}."""
    c = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    resp = c.models.generate_content(
        model=MODEL, contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"))
    out = json.loads(resp.text)
    return {k: v for k, v in out.items() if k in SYN and v in list(df.columns)}

def auto_map(df, label):
    key = "map::" + label + "|" + "|".join(df.columns)
    if key in st.session_state:
        return st.session_state[key]
    m = rule_map(df)
    if "company" not in m or "email" not in m:   # only call LLM when rules fall short
        try:
            for k, v in llm_map(df).items():
                if k not in m and v not in m.values():
                    m[k] = v
        except Exception:
            pass
    if "company" not in m:                       # last resort: first unused column
        free = [c for c in df.columns if c not in m.values()]
        if free:
            m["company"] = free[0]
    st.session_state[key] = m
    return m

def standardize(df, m):
    out = pd.DataFrame({f: (df[m[f]].fillna("").astype(str).str.strip() if m.get(f) else "")
                        for f in COLS}, index=df.index)
    raw_email = out["email"].copy()              # handles "Name <a@b.com>" cells
    out["email"] = raw_email.map(lambda x: MAIL_RE.search(x).group(0).lower() if MAIL_RE.search(x) else "")
    nm = raw_email.map(lambda x: x.split("<")[0].strip(' "') if "<" in x else "")
    out["founder_name"] = out["founder_name"].where(out["founder_name"] != "", nm)
    return out

files = st.file_uploader("Upload CSV / Excel (any column names, multiple files OK)",
                         type=["csv", "xlsx"], accept_multiple_files=True)
pasted = st.text_area("Or paste company names, one per line (optional)")

frames = []
for f in files or []:
    for sheet, raw in read_tables(f).items():
        d0 = apply_header(raw)
        if d0.empty:
            continue
        label = f"{f.name} / {sheet}"
        guess = auto_map(d0, label)
        final_map = {}
        with st.expander(f"Column mapping: {label}", expanded=True):
            opts = ["(none)"] + list(d0.columns)
            cc = st.columns(len(COLS))
            for col, field in zip(cc, COLS):
                idx = opts.index(guess[field]) if guess.get(field) in opts else 0
                pick = col.selectbox(field, opts, index=idx, key=f"{label}-{field}")
                final_map[field] = None if pick == "(none)" else pick
        frames.append(standardize(d0, final_map))

if pasted.strip():
    names = [n.strip() for n in pasted.splitlines() if n.strip()]
    frames.append(standardize(pd.DataFrame({"company": names}), {"company": "company"}))

if not frames:
    st.info("Upload files or paste company names to start.")
    st.stop()

df = pd.concat(frames, ignore_index=True)[COLS].fillna("")
df = df[(df["company"] != "") | (df["email"] != "")]
df["company"] = df.apply(            # blank company -> guess from email domain
    lambda r: r["company"] or (r["email"].split("@")[1].split(".")[0].title() if r["email"] else ""),
    axis=1)
key = df["email"].where(df["email"] != "", df["company"].str.lower())
df = df[~key.duplicated()].reset_index(drop=True)   # dedupe across files

st.subheader("Fix anything here before generating")
edited = st.data_editor(df, num_rows="dynamic", use_container_width=True)

# ---------- RESEARCH + LLM ----------
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def find_website(company):
    try:
        res = DDGS().text(f"{company} official website", max_results=3)
        return res[0]["href"] if res else ""
    except Exception:
        return ""

def get_context(company, website):
    parts = []
    if website:
        try:
            html = requests.get(website, timeout=10, headers={"User-Agent": "Mozilla/5.0"}).text
            text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
            parts.append("WEBSITE TEXT: " + text[:5000])
        except Exception:
            pass
    try:
        hits = DDGS().text(f"{company} founder startup funding", max_results=5)
        parts.append("SEARCH SNIPPETS: " + " | ".join(h["body"] for h in hits))
    except Exception:
        pass
    return "\n".join(parts)

def get_models():
    return ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
            "gemini-3.5-flash", "gemini-flash-latest", "gemini-3.5-flash-lite"]

def generate(company, known_name, notes, context):
    guidelines = open("guidelines.txt", encoding="utf-8").read()
    prompt = f"""{guidelines}

Company: {company}
Founder name (may be blank): {known_name}
My notes: {notes}
Context:
{context}

Return ONLY JSON with keys:
"founder_first_name": first name of a founder ONLY if clearly stated in the context or above, else "",
"company_line": the 2 sentences."""
    last_err = None
    for model in get_models():
        for wait in (0, 5, 12):
            time.sleep(wait)
            try:
                resp = client.models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json"))
                return json.loads(resp.text)
            except Exception as e:
                last_err = e
                if "404" in str(e) or "NOT_FOUND" in str(e):
                    break
    raise last_err

# ---------- GMAIL ----------
def gmail_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            creds = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES).run_local_server(port=0)
        open("token.json", "w").write(creds.to_json())
    return build("gmail", "v1", credentials=creds)

def linkify(s):
    return re.sub(r"(https?://[^\s<]+)", r'<a href="\1">\1</a>', s)

def to_html(text):
    out, in_list = [], False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("- "):
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append(f"<li style='margin-bottom:6px'>{linkify(_html.escape(s[2:]))}</li>")
        else:
            if in_list:
                out.append("</ul>"); in_list = False
            if s:
                out.append(f"<p style='margin:0 0 10px'>{linkify(_html.escape(s))}</p>")
    if in_list:
        out.append("</ul>")
    return "".join(out)

def build_email(to, name, line):
    raw = open("template.txt", encoding="utf-8").read().strip().splitlines()
    subject = raw[0].replace("Subject:", "").strip()
    body = "\n".join(raw[1:]).strip()
    body = body.replace("{name}", name or "there").replace("{company_line}", line)
    m = EmailMessage()
    if to:
        m["To"] = to
    m["Subject"] = subject
    m.set_content(body)
    m.add_alternative(to_html(body), subtype="html")
    with open("resume.pdf", "rb") as f:
        m.add_attachment(f.read(), maintype="application", subtype="pdf",
                         filename="Sanika_Dhuri_Resume.pdf")
    return m

def save_draft(service, msg):
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    service.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()

# ---------- STEP A: GENERATE ----------
if st.button("1. Generate lines"):
    rows, bar = [], st.progress(0.0)
    todo = [r for _, r in edited.iterrows() if clean(r["company"])]
    for i, r in enumerate(todo):
        company = clean(r["company"])
        site = clean(r["website"]) or find_website(company)
        ctx = get_context(company, site)
        try:
            out = generate(company, clean(r["founder_name"]), clean(r["notes"]), ctx)
        except Exception as e:
            out = {"founder_first_name": "", "company_line": f"[ERROR: {e}]"}
        rows.append({
            "company": company,
            "email": clean(r["email"]),
            "name": clean(r["founder_name"]).split(" ")[0] or out.get("founder_first_name", ""),
            "company_line": out.get("company_line", ""),
            "create_draft": True,
        })
        bar.progress((i + 1) / len(todo))
        time.sleep(4)  # free-tier rate limit
    st.session_state["results"] = pd.DataFrame(rows)

# ---------- STEP B: REVIEW + DRAFT ----------
if "results" in st.session_state:
    st.subheader("Review and edit the lines. Untick any you want to skip")
    final = st.data_editor(st.session_state["results"], use_container_width=True)
    if st.button("2. Create Gmail drafts"):
        service = gmail_service()
        n = 0
        for _, r in final.iterrows():
            if r["create_draft"] and not r["company_line"].startswith("[ERROR"):
                save_draft(service, build_email(r["email"], r["name"], r["company_line"]))
                n += 1
        st.success(f"{n} drafts created. Check Gmail > Drafts.")