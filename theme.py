"""Visual theme for the Term Deposit Predictor: typography, palette, chart styling."""
import re
import streamlit as st

INK, PAPER, CARD, RULE = "#1D2321", "#F5F1E8", "#FBF9F4", "#D9D2C1"
TEAL, BRASS, YES, NO = "#0F4C4A", "#B8741A", "#1F7A5C", "#8A989A"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, .stApp, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; color: {INK}; }}
.stApp {{ background: {PAPER}; }}
#MainMenu, footer, [data-testid="stSidebar"], [data-testid="collapsedControl"],
[data-testid="stToolbar"], [data-testid="stDecoration"] {{ display: none; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
.block-container {{ max-width: 1040px; padding: 2.2rem 1.5rem 4rem; }}
h1, h2, h3 {{ font-family: 'Fraunces', serif !important; font-weight: 500 !important; letter-spacing: -0.015em; color: {INK}; }}
h1 {{ font-size: 2.3rem !important; line-height: 1.15; padding-top: .4rem; }}
h2, h3 {{ font-size: 1.35rem !important; margin-top: 1.6rem; }}
p, li {{ line-height: 1.65; }}
code {{ font-family: 'IBM Plex Mono', monospace; font-size: .85em; background: #ECE6D6; color: {TEAL}; padding: 1px 5px; border-radius: 2px; }}
a {{ color: {TEAL}; }}

.mast {{ border-bottom: 1px solid {INK}; padding-bottom: .9rem; margin-bottom: .2rem; display:flex; justify-content:space-between; align-items:baseline; flex-wrap:wrap; gap:.5rem; }}
.mast .name {{ font-family:'Fraunces',serif; font-size:1.25rem; font-weight:600; }}
.mast .meta {{ font-family:'IBM Plex Mono',monospace; font-size:.72rem; letter-spacing:.06em; text-transform:uppercase; color:#6B6F66; }}

div[role="radiogroup"] {{ gap: 1.6rem; border-bottom: 1px solid {RULE}; margin-bottom: 1.6rem; padding: .5rem 0 0; }}
div[role="radiogroup"] label {{ padding: 0 0 .55rem; margin: 0; cursor: pointer; border-bottom: 2px solid transparent; }}
div[role="radiogroup"] label > div:first-child {{ display: none; }}
div[role="radiogroup"] label p {{ font-size: .95rem; color: #6B6F66; }}
div[role="radiogroup"] label:has(input:checked) {{ border-bottom-color: {BRASS}; }}
div[role="radiogroup"] label:has(input:checked) p {{ color: {INK}; font-weight: 600; }}

.kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:1.5rem; margin:1.4rem 0 .6rem; }}
.kpi {{ border-top: 2px solid {INK}; padding-top: .55rem; }}
.kpi .v {{ font-family:'Fraunces',serif; font-size:2.2rem; font-weight:500; line-height:1.1; }}
.kpi .l {{ font-family:'IBM Plex Mono',monospace; font-size:.7rem; letter-spacing:.06em; text-transform:uppercase; color:#6B6F66; margin-top:.25rem; }}
.kpi .n {{ font-size:.82rem; color:{YES}; margin-top:.2rem; }}

.step {{ display:grid; grid-template-columns:2.6rem 11rem 1fr; gap:.5rem; padding:.7rem 0; border-bottom:1px solid {RULE}; align-items:baseline; }}
.step .i {{ font-family:'IBM Plex Mono',monospace; color:{BRASS}; font-size:.85rem; }}
.step .t {{ font-weight:600; }}
.step .d {{ color:#43494A; }}

.verdict {{ background:{CARD}; border:1px solid {RULE}; border-left:5px solid var(--c); padding:.9rem 1.1rem; margin-top:.4rem; }}
.verdict .h {{ font-family:'Fraunces',serif; font-size:1.25rem; }}
.verdict .s {{ color:#43494A; font-size:.92rem; margin-top:.15rem; }}
.foot {{ margin-top:3rem; padding-top:.8rem; border-top:1px solid {RULE}; font-size:.78rem; color:#6B6F66; }}

.stButton button, .stFormSubmitButton button {{ border-radius:2px; font-weight:500; border:1px solid {TEAL}; }}
.stFormSubmitButton button[kind="primary"] {{ background:{TEAL}; color:#fff; padding:.5rem 1.8rem; }}
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"] {{ background:{CARD}; border-radius:2px !important; }}
[data-testid="stForm"] {{ background:{CARD}; border:1px solid {RULE}; border-radius:2px; padding:1.4rem 1.4rem .8rem; }}
[data-testid="stForm"] p strong {{ font-family:'IBM Plex Mono',monospace; font-size:.74rem; letter-spacing:.07em; text-transform:uppercase; color:{BRASS}; }}
[data-testid="stExpander"] {{ border:1px solid {RULE}; border-radius:2px; background:{CARD}; }}
@media (max-width: 760px) {{ .kpis {{ grid-template-columns:1fr 1fr; }} .step {{ grid-template-columns:2rem 1fr; }} .step .d {{ grid-column:2; }} }}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def masthead():
    st.markdown(
        '<div class="mast"><span class="name">Term Deposit Predictor</span>'
        '<span class="meta">Data Science mini project · Siddhi Pandhere · Pillai College of Engineering</span></div>',
        unsafe_allow_html=True)


def kpis(items):
    html = "".join(
        f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div>'
        + (f'<div class="n">{n}</div>' if n else "") + "</div>" for l, v, n in items)
    st.markdown(f'<div class="kpis">{html}</div>', unsafe_allow_html=True)


def steps_list(steps):
    code = lambda s: re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    html = "".join(
        f'<div class="step"><span class="i">{i:02d}</span><span class="t">{t}</span>'
        f'<span class="d">{code(d)}</span></div>' for i, (t, d) in enumerate(steps, 1))
    st.markdown(html, unsafe_allow_html=True)


def verdict(p, base):
    yes = p >= 0.5
    head = "Likely to subscribe" if yes else "Unlikely to subscribe"
    sub = (f"{p:.1%} probability. Worth a call." if yes else
           f"{p:.1%} probability, {p / base:.1f}× the {base:.1%} base rate. Lower priority.")
    st.markdown(f'<div class="verdict" style="--c:{YES if yes else BRASS}">'
                f'<div class="h">{head}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)


def footer():
    st.markdown(
        '<div class="foot">Data: Moro, Cortez &amp; Rita (2014), '
        '<a href="https://archive.ics.uci.edu/dataset/222/bank+marketing">UCI Bank Marketing</a>. '
        'Models: AdaBoost, Gradient Boosting, XGBoost and a soft-voting ensemble.</div>',
        unsafe_allow_html=True)


def show(fig):
    """Apply the house chart style, then render."""
    fig.update_layout(
        font=dict(family="IBM Plex Sans, sans-serif", color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(x=0, font=dict(family="Fraunces, serif", size=17)),
        legend=dict(bgcolor="rgba(0,0,0,0)"), hoverlabel=dict(font_family="IBM Plex Sans"))
    fig.update_xaxes(gridcolor="#E4DECF", zerolinecolor=RULE, linecolor=INK)
    fig.update_yaxes(gridcolor="#E4DECF", zerolinecolor=RULE)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
