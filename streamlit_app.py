
import re
import time
from urllib.parse import urlparse, urljoin, parse_qs, unquote

import requests
import streamlit as st
from bs4 import BeautifulSoup

# --------------------------------------------------
# AYARLAR
# --------------------------------------------------

st.set_page_config(
    page_title="Yacht Broker Email Extractor",
    page_icon="🛥️",
    layout="wide"
)

EMAIL_PATTERN = re.compile(
    r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
    )
}

SEARCH_QUERIES = [
    '"yacht broker" contact email',
    '"yacht brokerage" contact',
    '"yacht sales" broker contact',
    '"yacht charter broker" contact',
    'yacht brokers Europe contact',
    'yacht brokers United States contact',
    'yacht brokers United Kingdom contact',
    'yacht brokers Spain contact',
    'yacht brokers Italy contact',
    'yacht brokers Germany contact',
    'yacht brokers France contact',
    'yacht brokers Australia contact',
]

SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif",
    ".zip", ".webp", ".svg", ".mp4", ".mp3"
)

CONTACT_KEYWORDS = (
    "contact", "about", "team", "broker", "office",
    "impressum", "kontakt", "our-company"
)


# --------------------------------------------------
# HTTP VE ARAMA FONKSİYONLARI
# --------------------------------------------------

def get_domain(url):
    return urlparse(url).netloc.lower().removeprefix("www.")


def search_web(query, max_results=15):
    """DuckDuckGo HTML arama sonuçlarından URL toplar."""
    results = []

    try:
        response = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=HEADERS,
            timeout=20
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for link in soup.select("a.result__a"):
            href = link.get("href", "").strip()

            if not href:
                continue

            parsed = urlparse(href)

            if "duckduckgo.com" in parsed.netloc:
                targets = parse_qs(parsed.query).get("uddg", [])
                if targets:
                    href = unquote(targets[0])

            parsed = urlparse(href)

            if parsed.scheme not in ("http", "https"):
                continue

            if any(parsed.path.lower().endswith(ext)
                   for ext in SKIP_EXTENSIONS):
                continue

            if href not in results:
                results.append(href)

            if len(results) >= max_results:
                break

    except requests.RequestException as exc:
        st.warning(f"Arama yapılamadı: {exc}")

    return results


def fetch_html(url):
    """Yalnızca HTML sayfalarını indirir."""
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=15
        )

        if response.status_code != 200:
            return None

        content_type = response.headers.get(
            "Content-Type", ""
        ).lower()

        if "text/html" not in content_type:
            return None

        return response.text

    except requests.RequestException:
        return None


# --------------------------------------------------
# E-POSTA ÇIKARMA
# --------------------------------------------------

def extract_emails(html):
    soup = BeautifulSoup(html, "html.parser")
    found = set()

    # Sayfa metninden
    text = soup.get_text(" ", strip=True)
    found.update(EMAIL_PATTERN.findall(text))

    # mailto bağlantılarından
    for link in soup.select('a[href^="mailto:"]'):
        href = link.get("href", "")
        address = href.split(":", 1)[-1].split("?", 1)[0]
        found.update(EMAIL_PATTERN.findall(address))

    cleaned = set()

    for email in found:
        email = email.strip(".,;:()[]{}<>").lower()
        domain = email.rsplit("@", 1)[-1]

        if ".." in email or ".." in domain:
            continue

        if domain.endswith((
            ".png", ".jpg", ".jpeg", ".gif",
            ".webp", ".example", ".invalid", ".test"
        )):
            continue

        cleaned.add(email)

    return cleaned


def find_contact_pages(base_url, html, limit=4):
    soup = BeautifulSoup(html, "html.parser")
    pages = []

    for link in soup.find_all("a", href=True):
        href = link["href"].strip()
        label = link.get_text(" ", strip=True).lower()

        if not any(
            word in (href + " " + label).lower()
            for word in CONTACT_KEYWORDS
        ):
            continue

        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)

        if parsed.scheme not in ("http", "https"):
            continue

        if get_domain(absolute) != get_domain(base_url):
            continue

        if any(parsed.path.lower().endswith(ext)
               for ext in SKIP_EXTENSIONS):
            continue

        if absolute not in pages:
            pages.append(absolute)

        if len(pages) >= limit:
            break

    return pages


# --------------------------------------------------
# STREAMLIT ARAYÜZÜ
# --------------------------------------------------

st.title("🛥️ Yacht Broker Email Extractor")

st.write(
    "Yat brokerları ve yat satış şirketlerinin herkese açık "
    "web sayfalarını araştırın, bulunan e-posta adreslerini "
    "TXT veya CSV olarak dışa aktarın."
)

with st.sidebar:
    st.header("Arama ayarları")

    max_sites = st.slider(
        "İncelenecek maksimum web sitesi",
        min_value=5,
        max_value=100,
        value=25,
        step=5
    )

    queries_to_use = st.slider(
        "Arama sorgusu sayısı",
        min_value=1,
        max_value=len(SEARCH_QUERIES),
        value=6
    )

    pages_per_site = st.slider(
        "Her sitedeki ek iletişim sayfası",
        min_value=0,
        max_value=4,
        value=2
    )

    custom_query = st.text_input(
        "Ek arama terimi (isteğe bağlı)",
        placeholder="Örn. yacht brokers Greece contact"
    )

st.caption(
    "Not: Sonuç sayısı arama motorunun erişimine, "
    "web sitelerinin içeriğine ve engellemelerine bağlıdır."
)

if "emails" not in st.session_state:
    st.session_state.emails = set()

if "visited_sites" not in st.session_state:
    st.session_state.visited_sites = []

if "scan_complete" not in st.session_state:
    st.session_state.scan_complete = False

col1, col2 = st.columns(2)

with col1:
    start_scan = st.button(
        "🔎 İnternette Ara",
        type="primary",
        use_container_width=True
    )

with col2:
    clear_results = st.button(
        "🗑️ Sonuçları Temizle",
        use_container_width=True
    )

if clear_results:
    st.session_state.emails = set()
    st.session_state.visited_sites = []
    st.session_state.scan_complete = False
    st.rerun()


# --------------------------------------------------
# TARAMA İŞLEMİ
# --------------------------------------------------

if start_scan:
    st.session_state.emails = set()
    st.session_state.visited_sites = []
    st.session_state.scan_complete = False

    queries = SEARCH_QUERIES[:queries_to_use]

    if custom_query.strip():
        queries.append(custom_query.strip())

    discovered_urls = []
    seen_domains = set()

    progress = st.progress(0)
    status = st.empty()

    status.info("Arama motorunda web siteleri aranıyor...")

    for query_index, query in enumerate(queries):
        status.write(
            f"Arama {query_index + 1}/{len(queries)}: {query}"
        )

        urls = search_web(query, max_results=15)

        for url in urls:
            domain = get_domain(url)

            if domain and domain not in seen_domains:
                seen_domains.add(domain)
                discovered_urls.append(url)

        progress.progress(
            min((query_index + 1) / len(queries), 1.0)
        )

        time.sleep(1)

    discovered_urls = discovered_urls[:max_sites]

    if not discovered_urls:
        st.warning(
            "Arama sonucu bulunamadı. Arama motoru erişimi "
            "engellenmiş olabilir. Biraz sonra tekrar deneyin."
        )
    else:
        st.info(
            f"{len(discovered_urls)} farklı web sitesi incelenecek."
        )

        site_progress = st.progress(0)

        for index, url in enumerate(discovered_urls, start=1):
            status.write(
                f"Site inceleniyor: {index}/{len(discovered_urls)} — {url}"
            )

            html = fetch_html(url)

            if html:
                found = extract_emails(html)
                st.session_state.emails.update(found)

                contact_pages = find_contact_pages(
                    url,
                    html,
                    limit=pages_per_site
                )

                for contact_url in contact_pages:
                    time.sleep(0.5)
                    contact_html = fetch_html(contact_url)

                    if contact_html:
                        st.session_state.emails.update(
                            extract_emails(contact_html)
                        )

            st.session_state.visited_sites.append(url)

            site_progress.progress(
                index / len(discovered_urls)
            )

        st.session_state.scan_complete = True
        status.success("Tarama tamamlandı!")


# --------------------------------------------------
# SONUÇLAR VE İNDİRME
# --------------------------------------------------

emails = sorted(st.session_state.emails)

st.divider()
st.subheader("📊 Tarama Sonuçları")

metric1, metric2, metric3 = st.columns(3)

metric1.metric("Benzersiz e-posta", len(emails))
metric2.metric(
    "İncelenen site",
    len(st.session_state.visited_sites)
)
metric3.metric(
    "Tarama durumu",
    "Tamamlandı" if st.session_state.scan_complete
    else "Hazır"
)

if emails:
    st.success(f"{len(emails)} benzersiz e-posta adresi bulundu.")

    search_filter = st.text_input(
        "Sonuçlarda ara",
        placeholder="Örn. info, sales, broker..."
    )

    filtered_emails = [
        email for email in emails
        if search_filter.lower() in email.lower()
    ]

    st.dataframe(
        {"Email": filtered_emails},
        use_container_width=True,
        hide_index=True
    )

    txt_content = "\n".join(filtered_emails) + "\n"

    st.download_button(
        label="📄 TXT olarak indir",
        data=txt_content,
        file_name="yacht_broker_emails.txt",
        mime="text/plain",
        use_container_width=True
    )

    csv_content = "email\n" + "\n".join(filtered_emails) + "\n"

    st.download_button(
        label="📊 CSV olarak indir",
        data=csv_content,
        file_name="yacht_broker_emails.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    st.info(
        "Henüz e-posta bulunmadı. Yukarıdaki 'İnternette Ara' "
        "düğmesine basarak taramayı başlatın."
    )

if st.session_state.visited_sites:
    with st.expander("İncelenen web sitelerini göster"):
        for site in st.session_state.visited_sites:
            st.write(site)

st.caption(
    "Yalnızca erişilebilir, herkese açık sayfalar incelenir. "
    "E-posta adreslerinin geçerliliği ayrıca doğrulanmalıdır. "
    "Web sitelerinin kullanım koşullarına ve geçerli veri "
    "koruma kurallarına uyun."
)
