
import re
import time
import html
import csv
import io
from urllib.parse import (
    urlparse, urljoin, parse_qs, unquote
)

import requests
import streamlit as st
from html.parser import HTMLParser


# ==================================================
# CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Yacht Broker Email Extractor",
    page_icon="🛥️",
    layout="wide"
)

APP_TITLE = "Yacht Broker Email Extractor"

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
    '"yacht broker" contact',
    '"yacht brokerage" email',
    '"yacht sales broker" contact',
    '"yacht charter broker" contact',
    'yacht brokers Europe',
    'yacht brokers United States',
    'yacht brokers United Kingdom',
    'yacht brokers Spain',
    'yacht brokers Italy',
    'yacht brokers Germany',
    'yacht brokers France',
    'yacht brokers Australia',
]

CONTACT_WORDS = (
    "contact", "about", "team", "broker",
    "office", "impressum", "kontakt"
)

SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif",
    ".zip", ".webp", ".svg", ".mp4", ".mp3"
)


# ==================================================
# HTML PARSER - NO BS4 REQUIRED
# ==================================================

class PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links = []
        self.text_parts = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            attrs_dict = dict(attrs)
            href = attrs_dict.get("href", "")
            if href:
                self.links.append(href)

    def handle_data(self, data):
        if data and data.strip():
            self.text_parts.append(data.strip())


def parse_html(document):
    parser = PageParser()

    try:
        parser.feed(document)
        parser.close()
    except Exception:
        pass

    return parser


# ==================================================
# HTTP HELPERS
# ==================================================

def get_domain(url):
    return urlparse(url).netloc.lower().removeprefix("www.")


def fetch_html(url):
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=15,
            allow_redirects=True
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


# ==================================================
# WEB SEARCH
# ==================================================

def search_web(query, max_results=15):
    urls = []

    try:
        response = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=HEADERS,
            timeout=20
        )

        response.raise_for_status()

        parser = parse_html(response.text)

        for href in parser.links:
            href = html.unescape(href.strip())

            if not href:
                continue

            parsed = urlparse(href)

            # Resolve DuckDuckGo redirect links
            if "duckduckgo.com" in parsed.netloc:
                targets = parse_qs(
                    parsed.query
                ).get("uddg", [])

                if targets:
                    href = unquote(targets[0])

            parsed = urlparse(href)

            if parsed.scheme not in ("http", "https"):
                continue

            if any(
                parsed.path.lower().endswith(ext)
                for ext in SKIP_EXTENSIONS
            ):
                continue

            if href not in urls:
                urls.append(href)

            if len(urls) >= max_results:
                break

    except requests.RequestException:
        pass

    return urls


# ==================================================
# EMAIL EXTRACTION
# ==================================================

def extract_emails(document):
    found = set()

    # Decode HTML entities
    document = html.unescape(document)

    # Search raw HTML, including mailto links
    found.update(EMAIL_PATTERN.findall(document))

    # Search visible text
    parser = parse_html(document)
    visible_text = " ".join(parser.text_parts)
    found.update(EMAIL_PATTERN.findall(visible_text))

    cleaned = set()

    for email in found:
        email = email.strip(
            ".,;:()[]{}<>\"' "
        ).lower()

        if "@" not in email:
            continue

        domain = email.rsplit("@", 1)[-1]

        if ".." in email or ".." in domain:
            continue

        if domain.endswith((
            ".example", ".invalid", ".test",
            ".png", ".jpg", ".jpeg", ".gif",
            ".svg", ".webp"
        )):
            continue

        cleaned.add(email)

    return cleaned


def find_contact_pages(base_url, document, limit=3):
    parser = parse_html(document)
    pages = []

    for href in parser.links:
        href = html.unescape(href.strip())

        if not href:
            continue

        if not any(
            word in href.lower()
            for word in CONTACT_WORDS
        ):
            continue

        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)

        if parsed.scheme not in ("http", "https"):
            continue

        if get_domain(absolute) != get_domain(base_url):
            continue

        if any(
            parsed.path.lower().endswith(ext)
            for ext in SKIP_EXTENSIONS
        ):
            continue

        if absolute not in pages:
            pages.append(absolute)

        if len(pages) >= limit:
            break

    return pages


# ==================================================
# SCAN LOGIC
# ==================================================

def run_scan(queries, max_sites, contact_pages_limit):
    discovered = []
    seen_domains = set()
    emails = set()
    visited_sites = []

    progress = st.progress(0)
    status = st.empty()

    # Search web
    for i, query in enumerate(queries):
        status.info(
            f"Searching {i + 1}/{len(queries)}: {query}"
        )

        for url in search_web(query):
            domain = get_domain(url)

            if domain and domain not in seen_domains:
                seen_domains.add(domain)
                discovered.append(url)

        progress.progress(
            (i + 1) / max(len(queries), 1) * 0.25
        )

        time.sleep(1)

    discovered = discovered[:max_sites]

    if not discovered:
        status.warning(
            "No search results were returned. "
            "The search service may be blocking automated requests."
        )
        return emails, visited_sites

    # Inspect websites
    for index, url in enumerate(discovered, start=1):
        status.info(
            f"Scanning website {index}/{len(discovered)}: {url}"
        )

        document = fetch_html(url)

        if document:
            emails.update(extract_emails(document))

            pages = find_contact_pages(
                url,
                document,
                limit=contact_pages_limit
            )

            for page_url in pages:
                time.sleep(0.5)

                contact_document = fetch_html(page_url)

                if contact_document:
                    emails.update(
                        extract_emails(contact_document)
                    )

        visited_sites.append(url)

        progress.progress(
            0.25 + 0.75 * index / len(discovered)
        )

        time.sleep(0.5)

    status.success("Scan completed.")
    return emails, visited_sites


# ==================================================
# STREAMLIT UI
# ==================================================

st.title("🛥️ Yacht Broker Email Extractor")

st.write(
    "Search publicly accessible yacht brokerage websites, "
    "extract published email addresses, remove duplicates, "
    "and export the results."
)

with st.sidebar:
    st.header("Search Settings")

    max_sites = st.slider(
        "Maximum websites to scan",
        min_value=5,
        max_value=100,
        value=25,
        step=5
    )

    query_count = st.slider(
        "Number of search queries",
        min_value=1,
        max_value=len(SEARCH_QUERIES),
        value=6
    )

    contact_pages_limit = st.slider(
        "Contact pages per website",
        min_value=0,
        max_value=3,
        value=2
    )

    custom_query = st.text_input(
        "Additional search term",
        placeholder="yacht brokers Greece contact"
    )

if "emails" not in st.session_state:
    st.session_state.emails = []

if "visited_sites" not in st.session_state:
    st.session_state.visited_sites = []

if "scan_complete" not in st.session_state:
    st.session_state.scan_complete = False

col1, col2 = st.columns(2)

with col1:
    start = st.button(
        "🔎 Start Search",
        type="primary",
        use_container_width=True
    )

with col2:
    clear = st.button(
        "🗑️ Clear Results",
        use_container_width=True
    )

if clear:
    st.session_state.emails = []
    st.session_state.visited_sites = []
    st.session_state.scan_complete = False
    st.rerun()

if start:
    queries = SEARCH_QUERIES[:query_count]

    if custom_query.strip():
        queries.append(custom_query.strip())

    with st.spinner("Searching the web..."):
        results, visited = run_scan(
            queries,
            max_sites,
            contact_pages_limit
        )

    st.session_state.emails = sorted(results)
    st.session_state.visited_sites = visited
    st.session_state.scan_complete = True


# ==================================================
# RESULTS
# ==================================================

st.divider()
st.subheader("📊 Results")

emails = st.session_state.emails
visited = st.session_state.visited_sites

c1, c2, c3 = st.columns(3)

c1.metric("Unique emails", len(emails))
c2.metric("Websites visited", len(visited))
c3.metric(
    "Status",
    "Completed" if st.session_state.scan_complete
    else "Ready"
)

if emails:
    st.success(
        f"Found {len(emails)} unique email addresses."
    )

    filter_text = st.text_input(
        "Filter email addresses",
        placeholder="Search by domain or keyword"
    )

    filtered = [
        email for email in emails
        if filter_text.lower() in email.lower()
    ]

    st.dataframe(
        {"Email": filtered},
        use_container_width=True,
        hide_index=True
    )

    txt_data = "\n".join(filtered) + "\n"

    st.download_button(
        "📄 Download TXT",
        data=txt_data,
        file_name="yacht_broker_emails.txt",
        mime="text/plain",
        use_container_width=True
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["email"])

    for email in filtered:
        writer.writerow([email])

    st.download_button(
        "📊 Download CSV",
        data=output.getvalue(),
        file_name="yacht_broker_emails.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    st.info(
        "No emails collected yet. Click Start Search to begin."
    )

if visited:
    with st.expander("View scanned websites"):
        for site in visited:
            st.write(site)

st.caption(
    "Search results depend on search-engine access and publicly "
    "available pages. Extracted addresses are not guaranteed to "
    "be active or verified. Respect website terms and applicable "
    "privacy and electronic marketing laws."
)
