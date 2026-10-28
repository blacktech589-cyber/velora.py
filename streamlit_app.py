
import re
import time
import random
from pathlib import Path
from urllib.parse import urlparse, urljoin, parse_qs, unquote

import requests
from bs4 import BeautifulSoup

OUTPUT_FILE = "yacht_broker_emails.txt"
MAX_SEARCH_PAGES = 10
MAX_SITES = 150
MAX_PAGES_PER_SITE = 5

SEARCH_QUERIES = [
    '"yacht broker" contact email',
    '"yacht brokerage" contact',
    '"yacht sales" yacht broker email',
    '"yacht broker" "contact us"',
    '"yacht charter broker" contact email',
    'yacht brokerage company Europe contact',
    'yacht brokers USA contact',
    'yacht brokers UK contact',
    'yacht brokers Spain contact',
    'yacht brokers Italy contact',
    'yacht brokers Germany contact',
    'yacht brokers France contact',
    'yacht brokers Australia contact',
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
    )
}

EMAIL_PATTERN = re.compile(
    r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b"
)

SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip",
    ".webp", ".svg", ".mp4", ".mp3", ".doc", ".docx"
)

session = requests.Session()
session.headers.update(HEADERS)


def search_web(query, max_results=20):
    """DuckDuckGo HTML sonuçlarından web adreslerini alır."""
    urls = []

    try:
        response = session.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            timeout=20
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for link in soup.select("a.result__a"):
            href = link.get("href", "")
            if not href:
                continue

            parsed = urlparse(href)

            # DuckDuckGo yönlendirme adreslerini çöz
            if "duckduckgo.com" in parsed.netloc:
                target = parse_qs(parsed.query).get("uddg", [])
                if target:
                    href = unquote(target[0])

            parsed = urlparse(href)

            if parsed.scheme not in ("http", "https"):
                continue

            if any(parsed.path.lower().endswith(ext)
                   for ext in SKIP_EXTENSIONS):
                continue

            urls.append(href)

    except requests.RequestException as exc:
        print(f"Arama hatası: {exc}")

    return urls[:max_results]


def get_domain(url):
    return urlparse(url).netloc.lower().removeprefix("www.")


def get_page(url):
    """HTML sayfasını indirir; desteklenmeyen içerikleri atlar."""
    try:
        response = session.get(
            url,
            timeout=15,
            allow_redirects=True
        )

        content_type = response.headers.get("Content-Type", "").lower()

        if response.status_code != 200:
            return None

        if "text/html" not in content_type:
            return None

        return response.text

    except requests.RequestException:
        return None


def extract_emails(html):
    soup = BeautifulSoup(html, "html.parser")
    found = set()

    # Sayfa metnindeki e-postalar
    text = soup.get_text(" ", strip=True)
    found.update(EMAIL_PATTERN.findall(text))

    # mailto bağlantıları
    for link in soup.select('a[href^="mailto:"]'):
        address = link.get("href", "").split(":", 1)[1]
        address = address.split("?", 1)[0].strip()
        found.update(EMAIL_PATTERN.findall(address))

    # Bariz örnek ve dosya uzantısı hatalarını ele
    cleaned = set()

    for email in found:
        email = email.strip(".,;:()[]{}<>").lower()
        domain = email.rsplit("@", 1)[-1]

        if domain.endswith((".png", ".jpg", ".gif", ".webp")):
            continue

        if ".." in email or ".." in domain:
            continue

        if email.endswith((".example", ".invalid", ".test")):
            continue

        cleaned.add(email)

    return cleaned


def find_contact_pages(base_url, html):
    soup = BeautifulSoup(html, "html.parser")
    candidates = []

    keywords = (
        "contact", "about", "team", "broker", "office",
        "impressum", "kontakt", "contact-us", "our-company"
    )

    for link in soup.find_all("a", href=True):
        href = link["href"].strip()
        label = link.get_text(" ", strip=True).lower()
        combined = (href + " " + label).lower()

        if not any(word in combined for word in keywords):
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

        if absolute not in candidates:
            candidates.append(absolute)

    return candidates


def main():
    all_urls = []
    seen_domains = set()
    emails = set()

    print("=" * 55)
    print("     YACHT BROKER EMAIL EXTRACTOR")
    print("=" * 55)

    # 1. İnternet araması
    for query in SEARCH_QUERIES[:MAX_SEARCH_PAGES]:
        print(f"\nAranıyor: {query}")

        urls = search_web(query, max_results=20)

        for url in urls:
            domain = get_domain(url)

            if domain and domain not in seen_domains:
                seen_domains.add(domain)
                all_urls.append(url)

        print(f"Bulunan farklı web sitesi: {len(all_urls)}")
        time.sleep(random.uniform(2, 4))

    all_urls = all_urls[:MAX_SITES]

    # 2. Şirket sitelerini incele
    for index, url in enumerate(all_urls, start=1):
        print(f"\n[{index}/{len(all_urls)}] {url}")

        html = get_page(url)
        if not html:
            print("Sayfa okunamadı.")
            continue

        found = extract_emails(html)
        emails.update(found)

        # İletişim ve hakkında sayfalarını incele
        contact_pages = find_contact_pages(url, html)

        for contact_url in contact_pages[:MAX_PAGES_PER_SITE - 1]:
            time.sleep(random.uniform(1.5, 3))

            contact_html = get_page(contact_url)
            if contact_html:
                emails.update(extract_emails(contact_html))

        # Her yeni sonuçtan sonra TXT dosyasını güncelle
        Path(OUTPUT_FILE).write_text(
            "".join(email + "\n" for email in sorted(emails)),
            encoding="utf-8"
        )

        print(f"Toplam benzersiz e-posta: {len(emails)}")
        time.sleep(random.uniform(1.5, 3))

    # 3. Son kayıt
    Path(OUTPUT_FILE).write_text(
        "".join(email + "\n" for email in sorted(emails)),
        encoding="utf-8"
    )

    print("\n" + "=" * 55)
    print("İŞLEM TAMAMLANDI")
    print(f"İncelenen web sitesi: {len(all_urls)}")
    print(f"Bulunan benzersiz e-posta: {len(emails)}")
    print(f"TXT dosyası: {Path(OUTPUT_FILE).resolve()}")
    print("=" * 55)


if __name__ == "__main__":
    main()
