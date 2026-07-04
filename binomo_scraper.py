"""Binomo selenium collector scaffold.

Bu dosya Binomo veya benzeri web ekranlarından veri okumak için
hazır iskelet sağlar. Gerçek selector ve URL bilgileri kullanıcıya göre
özelleştirilmelidir.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except Exception:
    SELENIUM_AVAILABLE = False


BINOMO_URL = os.getenv("BINOMO_URL", "https://binomo.com/trading")
BINOMO_EMAIL = os.getenv("BINOMO_EMAIL", "")
BINOMO_PASSWORD = os.getenv("BINOMO_PASSWORD", "")
SELENIUM_HEADLESS = os.getenv("SELENIUM_HEADLESS", "false").lower() == "true"
CHROME_DEBUGGER_ADDRESS = os.getenv("CHROME_DEBUGGER_ADDRESS", "127.0.0.1:9222")
CHROMEDRIVER_PATH = os.getenv("CHROMEDRIVER_PATH", "")


@dataclass
class BinomoReadResult:
    asset: str
    prices: List[float]
    source: str
    note: str = ""


class BinomoSeleniumCollector:
    def __init__(self, base_url: Optional[str] = None, headless: bool = True):
        self.base_url = base_url or BINOMO_URL
        self.headless = headless
        self.driver = None

    def start(self):
        if not SELENIUM_AVAILABLE:
            raise RuntimeError("selenium yüklü değil")

        options = Options()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1600,1200")
        options.add_argument("--disable-gpu")

        if CHROMEDRIVER_PATH:
            service = Service(CHROMEDRIVER_PATH)
            self.driver = webdriver.Chrome(service=service, options=options)
        else:
            self.driver = webdriver.Chrome(options=options)
        return self.driver

    def attach_to_existing_chrome(self):
        if not SELENIUM_AVAILABLE:
            raise RuntimeError("selenium yüklü değil")

        options = Options()
        options.add_experimental_option("debuggerAddress", CHROME_DEBUGGER_ADDRESS)

        if CHROMEDRIVER_PATH:
            service = Service(CHROMEDRIVER_PATH)
            self.driver = webdriver.Chrome(service=service, options=options)
        else:
            self.driver = webdriver.Chrome(options=options)
        return self.driver

    def stop(self):
        if self.driver is not None:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None

    def open_platform(self):
        if not self.driver:
            self.start()
        if not self.base_url:
            raise RuntimeError("BINOMO_URL tanımlı değil")
        self.driver.get(self.base_url)
        time.sleep(3)

    def use_existing_or_open_platform(self):
        if not self.driver:
            self.attach_to_existing_chrome()

        current = ""
        try:
            current = self.driver.current_url or ""
        except Exception:
            current = ""

        if "binomo.com" not in current:
            self.driver.get(self.base_url)
            time.sleep(3)

    def login_if_needed(self):
        """Manuel giriş önerilir. Otomatik login varsayılan değildir."""
        return

    def select_asset(self, asset_name: str):
        if not self.driver:
            raise RuntimeError("driver başlatılmadı")

        search_candidates = [
            (By.CSS_SELECTOR, "input[placeholder*='Search']"),
            (By.CSS_SELECTOR, "input[type='search']"),
            (By.CSS_SELECTOR, "input"),
        ]

        for by, selector in search_candidates:
            try:
                elems = self.driver.find_elements(by, selector)
                for search in elems:
                    try:
                        if not search.is_displayed():
                            continue
                        search.clear()
                        search.send_keys(asset_name)
                        time.sleep(1)
                        return
                    except Exception:
                        continue
            except Exception:
                continue

    def read_visible_price_points(self, asset_name: str, limit: int = 120) -> BinomoReadResult:
        if not self.driver:
            raise RuntimeError("driver başlatılmadı")

        text_candidates = [
            ".price",
            ".quote",
            "[data-price]",
            ".chart-price",
            ".candle-price",
            ".assets-price",
            ".trading-price",
            "span",
            "div",
        ]

        found_values: List[float] = []
        for selector in text_candidates:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                values = []
                for el in elements:
                    raw = (el.text or "").strip().replace(",", "")
                    if not raw:
                        continue
                    try:
                        values.append(float(raw))
                    except Exception:
                        continue
                if len(values) >= 20:
                    found_values = values[-limit:]
                    break
            except Exception:
                continue

        if found_values:
            return BinomoReadResult(
                asset=asset_name,
                prices=found_values,
                source="Binomo Selenium DOM",
                note="Visible DOM values"
            )

        return BinomoReadResult(
            asset=asset_name,
            prices=[],
            source="Binomo Selenium DOM",
            note="Selector bulunamadı veya yeterli veri yok"
        )


def fetch_binomo_prices(asset_name: str, limit: int = 120) -> BinomoReadResult:
    collector = BinomoSeleniumCollector(headless=SELENIUM_HEADLESS)
    try:
        collector.use_existing_or_open_platform()
        collector.login_if_needed()
        collector.select_asset(asset_name)
        return collector.read_visible_price_points(asset_name, limit=limit)
    finally:
        collector.stop()
