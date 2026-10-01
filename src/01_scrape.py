#!/usr/bin/env python3
"""
Njuškalo Scraper v2 - Zagreb Apartments
- Selenium based (radi, ne blokira)
- Auto-detect broj stranica + early exit kad nema više oglasa
- Checkpoint sistem (nastavi odakle si stao)
- Retry logika po oglasu
- tqdm progress bar
- Bez duplikat filtriranja (to ide u cleaning fazu)
"""

import time
import random
import json
import csv
import re
import os
import logging
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from bs4 import BeautifulSoup
from tqdm import tqdm

# ============================================================================
# KONFIGURACIJA
# ============================================================================

BASE_URL = "https://www.njuskalo.hr/prodaja-stanova/zagreb"
CHECKPOINT_FILE = "scraper_checkpoint.json"
PARTIAL_FILE = "njuskalo_apartments_partial.json"
OUTPUT_FILE = "njuskalo_apartments_FULL.json"
ERROR_LOG = "scraper_errors.log"

# Koliko stranica da proba auto-detect? (fallback ako ne nađe paginaciju)
MAX_PAGES_FALLBACK = 350

# Koliko uzastopnih praznih stranica prije nego stane
EMPTY_PAGE_LIMIT = 3

# ============================================================================
# LOGGING
# ============================================================================

log_filename = f"scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# ============================================================================
# CHECKPOINT SISTEM
# ============================================================================

def save_checkpoint(page, properties, all_features):
    """Spremi trenutni napredak na disk."""
    checkpoint = {
        "last_completed_page": page,
        "timestamp": datetime.now().isoformat(),
        "total_scraped": len(properties),
        "all_features": list(all_features)
    }
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    with open(PARTIAL_FILE, "w", encoding="utf-8") as f:
        json.dump(properties, f, ensure_ascii=False, indent=2)

    logging.info(f"Checkpoint spremljen — stranica {page}, ukupno {len(properties)} oglasa")


def load_checkpoint():
    """Učitaj checkpoint ako postoji, inače kreni od početka."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)

            properties = []
            if os.path.exists(PARTIAL_FILE):
                with open(PARTIAL_FILE, "r", encoding="utf-8") as f:
                    properties = json.load(f)

            start_page = checkpoint["last_completed_page"] + 1
            all_features = set(checkpoint.get("all_features", []))

            logging.info(f"Checkpoint pronađen — nastavljam od stranice {start_page}, već scraped: {len(properties)}")
            return start_page, properties, all_features

        except Exception as e:
            logging.warning(f"Greška pri učitavanju checkpointa, krećem od početka: {e}")

    return 1, [], set()


# ============================================================================
# SCRAPER KLASA
# ============================================================================

class NjuskaloScraper:

    def __init__(self, headless=True):
        self.base_url = BASE_URL
        self.all_features = set()
        self.properties = []
        self.driver = None
        self.headless = headless

    # ------------------------------------------------------------------
    # DRIVER SETUP
    # ------------------------------------------------------------------

    def setup_driver(self):
        """Inicijaliziraj Selenium WebDriver s anti-detection mjerama."""
        options = webdriver.ChromeOptions()

        if self.headless:
            options.add_argument("--headless")

        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        self.driver = webdriver.Chrome(options=options)
        self.driver.execute_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        logging.info("WebDriver inicijaliziran")

    def restart_driver(self):
        """Restart drivera ako se nešto pokvari."""
        logging.warning("Restartam WebDriver...")
        try:
            if self.driver:
                self.driver.quit()
        except Exception:
            pass
        time.sleep(5)
        self.setup_driver()
        logging.info("WebDriver restartан")

    # ------------------------------------------------------------------
    # AUTO-DETECT BROJ STRANICA
    # ------------------------------------------------------------------

    def get_total_pages(self):
        """Pokušaj detektirati ukupan broj stranica iz paginacije."""
        try:
            self.driver.get(self.base_url)
            time.sleep(random.uniform(3, 5))

            soup = BeautifulSoup(self.driver.page_source, "html.parser")

            # Pokušaj više selektora za paginaciju
            pagination = (
                soup.find("ul", class_="Pagination-items")
                or soup.find("div", class_="pagination")
                or soup.find("nav", class_=re.compile(r"[Pp]agination"))
            )

            if pagination:
                page_links = pagination.find_all("a")
                pages = [int(link.text.strip()) for link in page_links if link.text.strip().isdigit()]
                if pages:
                    total = max(pages)
                    logging.info(f"Auto-detect: {total} stranica")
                    return total

            # Fallback: pogledaj da li ima podatak o broju oglasa
            count_tag = soup.find(class_=re.compile(r"[Cc]ount|[Rr]esult|[Tt]otal"))
            if count_tag:
                numbers = re.findall(r"\d+", count_tag.text.replace(".", ""))
                if numbers:
                    biggest = max(int(n) for n in numbers)
                    if biggest > 100:
                        estimated_pages = (biggest // 25) + 1
                        logging.info(f"Auto-detect via count: ~{biggest} oglasa → {estimated_pages} stranica")
                        return estimated_pages

            logging.warning("Nije moguće detektirati broj stranica, koristim fallback")
            return MAX_PAGES_FALLBACK

        except Exception as e:
            logging.error(f"Greška pri detekciji stranica: {e}")
            return MAX_PAGES_FALLBACK

    # ------------------------------------------------------------------
    # DOHVAT LINKOVA SA STRANICE
    # ------------------------------------------------------------------

    def get_ad_links_from_page(self, page_num):
        """Dohvati sve linkove oglasa s jedne stranice."""
        url = self.base_url if page_num == 1 else f"{self.base_url}?page={page_num}"

        try:
            self.driver.get(url)
            time.sleep(random.uniform(2, 4))

            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CLASS_NAME, "EntityList-items"))
            )

            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            ad_items = soup.find_all("li", class_="EntityList-item")

            links = []
            for item in ad_items:
                link_tag = item.find("a", class_="link") or item.find("a", href=True)
                if link_tag and link_tag.get("href"):
                    full_link = link_tag["href"]
                    if not full_link.startswith("http"):
                        full_link = "https://www.njuskalo.hr" + full_link
                    # Samo pravi oglasi nekretnina
                    if "/nekretnine/" in full_link and "oglas-" in full_link:
                        links.append(full_link)

            logging.info(f"Stranica {page_num}: {len(links)} oglasa")
            return links

        except TimeoutException:
            logging.warning(f"Timeout na stranici {page_num}")
            return []
        except Exception as e:
            logging.error(f"Greška na stranici {page_num}: {e}")
            return []

    # ------------------------------------------------------------------
    # PARSIRANJE JEDNOG OGLASA (s retry logikom)
    # ------------------------------------------------------------------

    def parse_single_ad(self, url, retries=3):
        """Parsiraj jedan oglas. Pokušaj do `retries` puta prije skipanja."""
        for attempt in range(1, retries + 1):
            try:
                self.driver.get(url)

                # Čekaj da se DOM potpuno učita — naslov je signal da je stranica renderirana
                try:
                    WebDriverWait(self.driver, 15).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "h1.ClassifiedDetailSummary-title"))
                    )
                except TimeoutException:
                    logging.warning(f"Timeout čekajući DOM za {url}, pokušavam svejedno")

                # Mali dodatni sleep za sporije stranice
                time.sleep(random.uniform(1, 2))

                soup = BeautifulSoup(self.driver.page_source, "html.parser")
                data = {"url": url}

                # --- Naslov ---
                title_tag = soup.find("h1", class_="ClassifiedDetailSummary-title")
                if title_tag:
                    data["title"] = title_tag.text.strip()
                    self.all_features.add("title")

                # --- Cijena ---
                price_tag = soup.find("dd", class_="ClassifiedDetailSummary-priceDomestic")
                if price_tag:
                    price_text = price_tag.text.strip()
                    price_clean = price_text.replace("€", "").replace(".", "").replace(" ", "").strip()
                    try:
                        data["price_eur"] = float(price_clean)
                        self.all_features.add("price_eur")
                    except ValueError:
                        data["price_raw"] = price_text
                        self.all_features.add("price_raw")

                # --- ID iz URL-a ---
                match = re.search(r"oglas-(\d+)", url)
                if match:
                    data["oglas_id"] = match.group(1)
                    self.all_features.add("oglas_id")

                # --- Osnovne informacije (dl lista) ---
                basic_details = soup.find("dl", class_="ClassifiedDetailBasicDetails-list")
                if basic_details:
                    terms = basic_details.find_all("dt", class_="ClassifiedDetailBasicDetails-listTerm")
                    definitions = basic_details.find_all("dd", class_="ClassifiedDetailBasicDetails-listDefinition")
                    for dt, dd in zip(terms, definitions):
                        key = dt.text.strip()
                        value = dd.text.strip()
                        key_norm = key.lower().replace(" ", "_").replace("²", "2").replace("/", "_").replace(":", "")
                        data[key_norm] = value
                        self.all_features.add(key_norm)

                # --- Opis ---
                desc_div = soup.find("div", class_="ClassifiedDetailDescription-text")
                if desc_div:
                    data["description"] = desc_div.text.strip()
                    self.all_features.add("description")

                # --- Property grupe (grijanje, podaci o objektu, itd.) ---
                property_groups = soup.find_all("section", class_="ClassifiedDetailPropertyGroups-group")
                for section in property_groups:
                    section_title_tag = section.find("h3", class_="ClassifiedDetailPropertyGroups-groupTitle")
                    if not section_title_tag:
                        continue
                    section_title = section_title_tag.text.strip()
                    list_items = section.find_all("li", class_="ClassifiedDetailPropertyGroups-groupListItem")

                    if "Grijanje" in section_title or "Kupaonica" in section_title or "WC" in section_title:
                        for item in list_items:
                            text = item.text.strip()
                            if ":" in text:
                                k, v = text.split(":", 1)
                                k_norm = k.strip().lower().replace(" ", "_").replace("č", "c").replace("ć", "c").replace("đ", "d").replace("š", "s").replace("ž", "z")
                                data[k_norm] = v.strip()
                                self.all_features.add(k_norm)
                    else:
                        section_key = section_title.lower().replace(" ", "_").replace("²", "2")
                        items_text = [item.text.strip() for item in list_items]
                        data[section_key] = ", ".join(items_text)
                        self.all_features.add(section_key)

                # --- System details (datum objave, pregledi) ---
                system_details = soup.find("dl", class_="ClassifiedDetailSystemDetails-list")
                if system_details:
                    terms = system_details.find_all("dt", class_="ClassifiedDetailSystemDetails-listTerm")
                    datas = system_details.find_all("dd", class_="ClassifiedDetailSystemDetails-listData")
                    for dt, dd in zip(terms, datas):
                        key = dt.text.strip()
                        value = dd.text.strip()
                        if "Oglas objavljen" in key:
                            data["datum_objave"] = value
                            self.all_features.add("datum_objave")
                        elif "prikazan" in key.lower():
                            m = re.search(r"(\d+)", value)
                            if m:
                                data["broj_pregleda"] = m.group(1)
                                self.all_features.add("broj_pregleda")
                        elif "isteka" in key.lower():
                            data["istek_oglasa"] = value
                            self.all_features.add("istek_oglasa")

                # --- Highlighted attributes (m2, sobe) ---
                highlighted = soup.find("ul", class_="ClassifiedDetailHighlightedAttributes-list")
                if highlighted:
                    items = highlighted.find_all("li", class_="ClassifiedDetailHighlightedAttributes-listItem")
                    for item in items:
                        label = item.find("dt", class_="ClassifiedDetailHighlightedAttributes-label")
                        text = item.find("dd", class_="ClassifiedDetailHighlightedAttributes-text")
                        if label and text:
                            k = label.text.strip().lower().replace(" ", "_").replace("²", "2")
                            v = text.text.strip()
                            if k not in data:
                                data[k] = v
                                self.all_features.add(k)

                # --- Vlasnik ---
                owner_name = soup.find("h2", class_="ClassifiedDetailOwnerDetails-title")
                if owner_name:
                    link = owner_name.find("a")
                    if link:
                        data["owner_name"] = link.text.strip()
                        self.all_features.add("owner_name")

                data["scraped_at"] = datetime.now().isoformat()
                self.all_features.add("scraped_at")

                return data

            except WebDriverException as e:
                logging.warning(f"WebDriver greška na {url} (pokušaj {attempt}/{retries}): {e}")
                if attempt < retries:
                    time.sleep(5 * attempt)
                    try:
                        self.restart_driver()
                    except Exception:
                        pass
                else:
                    self._log_error(url, str(e))
                    return {"url": url, "error": str(e)}

            except Exception as e:
                logging.warning(f"Greška na {url} (pokušaj {attempt}/{retries}): {e}")
                if attempt < retries:
                    time.sleep(3 * attempt)
                else:
                    self._log_error(url, str(e))
                    return {"url": url, "error": str(e)}

    def _log_error(self, url, error):
        """Logiraj neuspješan oglas u error file."""
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} | {url} | {error}\n")

    # ------------------------------------------------------------------
    # GLAVNI SCRAPE
    # ------------------------------------------------------------------

    def scrape_all(self, start_page=1, existing_properties=None, existing_features=None):
        """Glavna funkcija scrapinga."""
        if existing_properties:
            self.properties = existing_properties
        if existing_features:
            self.all_features = existing_features

        self.setup_driver()

        try:
            # Auto-detect broj stranica
            total_pages = self.get_total_pages()
            logging.info(f"Ciljamo {total_pages} stranica | Već scraped: {len(self.properties)}")

            # Prikupljanje linkova po stranicama
            all_links = []
            empty_streak = 0

            page_pbar = tqdm(range(start_page, total_pages + 1), desc="Stranice", unit="str")
            for page in page_pbar:
                links = self.get_ad_links_from_page(page)
                page_pbar.set_postfix({"oglasi_na_str": len(links), "ukupno_linkova": len(all_links)})

                if len(links) == 0:
                    empty_streak += 1
                    logging.warning(f"Prazna stranica {page} ({empty_streak}/{EMPTY_PAGE_LIMIT})")
                    if empty_streak >= EMPTY_PAGE_LIMIT:
                        logging.info("Dosegnut limit praznih stranica, zaustavljam paginaciju")
                        break
                else:
                    empty_streak = 0
                    all_links.extend(links)

                time.sleep(random.uniform(2, 4))

            logging.info(f"Ukupno pronađeno linkova: {len(all_links)}")

            # Parsiranje svakog oglasa
            ad_pbar = tqdm(enumerate(all_links, 1), total=len(all_links), desc="Oglasi", unit="oglas")
            for idx, link in ad_pbar:
                ad_pbar.set_postfix({"scraped": len(self.properties)})

                ad_data = self.parse_single_ad(link)
                if ad_data:
                    self.properties.append(ad_data)

                # Checkpoint svakih 50 oglasa
                if idx % 50 == 0:
                    save_checkpoint(f"ads_{idx}", self.properties, self.all_features)

                # Duža pauza svakih 100 oglasa da izbjegnemo blokadu
                if idx % 100 == 0 and idx > 0:
                    logging.info(f"Pauza 2 minute na oglasu {idx}...")
                    time.sleep(120)
                else:
                    time.sleep(random.uniform(5, 10))

            logging.info(f"Scraping završen! Ukupno: {len(self.properties)} oglasa")

        except Exception as e:
            logging.error(f"Fatalna greška: {e}")
            import traceback
            logging.error(traceback.format_exc())
        finally:
            if self.driver:
                self.driver.quit()
                logging.info("WebDriver zatvoren")

    # ------------------------------------------------------------------
    # SPREMANJE
    # ------------------------------------------------------------------

    def normalize_data(self):
        """Normalizacija — svi redovi imaju iste kolumne."""
        all_keys = set()
        for prop in self.properties:
            all_keys.update(prop.keys())

        return [{key: prop.get(key, None) for key in all_keys} for prop in self.properties]

    def save_to_csv(self, filename=None):
        filename = filename or OUTPUT_FILE.replace(".json", ".csv")
        if not self.properties:
            logging.warning("Nema podataka za spremanje")
            return

        normalized = self.normalize_data()
        fieldnames = list(normalized[0].keys())

        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(normalized)

        logging.info(f"CSV spremljen: {filename} ({len(normalized)} redova, {len(fieldnames)} kolumni)")

    def save_to_json(self, filename=None):
        filename = filename or OUTPUT_FILE
        if not self.properties:
            logging.warning("Nema podataka za spremanje")
            return

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.properties, f, ensure_ascii=False, indent=2)

        logging.info(f"JSON spremljen: {filename} ({len(self.properties)} oglasa)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("  NJUŠKALO SCRAPER v2 — Zagreb nekretnine")
    print("=" * 60)

    # Provjeri postoji li checkpoint
    start_page, existing_properties, existing_features = load_checkpoint()

    if existing_properties:
        print(f"  Nastavljam od stranice {start_page} | Već scraped: {len(existing_properties)}")
    else:
        print("  Krećem od početka (nema checkpointa)")

    print("=" * 60 + "\n")

    scraper = NjuskaloScraper(headless=False)  # headless=True za tiho pokretanje
    scraper.scrape_all(
        start_page=start_page,
        existing_properties=existing_properties,
        existing_features=existing_features
    )

    scraper.save_to_json("njuskalo_apartments_FULL_v3.json")
    scraper.save_to_csv("njuskalo_apartments_FULL_v3.csv")

    print("\n" + "=" * 60)
    print(f"  Završeno! Ukupno oglasa: {len(scraper.properties)}")
    print(f"  Ukupno featura: {len(scraper.all_features)}")
    print(f"  Output: {OUTPUT_FILE}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()