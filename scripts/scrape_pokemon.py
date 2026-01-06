#!/usr/bin/env python3
"""
Scrape Pokemon images from Bulbapedia and follow Serebii external links when present.

Usage examples:
  python scripts/scrape_pokemon.py --names Pikachu Bulbasaur
  python scripts/scrape_pokemon.py --names-file pokemon_list.txt --max-images 5

Notes:
- Respects robots.txt for each host (will skip if disallowed).
- Use responsibly and check site Terms of Use before scraping large amounts.
"""
import argparse
import os
import random
import time
import logging
from pathlib import Path
from urllib.parse import urljoin, quote, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; ImageScraper/1.0; +https://example.org)"
RETRY_COUNT = 3
RETRY_BACKOFF = 1.5


def check_allowed(url, user_agent=DEFAULT_USER_AGENT):
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        # If robots.txt can't be fetched, be conservative and allow (or you may choose to disallow)
        return True


def fetch_url(session, url, timeout=15):
    for attempt in range(RETRY_COUNT):
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            wait = RETRY_BACKOFF * (2 ** attempt) + random.random() * 0.5
            logging.debug(f"Fetch failed {url}: {e}; retrying in {wait:.1f}s")
            time.sleep(wait)
    logging.warning(f"Giving up fetching {url}")
    return None


def make_abs(src, base_url):
    if not src:
        return None
    src = src.strip()
    if src.startswith("//"):
        return "https:" + src
    if src.startswith("/"):
        parsed = urlparse(base_url)
        return f"{parsed.scheme}://{parsed.netloc}" + src
    if src.startswith("http://") or src.startswith("https://"):
        return src
    return urljoin(base_url, src)


def image_matches_name(img_tag, url, name):
    """Return True if image filename/alt/title/caption contains the target name."""
    if not url:
        return False
    name_norm = ''.join(c.lower() for c in name if c.isalnum())

    # check filename
    try:
        fname = os.path.basename(urlparse(url).path)
    except Exception:
        fname = ''
    fname_norm = ''.join(c.lower() for c in fname if c.isalnum())
    if name_norm and name_norm in fname_norm:
        return True

    # check alt/title attributes
    for attr in ('alt', 'title'):
        val = img_tag.get(attr) if img_tag else None
        if val and name.lower() in val.lower():
            return True

    # check nearby caption/figure
    fig = None
    # if image is inside a figure or gallerybox, try to find caption text
    parent = img_tag.parent if img_tag is not None else None
    for _ in range(4):
        if parent is None:
            break
        if parent.name in ('figure', 'div', 'td'):
            fig = parent
            break
        parent = parent.parent
    if fig:
        caption_text = ' '.join(t.strip() for t in fig.stripped_strings)
        if caption_text and name.lower() in caption_text.lower():
            return True

    return False


def parse_bulbapedia_images(soup, base_url, name):
    images = []
    # 1) Infobox main image: first <a class="image"> inside a roundy table or any image near top
    infobox = soup.find('table', class_='roundy') or soup.find('table', class_='infobox')
    if infobox:
        a = infobox.find('a', class_='image')
        if a:
            img = a.find('img')
            if img:
                src = img.get('data-src') or img.get('src')
                url = make_abs(src, base_url)
                if url:
                    images.append(url)

    # 2) Gallery images: prefer those that explicitly mention the Pokemon name
    for gallery in soup.find_all('div', class_='gallery'):
        for img in gallery.find_all('img'):
            src = img.get('data-src') or img.get('src')
            url = make_abs(src, base_url)
            if not url:
                continue
            if image_matches_name(img, url, name):
                images.append(url)

    # 3) Article images: look for images whose filename/alt contains the name
    for img in soup.find_all('img'):
        src = img.get('data-src') or img.get('src')
        if not src:
            continue
        url = make_abs(src, base_url)
        if image_matches_name(img, url, name):
            images.append(url)

    # unique and preserve order
    seen = set()
    out = []
    for u in images:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def find_external_serebii_link(soup):
    # search any anchor linking to serebii.net
    a = soup.select_one('a[href*="serebii.net"]')
    if a and a.get('href'):
        return a['href']
    return None


def download_image(session, url, dest_path):
    if not check_allowed(url):
        logging.info(f"Robots.txt disallowed fetching {url}")
        return False
    r = fetch_url(session, url)
    if not r:
        return False
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, 'wb') as f:
            f.write(r.content)
        return True
    except Exception as e:
        logging.warning(f"Failed saving {url} -> {dest_path}: {e}")
        return False


def safe_name(name: str):
    # create filesystem-safe name
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    return ''.join(c if c in keep else '_' for c in name).strip('_')


def scrape_one(session, name, out_dir, max_images=5, delay=(1.0, 2.5)):
    """Scrape Bulbapedia for a Pokemon name and follow Serebii external link if present."""
    base_bulba = 'https://bulbapedia.bulbagarden.net'
    # Construct page name: spaces -> '_', quote it
    page = f"{quote(name.replace(' ', '_'))}_(Pok%C3%A9mon)"
    bulba_url = f"{base_bulba}/wiki/{page}"

    if not check_allowed(bulba_url):
        logging.warning(f"Robots disallow {bulba_url}; skipping {name}")
        return 0

    r = fetch_url(session, bulba_url)
    if not r:
        return 0
    soup = BeautifulSoup(r.text, 'html.parser')
    images = parse_bulbapedia_images(soup, bulba_url, name)

    serebii_link = find_external_serebii_link(soup)
    serebii_images = []
    if serebii_link:
        if not serebii_link.startswith('http'):
            serebii_link = urljoin(base_bulba, serebii_link)
        if check_allowed(serebii_link):
            r2 = fetch_url(session, serebii_link)
            if r2:
                s2 = BeautifulSoup(r2.text, 'html.parser')
                # collect large images
                for img in s2.find_all('img'):
                    src = img.get('data-src') or img.get('src')
                    if not src:
                        continue
                    u = make_abs(src, serebii_link)
                    # heuristics: pick images hosted under serebii that look like artwork
                    if u and ('serebii' in u or '/artwork/' in u or '/pokemon/' in u):
                        serebii_images.append(u)

    # combine lists, Bulbapedia first then Serebii
    all_images = images + serebii_images

    # Download up to max_images
    saved = 0
    pn = safe_name(name)
    target_dir = Path(out_dir) / pn
    target_dir.mkdir(parents=True, exist_ok=True)

    for i, img_url in enumerate(all_images):
        if saved >= max_images:
            break
        # normalize
        img_url = make_abs(img_url, bulba_url)
        ext = os.path.splitext(urlparse(img_url).path)[1]
        if not ext:
            ext = '.jpg'
        filename = f"img_{i+1}{ext}"
        dest = target_dir / filename
        success = download_image(session, img_url, dest)
        if success:
            saved += 1
            time.sleep(random.uniform(*delay))

    return saved


def load_names_from_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines()]
    return [l for l in lines if l]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--names', nargs='*', help='Pokemon names to scrape (e.g. Pikachu)')
    parser.add_argument('--names-file', help='File with one Pokemon name per line')
    parser.add_argument('--out', default='data/pokemon', help='Output directory')
    parser.add_argument('--max-images', type=int, default=5, help='Max images per Pokemon')
    parser.add_argument('--delay-min', type=float, default=1.0)
    parser.add_argument('--delay-max', type=float, default=2.5)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='[%(levelname)s] %(message)s')

    names = []
    if args.names_file:
        names = load_names_from_file(args.names_file)
    if args.names:
        names.extend(args.names)
    if not names:
        logging.error('No names provided. Use --names or --names-file')
        return

    session = requests.Session()
    session.headers.update({'User-Agent': DEFAULT_USER_AGENT})

    total = 0
    for name in tqdm(names, desc='Pokemon'):
        try:
            saved = scrape_one(session, name, args.out, max_images=args.max_images,
                               delay=(args.delay_min, args.delay_max))
            logging.info(f"{name}: saved {saved} images")
            total += saved
        except Exception as e:
            logging.warning(f"Error scraping {name}: {e}")

    logging.info(f"Done. Total images saved: {total}")


if __name__ == '__main__':
    main()
