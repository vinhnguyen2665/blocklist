import asyncio
import logging
import re
import sys
import time
from collections import defaultdict
from urllib.parse import urljoin, urldefrag, urlparse

import aiohttp
import requests
import tldextract
from aiohttp import ClientConnectorError, ClientResponseError
from bs4 import BeautifulSoup
from bs4.dammit import EncodingDetector

# ----------------------------- CONFIG -----------------------------
DEFAULT_USER_AGENT = "AGFilterCrawler/1.0 (+https://github.com/)"
REQUEST_TIMEOUT = 20
CONCURRENT_REQUESTS = 8
DELAY_PER_DOMAIN = 1.0  # seconds between requests to same domain

# ----------------------------- HELPERS -----------------------------

logger = logging.getLogger("ag_crawler")


def get_domain(url: str) -> str:
    ext = tldextract.extract(url)
    if not ext.registered_domain:
        return ""
    scheme = url[:url.index('://')]
    new_url = scheme + '://' + ext.registered_domain.lower()
    return new_url

def normalize_url(base: str, href: str) -> str | None:
    if not href:
        return None
    href = href.strip()
    # ignore javascript: mailto: tel:
    if href.startswith("javascript:") or href.startswith("mailto:") or href.startswith("tel:"):
        return None
    try:
        joined = urljoin(base, href)
        # remove fragments
        joined = urldefrag(joined)[0]
        parsed = urlparse(joined)
        if parsed.scheme not in ("http", "https"):
            return None
        return joined
    except Exception:
        return None


def looks_like_gambling_link(url: str) -> bool:
    # headers = {"User-Agent": USERAGENT}
    resp = requests.get(url)
    http_encoding = resp.encoding if 'charset' in resp.headers.get('content-type', '').lower() else None
    html_encoding = EncodingDetector.find_declared_encoding(resp.content, is_html=True)
    encoding = html_encoding or http_encoding
    soup = BeautifulSoup(resp.content, from_encoding=encoding)

    print(soup)
    print(soup.body.text)
    body = BeautifulSoup(soup.body.text, from_encoding=encoding)
    print(body)
    ans = llama.question(llama.clean_html(body.text))
    print("------------------")
    print(ans)


# ----------------------------- CRAWLER -----------------------------
class Crawler:
    def __init__(
            self,
            seeds=None,
            user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36'
    ):
        self.session = None
        if seeds is None:
            seeds = []
        self.collected = set()
        self.seeds = seeds
        self.domain_last_time = defaultdict(lambda: 0.0)
        self.user_agent = user_agent

    async def _throttle(self, url: str):
        dom = urlparse(url).netloc
        now = time.monotonic()
        last = self.domain_last_time[dom]
        wait = max(0.0, DELAY_PER_DOMAIN - (now - last))
        if wait > 0:
            await asyncio.sleep(wait)
        self.domain_last_time[dom] = time.monotonic()

    async def fetch(self, url: str, session=None) -> str | None:
        if self.collected.__contains__(url):
            logger.info(f"URL: {url} - status: collected")
            return "collected"
        if not session: session = self.session
        await self._throttle(url)
        headers = {"User-Agent": self.user_agent}
        try:
            async with session.get(url, timeout=REQUEST_TIMEOUT, headers=headers) as resp:
                logger.info(f"URL: {url} - status: {resp.status}")
                if resp.status == 200 and resp.content_type and "html" in resp.content_type:
                    text = await resp.text(errors="ignore")
                    self.collected.add(url)
                    return text
                else:
                    return None
        except (asyncio.TimeoutError, ClientConnectorError, ClientResponseError) as e:
            logger.debug(f"Failed to fetch {url}: {e}")
            return None
        except Exception as e:
            logger.debug(f"Unexpected error fetching {url}: {e}")
            return None

    async def get_body(self, html: str) -> set:
        soup = BeautifulSoup(html, "html.parser")

    async def parse_links(self, base_url: str, soup: BeautifulSoup) -> set:
        links = set()
        links.add(base_url)
        try:
            for a in soup.find_all("a", href=True):
                norm = normalize_url(base_url, a.get("href"))
                if norm:
                    domain = get_domain(norm)
                    links.add(domain)
        except Exception as e:
            logger.debug(f"Error parsing HTML from {base_url}: {e}")
        return links

    async def read_seeds(self, seeds):
        conn = aiohttp.TCPConnector(limit_per_host=2)
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=REQUEST_TIMEOUT, sock_read=REQUEST_TIMEOUT)
        _total_links = set()
        async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
            for seed in seeds:
                seed_text = await self.fetch(seed, session)
                if seed_text.__eq__('collected'):
                    continue
                _soup = BeautifulSoup(seed_text, "html.parser")
                _links = await self.parse_links(seed, _soup)
                # _s = await self.read_seeds(_links)
                # _total_links.update(_s)
                _total_links.update(_links)
        return _total_links

    async def craw(self):
        conn = aiohttp.TCPConnector(limit_per_host=2)
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=REQUEST_TIMEOUT, sock_read=REQUEST_TIMEOUT)
        _total_links = set()
        _read_seeds = await self.read_seeds(self.seeds)
        _total_links.update(_read_seeds)
        print(_total_links)
        with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
            for _link in _total_links:
                seed_text = await self.fetch(_link, session)
                if seed_text.__eq__('collected'):
                    continue
                _soup = BeautifulSoup(seed_text, "html.parser")
                _body = _soup.find('body')


# ----------------------------- OUTPUT -----------------------------
def export_hosts(domains: list[str], out_file: str):
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# Generated hosts-like blocklist\n")
        for d in domains:
            f.write(f"0.0.0.0 {d}\n")


def export_adguard(domains: list[str], out_file: str):
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("! AdGuard filter list generated by ag_filter_crawler\n")
        for d in domains:
            f.write(f"||{d}^\n")


def read_block_list(block_file):
    _blocked = []
    with open(block_file, "r", encoding="utf-8") as f:
        for line in f:
            _line = line.strip()
            if _line:
                match = re.search(r"\|\|([^/\^]+)", line)
                if match:
                    domain = match.group(1)
                    _blocked.append(domain)
                else:
                    continue
            else:
                continue
    return _blocked


if __name__ == '__main__':
    log_format = '[%(asctime)s] [%(levelname)8s] --- %(message)s (%(filename)s:%(lineno)s)'
    formatter = logging.Formatter(log_format)
    logging.basicConfig(level=logging.DEBUG, format=log_format, filename='ag_crawler')
    # logger.addHandler(timed_rotating_log_handler)
    handler_stdout = logging.StreamHandler(sys.stdout)
    handler_stdout.setLevel(logging.DEBUG)
    handler_stdout.setFormatter(formatter)
    # logging.getLogger("ag_crawler")
    logger.addHandler(handler_stdout)
    # llama = LlamaAI()
    blocked = read_block_list('../blocklist.txt')
    _seeds = ["https://www.casino.org/", "https://www.gamblingsites.com/", "https://www.topbettingsites.com/"]
    # _seeds = ["https://www.casino.org/"]
    crawler = Crawler(_seeds)
    asyncio.run(crawler.craw())
