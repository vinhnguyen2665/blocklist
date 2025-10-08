import asyncio
import argparse
import logging
import re
import sys
import time
from collections import defaultdict, deque
from urllib.parse import urljoin, urldefrag, urlparse

import aiohttp
import tldextract
from aiohttp import ClientConnectorError, ClientResponseError
from bs4 import BeautifulSoup
from urllib import robotparser

# ----------------------------- CONFIG -----------------------------
DEFAULT_USER_AGENT = "AGFilterCrawler/1.0 (+https://github.com/)"
REQUEST_TIMEOUT = 20
CONCURRENT_REQUESTS = 8
DELAY_PER_DOMAIN = 1.0  # seconds between requests to same domain

GAMBLING_KEYWORDS = [
    "casino",
    "bet",
    "betting",
    "poker",
    "gamble",
    "sportsbook",
    "odds",
    "slots",
    "dice",
    "blackjack",
    "roulette",
]

# ----------------------------- HELPERS -----------------------------

logger = logging.getLogger("ag_crawler")


def get_domain(url: str) -> str:
    ext = tldextract.extract(url)
    if not ext.registered_domain:
        return ""
    return ext.registered_domain.lower()


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
    url_lower = url.lower()
    for kw in GAMBLING_KEYWORDS:
        if kw in url_lower:
            return True
    return False


# ----------------------------- ROBOTS -----------------------------
class RobotsCache:
    def __init__(self):
        self._cache = {}

    async def allowed(self, session: aiohttp.ClientSession, url: str, user_agent: str) -> bool:
        dom = urlparse(url).netloc
        if dom in self._cache:
            rp = self._cache[dom]
            return rp.can_fetch(user_agent, url)

        robots_url = f"{urlparse(url).scheme}://{dom}/robots.txt"
        rp = robotparser.RobotFileParser()
        try:
            async with session.get(robots_url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": user_agent}) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    rp.parse(text.splitlines())
                else:
                    # if robots not present or error, allow by default
                    rp.parse(["User-agent: *", "Allow: /"])
        except Exception:
            rp.parse(["User-agent: *", "Allow: /"])
        self._cache[dom] = rp
        return rp.can_fetch(user_agent, url)


# ----------------------------- CRAWLER -----------------------------
class Crawler:
    def __init__(
            self,
            seeds,
            max_pages=1000,
            max_depth=3,
            concurrency=CONCURRENT_REQUESTS,
            user_agent=DEFAULT_USER_AGENT,
            only_gambling_links=True,
    ):
        self.seeds = seeds
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.user_agent = user_agent
        self.only_gambling_links = only_gambling_links

        self.session: aiohttp.ClientSession | None = None
        self.robots = RobotsCache()

        # scheduling
        self.queue = asyncio.Queue()
        self.visited = set()
        self.found_external_domains = set()
        self.found_external_urls = set()

        # domain -> last request time for rate limiting
        self.domain_last_time = defaultdict(lambda: 0.0)

        self.sem = asyncio.Semaphore(concurrency)

    async def _throttle(self, url: str):
        dom = urlparse(url).netloc
        now = time.monotonic()
        last = self.domain_last_time[dom]
        wait = max(0.0, DELAY_PER_DOMAIN - (now - last))
        if wait > 0:
            await asyncio.sleep(wait)
        self.domain_last_time[dom] = time.monotonic()

    async def fetch(self, url: str) -> str | None:
        await self._throttle(url)
        headers = {"User-Agent": self.user_agent}
        try:
            async with self.session.get(url, timeout=REQUEST_TIMEOUT, headers=headers) as resp:
                logger.info(f"URL: {url} - status: {resp.status}")
                if resp.status == 200 and resp.content_type and "html" in resp.content_type:
                    text = await resp.text(errors="ignore")
                    return text
                else:
                    return None
        except (asyncio.TimeoutError, ClientConnectorError, ClientResponseError) as e:
            logger.debug(f"Failed to fetch {url}: {e}")
            return None
        except Exception as e:
            logger.debug(f"Unexpected error fetching {url}: {e}")
            return None

    async def parse_links(self, base_url: str, html: str) -> set:
        links = set()
        try:
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                norm = normalize_url(base_url, a.get("href"))
                if norm:
                    links.add(norm)
        except Exception as e:
            logger.debug(f"Error parsing HTML from {base_url}: {e}")
        return links

    async def worker(self):
        while True:
            try:
                url, depth = await asyncio.wait_for(self.queue.get(), timeout=5)
            except asyncio.TimeoutError:
                return

            if len(self.visited) >= self.max_pages:
                self.queue.task_done()
                continue

            if url in self.visited:
                self.queue.task_done()
                continue

            # robots
            allowed = await self.robots.allowed(self.session, url, self.user_agent)
            # if not allowed:
            #     logger.debug(f"Disallowed by robots: {url}")
            #     self.visited.add(url)
            #     self.queue.task_done()
            #     continue

            async with self.sem:
                logger.debug(f"Fetching: {url} (depth {depth})")
                html = await self.fetch(url)

            self.visited.add(url)

            if not html:
                self.queue.task_done()
                continue

            links = await self.parse_links(url, html)

            for link in links:
                # if external
                link_dom = get_domain(link)
                base_dom = get_domain(url)

                if link_dom and link_dom != base_dom:
                    # external link: record if looks like gambling or if not restricting
                    if looks_like_gambling_link(link) or not self.only_gambling_links:
                        self.found_external_urls.add(link)
                        if link_dom:
                            self.found_external_domains.add(link_dom)
                else:
                    # internal link: enqueue if depth allows
                    if depth + 1 <= self.max_depth and link not in self.visited:
                        await self.queue.put((link, depth + 1))

            self.queue.task_done()

    async def run(self):
        conn = aiohttp.TCPConnector(limit_per_host=2)
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=REQUEST_TIMEOUT, sock_read=REQUEST_TIMEOUT)
        async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
            self.session = session
            # seed the queue
            for s in self.seeds:
                await self.queue.put((s, 0))

            workers = [asyncio.create_task(self.worker()) for _ in range(CONCURRENT_REQUESTS)]

            # wait until queue done or max pages reached
            await self.queue.join()

            # cancel workers
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        return {
            "visited_count": len(self.visited),
            "external_domains": sorted(self.found_external_domains),
            "external_urls": sorted(self.found_external_urls),
        }


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


# ----------------------------- CLI -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Crawl seed sites to generate AdGuard Home filter for gambling domains")
    p.add_argument("--seeds", nargs="+", required=False, default=[
        "https://www.casino.org/",
        "https://www.gamblingsites.com/",
        "https://www.topbettingsites.com/",
    ], help="seed URLs to start crawling from")
    p.add_argument("--max-pages", type=int, default=500, help="max pages to visit")
    p.add_argument("--depth", type=int, default=3, help="max internal link depth to follow")
    p.add_argument("--concurrency", type=int, default=CONCURRENT_REQUESTS, help="concurrent workers")
    p.add_argument("--delay", type=float, default=DELAY_PER_DOMAIN, help="delay per domain in seconds")
    p.add_argument("--only-gambling-links", action="store_true",
                   help="only record external links that match gambling keywords")
    p.add_argument("--out-hosts", default="hosts_like.txt")
    p.add_argument("--out-adguard", default="adguard_filter.txt")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


async def main_async(seeds=[], blocked=[], max_pages=500, depth=3, concurrency=8):
    global DELAY_PER_DOMAIN, CONCURRENT_REQUESTS
    CONCURRENT_REQUESTS = concurrency

    crawler = Crawler(
        seeds=seeds,
        max_pages=max_pages,
        max_depth=depth,
        concurrency=concurrency,
        only_gambling_links=True,
    )

    result = await crawler.run()

    # dedupe and sort domains
    domains = sorted(set(result["external_domains"]))

    # Extra heuristic: keep domains that contain gambling keywords OR have many matching URLs
    selected = []
    for d in domains:
        if d in blocked:
            continue
        if any(kw in d for kw in GAMBLING_KEYWORDS):
            selected.append(d)
        else:
            # check if there are many external URLs with this domain
            cnt = sum(1 for u in result["external_urls"] if get_domain(u) == d)
            if cnt >= 2:
                selected.append(d)

    selected = sorted(set(selected))

    export_hosts(selected, './out_hosts.txt')
    export_adguard(selected, './out_adguard.txt')

    print(f"Visited pages: {result['visited_count']}")
    print(f"Found external domains: {len(domains)} (selected {len(selected)})")
    # print(f"Hosts-like output -> {out_hosts}")
    # print(f"AdGuard filter -> {out_adguard}")


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


def main():
    args = parse_args()

    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("Interrupted")


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
    blocked = read_block_list('../blocklist.txt')
    asyncio.run(
        # main_async(["https://www.casino.org/", "https://www.gamblingsites.com/", "https://www.topbettingsites.com/"],blocked))
        # main_async(["https://www.casino.org/","https://www.gamblingsites.com/"], blocked))
        main_async(["https://google.com"], blocked))
