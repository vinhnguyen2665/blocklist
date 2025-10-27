import requests
from bs4 import BeautifulSoup
from bs4.dammit import EncodingDetector

from src.llama_ai import LlamaAI
from src.zero_shot import ZeroShot
from src.embedder import Embedder

if __name__ == '__main__':
    # llama = ZeroShot()
    # print(llama.question(
    #     "vu88.net"))
    # em = Embedder()
    # em.question("bet365.com")
    llama = LlamaAI()

    url = "http://vu88.net"
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
