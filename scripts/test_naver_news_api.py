"""Quick test of the NAVER API HUB news search endpoint."""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.environ["NAVER_API_KEY_ID"]
CLIENT_SECRET = os.environ["NAVER_API_KEY"]

url = "https://naverapihub.apigw.ntruss.com/search/v1/news"
headers = {
    "X-NCP-APIGW-API-KEY-ID": CLIENT_ID,
    "X-NCP-APIGW-API-KEY": CLIENT_SECRET,
}
params = {
    "query": "삼성전자",
    "display": 10,
    "sort": "date",
}

resp = requests.get(url, headers=headers, params=params, timeout=10)
print("status:", resp.status_code)
print(resp.text[:2000])
