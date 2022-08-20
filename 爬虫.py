import requests

headers = {'User-Agent': 'https://pic.netbian.com/4kfengjing/', 'Accept-Encoding': 'gzip, deflate', 'Accept': '*/*', 'Connection': 'keep-alive'}
response = requests.get('http://github.com/',headers=headers)
print(response.request.headers)
