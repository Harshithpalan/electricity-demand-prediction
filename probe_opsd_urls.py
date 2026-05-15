import re
import requests

url = 'https://data.open-power-system-data.org/time_series/'
resp = requests.get(url, timeout=20)
print('status', resp.status_code)
regex = re.compile(r'href="([^"]+)"')
hrefs = regex.findall(resp.text)
opsd = [h for h in hrefs if 'time_series' in h]
print('found', len(opsd), 'links')
print(opsd[:50])
print('--- snippet ---')
print(resp.text[:2000])
