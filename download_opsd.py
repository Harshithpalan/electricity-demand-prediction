import requests
from pathlib import Path

candidates = [
    '2025-04-30',
    '2025-03-31',
    '2025-02-28',
    '2025-01-31',
    '2024-12-31',
    '2024-11-30',
    '2024-10-31',
    '2024-09-30',
]
output_dir = Path('data')
output_dir.mkdir(parents=True, exist_ok=True)

for d in candidates:
    url = f'https://data.open-power-system-data.org/time_series/{d}/time_series_60min.csv'
    try:
        r = requests.head(url, allow_redirects=True, timeout=20)
        print(d, url, r.status_code)
        if r.status_code == 200:
            result = requests.get(url, timeout=120)
            result.raise_for_status()
            out_path = output_dir / 'opsd_time_series_60min.csv'
            out_path.write_bytes(result.content)
            print(f'Saved {out_path} ({len(result.content)} bytes)')
            break
    except Exception as e:
        print('error', d, e)
else:
    raise SystemExit('No valid OPSD candidate URL found')
