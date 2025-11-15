import requests
import time

API = 'http://localhost:8001/query'
qs = ['What is evidence?', 'Define contract', 'Consumer rights', 'What is RTI?']

print('CONVERSATIONAL ANSWER TEST')
print('='*70)
for q in qs:
    try:
        r = requests.post(API, json={'question': q}, timeout=10).json()
        ans = r.get('answer', 'NO ANSWER')
        srcs = len(r.get('sources', []))
        conf = r.get('confidence', 0)
        print(f'\nQ: {q}')
        print(f'A: {ans}')
        print(f'Sources: {srcs}, Confidence: {conf:.2f}')
    except Exception as e:
        print(f'ERROR: {e}')
        time.sleep(2)
