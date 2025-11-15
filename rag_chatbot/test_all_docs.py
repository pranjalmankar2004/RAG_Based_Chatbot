import requests
import json

API = 'http://localhost:8001/query'

queries = [
    'What is evidence?',
    'What is a contract?',
    'What are consumer rights?',
    'What does RTI mean?',
    'What is the Consumer Protection Act?'
]

print('TESTING COMPREHENSIVE SEARCH - ALL 10 DOCUMENTS')
print('=' * 80)

for q in queries:
    try:
        r = requests.post(API, json={'question': q}, timeout=10)
        data = r.json()
        print(f'\nQ: {q}')
        print(f'Status: {r.status_code}')
        answer = data.get('answer', '')
        # Show first 200 chars of actual answer
        preview = answer[:200] if len(answer) > 200 else answer
        print(f'Answer: {preview}')
        if len(answer) > 200:
            print(f'        ... ({len(answer)} chars total)')
        print(f'Confidence: {data.get("confidence", 0)}')
        sources = data.get('sources', [])
        print(f'Sources: {len(sources)} documents')
        for src in sources[:2]:
            print(f'  - {src[:70]}')
    except Exception as e:
        print(f'Error: {e}')

print('\n' + '=' * 80)
print(' TEST COMPLETE - Searching all 10 documents')
