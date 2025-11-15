import requests
import json

API = 'http://localhost:8001/query'
test_questions = [
    'What is the Consumer Protection Act?',
    'How do I access information under RTI?',
    'What are my rights as a citizen?'
]

print('=' * 70)
print('LEGAL RAG CHATBOT - FINAL VERIFICATION TEST')
print('=' * 70)

for i, q in enumerate(test_questions, 1):
    try:
        r = requests.post(API, json={'question': q}, timeout=10)
        data = r.json()
        status_str = 'OK' if r.status_code == 200 else 'ERROR'
        print(f'\n[Test {i}] {q}')
        print(f'Status: {r.status_code} {status_str}')
        print(f'Confidence: {data.get("confidence", "N/A")}')
        print(f'Sources: {len(data.get("sources", []))} documents cited')
        if data.get('sources'):
            for src in data.get('sources', [])[:2]:
                print(f'  - {src}')
    except Exception as e:
        print(f'\n[Test {i}] FAILED: {e}')

print('\n' + '=' * 70)
print('VERIFICATION COMPLETE')
print('=' * 70)
