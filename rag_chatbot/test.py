import requests
import json
import sys
from time import sleep

API = "http://localhost:8001/query"

def test_query(question):
    try:
        r = requests.post(API, json={"question": question}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            print(f"\n✓ Query: {question}")
            print(f"✓ Status: {r.status_code} OK")
            print(f"✓ Answer: {data.get('answer', 'N/A')[:200]}...")
            print(f"✓ Confidence: {data.get('confidence', 'N/A')}")
            return True
        else:
            print(f"\n✗ Error: {r.status_code}")
            return False
    except Exception as e:
        print(f"\n✗ Connection failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing backend connection...")
    sleep(2)
    
    questions = [
        "What is RTI?",
        "What are consumer rights?",
        "What is the Indian Evidence Act?"
    ]
    
    for q in questions:
        if not test_query(q):
            break
        sleep(1)
