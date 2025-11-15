import os
import sys
import json
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from main import app, query_handler

async def handler(event, context):
    """Netlify Functions handler for RAG query endpoint"""
    
    try:
        # Parse request
        if event['httpMethod'] == 'POST':
            body = json.loads(event.get('body', '{}'))
            question = body.get('question', '')
            
            if not question:
                return {
                    'statusCode': 400,
                    'body': json.dumps({'error': 'No question provided'})
                }
            
            # Call the RAG query handler
            result = await query_handler(question)
            
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps(result)
            }
        
        elif event['httpMethod'] == 'OPTIONS':
            # Handle CORS preflight
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type'
                }
            }
        
        else:
            return {
                'statusCode': 405,
                'body': json.dumps({'error': 'Method not allowed'})
            }
    
    except Exception as e:
        print(f"Error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
