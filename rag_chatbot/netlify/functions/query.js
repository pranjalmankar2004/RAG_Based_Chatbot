const fetch = require('node-fetch');
const fs = require('fs');
const path = require('path');

// Mock implementation - in production, you'd integrate with your Python backend
exports.handler = async (event, context) => {
  // Enable CORS
  const headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Content-Type': 'application/json'
  };

  // Handle CORS preflight
  if (event.httpMethod === 'OPTIONS') {
    return {
      statusCode: 200,
      headers,
      body: ''
    };
  }

  // Handle POST requests
  if (event.httpMethod === 'POST') {
    try {
      const body = JSON.parse(event.body);
      const question = body.question;

      if (!question) {
        return {
          statusCode: 400,
          headers,
          body: JSON.stringify({ error: 'No question provided' })
        };
      }

      // Call your backend API (localhost for local dev, production URL for deployed)
      const backendUrl = process.env.BACKEND_URL || 'http://localhost:8001';
      
      try {
        const response = await fetch(`${backendUrl}/query`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question })
        });

        if (!response.ok) {
          throw new Error(`Backend error: ${response.status}`);
        }

        const data = await response.json();
        return {
          statusCode: 200,
          headers,
          body: JSON.stringify(data)
        };
      } catch (err) {
        // Fallback demo response if backend not available
        console.log('Backend not available, using demo response');
        return {
          statusCode: 200,
          headers,
          body: JSON.stringify({
            answer: 'Demo: The Indian Evidence Act 1872 defines the rules of admissibility of evidence in courts. Please deploy backend for live responses.',
            sources: [
              { source: 'THE-INDIAN-EVIDENCE-ACT-1872.pdf', section: 'Introduction' }
            ],
            confidence: 75
          })
        };
      }
    } catch (error) {
      console.error('Function error:', error);
      return {
        statusCode: 500,
        headers,
        body: JSON.stringify({ error: error.message })
      };
    }
  }

  return {
    statusCode: 405,
    headers,
    body: JSON.stringify({ error: 'Method not allowed' })
  };
};
