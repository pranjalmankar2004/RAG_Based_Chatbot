// ============================================
// Legal RAG Assistant - JavaScript
// ============================================

class ChatManager {
    constructor() {
        this.messagesContainer = document.getElementById('messagesContainer');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.settingsBtn = document.getElementById('settingsBtn');
        this.typingTemplate = document.getElementById('typingIndicatorTemplate');
        this.messageHistory = [];
        this.isWaitingForResponse = false;
        
        // Detect environment and set appropriate backend URL
        const hostname = window.location.hostname;
        const port = window.location.port;
        
        // Check if running locally
        if (hostname === 'localhost' || hostname === '127.0.0.1' || port === '5500') {
            // Local development on same computer
            this.backendUrl = 'http://127.0.0.1:8001';
        } else if (hostname.startsWith('192.168.') || hostname.startsWith('10.') || hostname.startsWith('172.')) {
            // Local network IP (mobile access or other device)
            this.backendUrl = `http://${hostname}:8001`;
        } else if (hostname.includes('netlify') || hostname.includes('localhost')) {
            // Netlify production - use configured IP
            this.backendUrl = 'http://10.57.205.169:8001';
        } else {
            // Fallback
            this.backendUrl = `http://${hostname}:8001`;
        }

        this.initializeEventListeners();
        this.initializeModal();
        this.showWelcomeMessage();
    }

    // Initialize modal event listeners
    initializeModal() {
        const modal = document.getElementById('documentsModal');
        const closeBtn = document.getElementById('modalClose');
        
        closeBtn.addEventListener('click', () => {
            modal.classList.remove('active');
        });
        
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.classList.remove('active');
            }
        });
    }

    // Initialize event listeners
    initializeEventListeners() {
        this.sendBtn.addEventListener('click', () => this.handleSendMessage());
        this.messageInput.addEventListener('keydown', (e) => this.handleKeyDown(e));
        this.messageInput.addEventListener('input', () => this.adjustTextareaHeight());
        this.settingsBtn.addEventListener('click', () => this.handleSettings());
        
        // Mobile-specific improvements
        this.messageInput.addEventListener('focus', () => {
            // Scroll into view on mobile when keyboard appears
            if (this.isMobileDevice()) {
                setTimeout(() => this.messageInput.scrollIntoView({ behavior: 'smooth' }), 500);
            }
        });
        
        // Prevent double-tap zoom on buttons (mobile)
        document.querySelectorAll('button').forEach(btn => {
            btn.addEventListener('touchstart', () => {}, { passive: true });
        });
    }

    // Handle key press events
    handleKeyDown(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.handleSendMessage();
        }
    }

    // Adjust textarea height based on content
    adjustTextareaHeight() {
        this.messageInput.style.height = 'auto';
        const maxHeight = this.isMobileDevice() ? 100 : 120;
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, maxHeight) + 'px';
    }

    // Handle send message
    async handleSendMessage() {
        const message = this.messageInput.value.trim();

        if (!message) return;

        // Disable send button while processing
        this.sendBtn.disabled = true;
        this.isWaitingForResponse = true;

        // Add user message to chat
        this.addMessage(message, 'user');

        // Clear input
        this.messageInput.value = '';
        this.adjustTextareaHeight();

        // Add typing indicator
        this.showTypingIndicator();

        try {
            // Call backend RAG API
            const response = await this.sendMessageToBackend(message);

            // Remove typing indicator
            this.removeTypingIndicator();

            // Add bot response
            this.addMessage(response, 'bot');

            // Store in history
            this.messageHistory.push({
                user: message,
                bot: response,
                timestamp: new Date()
            });
        } catch (error) {
            this.removeTypingIndicator();
            this.addMessage(
                `⚠️ Error: ${error.message || 'Unable to process your question. Make sure the backend server is running on port 8001.'}`,
                'bot',
                'error'
            );
        } finally {
            this.sendBtn.disabled = false;
            this.isWaitingForResponse = false;
            this.messageInput.focus();
        }
    }

    // Send message to backend
    async sendMessageToBackend(message) {
        try {
            const response = await fetch(`${this.backendUrl}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: message })
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();
            
            // Format response with citations/sources
            let formatted = data.answer || 'No response';
            
            // Always show all sources/citations
            if (data.sources && data.sources.length > 0) {
                formatted += '\n\n📚 SOURCES/CITATIONS:';
                data.sources.forEach((source, index) => {
                    formatted += `\n[${index + 1}] ${source}`;
                });
                
                // Add confidence score if available
                if (data.confidence !== undefined && data.confidence > 0) {
                    formatted += `\n\nConfidence: ${Math.round(data.confidence * 100)}%`;
                }
            }
            
            return formatted;
        } catch (error) {
            console.warn('Backend not available. Using demo mode.');
            return this.getDemoResponse(message);
        }
    }

    // Get demo response (fallback when backend unavailable)
    getDemoResponse(question) {
        const lowerQuestion = question.toLowerCase();
        
        const responses = {
            'contract': 'A contract is an agreement between two or more parties that creates mutual obligations and is enforceable by law. Under Indian Contract Act 2019, essential elements of a contract include: (1) Offer, (2) Acceptance, (3) Consideration, (4) Intention to create legal relations, and (5) Capacity of parties.',
            'consumer': 'Consumer Rights are protected under the Consumer Protection Act 1986. Key consumer rights include: (1) Right to safety, (2) Right to information, (3) Right to choose, (4) Right to be heard, (5) Right to seek redressal, and (6) Right to consumer education.',
            'rti': 'The Right to Information (RTI) Act is a central legislation that provides citizens the right to access information held by or under the control of the Government. It aims to promote transparency and accountability in administration.',
            'evidence': 'The Indian Evidence Act 1872 defines the rules of admissibility of evidence in courts of law. Evidence includes facts, documents, and things that help prove or disprove matters in issue. It must be relevant, material, and competent to be admissible.',
        };

        // Find matching response
        for (const [key, value] of Object.entries(responses)) {
            if (lowerQuestion.includes(key)) {
                return '📚 Demo Response:\n\n' + value + '\n\n⚠️ Note: Backend not connected. Please start the Python server on port 8001 for real document search.';
            }
        }

        // Generic response
        return 'I can help with questions about Contract Act, Consumer Protection Act, RTI Act, and Indian Evidence Act. Please start the backend server (port 8001) for full RAG functionality with your documents.';
    }

    // Add message to chat
    addMessage(text, sender = 'bot', type = 'normal') {
        const messageEl = document.createElement('div');
        messageEl.className = `message ${sender}-message`;

        const avatar = sender === 'user' ? '👤' : '⚖️';
        
        // Convert newlines to actual line breaks and preserve formatting
        const formattedText = this.escapeHtml(text).replace(/\n/g, '<br>');
        
        messageEl.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content" style="white-space: pre-wrap; word-break: break-word;">${formattedText}</div>
        `;

        if (type === 'error') {
            messageEl.classList.add('error');
        }

        this.messagesContainer.appendChild(messageEl);
        this.scrollToBottom();
    }

    // Show welcome message
    showWelcomeMessage() {
        const welcomeMsg = `👋 Welcome to Legal RAG Assistant!\n\nI have loaded 5 Indian legal documents:\n• Contract Act 2019\n• Consumer Protection Act 1986\n• Right to Information Act\n• Indian Evidence Act 1872\n\nAsk any legal questions and I'll search the documents for relevant information.\n\n💡 Try asking:\n• "What is a contract?"\n• "What are consumer rights?"\n• "Explain the RTI Act"\n• "What is evidence?"`;
        
        this.addMessage(welcomeMsg, 'bot');
    }

    // Show typing indicator
    showTypingIndicator() {
        const template = this.typingTemplate;
        const typingEl = template.content.cloneNode(true);
        this.messagesContainer.appendChild(typingEl);
        this.scrollToBottom();
    }

    // Remove typing indicator
    removeTypingIndicator() {
        const indicator = this.messagesContainer.querySelector('.typing-indicator');
        if (indicator) {
            indicator.closest('.message').remove();
        }
    }

    // Scroll to bottom of messages
    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        // For mobile, add extra scroll to ensure visibility
        if (this.isMobileDevice()) {
            setTimeout(() => {
                this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
            }, 100);
        }
    }

    // Detect if device is mobile
    isMobileDevice() {
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ||
               (window.innerWidth <= 768);
    }

    // Handle settings button click
    handleSettings() {
        const modal = document.getElementById('documentsModal');
        modal.classList.add('active');
    }

    // Escape HTML to prevent XSS
    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, m => map[m]);
    }
}

// Initialize chat manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ChatManager();
});
