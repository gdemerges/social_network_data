/**
 * Auto-scroll pour le chat history
 * Scroll automatiquement vers le bas quand un nouveau message est ajouté
 */

// Fonction pour scroller vers le bas
function scrollChatToBottom() {
    const chatHistory = document.getElementById('chat-history');
    if (chatHistory) {
        // Smooth scroll vers le bas
        chatHistory.scrollTo({
            top: chatHistory.scrollHeight,
            behavior: 'smooth'
        });
    }
}

// Observer les changements dans le chat history
function setupChatObserver() {
    const chatHistory = document.getElementById('chat-history');

    if (chatHistory) {
        // Créer un MutationObserver pour détecter les nouveaux messages
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                    // Un nouveau message a été ajouté, scroller vers le bas
                    scrollChatToBottom();
                }
            });
        });

        // Observer les changements dans les enfants du chat history
        observer.observe(chatHistory, {
            childList: true,
            subtree: true
        });

        // Scroll initial
        scrollChatToBottom();
    }
}

// Attendre que le DOM soit chargé
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupChatObserver);
} else {
    setupChatObserver();
}

// Re-setup si Dash recharge le layout
setInterval(() => {
    const chatHistory = document.getElementById('chat-history');
    if (chatHistory && !chatHistory.dataset.observerSetup) {
        chatHistory.dataset.observerSetup = 'true';
        setupChatObserver();
    }
}, 1000);
