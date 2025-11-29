// LLM Chat Agent Integration for Flight Price Forecast

class ChatAgent {
  constructor() {
    this.isOpen = false;
    this.messages = [];
    this.isTyping = false;
    this.init();
  }

  init() {
    this.createChatWidget();
    this.attachEventListeners();
    this.loadWelcomeMessage();
  }

  createChatWidget() {
    const chatHTML = `
      <div class="chat-widget">
        <button class="chat-button" id="chatToggle" title="Chat with AI Assistant">
          💬
        </button>
        
        <div class="chat-window" id="chatWindow">
          <div class="chat-header">
            <div class="chat-header-info">
              <h4>Flight Assistant 🤖</h4>
              <div class="chat-status" id="chatStatus">Online</div>
            </div>
            <div class="chat-header-buttons">
              <button class="chat-header-btn" id="chatReset" title="Reset conversation">🔄</button>
              <button class="chat-header-btn chat-close-btn" id="chatClose" title="Close">✕</button>
            </div>
          </div>
          
          <div class="chat-suggestions" id="chatSuggestions">
            <div class="suggestion-chip" data-suggestion="Find me flights to California">
              Find flights to CA
            </div>
            <div class="suggestion-chip" data-suggestion="What's the best time to book?">
              Best booking time?
            </div>
            <div class="suggestion-chip" data-suggestion="Compare airlines">
              Compare airlines
            </div>
          </div>
          
          <div class="chat-messages" id="chatMessages">
            <!-- Messages will be inserted here -->
          </div>
          
          <div class="chat-typing" id="chatTyping">
            <div class="typing-dots">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
          
          <div class="chat-input-container">
            <div class="chat-input-wrapper">
              <textarea 
                class="chat-input" 
                id="chatInput" 
                placeholder="Ask about flight prices, booking tips, or travel advice..."
                rows="1"
              ></textarea>
              <button class="chat-send" id="chatSend" title="Send message">
                ➤
              </button>
            </div>
          </div>
        </div>
      </div>
    `;

    document.body.insertAdjacentHTML("beforeend", chatHTML);
  }

  attachEventListeners() {
    const toggleBtn = document.getElementById("chatToggle");
    const closeBtn = document.getElementById("chatClose");
    const sendBtn = document.getElementById("chatSend");
    const input = document.getElementById("chatInput");
    const suggestions = document.querySelectorAll(".suggestion-chip");

    toggleBtn.addEventListener("click", () => this.toggleChat());
    closeBtn.addEventListener("click", () => this.toggleChat());
    sendBtn.addEventListener("click", () => this.sendMessage());
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });

    // Auto-resize textarea
    input.addEventListener("input", function () {
      this.style.height = "40px";
      const newHeight = Math.min(this.scrollHeight, 100);
      this.style.height = newHeight + "px";

      // Show scrollbar only when at max height
      if (this.scrollHeight > 100) {
        this.classList.add("expanded");
      } else {
        this.classList.remove("expanded");
      }
    });

    // Suggestion chips
    suggestions.forEach((chip) => {
      chip.addEventListener("click", (e) => {
        const suggestion = e.target.dataset.suggestion;
        document.getElementById("chatInput").value = suggestion;
        this.sendMessage();
      });
    });
  }

  toggleChat() {
    this.isOpen = !this.isOpen;
    const window = document.getElementById("chatWindow");
    const button = document.getElementById("chatToggle");

    if (this.isOpen) {
      window.classList.add("active");
      button.classList.add("active");
      document.getElementById("chatInput").focus();
    } else {
      window.classList.remove("active");
      button.classList.remove("active");
    }
  }

  loadWelcomeMessage() {
    const welcomeMsg = `Hello! I'm your AI flight assistant. I can help you with:

**Flight Prices** - Get predictions for any route
**Booking Tips** - Learn when to book for best prices
**Travel Advice** - Airlines, routes, and cost-saving tips

Try asking: "What's the price from New York to LA?" or "When should I book my flight?"`;

    this.addMessage("assistant", welcomeMsg);
  }

  async sendMessage() {
    const input = document.getElementById("chatInput");
    const message = input.value.trim();

    if (!message || this.isTyping) return;

    // Add user message
    this.addMessage("user", message);
    input.value = "";
    input.style.height = "auto";

    // Show typing indicator
    this.setTyping(true);

    try {
      // Check if LLM is available
      const healthResponse = await fetch("/api/health");
      const healthData = await healthResponse.json();

      if (!healthData.llm_available) {
        this.addMessage(
          "assistant",
          "I'm sorry, but the AI assistant is not currently available. The administrator needs to set up API keys. However, you can still use the flight price prediction form above!"
        );
        this.setTyping(false);
        return;
      }

      // Send to chat API
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: message,
          context: this.getContext(),
        }),
      });

      const data = await response.json();

      if (data.status === "success") {
        this.addMessage("assistant", data.response);

        // If there's prediction data, sync with the main form
        if (data.prediction && data.flight_info) {
          this.syncWithMainForm(data.flight_info, data.prediction);
        }
      } else {
        this.addMessage(
          "assistant",
          `I encountered an error: ${data.error}. Please try rephrasing your question.`
        );
      }
    } catch (error) {
      console.error("Chat error:", error);
      this.addMessage(
        "assistant",
        "I'm having trouble connecting right now. Please try again in a moment."
      );
    } finally {
      this.setTyping(false);
    }
  }

  addMessage(role, content) {
    const messagesDiv = document.getElementById("chatMessages");
    const messageDiv = document.createElement("div");
    messageDiv.className = `chat-message ${role}`;

    const bubble = document.createElement("div");
    bubble.className = `message-bubble ${role}`;

    // Convert markdown-style formatting to HTML
    const formattedContent = this.formatMessage(content);
    bubble.innerHTML = formattedContent;

    messageDiv.appendChild(bubble);
    messagesDiv.appendChild(messageDiv);

    // Scroll to bottom
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    // Store message
    this.messages.push({ role, content, timestamp: new Date() });
  }

  formatMessage(content) {
    // Convert markdown-style formatting to HTML
    let formatted = content
      // Escape HTML first (except our formatting)
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      // Bold text - inline styling (flows with text)
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      // Italic text
      .replace(/\*([^*]+)\*/g, "<em>$1</em>")
      // Convert double newlines to paragraph breaks
      .replace(/\n\n/g, "</p><p>")
      // Convert single newlines to line breaks
      .replace(/\n/g, "<br>")
      // Bullet lists (lines starting with -)
      .replace(/^- (.+)$/gm, '<span class="bullet-item">$1</span>');

    // Wrap in paragraph tags
    formatted = "<p>" + formatted + "</p>";

    // Clean up empty paragraphs
    formatted = formatted.replace(/<p><\/p>/g, "");

    return formatted;
  }

  setTyping(isTyping) {
    this.isTyping = isTyping;
    const typingDiv = document.getElementById("chatTyping");
    const sendBtn = document.getElementById("chatSend");

    if (isTyping) {
      typingDiv.classList.add("active");
      sendBtn.disabled = true;
      document.getElementById("chatStatus").textContent = "Typing...";
    } else {
      typingDiv.classList.remove("active");
      sendBtn.disabled = false;
      document.getElementById("chatStatus").textContent = "Online";
    }

    // Scroll to show typing indicator
    const messagesDiv = document.getElementById("chatMessages");
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
  }

  getContext() {
    // Gather context from the page that might be useful for the agent
    const context = {
      page: "flight_prediction",
      timestamp: new Date().toISOString(),
    };

    // If there's a form filled out, include that context
    const originCity = document.getElementById("originCity")?.value;
    const destCity = document.getElementById("destCity")?.value;
    const travelDate = document.getElementById("travelDate")?.value;

    if (originCity || destCity || travelDate) {
      context.current_search = {
        origin: originCity,
        destination: destCity,
        travel_date: travelDate,
      };
    }

    return context;
  }

  reset() {
    this.messages = [];
    document.getElementById("chatMessages").innerHTML = "";
    this.loadWelcomeMessage();

    fetch("/api/chat/reset", { method: "POST" }).catch((error) =>
      console.error("Reset error:", error)
    );
  }

  syncWithMainForm(flightInfo, prediction) {
    // Populate the main form with the extracted flight info
    try {
      // Set origin city
      if (flightInfo.origin) {
        const originSelect = document.getElementById("originCity");
        this.setSelectValue(originSelect, flightInfo.origin);
      }

      // Set destination city
      if (flightInfo.destination) {
        const destSelect = document.getElementById("destCity");
        this.setSelectValue(destSelect, flightInfo.destination);
      }

      // Set travel date
      if (flightInfo.travel_date) {
        const dateInput = document.getElementById("travelDate");
        if (dateInput) {
          dateInput.value = flightInfo.travel_date;
        }
      }

      // Set booking advance
      if (
        flightInfo.days_advance !== null &&
        flightInfo.days_advance !== undefined
      ) {
        const advanceSelect = document.getElementById("bookingAdvance");
        if (advanceSelect) {
          // Find closest matching option
          const options = [0, 7, 14, 30, 60, 90];
          const closest = options.reduce((prev, curr) =>
            Math.abs(curr - flightInfo.days_advance) <
            Math.abs(prev - flightInfo.days_advance)
              ? curr
              : prev
          );
          advanceSelect.value = closest.toString();
        }
      }

      // Set airline if specified
      if (flightInfo.airline) {
        const airlineSelect = document.getElementById("airline");
        this.setSelectValue(airlineSelect, flightInfo.airline);
      }

      // Set trip type
      if (flightInfo.trip_type) {
        const tripSelect = document.getElementById("tripType");
        if (tripSelect) {
          tripSelect.value = flightInfo.trip_type;
        }
      }

      // Display the prediction results in the main results area
      this.displayPredictionOnPage(prediction, flightInfo);

      // Show a notification that form was updated
      this.showFormSyncNotification();
    } catch (error) {
      console.error("Error syncing with form:", error);
    }
  }

  setSelectValue(selectElement, value) {
    if (!selectElement || !value) return;

    // Try exact match first
    for (let option of selectElement.options) {
      if (
        option.value.toLowerCase() === value.toLowerCase() ||
        option.text.toLowerCase() === value.toLowerCase()
      ) {
        selectElement.value = option.value;
        return;
      }
    }

    // Try partial match (city name without state)
    const searchTerm = value.split(",")[0].toLowerCase().trim();
    for (let option of selectElement.options) {
      if (
        option.value.toLowerCase().includes(searchTerm) ||
        option.text.toLowerCase().includes(searchTerm)
      ) {
        selectElement.value = option.value;
        return;
      }
    }
  }

  displayPredictionOnPage(prediction, flightInfo) {
    const resultsCard = document.getElementById("resultsCard");
    const predictionResults = document.getElementById("predictionResults");

    if (!resultsCard || !predictionResults || !prediction) return;

    const fare = prediction.predicted_fare || prediction.price || 0;
    const confidence =
      prediction.market_confidence || prediction.confidence || 75;
    const fareClasses = prediction.fare_classes || {};

    const resultsHTML = `
      <div class="price-prediction">
        <div class="d-flex align-items-center mb-2">
          <span class="badge bg-success me-2">🤖 AI Prediction</span>
          <small class="text-muted">via Chat Assistant</small>
        </div>
        <h3>Predicted Flight Price</h3>
        <div class="price-amount">$${Math.round(fare).toLocaleString()}</div>
        <div class="price-range mt-2">
          <small>Market range: $${Math.round(
            fareClasses.market_low || fare * 0.85
          ).toLocaleString()} - $${Math.round(
      fareClasses.market_high || fare * 1.15
    ).toLocaleString()}</small>
        </div>
        <div class="confidence-meter">
          <div class="d-flex justify-content-between align-items-center">
            <small>Confidence Level</small>
            <small><strong>${Math.round(confidence)}%</strong></small>
          </div>
          <div class="confidence-bar" style="width: ${confidence}%"></div>
        </div>
      </div>
      
      <div class="row mt-4">
        <div class="col-md-6">
          <h5>✈️ Flight Details</h5>
          <ul class="list-unstyled">
            <li class="mb-2"><strong>From:</strong> ${
              flightInfo.origin || "N/A"
            }</li>
            <li class="mb-2"><strong>To:</strong> ${
              flightInfo.destination || "N/A"
            }</li>
            <li class="mb-2"><strong>Date:</strong> ${
              flightInfo.travel_date || "N/A"
            }</li>
            <li class="mb-2"><strong>Trip:</strong> ${
              flightInfo.trip_type || "roundtrip"
            }</li>
          </ul>
        </div>
        <div class="col-md-6">
          <h5>💰 Fare Classes</h5>
          <ul class="list-unstyled">
            <li class="mb-2"><strong>Economy:</strong> $${Math.round(
              fareClasses.economy || fare
            ).toLocaleString()}</li>
            <li class="mb-2"><strong>Premium:</strong> $${Math.round(
              fareClasses.premium_economy || fare * 1.4
            ).toLocaleString()}</li>
            <li class="mb-2"><strong>Business:</strong> $${Math.round(
              fareClasses.business || fare * 2.5
            ).toLocaleString()}</li>
          </ul>
        </div>
      </div>
      
      <div class="alert alert-info mt-3" role="alert">
        <strong>💬 Chat-Synced Result:</strong> This prediction was generated through the AI assistant and synced to the main form.
      </div>
    `;

    predictionResults.innerHTML = resultsHTML;
    resultsCard.style.display = "block";

    // Smooth scroll to results
    resultsCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  showFormSyncNotification() {
    // Create a temporary notification
    const notification = document.createElement("div");
    notification.className = "chat-sync-notification";
    notification.innerHTML = `
      <div class="sync-icon">✓</div>
      <div class="sync-text">Form updated with flight details</div>
    `;
    document.body.appendChild(notification);

    // Animate and remove
    setTimeout(() => notification.classList.add("show"), 10);
    setTimeout(() => {
      notification.classList.remove("show");
      setTimeout(() => notification.remove(), 300);
    }, 2500);
  }
}

// Initialize chat agent when page loads
document.addEventListener("DOMContentLoaded", () => {
  window.chatAgent = new ChatAgent();

  // Attach reset button handler
  const resetBtn = document.getElementById("chatReset");
  if (resetBtn) {
    resetBtn.onclick = () => window.chatAgent.reset();
  }
});
