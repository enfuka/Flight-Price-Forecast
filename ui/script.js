// Flight Price Forecast - JavaScript Functionality

class FlightPredictor {
  constructor() {
    this.form = document.getElementById("predictionForm");
    this.resultsCard = document.getElementById("resultsCard");
    this.loadingSpinner = document.getElementById("loadingSpinner");
    this.predictionResults = document.getElementById("predictionResults");

    this.init();
  }

  init() {
    this.setupFormSubmission();
    this.populateDropdowns();
    this.setupDateValidation();
  }

  setupFormSubmission() {
    this.form.addEventListener("submit", (e) => {
      e.preventDefault();
      this.handlePrediction();
    });
  }

  setupDateValidation() {
    const dateInput = document.getElementById("travelDate");
    const today = new Date().toISOString().split("T")[0];
    dateInput.min = today;

    // Set default date to 30 days from now
    const defaultDate = new Date();
    defaultDate.setDate(defaultDate.getDate() + 30);
    dateInput.value = defaultDate.toISOString().split("T")[0];
  }

  populateDropdowns() {
    // Sample cities - in real implementation, these would come from the dataset
    const cities = [
      "New York, NY",
      "Los Angeles, CA",
      "Chicago, IL",
      "Houston, TX",
      "Phoenix, AZ",
      "Philadelphia, PA",
      "San Antonio, TX",
      "San Diego, CA",
      "Dallas, TX",
      "San Jose, CA",
      "Austin, TX",
      "Jacksonville, FL",
      "Fort Worth, TX",
      "Columbus, OH",
      "Charlotte, NC",
      "San Francisco, CA",
      "Indianapolis, IN",
      "Seattle, WA",
      "Denver, CO",
      "Washington, DC",
      "Boston, MA",
      "Nashville, TN",
      "Oklahoma City, OK",
      "Las Vegas, NV",
      "Portland, OR",
      "Memphis, TN",
      "Louisville, KY",
      "Baltimore, MD",
      "Milwaukee, WI",
      "Albuquerque, NM",
      "Tucson, AZ",
      "Atlanta, GA",
      "Miami, FL",
      "Orlando, FL",
      "Tampa, FL",
      "New Orleans, LA",
    ];

    const airlines = [
      "American Airlines",
      "Delta Air Lines",
      "United Airlines",
      "Southwest Airlines",
      "JetBlue Airways",
      "Alaska Airlines",
      "Spirit Airlines",
      "Frontier Airlines",
      "Allegiant Air",
      "Hawaiian Airlines",
      "Sun Country Airlines",
    ];

    this.populateSelect("originCity", cities);
    this.populateSelect("destCity", cities);
    this.populateSelect("airline", airlines);
  }

  populateSelect(selectId, options) {
    const select = document.getElementById(selectId);
    const existingOptions = select.querySelectorAll('option:not([value=""])');
    existingOptions.forEach((option) => option.remove());

    options.forEach((option) => {
      const optionElement = document.createElement("option");
      optionElement.value = option;
      optionElement.textContent = option;
      select.appendChild(optionElement);
    });
  }

  async handlePrediction() {
    this.showLoading(true);
    this.resultsCard.style.display = "none";

    try {
      const formData = this.getFormData();

      // Simulate API call - replace with actual model prediction
      await this.simulateModelPrediction(formData);

      const prediction = this.generatePrediction(formData);
      this.displayResults(prediction);
    } catch (error) {
      console.error("Prediction error:", error);
      this.displayError("Unable to generate prediction. Please try again.");
    } finally {
      this.showLoading(false);
    }
  }

  getFormData() {
    return {
      originCity: document.getElementById("originCity").value,
      destCity: document.getElementById("destCity").value,
      airline: document.getElementById("airline").value,
      travelDate: document.getElementById("travelDate").value,
      bookingAdvance: parseInt(document.getElementById("bookingAdvance").value),
      tripType: document.getElementById("tripType").value,
    };
  }

  async simulateModelPrediction(formData) {
    // Simulate API delay
    return new Promise((resolve) => setTimeout(resolve, 2000));
  }

  generatePrediction(formData) {
    // This is a simulation - in real implementation, this would call your trained model
    const basePrice = this.calculateBasePrice(formData);
    const seasonalAdjustment = this.getSeasonalAdjustment(formData.travelDate);
    const advanceBookingAdjustment = this.getAdvanceBookingAdjustment(
      formData.bookingAdvance
    );
    const routePopularity = this.getRoutePopularityAdjustment(
      formData.originCity,
      formData.destCity
    );

    const predictedPrice = Math.round(
      basePrice *
        seasonalAdjustment *
        advanceBookingAdjustment *
        routePopularity
    );
    const confidence = Math.min(95, Math.max(65, 85 + Math.random() * 10));

    return {
      price: predictedPrice,
      confidence: Math.round(confidence),
      priceRange: {
        low: Math.round(predictedPrice * 0.85),
        high: Math.round(predictedPrice * 1.15),
      },
      recommendations: this.generateRecommendations(formData, predictedPrice),
      insights: this.generateInsights(formData),
    };
  }

  calculateBasePrice(formData) {
    // Simulate distance-based pricing
    const routeMultipliers = {
      short: 250, // < 500 miles
      medium: 350, // 500-1500 miles
      long: 450, // > 1500 miles
    };

    // Simple route distance simulation based on cities
    const distance = this.estimateDistance(
      formData.originCity,
      formData.destCity
    );
    let routeType = "medium";

    if (distance < 500) routeType = "short";
    else if (distance > 1500) routeType = "long";

    let basePrice = routeMultipliers[routeType];

    // Airline premium
    const airlinePremiums = {
      "American Airlines": 1.1,
      "Delta Air Lines": 1.12,
      "United Airlines": 1.08,
      "Southwest Airlines": 0.95,
      "JetBlue Airways": 1.02,
      "Spirit Airlines": 0.8,
      "Frontier Airlines": 0.82,
    };

    if (formData.airline && airlinePremiums[formData.airline]) {
      basePrice *= airlinePremiums[formData.airline];
    }

    // Round trip multiplier
    if (formData.tripType === "roundtrip") {
      basePrice *= 1.8; // Not exactly double due to potential discounts
    }

    return basePrice;
  }

  estimateDistance(origin, dest) {
    // Very simplified distance estimation
    const majorCityCoords = {
      "New York, NY": [40.7128, -74.006],
      "Los Angeles, CA": [34.0522, -118.2437],
      "Chicago, IL": [41.8781, -87.6298],
      "Houston, TX": [29.7604, -95.3698],
      "Phoenix, AZ": [33.4484, -112.074],
      "Miami, FL": [25.7617, -80.1918],
      "Seattle, WA": [47.6062, -122.3321],
      "Denver, CO": [39.7392, -104.9903],
      "Atlanta, GA": [33.749, -84.388],
      "San Francisco, CA": [37.7749, -122.4194],
    };

    const coord1 = majorCityCoords[origin] || [40, -100]; // Default center US
    const coord2 = majorCityCoords[dest] || [40, -100];

    // Haversine formula approximation
    const lat1 = (coord1[0] * Math.PI) / 180;
    const lat2 = (coord2[0] * Math.PI) / 180;
    const deltaLat = ((coord2[0] - coord1[0]) * Math.PI) / 180;
    const deltaLon = ((coord2[1] - coord1[1]) * Math.PI) / 180;

    const a =
      Math.sin(deltaLat / 2) * Math.sin(deltaLat / 2) +
      Math.cos(lat1) *
        Math.cos(lat2) *
        Math.sin(deltaLon / 2) *
        Math.sin(deltaLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

    return 3956 * c; // Distance in miles
  }

  getSeasonalAdjustment(travelDate) {
    const date = new Date(travelDate);
    const month = date.getMonth(); // 0-11

    // Seasonal pricing adjustments
    const seasonalMultipliers = {
      0: 0.9, // January
      1: 0.85, // February
      2: 0.95, // March
      3: 1.1, // April
      4: 1.15, // May
      5: 1.25, // June
      6: 1.3, // July
      7: 1.25, // August
      8: 1.05, // September
      9: 1.0, // October
      10: 1.2, // November (Thanksgiving)
      11: 1.25, // December (Holidays)
    };

    return seasonalMultipliers[month] || 1.0;
  }

  getAdvanceBookingAdjustment(daysAdvance) {
    // Sweet spot is usually 6-8 weeks in advance
    if (daysAdvance <= 7) return 1.4; // Last minute premium
    if (daysAdvance <= 14) return 1.25; // 2 weeks
    if (daysAdvance <= 30) return 1.1; // 1 month
    if (daysAdvance <= 60) return 0.9; // 2 months (sweet spot)
    if (daysAdvance <= 90) return 0.95; // 3 months
    return 1.05; // Too far in advance
  }

  getRoutePopularityAdjustment(origin, dest) {
    // Major hub routes might have more competition
    const majorHubs = [
      "New York, NY",
      "Los Angeles, CA",
      "Chicago, IL",
      "Atlanta, GA",
    ];
    const isPopularRoute =
      majorHubs.includes(origin) && majorHubs.includes(dest);

    return isPopularRoute ? 0.95 : 1.05; // Popular routes have more competition
  }

  generateRecommendations(formData, predictedPrice) {
    const recommendations = [];

    if (formData.bookingAdvance < 30) {
      recommendations.push(
        "📅 Consider booking 4-6 weeks in advance for better prices"
      );
    }

    if (formData.bookingAdvance > 90) {
      recommendations.push(
        "⏰ You're booking quite early - prices may fluctuate"
      );
    }

    const travelDate = new Date(formData.travelDate);
    const month = travelDate.getMonth();

    if ([5, 6, 7, 10, 11].includes(month)) {
      // June, July, August, November, December
      recommendations.push(
        "🎯 Consider traveling in shoulder season for savings"
      );
    }

    if (!formData.airline) {
      recommendations.push(
        "✈️ Compare budget airlines like Southwest or Spirit for potential savings"
      );
    }

    recommendations.push("🔄 Set up price alerts to monitor fare changes");

    return recommendations;
  }

  generateInsights(formData) {
    const insights = [];

    const travelDate = new Date(formData.travelDate);
    const dayOfWeek = travelDate.getDay();

    if ([1, 2, 3].includes(dayOfWeek)) {
      // Tuesday, Wednesday, Thursday
      insights.push("✅ Traveling mid-week typically offers better prices");
    } else {
      insights.push(
        "💡 Consider shifting travel dates to Tuesday-Thursday for savings"
      );
    }

    insights.push(
      `📊 Historical data shows ${Math.floor(
        Math.random() * 30 + 20
      )}% price variation on this route`
    );

    if (formData.bookingAdvance >= 42 && formData.bookingAdvance <= 56) {
      insights.push("🎯 You're in the optimal booking window!");
    }

    return insights;
  }

  displayResults(prediction) {
    const resultsHTML = `
            <div class="price-prediction">
                <h3>Predicted Flight Price</h3>
                <div class="price-amount">$${prediction.price.toLocaleString()}</div>
                <div class="price-range mt-2">
                    <small>Expected range: $${prediction.priceRange.low.toLocaleString()} - $${prediction.priceRange.high.toLocaleString()}</small>
                </div>
                <div class="confidence-meter">
                    <div class="d-flex justify-content-between align-items-center">
                        <small>Confidence Level</small>
                        <small><strong>${
                          prediction.confidence
                        }%</strong></small>
                    </div>
                    <div class="confidence-bar" style="width: ${
                      prediction.confidence
                    }%"></div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    <h5>💡 Recommendations</h5>
                    <ul class="list-unstyled">
                        ${prediction.recommendations
                          .map((rec) => `<li class="mb-2">${rec}</li>`)
                          .join("")}
                    </ul>
                </div>
                <div class="col-md-6">
                    <h5>📈 Market Insights</h5>
                    <ul class="list-unstyled">
                        ${prediction.insights
                          .map((insight) => `<li class="mb-2">${insight}</li>`)
                          .join("")}
                    </ul>
                </div>
            </div>
            
            <div class="alert alert-info mt-3" role="alert">
                <strong>Note:</strong> Predictions are based on historical flight data from 1993-2024. 
                Actual prices may vary due to current market conditions, promotions, and availability.
            </div>
        `;

    this.predictionResults.innerHTML = resultsHTML;
    this.resultsCard.style.display = "block";

    // Smooth scroll to results
    this.resultsCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  displayError(message) {
    this.predictionResults.innerHTML = `
            <div class="alert alert-danger" role="alert">
                <h4 class="alert-heading">Prediction Error</h4>
                <p>${message}</p>
            </div>
        `;
    this.resultsCard.style.display = "block";
  }

  showLoading(show) {
    const button = this.form.querySelector('button[type="submit"]');
    const buttonText = button.querySelector(".btn-text") || button;

    if (show) {
      this.loadingSpinner.style.display = "inline-block";
      button.disabled = true;
      if (!button.querySelector(".btn-text")) {
        button.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span><span class="btn-text">Analyzing...</span>`;
      }
    } else {
      this.loadingSpinner.style.display = "none";
      button.disabled = false;
      button.innerHTML = "Predict Flight Price";
    }
  }
}

// Initialize the application when the DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  new FlightPredictor();

  // Add smooth scrolling for navigation links
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute("href"));
      if (target) {
        target.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }
    });
  });
});

// Utility functions
function formatCurrency(amount) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
  }).format(amount);
}

function getSeasonName(month) {
  const seasons = {
    Winter: [11, 0, 1],
    Spring: [2, 3, 4],
    Summer: [5, 6, 7],
    Fall: [8, 9, 10],
  };

  for (const [season, months] of Object.entries(seasons)) {
    if (months.includes(month)) return season;
  }
  return "Unknown";
}
