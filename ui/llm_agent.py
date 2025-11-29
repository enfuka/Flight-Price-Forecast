"""
LLM Agent for Flight Price Forecast
Provides intelligent conversational interface using Google Gemini for flight price predictions and travel advice
"""

import os
from typing import Dict, List, Optional
import json
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class FlightLLMAgent:
    """
    Intelligent agent that combines flight price predictions with natural language understanding
    Uses Google Gemini API for conversational AI
    """

    def __init__(self, predictor):
        """
        Initialize the LLM agent with Google Gemini

        Args:
            predictor: FlightPricePredictor instance from app.py
        """
        self.predictor = predictor
        self.conversation_history = []
        self.setup_llm()

    def setup_llm(self):
        """Setup Google Gemini client"""
        # Requires: pip install google-generativeai
        # Set environment variable: GEMINI_API_KEY
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        genai.configure(api_key=api_key)

        # Use Gemini 1.5 Flash for fast, cost-effective responses
        # Or use gemini-1.5-pro for higher quality (more expensive)
        self.model = genai.GenerativeModel(
            'models/gemini-2.5-flash',
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )

        # Initialize chat session
        self.chat_session = self.model.start_chat(history=[])

    def get_system_prompt(self) -> str:
        """Define the agent's role and capabilities"""
        return """You are a helpful flight booking assistant with access to real-time flight price predictions.

Your capabilities:
1. Predict flight prices based on routes, dates, and booking timing
2. Provide personalized travel recommendations
3. Explain price trends and market factors
4. Suggest optimal booking strategies
5. Compare different airlines and routes

Guidelines:
- Be conversational and friendly
- Provide specific, actionable advice
- Explain your reasoning when suggesting alternatives
- Use the prediction data to support your recommendations
- Always mention that predictions are estimates based on historical data
- If you need more information to make a prediction, ask clarifying questions

Current date: {current_date}

When users ask about flights, you can help them by:
- Asking for origin city, destination city, travel date, and how far in advance they're booking
- Providing price estimates when you have enough information
- Offering booking advice and cost-saving tips
- Explaining seasonal pricing trends
"""

    def create_tools(self) -> List[Dict]:
        """Define function calling schema for Gemini"""
        return [
            {
                "name": "predict_flight_price",
                "description": "Predict airline ticket price for a specific route and date. Use this when the user provides enough information about their desired flight.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {
                            "type": "string",
                            "description": "Origin city with state, e.g., 'New York, NY'"
                        },
                        "destination": {
                            "type": "string",
                            "description": "Destination city with state, e.g., 'Los Angeles, CA'"
                        },
                        "travel_date": {
                            "type": "string",
                            "description": "Travel date in YYYY-MM-DD format"
                        },
                        "days_advance": {
                            "type": "integer",
                            "description": "Number of days in advance of booking"
                        },
                        "airline": {
                            "type": "string",
                            "description": "Preferred airline name (optional, use empty string if not specified)"
                        },
                        "trip_type": {
                            "type": "string",
                            "enum": ["roundtrip", "oneway"],
                            "description": "Type of trip"
                        }
                    },
                    "required": ["origin", "destination", "travel_date", "days_advance"]
                }
            }
        ]

    def predict_flight_price(self, origin: str, destination: str, travel_date: str,
                             days_advance: int, airline: str = "", trip_type: str = "roundtrip") -> Dict:
        """Call the price predictor and format results for LLM"""
        result = self.predictor.predict_price(
            origin, destination, airline, travel_date, days_advance, trip_type
        )
        return result

    def chat(self, user_message: str, context: Optional[Dict] = None) -> Dict:
        """
        Process user message and generate response using Google Gemini

        Args:
            user_message: User's message
            context: Optional context (previous predictions, user preferences, etc.)

        Returns:
            Dict with 'response' text and optional 'prediction' data for UI sync
        """
        try:
            # Build the complete message with system prompt and context
            full_message = self.get_system_prompt().format(
                current_date=datetime.now().strftime("%Y-%m-%d")
            )

            if context:
                full_message += f"\n\nAdditional context: {json.dumps(context)}"

            full_message += f"\n\nUser: {user_message}"

            # Check if we need to extract flight information and make a prediction
            result = self._process_with_gemini(full_message, user_message)

            return result

        except Exception as e:
            error_msg = f"I encountered an error: {str(e)}. Please try again or rephrase your question."
            return {'response': error_msg, 'prediction': None, 'flight_info': None}

    def _process_with_gemini(self, full_message: str, user_message: str) -> Dict:
        """Process message with Gemini and handle flight predictions"""

        # Send message to Gemini
        response = self.chat_session.send_message(full_message)
        response_text = response.text

        # Check if the response suggests we need flight information
        # If user provided specific flight details, try to extract and predict
        if self._should_make_prediction(user_message, response_text):
            flight_info = self._extract_flight_info(user_message)

            if flight_info and self._has_required_fields(flight_info):
                # Make prediction
                prediction = self.predict_flight_price(**flight_info)
                formatted_result = self._format_prediction_result(prediction)

                # Ask Gemini to incorporate the prediction into a natural response
                follow_up = f"I got this prediction result:\n{formatted_result}\n\nPlease provide a conversational response to the user incorporating this flight price information."
                final_response = self.chat_session.send_message(follow_up)
                
                # Return structured data for UI sync
                return {
                    'response': final_response.text,
                    'prediction': prediction,
                    'flight_info': flight_info
                }

        return {'response': response_text, 'prediction': None, 'flight_info': None}

    def _should_make_prediction(self, user_message: str, response_text: str) -> bool:
        """Determine if we should attempt to make a price prediction"""
        # Keywords that suggest user wants a price prediction
        price_keywords = ['price', 'cost', 'fare',
                          'how much', 'predict', 'estimate']
        route_keywords = ['from', 'to', 'flying', 'flight', 'book']

        user_lower = user_message.lower()
        has_price_intent = any(
            keyword in user_lower for keyword in price_keywords)
        has_route_intent = any(
            keyword in user_lower for keyword in route_keywords)

        return has_price_intent or has_route_intent

    def _extract_flight_info(self, user_message: str) -> Optional[Dict]:
        """Use Gemini to extract flight information from user message"""
        extraction_prompt = f"""Extract flight booking information from this message: "{user_message}"

Return ONLY a JSON object with these fields (use null for missing fields):
{{
    "origin": "City, State",
    "destination": "City, State", 
    "travel_date": "YYYY-MM-DD",
    "days_advance": number,
    "airline": "Airline Name or empty string",
    "trip_type": "roundtrip or oneway"
}}

Current date is {datetime.now().strftime("%Y-%m-%d")} for calculating dates and advance booking days.
If the user doesn't specify days_advance but gives a travel date, calculate it as the difference between travel date and today.
Only respond with the JSON, no other text."""

        try:
            extraction_response = self.model.generate_content(
                extraction_prompt)
            # Clean up response and parse JSON
            json_text = extraction_response.text.strip()
            # Remove markdown code blocks if present
            if json_text.startswith('```'):
                json_text = json_text.split('```')[1]
                if json_text.startswith('json'):
                    json_text = json_text[4:]
            json_text = json_text.strip()

            flight_info = json.loads(json_text)
            return flight_info
        except Exception as e:
            print(f"Could not extract flight info: {e}")
            return None

    def _has_required_fields(self, flight_info: Dict) -> bool:
        """Check if we have minimum required fields for prediction"""
        required = ['origin', 'destination', 'travel_date', 'days_advance']
        return all(
            flight_info.get(field) is not None and
            flight_info.get(field) != "" and
            flight_info.get(field) != "null"
            for field in required
        )

    def _format_prediction_result(self, result: Dict) -> str:
        """Format prediction result for user-friendly display"""
        if result.get('status') == 'error':
            return f"I couldn't get a price prediction: {result.get('error')}"

        fare = result['predicted_fare']
        confidence = result['market_confidence']
        fare_classes = result.get('fare_classes', {})

        response = f"""Based on my analysis:

**Predicted Fare**: ${fare:,}
**Confidence Level**: {confidence}%

**Fare Range by Class**:
- Economy: ${fare_classes.get('economy', 'N/A'):,}
- Premium Economy: ${fare_classes.get('premium_economy', 'N/A'):,}
- Business: ${fare_classes.get('business', 'N/A'):,}

**Market Range**: ${fare_classes.get('market_low', 'N/A'):,} - ${fare_classes.get('market_high', 'N/A'):,}
"""

        # Add market analysis if available
        if 'market_analysis' in result:
            analysis = result['market_analysis']
            response += f"\n**Market Insights**:\n"
            response += f"- Route Type: {analysis.get('route_type', 'N/A')}\n"
            response += f"- Season: {analysis.get('season', 'N/A')}\n"
            response += f"- Booking Timing: {analysis.get('booking_timing', 'N/A')}\n"

        return response

    def reset_conversation(self):
        """Clear conversation history and start fresh chat session"""
        self.chat_session = self.model.start_chat(history=[])


# Example usage
if __name__ == "__main__":
    # This would normally be imported from app.py
    from app import FlightPricePredictor

    # Make sure to set GEMINI_API_KEY environment variable
    predictor = FlightPricePredictor()
    agent = FlightLLMAgent(predictor)

    # Test conversation
    print("Flight Assistant: Hello! I can help you find the best flight prices and travel recommendations.")
    print("Flight Assistant: Ask me about flight prices, booking strategies, or travel advice!\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Flight Assistant: Safe travels! Goodbye!")
            break

        response = agent.chat(user_input)
        print(f"\nFlight Assistant: {response}\n")
