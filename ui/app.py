# Flight Price Forecast - Flask Backend Application
# This file serves the trained machine learning models via REST API

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import LLM agent (optional - will gracefully handle if dependencies not installed)
try:
    from llm_agent import FlightLLMAgent
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("LLM agent not available. Install google-generativeai to enable AI assistant: pip install google-generativeai")

app = Flask(__name__,
            static_folder='.',
            template_folder='.')
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlightPricePredictor:
    """
    Flight price predictor using trained ML model on DOT airline market data.

    The model predicts fares based on route market characteristics:
    - large_ms: Market share of largest carrier
    - lf_ms: Market share of lowest fare carrier  
    - fare_low: Lowest fare on route
    - distance_cat_encoded: Distance category (0=long, 1=medium, 2=short)
    - nsmiles: Distance in miles
    - combined_ms: Combined market share
    - fare_spread: Difference between highest and lowest fares
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.route_data = None
        self.city_mapping = {}
        self.load_models()
        self.load_route_data()

    def load_models(self):
        """Load trained models and preprocessors"""
        try:
            # Get the directory containing app.py
            app_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(app_dir)
            
            model_path = os.path.join(project_root, 'models', 'best_model.pkl')
            scaler_path = os.path.join(project_root, 'models', 'scaler.pkl')
            features_path = os.path.join(project_root, 'models', 'feature_columns.json')

            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info("Model loaded successfully")
            else:
                logger.warning(f"Model not found at {model_path}")
                self.model = "mock"

            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("Scaler loaded successfully")

            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    self.feature_columns = json.load(f)
                # Remove 'fare' if accidentally included (it's the target, not a feature)
                if 'fare' in self.feature_columns:
                    self.feature_columns.remove('fare')
                    logger.warning(
                        "Removed 'fare' from features - it's the target variable!")
                logger.info(f"Feature columns loaded: {self.feature_columns}")
            else:
                # Default feature columns based on feature_metadata.json
                self.feature_columns = ['large_ms', 'lf_ms', 'fare_lg', 'fare_low',
                                        'distance_cat_encoded', 'nsmiles', 'combined_ms', 'fare_spread']

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.model = "mock"

    def load_route_data(self):
        """Load route data for looking up market characteristics"""
        try:
            # Get the directory containing app.py
            app_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(app_dir)
            
            data_path = os.path.join(project_root, 'data', 'cleaned_data.csv')
            if os.path.exists(data_path):
                # Load route data and aggregate by route (most recent data)
                df = pd.read_csv(data_path)

                # Get the most recent data for each route
                df_recent = df.sort_values(
                    ['Year', 'quarter'], ascending=False)

                # Create route key and aggregate market stats
                df_recent['route_key'] = df_recent.apply(
                    lambda x: tuple(sorted([str(x['city1']).lower().strip(),
                                           str(x['city2']).lower().strip()])), axis=1
                )

                # Aggregate by route - take mean of recent quarters
                self.route_data = df_recent.groupby('route_key').agg({
                    'nsmiles': 'first',
                    'large_ms': 'mean',
                    'lf_ms': 'mean',
                    'fare_lg': 'mean',
                    'fare_low': 'mean',
                    'fare': 'mean',
                    'city1': 'first',
                    'city2': 'first'
                }).reset_index()

                # Calculate derived features
                self.route_data['fare_spread'] = self.route_data['fare_lg'] - \
                    self.route_data['fare_low']
                self.route_data['combined_ms'] = self.route_data['large_ms'] + \
                    self.route_data['lf_ms']

                # Distance category encoding (0=long, 1=medium, 2=short based on feature_metadata.json)
                self.route_data['distance_cat_encoded'] = pd.cut(
                    self.route_data['nsmiles'],
                    bins=[0, 500, 1500, float('inf')],
                    labels=[2, 1, 0]  # short=2, medium=1, long=0
                ).astype(int)

                # Build city name mapping for flexible matching
                self._build_city_mapping(df)

                logger.info(
                    f"Route data loaded: {len(self.route_data)} unique routes")
            else:
                logger.warning(f"Route data not found at {data_path}")
                self.route_data = None

        except Exception as e:
            logger.error(f"Error loading route data: {e}")
            self.route_data = None

    def _build_city_mapping(self, df):
        """Build mapping of city names for flexible matching"""
        cities = set(df['city1'].unique()) | set(df['city2'].unique())
        for city in cities:
            if pd.notna(city):
                city_lower = str(city).lower().strip()
                self.city_mapping[city_lower] = city
                # Also map common variations
                parts = city_lower.split(',')
                if len(parts) > 0:
                    self.city_mapping[parts[0].strip()] = city

    def _normalize_city(self, city_name):
        """Normalize city name for matching"""
        city_lower = str(city_name).lower().strip()

        # Direct match
        if city_lower in self.city_mapping:
            return self.city_mapping[city_lower]

        # Try without state suffix
        parts = city_lower.split(',')
        if len(parts) > 0 and parts[0].strip() in self.city_mapping:
            return self.city_mapping[parts[0].strip()]

        # Fuzzy match - find closest city
        for key in self.city_mapping:
            if parts[0].strip() in key or key in parts[0].strip():
                return self.city_mapping[key]

        return city_name

    def _get_route_features(self, origin, destination):
        """Look up market features for a route from historical data"""
        if self.route_data is None:
            return None

        # Normalize city names
        origin_norm = self._normalize_city(origin)
        dest_norm = self._normalize_city(destination)

        # Create route key (sorted for bidirectional matching)
        route_key = tuple(sorted([str(origin_norm).lower().strip(),
                                  str(dest_norm).lower().strip()]))

        # Look up route
        route_match = self.route_data[self.route_data['route_key'] == route_key]

        if len(route_match) > 0:
            return route_match.iloc[0]

        return None

    def _estimate_route_features(self, origin, destination):
        """Estimate market features when route is not in dataset"""
        # Use median values from dataset as baseline
        if self.route_data is not None and len(self.route_data) > 0:
            median_values = {
                'nsmiles': self.route_data['nsmiles'].median(),
                'large_ms': self.route_data['large_ms'].median(),
                'lf_ms': self.route_data['lf_ms'].median(),
                'fare_low': self.route_data['fare_low'].median(),
                'fare_spread': self.route_data['fare_spread'].median(),
                'combined_ms': self.route_data['combined_ms'].median(),
                'distance_cat_encoded': 1,  # medium distance
                'fare': self.route_data['fare'].median()
            }
        else:
            # Fallback defaults based on typical US domestic routes
            median_values = {
                'nsmiles': 1000,
                'large_ms': 0.45,
                'lf_ms': 0.15,
                'fare_low': 150,
                'fare_spread': 80,
                'combined_ms': 0.60,
                'distance_cat_encoded': 1,
                'fare': 220
            }

        return pd.Series(median_values)

    def prepare_features(self, origin, destination):
        """Prepare features for the ML model"""
        # Try to get actual route data
        route_features = self._get_route_features(origin, destination)

        if route_features is None:
            logger.info(
                f"Route {origin} -> {destination} not found, using estimates")
            route_features = self._estimate_route_features(origin, destination)
            is_estimated = True
        else:
            is_estimated = False

        # Build feature vector matching model's expected columns
        # Model was trained with: ['large_ms', 'lf_ms', 'fare_low', 'distance_cat_encoded',
        #                          'nsmiles', 'combined_ms', 'fare_spread', 'fare']
        # Note: 'fare' is included as a feature (historical average) even though it's also target

        feature_cols = self.feature_columns.copy()

        features = {}
        for col in feature_cols:
            if col in route_features.index:
                features[col] = float(route_features[col])
            else:
                # Provide sensible defaults
                defaults = {
                    'large_ms': 0.45,
                    'lf_ms': 0.15,
                    'fare_low': 150,
                    'distance_cat_encoded': 1,
                    'nsmiles': 1000,
                    'combined_ms': 0.60,
                    'fare_spread': 80,
                    'fare_lg': 230,
                    'fare': 220  # Historical average fare for route
                }
                features[col] = defaults.get(col, 0)

        return pd.DataFrame([features])[feature_cols], route_features, is_estimated

    def get_season(self, month):
        """Get season from month"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    def get_seasonal_multiplier(self, month):
        """Get seasonal price multiplier"""
        # Summer and holidays typically more expensive
        multipliers = {
            1: 0.95,   # January - post-holiday lull
            2: 0.90,   # February - low season
            3: 1.00,   # March - spring break starts
            4: 1.05,   # April - spring break
            5: 1.00,   # May - shoulder season
            6: 1.15,   # June - summer begins
            7: 1.20,   # July - peak summer
            8: 1.15,   # August - summer
            9: 0.95,   # September - back to school
            10: 1.00,  # October - fall
            11: 1.10,  # November - Thanksgiving
            12: 1.15   # December - holidays
        }
        return multipliers.get(month, 1.0)

    def get_booking_multiplier(self, days_advance):
        """Get price multiplier based on advance booking days"""
        if days_advance >= 60:
            return 0.85   # Early booking discount
        elif days_advance >= 30:
            return 0.90   # Good advance booking
        elif days_advance >= 14:
            return 1.00   # Standard pricing
        elif days_advance >= 7:
            return 1.15   # Short notice premium
        else:
            return 1.35   # Last minute premium

    def predict_price(self, origin, destination, airline, travel_date, days_advance, trip_type='roundtrip'):
        """Predict airline fare based on route and market characteristics"""
        try:
            # Parse travel date for seasonal adjustments
            travel_dt = datetime.strptime(travel_date, '%Y-%m-%d')
            month = travel_dt.month
            quarter = (month - 1) // 3 + 1

            # Prepare features for model
            features_df, route_features, is_estimated = self.prepare_features(
                origin, destination)

            if self.model == "mock" or self.model is None:
                # Use mock prediction based on route features
                base_fare = self._calculate_mock_price(
                    route_features, is_estimated)
                market_confidence = 75 if is_estimated else 85
            else:
                # Use trained ML model
                try:
                    if self.scaler is not None:
                        features_scaled = self.scaler.transform(features_df)
                    else:
                        features_scaled = features_df.values

                    prediction = self.model.predict(features_scaled)[0]
                    base_fare = max(75, float(prediction))
                    market_confidence = 70 if is_estimated else 88
                except Exception as e:
                    logger.error(f"Model prediction error: {e}")
                    base_fare = self._calculate_mock_price(
                        route_features, is_estimated)
                    market_confidence = 75 if is_estimated else 85

            # Apply seasonal adjustment
            seasonal_mult = self.get_seasonal_multiplier(month)
            base_fare *= seasonal_mult

            # Apply advance booking adjustment
            booking_mult = self.get_booking_multiplier(days_advance)
            base_fare *= booking_mult

            # Round-trip adjustment
            if trip_type == 'roundtrip':
                base_fare *= 1.85

            # Get fare range from route data if available
            fare_low = float(route_features.get(
                'fare_low', base_fare * 0.7)) if not is_estimated else base_fare * 0.7
            fare_spread = float(route_features.get(
                'fare_spread', base_fare * 0.3)) if not is_estimated else base_fare * 0.3

            # Classify route distance
            nsmiles = float(route_features.get('nsmiles', 1000))
            if nsmiles < 500:
                route_type = 'Short-haul'
            elif nsmiles < 1500:
                route_type = 'Medium-haul'
            else:
                route_type = 'Long-haul'

            # Competition level based on market share
            large_ms = float(route_features.get('large_ms', 0.5))
            if large_ms > 0.7:
                competition = 'Low (Monopoly route)'
            elif large_ms > 0.5:
                competition = 'Medium'
            else:
                competition = 'High (Competitive route)'

            return {
                'predicted_fare': int(base_fare),
                'market_confidence': market_confidence,
                'fare_classes': {
                    'economy': int(base_fare * 0.85),
                    'premium_economy': int(base_fare * 1.15),
                    'business': int(base_fare * 2.5),
                    'market_low': int(fare_low * booking_mult * (1.85 if trip_type == 'roundtrip' else 1)),
                    'market_high': int((fare_low + fare_spread) * booking_mult * (1.85 if trip_type == 'roundtrip' else 1))
                },
                'market_analysis': {
                    'route_type': route_type,
                    'distance_miles': int(nsmiles),
                    'season': self.get_season(month),
                    'booking_timing': 'Optimal' if 14 <= days_advance <= 60 else ('Early' if days_advance > 60 else 'Last-minute'),
                    'competition_level': competition,
                    'data_source': 'Estimated (similar routes)' if is_estimated else 'Historical route data'
                },
                'pricing_insights': {
                    'seasonal_impact': f"{'+' if seasonal_mult > 1 else ''}{int((seasonal_mult - 1) * 100)}%",
                    'booking_impact': f"{'+' if booking_mult > 1 else ''}{int((booking_mult - 1) * 100)}%",
                    'market_share_largest': f"{large_ms * 100:.0f}%"
                },
                'status': 'success'
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'status': 'error'
            }

    def _calculate_mock_price(self, route_features, is_estimated):
        """Calculate price based on route features when model isn't available"""
        # Base price from historical average or estimate
        if 'fare' in route_features.index and not is_estimated:
            base_fare = float(route_features['fare'])
        else:
            # Estimate based on distance
            nsmiles = float(route_features.get('nsmiles', 1000))
            if nsmiles < 500:
                base_fare = 150 + nsmiles * 0.15
            elif nsmiles < 1500:
                base_fare = 175 + nsmiles * 0.12
            else:
                base_fare = 200 + nsmiles * 0.08

        # Adjust for competition (lower market share = more competition = lower prices)
        large_ms = float(route_features.get('large_ms', 0.5))
        # Higher monopoly = higher prices
        competition_factor = 1 + (large_ms - 0.5) * 0.2
        base_fare *= competition_factor

        return base_fare


# Initialize predictor
predictor = FlightPricePredictor()

# Initialize LLM agent if available
llm_agent = None
if LLM_AVAILABLE:
    try:
        # Initialize with Google Gemini
        llm_agent = FlightLLMAgent(predictor)
        logger.info("✓ LLM agent initialized successfully with Google Gemini")
    except Exception as e:
        logger.warning(f"Could not initialize LLM agent: {e}")
        logger.warning("Make sure GEMINI_API_KEY environment variable is set")
        llm_agent = None


@app.route('/')
def home():
    """Serve the main UI"""
    return send_from_directory('.', 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('.', filename)


@app.route('/api/predict', methods=['POST'])
def predict_flight_price():
    """API endpoint for price prediction"""
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['originCity', 'destCity',
                           'travelDate', 'bookingAdvance']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}', 'status': 'error'}), 400

        # Extract parameters
        origin = data['originCity']
        destination = data['destCity']
        airline = data.get('airline', '')
        travel_date = data['travelDate']
        days_advance = int(data['bookingAdvance'])
        trip_type = data.get('tripType', 'roundtrip')

        # Validate travel date
        try:
            travel_dt = datetime.strptime(travel_date, '%Y-%m-%d')
            if travel_dt < datetime.now():
                return jsonify({'error': 'Travel date cannot be in the past', 'status': 'error'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid date format', 'status': 'error'}), 400

        # Make prediction
        result = predictor.predict_price(
            origin, destination, airline, travel_date, days_advance, trip_type)

        if result['status'] == 'error':
            return jsonify(result), 500

        return jsonify(result)

    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@app.route('/api/cities', methods=['GET'])
def get_cities():
    """Get list of available cities from the dataset"""
    if predictor.city_mapping:
        # Return unique cities from actual data
        cities = sorted(set(predictor.city_mapping.values()))
        return jsonify({'cities': cities})
    else:
        # Fallback to common US cities
        cities = [
            'New York, NY', 'Los Angeles, CA', 'Chicago, IL', 'Houston, TX',
            'Phoenix, AZ', 'Philadelphia, PA', 'San Antonio, TX', 'San Diego, CA',
            'Dallas, TX', 'San Jose, CA', 'Austin, TX', 'Jacksonville, FL',
            'Fort Worth, TX', 'Columbus, OH', 'Charlotte, NC', 'San Francisco, CA',
            'Indianapolis, IN', 'Seattle, WA', 'Denver, CO', 'Washington, DC',
            'Boston, MA', 'Nashville, TN', 'Oklahoma City, OK', 'Las Vegas, NV',
            'Portland, OR', 'Memphis, TN', 'Louisville, KY', 'Baltimore, MD',
            'Milwaukee, WI', 'Albuquerque, NM', 'Tucson, AZ', 'Atlanta, GA',
            'Miami, FL', 'Orlando, FL', 'Tampa, FL', 'New Orleans, LA'
        ]
        return jsonify({'cities': cities})


@app.route('/api/airlines', methods=['GET'])
def get_airlines():
    """Get list of available airlines"""
    airlines = [
        'American Airlines', 'Delta Air Lines', 'United Airlines',
        'Southwest Airlines', 'JetBlue Airways', 'Alaska Airlines',
        'Spirit Airlines', 'Frontier Airlines', 'Allegiant Air',
        'Hawaiian Airlines', 'Sun Country Airlines'
    ]
    return jsonify({'airlines': airlines})


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None and predictor.model != "mock",
        'route_data_loaded': predictor.route_data is not None,
        'routes_available': len(predictor.route_data) if predictor.route_data is not None else 0,
        'llm_available': llm_agent is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/chat', methods=['POST'])
def chat_with_agent():
    """Chat endpoint for LLM agent"""
    if not llm_agent:
        return jsonify({
            'error': 'LLM agent not available. Please set up API keys and install required packages.',
            'status': 'error'
        }), 503

    try:
        data = request.get_json()
        message = data.get('message', '')
        context = data.get('context', {})

        if not message:
            return jsonify({'error': 'Message is required', 'status': 'error'}), 400

        # Get response from agent (now returns dict with response, prediction, flight_info)
        result = llm_agent.chat(message, context)

        return jsonify({
            'response': result.get('response', ''),
            'prediction': result.get('prediction'),
            'flight_info': result.get('flight_info'),
            'status': 'success'
        })

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/chat/reset', methods=['POST'])
def reset_chat():
    """Reset conversation history"""
    if not llm_agent:
        return jsonify({'error': 'LLM agent not available', 'status': 'error'}), 503

    try:
        llm_agent.reset_conversation()
        return jsonify({'status': 'success', 'message': 'Conversation reset'})
    except Exception as e:
        logger.error(f"Reset error: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs(os.path.join('..', 'models'), exist_ok=True)

    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
