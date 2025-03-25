# smart_travel_planner.py - Integrated travel planning application

import os
import streamlit as st
import plotly.express as px
from typing import Dict, Any, List
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from embeddings import JinaEmbeddings

# Load environment variables
load_dotenv()

# App configuration
st.set_page_config(
    page_title="Smart Travel Planner",
    page_icon="🌴",
    layout="wide"
)

# Constants
INDEX_HOTELS = "hotels"
INDEX_TRANSPORT = "transport" 
INDEX_PLACES = "tourist-places"
INDEX_EXPENSES = "expenses"
INDEX_DESTINATIONS = "destinations"

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Initialize embeddings
embeddings = JinaEmbeddings()

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.getenv("GROQ_API")
)

# Destination data for seeding
DESTINATION_DETAILS = {
    "Kodaikanal": {
        "latitude": 10.2381,
        "longitude": 77.4892,
        "best_season": "October to March",
        "altitude": "2133 meters",
        "known_for": ["Princess of Hill stations", "Pine forests", "Kurinji flowers"],
        "description": """Kodaikanal, known as the Princess of Hill stations, is a hill town in Tamil Nadu. 
        Famous for its pine forests, cool climate, and the star-shaped Kodaikanal Lake."""
    },
    "Munnar": {
        "latitude": 10.0889,
        "longitude": 77.0595,
        "best_season": "September to May",
        "altitude": "1600 meters",
        "known_for": ["Tea plantations", "Neelakurinji flowers", "Wildlife"],
        "description": """Munnar is a town in the Western Ghats mountain range in Kerala. A hill station 
        and former resort for the British Raj elite, it's surrounded by rolling hills dotted with tea plantations."""
    },
    "Ooty": {
        "latitude": 11.4102,
        "longitude": 76.6950,
        "best_season": "October to June",
        "altitude": "2240 meters",
        "known_for": ["Queen of hill stations", "Nilgiri Mountain Railway", "Botanical Gardens"],
        "description": """Ooty, also known as Udhagamandalam, is a hill station in Tamil Nadu. It is situated 
        at an altitude of 2,240 meters above sea level in the Nilgiri Hills."""
    }
}

# ===== Database Functions =====

def ensure_indexes_exist() -> bool:
    """Create all required indexes if they don't exist."""
    required_indexes = [INDEX_HOTELS, INDEX_TRANSPORT, INDEX_PLACES, INDEX_EXPENSES, INDEX_DESTINATIONS]
    existing_indexes = pc.list_indexes().names()
    
    created = []
    for index_name in required_indexes:
        if index_name not in existing_indexes:
            try:
                pc.create_index(
                    name=index_name,
                    dimension=1024,  # Jina embeddings dimension
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                created.append(index_name)
            except Exception as e:
                st.error(f"Error creating index {index_name}: {str(e)}")
                return False
    
    if created:
        st.success(f"Created indexes: {', '.join(created)}")
    return True

def add_data_to_index(index_name: str, text: str, metadata: dict) -> bool:
    """Add data to the specified index."""
    try:
        # Get embeddings for the text
        vector = embeddings.embed_documents([text])[0]
        
        # Add to Pinecone
        index = pc.Index(index_name)
        index.upsert(
            vectors=[{
                'id': os.urandom(12).hex(),  # Generate random ID
                'values': vector,
                'metadata': {
                    'text': text,
                    **metadata
                }
            }]
        )
        return True
    except Exception as e:
        st.error(f"Error adding data: {str(e)}")
        return False

def seed_destination_data() -> bool:
    """Seed predefined destination data."""
    try:
        # Ensure destinations index exists
        if INDEX_DESTINATIONS not in pc.list_indexes().names():
            pc.create_index(
                name=INDEX_DESTINATIONS,
                dimension=1024,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        
        index = pc.Index(INDEX_DESTINATIONS)
        
        for destination, details in DESTINATION_DETAILS.items():
            # Create a rich text description for embedding
            description = f"""
            Destination: {destination}
            Best Season: {details['best_season']}
            Altitude: {details['altitude']}
            Known For: {', '.join(details['known_for'])}
            {details.get('description', '')}
            """
            
            # Get embeddings for the description
            vector = embeddings.embed_documents([description])[0]
            
            # Add to Pinecone
            index.upsert(
                vectors=[{
                    'id': f"dest_{destination.lower()}",
                    'values': vector,
                    'metadata': {
                        'name': destination,
                        'text': description,
                        **details
                    }
                }]
            )
        return True
    except Exception as e:
        st.error(f"Error seeding destination data: {str(e)}")
        return False

def get_index_stats(index_name: str) -> Dict[str, Any]:
    """Get statistics for the specified index."""
    try:
        index = pc.Index(index_name)
        return index.describe_index_stats()
    except Exception as e:
        st.error(f"Error fetching stats for {index_name}: {str(e)}")
        return {}

# ===== Travel Agent Classes =====

class TravelAgent:
    """Base class for travel recommendation agents."""
    def __init__(self, name: str, index_name: str = None):
        self.name = name
        self.index_name = index_name
        
        if index_name:  # Some agents might not need a vector index
            try:
                self.index = pc.Index(index_name)
                self.vectorstore = LangchainPinecone.from_existing_index(
                    index_name=index_name,
                    embedding=embeddings,
                    text_key="text"
                )
            except Exception as e:
                st.error(f"Error initializing {name}: {str(e)}")
                self.vectorstore = None

    def get_recommendations(self, query: Dict[str, Any]) -> str:
        """Get recommendations based on the query parameters."""
        try:
            # Get relevant information from vector store if available
            results = []
            if hasattr(self, 'vectorstore') and self.vectorstore:
                search_query = self._build_search_query(query)
                results = self.vectorstore.similarity_search(search_query)
            
            # Build prompt template and format with results
            prompt_template = self._build_prompt_template()
            prompt = PromptTemplate.from_template(prompt_template)
            
            # Invoke LLM
            response = llm.invoke(prompt.format(
                results=results,
                **query
            ))
            return response.content
            
        except Exception as e:
            st.error(f"Error in {self.name}: {str(e)}")
            return f"Unable to fetch {self.name} recommendations. Please try again later."
    
    def _build_search_query(self, query: Dict[str, Any]) -> str:
        """Build a search query string - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")
        
    def _build_prompt_template(self) -> str:
        """Build a prompt template - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")


class HotelAgent(TravelAgent):
    """Agent for hotel recommendations."""
    def __init__(self):
        super().__init__("Hotel Recommendations", INDEX_HOTELS)
        
    def _build_search_query(self, query: Dict[str, Any]) -> str:
        return f"Find hotels in {query['destination']} for {query['num_people']} people with budget {query['budget']}"
        
    def _build_prompt_template(self) -> str:
        return """Based on these hotel options: {results}
        Provide hotel recommendations for {num_people} travelers to {destination} with a budget of Rs. {budget}.
        
        Format the response as:
        
        🏨 Available Hotels:
        - [Hotel name] - [Price range] - [Brief description]
        - [Hotel name] - [Price range] - [Brief description]
        
        💡 Best Options:
        - For luxury: [recommendation]
        - For budget: [recommendation]
        - For families: [recommendation]
        """


class TransportAgent(TravelAgent):
    """Agent for transport recommendations."""
    def __init__(self):
        super().__init__("Transport Options", INDEX_TRANSPORT)
        
    def _build_search_query(self, query: Dict[str, Any]) -> str:
        return f"Find transport from {query['boarding']} to {query['destination']}"
        
    def _build_prompt_template(self) -> str:
        return """Based on these transport options: {results}
        Provide transport recommendations from {boarding} to {destination} for {num_people} travelers.
        
        Format the response as:
        
        🚌 Available Options:
        - [Transport mode] - [Duration] - [Price range]
        - [Transport mode] - [Duration] - [Price range]
        
        💡 Recommended Route:
        - Fastest: [recommendation]
        - Most comfortable: [recommendation]
        - Budget-friendly: [recommendation]
        """


class TouristAgent(TravelAgent):
    """Agent for tourist attractions recommendations."""
    def __init__(self):
        super().__init__("Tourist Guide", INDEX_PLACES)
        
    def _build_search_query(self, query: Dict[str, Any]) -> str:
        return f"Find tourist places in {query['destination']}"
        
    def _build_prompt_template(self) -> str:
        return """Based on these attractions: {results}
        Provide tourist recommendations for {destination}.
        
        Format the response as:
        
        🗺️ Must Visit Places:
        - [Attraction] - [Why it's special]
        - [Attraction] - [Why it's special]
        
        🍴 Local Food:
        - [Food item] - [Brief description]
        - [Food item] - [Brief description]
        
        💡 Travel Tips:
        - [Practical tip for the destination]
        - [Practical tip for the destination]
        """


class ExpenseAgent(TravelAgent):
    """Agent for expense breakdown recommendations."""
    def __init__(self):
        # This agent doesn't need a vector index
        super().__init__("Expense Breakdown", None)
        
    def _build_search_query(self, query: Dict[str, Any]) -> str:
        # Not used for this agent
        return ""
        
    def _build_prompt_template(self) -> str:
        return """Create a detailed expense breakdown for a trip to {destination}.
        
        Trip Details:
        - Total Budget: Rs. {budget}
        - Number of Travelers: {num_people}
        
        Format the response as:
        
        💰 Daily Expenses (per person):
        • Food: Rs. [amount]/day
        • Local Transport: Rs. [amount]/day
        • Activities: Rs. [amount]/day
        • Miscellaneous: Rs. [amount]/day
        
        🏨 Accommodation:
        • Budget options: Rs. [amount] per night
        • Mid-range options: Rs. [amount] per night
        • Luxury options: Rs. [amount] per night
        
        ✈️ Travel Costs:
        • Round-trip transportation: Rs. [amount] per person
        
        💡 Money-Saving Tips:
        • [Practical tip for saving money]
        • [Practical tip for saving money]
        """

def get_all_recommendations(query: Dict[str, Any]) -> Dict[str, str]:
    """Get recommendations from all agents."""
    agents = [
        HotelAgent(),
        TransportAgent(),
        ExpenseAgent(),
        TouristAgent()
    ]
    
    results = {}
    for agent in agents:
        with st.spinner(f"Getting {agent.name}..."):
            results[agent.name] = agent.get_recommendations(query)
    
    return results

# ===== User Interface Functions =====

def db_manager_ui():
    """Database management user interface."""
    st.header("🗄️ Travel Database Manager")
    
    # Initialize session state variables for form inputs if they don't exist
    if 'form_submitted' not in st.session_state:
        st.session_state.form_submitted = False
    
    # Reset form fields if data was successfully submitted
    if st.session_state.form_submitted:
        # Clear all form field values in session state
        for key in list(st.session_state.keys()):
            if key.endswith("_input"):
                st.session_state[key] = "" if isinstance(st.session_state[key], str) else (
                    [] if isinstance(st.session_state[key], list) else 0
                )
        # Reset the submitted flag
        st.session_state.form_submitted = False
    
    # Simplified UI with just data entry
    st.write("Add travel data to your database")
    
    # Index Selection
    index_name = st.selectbox(
        "Choose a database category",
        [INDEX_HOTELS, INDEX_TRANSPORT, INDEX_PLACES, INDEX_EXPENSES],
        key='index_selector'
    )

    # Add Data
    with st.form("add_data_form"):
        st.subheader(f"Add New {index_name.title()} Data")
        description = ""  # Initialize description variable
        
        if index_name == INDEX_HOTELS:
            name = st.text_input("Hotel Name", key="name_input")
            location = st.text_input("Location", key="location_input")
            price = st.number_input("Price per night (Rs.)", min_value=0, key="price_input")
            amenities = st.multiselect(
                "Amenities",
                ["WiFi", "AC", "Restaurant", "Pool", "Gym", "Parking"],
                key="amenities_input"
            )
            description = st.text_area("Description", key="description_input")
            metadata = {
                "name": name,
                "location": location,
                "price": price,
                "amenities": amenities
            }
            
        elif index_name == INDEX_TRANSPORT:
            mode = st.selectbox("Mode", ["Bus", "Train", "Flight"], key="mode_input")
            from_city = st.text_input("From", key="from_input")
            to_city = st.text_input("To", key="to_input")
            price = st.number_input("Price (Rs.)", min_value=0, key="price_input")
            duration = st.number_input("Duration (hours)", min_value=0.0, key="duration_input")
            description = st.text_area("Description", key="description_input")
            metadata = {
                "mode": mode,
                "from": from_city,
                "to": to_city,
                "price": price,
                "duration": duration
            }
            
        elif index_name == INDEX_PLACES:
            name = st.text_input("Place Name", key="name_input")
            location = st.text_input("Location", key="location_input")
            entry_fee = st.number_input("Entry Fee (Rs.)", min_value=0, key="fee_input")
            best_time = st.text_input("Best Time to Visit", key="time_input")
            description = st.text_area("Description", key="description_input")
            metadata = {
                "name": name,
                "location": location,
                "entry_fee": entry_fee,
                "best_time": best_time
            }
            
        elif index_name == INDEX_EXPENSES:
            category = st.selectbox("Category", ["Food", "Activities", "Local Transport", "Miscellaneous"], key="category_input")
            location = st.text_input("Location", key="location_input")
            amount = st.number_input("Amount per day (Rs.)", min_value=0, key="amount_input")
            description = st.text_area("Description", key="description_input")
            metadata = {
                "category": category,
                "location": location,
                "amount": amount
            }
        
        # Add form submit button
        submit = st.form_submit_button("Add Data")
    
    # Handle form submission outside the form
    if submit:
        if not description:
            st.error("Please fill in the description field")
        else:
            if add_data_to_index(index_name, description, metadata):
                st.success("Data added successfully!")
                # Set the flag to reset form on next rerun
                st.session_state.form_submitted = True
                st.rerun()

def planner_ui():
    """Trip planner user interface."""
    st.header("🌴 Smart Travel Planner")
    st.write("Plan your perfect trip with AI assistance!")

    # User inputs
    st.subheader("Trip Details")
    col1, col2 = st.columns(2)
    
    with col1:
        boarding = st.text_input("From", placeholder="Enter starting city")
        num_people = st.number_input("Number of Travelers", min_value=1, value=1)
    
    with col2:
        destination = st.text_input("To", placeholder="Enter destination city")
        budget = st.number_input("Budget (Rs.)", min_value=1000, value=10000, step=1000)

    if st.button("Plan My Trip", type="primary"):
        if not all([boarding, destination]):
            st.error("Please fill in all required fields")
            return

        # Calculate budget distribution
        hotel_budget = budget * 0.4
        transport_budget = budget * 0.3
        misc_budget = budget * 0.3

        # Query parameters
        query = {
            "boarding": boarding,
            "destination": destination,
            "num_people": num_people,
            "budget": budget
        }

        with st.spinner("Planning your perfect trip..."):
            # Get recommendations
            results = get_all_recommendations(query)

            # Display results in tabs
            tabs = st.tabs(["🏨 Hotels", "🚌 Transport", "💰 Expenses", "🎯 Tourist Guide"])

            with tabs[0]:
                st.subheader("Hotel Recommendations")
                st.info(f"Hotel Budget: Rs. {hotel_budget:,.2f}")
                st.markdown(results["Hotel Recommendations"])

            with tabs[1]:
                st.subheader("Transport Options")
                st.info(f"Transport Budget: Rs. {transport_budget:,.2f}")
                st.markdown(results["Transport Options"])

            with tabs[2]:
                st.subheader("Expense Breakdown")
                st.info(f"Daily Budget: Rs. {misc_budget:,.2f}")
                
                # Display pie chart
                fig = px.pie(
                    values=[hotel_budget, transport_budget, misc_budget],
                    names=["Hotels", "Transport", "Daily Expenses"],
                    title="Budget Distribution",
                    color_discrete_sequence=px.colors.sequential.Blues_r
                )
                st.plotly_chart(fig)
                st.markdown(results["Expense Breakdown"])

            with tabs[3]:
                st.subheader(f"Tourist Guide - {destination}")
                st.markdown(results["Tourist Guide"])

# ===== Main Application =====

def main():
    """Main application function."""
    st.title("🌴 Smart Travel Planner")
    
    # Initialize app
    if not ensure_indexes_exist():
        st.error("Failed to initialize databases. Please check your configuration.")
        return
        
    # Create tabs for the main sections
    tab1, tab2 = st.tabs(["Plan Your Trip", "Manage Database"])
    
    with tab1:
        planner_ui()
        
    with tab2:
        db_manager_ui()

if __name__ == "__main__":
    main()
