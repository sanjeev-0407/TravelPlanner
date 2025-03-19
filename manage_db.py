import os
from typing import List
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from embeddings import JinaEmbeddings

# Add after existing imports
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
    # ... other destinations
}

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Initialize embeddings
embeddings = JinaEmbeddings()

def create_index(index_name: str):
    try:
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1024,  # Jina embeddings dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            st.success(f"Created index: {index_name}")
        else:
            st.info(f"Index {index_name} already exists")
    except Exception as e:
        st.error(f"Error creating index: {str(e)}")

def add_data_to_index(index_name: str, text: str, metadata: dict):
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
        st.success("Data added successfully!")
    except Exception as e:
        st.error(f"Error adding data: {str(e)}")

def seed_destination_data():
    try:
        # Create destinations index if it doesn't exist
        if "destinations" not in pc.list_indexes().names():
            pc.create_index(
                name="destinations",
                dimension=1024,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            st.success("Created destinations index")

        index = pc.Index("destinations")
        
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
            st.success(f"Added {destination} to database")
            
    except Exception as e:
        st.error(f"Error seeding destination data: {str(e)}")

def main():
    st.title("🗄️ Travel Database Manager")
    
    # Add a new tab for destination data
    tab1, tab2 = st.tabs(["Manage Data", "Seed Destinations"])
    
    with tab1:
        st.write("Manage your travel data in Pinecone")

        # Index Selection/Creation
        st.header("1. Select Database")
        index_name = st.selectbox(
            "Choose a database category",
            ["hotels", "transport", "tourist-places", "expenses"]
        )

        if st.button("Create/Check Index"):
            create_index(index_name)

        # Add Data
        st.header("2. Add New Data")
        
        # Add data form
        with st.form("add_data"):
            description = ""  # Initialize description variable
            
            if index_name == "hotels":
                name = st.text_input("Hotel Name")
                location = st.text_input("Location")
                price = st.number_input("Price per night", min_value=0)
                amenities = st.multiselect(
                    "Amenities",
                    ["WiFi", "AC", "Restaurant", "Pool", "Gym", "Parking"]
                )
                description = st.text_area("Description")
                metadata = {
                    "name": name,
                    "location": location,
                    "price": price,
                    "amenities": amenities
                }
                
            elif index_name == "transport":
                mode = st.selectbox("Mode", ["Bus", "Train", "Flight"])
                from_city = st.text_input("From")
                to_city = st.text_input("To")
                price = st.number_input("Price", min_value=0)
                duration = st.number_input("Duration (hours)", min_value=0.0)
                description = st.text_area("Description")
                metadata = {
                    "mode": mode,
                    "from": from_city,
                    "to": to_city,
                    "price": price,
                    "duration": duration
                }
                
            elif index_name == "tourist_places":
                name = st.text_input("Place Name")
                location = st.text_input("Location")
                entry_fee = st.number_input("Entry Fee", min_value=0)
                best_time = st.text_input("Best Time to Visit")
                description = st.text_area("Description")
                metadata = {
                    "name": name,
                    "location": location,
                    "entry_fee": entry_fee,
                    "best_time": best_time
                }
                
            elif index_name == "expenses":  # Add expenses form
                category = st.selectbox("Category", ["Food", "Activities", "Local Transport", "Miscellaneous"])
                location = st.text_input("Location")
                amount = st.number_input("Amount per day", min_value=0)
                description = st.text_area("Description")
                metadata = {
                    "category": category,
                    "location": location,
                    "amount": amount
                }
            
            # Add form submit button
            if st.form_submit_button("Add Data"):
                if not description:
                    st.error("Please fill in the description field")
                else:
                    add_data_to_index(index_name, description, metadata)

        # View Data
        st.header("3. View Database Stats")
        if st.button("Show Statistics"):
            try:
                index = pc.Index(index_name)
                stats = index.describe_index_stats()
                st.json(stats)
            except Exception as e:
                st.error(f"Error fetching stats: {str(e)}")
    
    with tab2:
        st.header("Seed Destination Data")
        st.write("Add predefined destination information to the database")
        
        if st.button("Seed Destination Data"):
            seed_destination_data()
        
        # View destination data
        if st.button("View Destination Data"):
            try:
                index = pc.Index("destinations")
                stats = index.describe_index_stats()
                st.json(stats)
                
                # Query example destination
                results = index.query(
                    vector=[0] * 1024,  # dummy vector
                    top_k=5,
                    include_metadata=True
                )
                
                if results.matches:
                    st.subheader("Sample Destination Data")
                    for match in results.matches:
                        with st.expander(match.metadata['name']):
                            st.write("**Best Season:**", match.metadata['best_season'])
                            st.write("**Altitude:**", match.metadata['altitude'])
                            st.write("**Known For:**", ", ".join(match.metadata['known_for']))
                            
            except Exception as e:
                st.error(f"Error viewing destination data: {str(e)}")

if __name__ == "__main__":
    main()