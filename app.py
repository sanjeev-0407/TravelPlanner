import streamlit as st
import plotly.express as px
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
import asyncio
from embeddings import JinaEmbeddings

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Initialize Groq LLM
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0.7,
    api_key=os.getenv("GROQ_API")
)

# Initialize embeddings
embeddings = JinaEmbeddings()

def ensure_indexes_exist():
    required_indexes = ["hotels", "transport", "tourist_places", "expenses"]
    existing_indexes = pc.list_indexes().names()
    
    for index_name in required_indexes:
        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=1024,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            st.info(f"Created index: {index_name}")

class TravelAgent:
    def __init__(self, name: str):
        self.name = name
        if name != "ExpenseAgent":  # ExpenseAgent doesn't need Pinecone
            self.index = pc.Index(name.lower().replace("agent", "s"))
            self.vectorstore = LangchainPinecone.from_existing_index(
                index_name=name.lower().replace("agent", "s"),
                embedding=embeddings,
                text_key="text"
            )

    async def get_recommendations(self, query: dict) -> str:
        try:
            if self.name == "HotelAgent":
                results = self.vectorstore.similarity_search(
                    f"Find hotels in {query['destination']} for {query['num_people']} people with budget {query['budget']}"
                )
                prompt = """Based on these hotel options: {results}
                Format the response as:
                
                🏨 Available Hotels:
                {hotel_list}
                
                💡 Best Options:
                {recommendations}
                """
                
            elif self.name == "TransportAgent":
                results = self.vectorstore.similarity_search(
                    f"Find transport from {query['boarding']} to {query['destination']}"
                )
                prompt = """Based on these transport options: {results}
                Format the response as:
                
                🚌 Available Options:
                {transport_list}
                
                💡 Recommended Route:
                {recommendations}
                """
                
            elif self.name == "TouristAgent":
                results = self.vectorstore.similarity_search(
                    f"Find tourist places in {query['destination']}"
                )
                prompt = """Based on these attractions: {results}
                Format the response as:
                
                🗺️ Must Visit Places:
                {places_list}
                
                🍴 Local Food:
                {food_suggestions}
                
                💡 Travel Tips:
                {tips}
                """
                
            else:  # ExpenseAgent
                prompt = """Create expense breakdown for {destination} trip.
                Budget: Rs. {budget}
                People: {num_people}
                
                Format as:
                
                💰 Daily Expenses:
                • Food: Rs. {food_cost}/day
                • Local Transport: Rs. {transport_cost}/day
                • Activities: Rs. {activities_cost}/day
                • Miscellaneous: Rs. {misc_cost}/day
                """

            template = PromptTemplate.from_template(prompt)
            response = await llm.ainvoke(template.format(
                results=results if self.name != "ExpenseAgent" else "",
                **query
            ))
            return response.content
            
        except Exception as e:
            st.error(f"Error in {self.name}: {str(e)}")
            return f"Unable to fetch {self.name.lower().replace('agent', '')} recommendations"

# ... rest of the code remains the same as in your example ...

async def get_all_recommendations(query: dict):
    agents = [
        TravelAgent("HotelAgent"),
        TravelAgent("TransportAgent"),
        TravelAgent("ExpenseAgent"),
        TravelAgent("TouristAgent")
    ]
    
    tasks = [agent.get_recommendations(query) for agent in agents]
    return await asyncio.gather(*tasks)

def main():
    st.title("🌴 Smart Travel Planner")
    st.write("Plan your perfect trip with AI assistance!")

    # User inputs
    with st.sidebar:
        st.header("Trip Details")
        budget = st.number_input("Budget (Rs.)", min_value=1000, value=10000)
        boarding = st.text_input("From", placeholder="Enter starting city")
        destination = st.text_input("To", placeholder="Enter destination city")
        num_people = st.number_input("Number of Travelers", min_value=1, value=1)

    if st.button("Plan My Trip"):
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
            results = asyncio.run(get_all_recommendations(query))

            # Display results in tabs
            tabs = st.tabs(["🏨 Hotels", "🚌 Transport", "💰 Expenses", "🎯 Tourist Guide"])

            with tabs[0]:
                st.header("Hotel Recommendations")
                st.info(f"Hotel Budget: Rs. {hotel_budget:,.2f}")
                st.markdown(results[0])

            with tabs[1]:
                st.header("Transport Options")
                st.info(f"Transport Budget: Rs. {transport_budget:,.2f}")
                st.markdown(results[1])

            with tabs[2]:
                st.header("Expense Breakdown")
                st.info(f"Daily Budget: Rs. {misc_budget:,.2f}")
                
                # Display pie chart
                fig = px.pie(
                    values=[hotel_budget, transport_budget, misc_budget],
                    names=["Hotels", "Transport", "Daily Expenses"],
                    title="Budget Distribution"
                )
                st.plotly_chart(fig)
                st.markdown(results[2])

            with tabs[3]:
                st.header(f"Tourist Guide - {destination}")
                st.markdown(results[3])



def initialize_app():
    try:
        ensure_indexes_exist()
        return True
    except Exception as e:
        st.error(f"Failed to initialize app: {str(e)}")
        return False

if __name__ == "__main__":
    if initialize_app():
        main()
    else:
        st.error("Application failed to initialize properly. Please check your configuration.")