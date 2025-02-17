import streamlit as st
import plotly.express as px
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-specdec",
    temperature=0.7,
    api_key=os.getenv("GROQ_API")
)

class TravelAgent:
    def __init__(self, name: str):
        self.name = name
        
    async def get_recommendations(self, query: dict) -> str:
        if self.name == "HotelAgent":
            prompt = """Suggest hotels in {destination} for {num_people} people with budget Rs. {budget}.
            Format as:
            
            🏨 Hotel Options:
            1. [Hotel Name]
               • Location: [area]
               • Room Types: [types]
               • Price Range: Rs. [range]
               • Amenities: [list]
            """
        
        elif self.name == "TransportAgent":
            prompt = """Suggest transport from {boarding} to {destination} for {num_people} people with budget Rs. {budget}.
            Format as:
            
            🚌 Available Options:
            1. [Transport Mode]
               • Departure: [times]
               • Duration: [hours]
               • Cost per person: Rs. [amount]
               • Total cost: Rs. [total]
            """
        
        elif self.name == "ExpenseAgent":
            prompt = """Create expense breakdown for {destination} trip for {num_people} people with budget Rs. {budget}.
            Format as:
            
            💰 Daily Expenses:
            • Food: Rs. [amount]/day
            • Local Transport: Rs. [amount]/day
            • Activities: Rs. [amount]/day
            • Miscellaneous: Rs. [amount]/day
            """
        
        else:  # TouristAgent
            prompt = """Provide tourist guide for {destination} for {num_people} people.
            Format as:
            
            🗺️ Must Visit Places:
            1. [Place Name]
               • Best time: [when]
               • Entry fee: Rs. [amount]
               • Time needed: [duration]
            
            🍴 Local Food:
            • [Dish names and where to try]
            
            💡 Travel Tips:
            • [Important tips]
            """
        
        template = PromptTemplate.from_template(prompt)
        response = await llm.ainvoke(template.format(**query))
        return response.content

async def get_all_recommendations(hotel_query, transport_query, expense_query, tourist_query):
    agents = [
        TravelAgent("HotelAgent"),
        TravelAgent("TransportAgent"),
        TravelAgent("ExpenseAgent"),
        TravelAgent("TouristAgent")
    ]
    
    tasks = [
        agents[0].get_recommendations(hotel_query),    # Hotel Agent
        agents[1].get_recommendations(transport_query),  # Transport Agent
        agents[2].get_recommendations(expense_query),   # Expense Agent
        agents[3].get_recommendations(tourist_query)    # Tourist Agent
    ]
    
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

        # Query parameters for each agent
        hotel_query = {
            "destination": destination,
            "num_people": num_people,
            "budget": hotel_budget
        }

        transport_query = {
            "boarding": boarding,
            "destination": destination,
            "num_people": num_people,
            "budget": transport_budget
        }

        expense_query = {
            "destination": destination,
            "num_people": num_people,
            "budget": misc_budget
        }

        tourist_query = {
            "destination": destination,
            "num_people": num_people
        }

        with st.spinner("Planning your perfect trip..."):
            # Get recommendations asynchronously
            results = asyncio.run(get_all_recommendations(hotel_query, transport_query, expense_query, tourist_query))

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

if __name__ == "__main__":
    main()
