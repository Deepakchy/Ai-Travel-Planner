import os
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI  
from langchain.schema import SystemMessage, HumanMessage
from deep_translator import GoogleTranslator


#  Set API key from Streamlit Secrets

API_KEY = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = API_KEY
genai.configure(api_key=API_KEY)

#  Input Validation

def is_valid_location(name):
    """Check if the location contains only valid characters (letters, spaces, basic punctuation)."""
    return bool(re.match(r"^[A-Za-z\s,.-]{2,}$", name.strip()))


#  LangChain + Gemini Travel Planner

def get_travel_plan(source, destination, travel_mode, travel_preference, language):
    # Validate inputs before making API call
    if not is_valid_location(source) or not is_valid_location(destination):
        return (
            "🙏 Sorry, the location names you entered don't seem valid.\n\n"
            "Please enter real place names without numbers or special characters."
        )
    
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash")  # Recommended stable model
    messages = [
        SystemMessage(content="You are an AI travel planner providing optimized travel plans."),
        HumanMessage(content=f"""
        Generate a detailed travel plan from {source} to {destination}.
        - Travel mode: {travel_mode}
        - Preference: {travel_preference}
        - Language: {language}
        
        Also suggest relevant websites or apps for more details.
        Include estimated travel time, cost, and comfort level.
        If the user enters anything other than a valid place or location name, politely deny the request.
        """)
    ]
    response = model.invoke(messages)
    return response.content


#  Language Translation

def translate_text(text, lang_code):
    return GoogleTranslator(source="auto", target=lang_code).translate(text)

lang_codes = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Maithili": "mai"
}

#  Streamlit UI

st.set_page_config(page_title="AI Travel Planner", page_icon="🌍")
st.title("🌍 AI Travel Planner")

source = st.text_input("Enter Source Location", placeholder="e.g., Delhi")
destination = st.text_input("Enter Destination", placeholder="e.g., Mumbai")
travel_mode = st.selectbox("Select Travel Mode", ["All", "Flight", "Train", "Bus", "Cab", "Bike"])
travel_preference = st.selectbox("Select Travel Preference", ["Any", "Budget", "Fastest", "Most Comfortable"])
language = st.selectbox("Select Language", ["English", "Hindi", "Tamil", "Telugu", "Maithili"])

if st.button("Plan My Travel"):
    if source and destination:
        try:
            with st.spinner("Generating your travel plan..."):
                travel_plan = get_travel_plan(source, destination, travel_mode, travel_preference, language)
                if language in lang_codes and language != "English":
                    travel_plan = translate_text(travel_plan, lang_codes[language])
                st.success("Here's your travel plan:")
                st.write(travel_plan)
        except Exception as e:
            st.error(f"❌ Something went wrong: {e}")
    else:
        st.warning("⚠️ Please enter both source and destination.")
