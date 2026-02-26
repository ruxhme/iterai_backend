import os

from dotenv import load_dotenv
from supabase import Client, create_client

# Load credentials from the .env file
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("⚠️ WARNING: Supabase credentials not found. Check your .env file.")

# Initialize the connection
supabase: Client = (
    create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
)

def fetch_all_titles():
    """
    Fetches all verified titles from the database.
    (Note: For the hackathon demo, this grabs the titles into memory. 
    For production with 160k titles, we would use Supabase RPCs for pagination).
    """
    try:
        if supabase is None:
            return set()
        response = supabase.table("publications").select("title").execute()
        # Extract just the title strings into a set
        return {row['title'].lower() for row in response.data}
    except Exception as e:
        print(f"Database error: {e}")
        return set()

def insert_new_application(title: str, language: str):
    """
    Saves a successful application into the database to track it 
    and prevent future duplicates.
    """
    try:
        if supabase is None:
            return False
        supabase.table("publications").insert({
            "title": title,
            "language": language,
            "status": "pending" # Mark as a pending application
        }).execute()
        return True
    except Exception as e:
        print(f"Failed to save application: {e}")
        return False
