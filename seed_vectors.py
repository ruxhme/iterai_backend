import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import Client, create_client

# 1. Setup Environment
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") # MUST be service_role key

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("Missing Supabase admin credentials in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# 2. Load multilingual encoder
print("Loading paraphrase-multilingual-MiniLM-L12-v2 model...")
# Use the smarter, multilingual model!
nlp_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def generate_and_upload_embeddings():
    batch_size = 500
    processed_count = 0
    
    print("Starting vector generation for 77,000 records...")
    
    while True:
        # Fetch a batch of titles that don't have an embedding yet
        # Adjust 'Title' if your friend used lowercase 'title' in the DB column
        response = supabase.table("existing_titles") \
            .select("id, Title") \
            .is_("embedding", "null") \
            .limit(batch_size) \
            .execute()
            
        records = response.data
        if not records:
            print(f"🎉 Complete! Processed a total of {processed_count} records.")
            break
            
        print(f"Processing batch of {len(records)} records...")
        
        titles = [record["Title"] for record in records]
        vectors = nlp_model.encode(
            titles,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        # Batch upsert is significantly faster than row-by-row updates.
        updates = [
            {"id": record["id"], "embedding": vector}
            for record, vector in zip(records, vectors)
        ]
        supabase.table("existing_titles").upsert(updates).execute()
        processed_count += len(updates)

        print(f"Embedded {processed_count} titles so far...")

if __name__ == "__main__":
    generate_and_upload_embeddings()
