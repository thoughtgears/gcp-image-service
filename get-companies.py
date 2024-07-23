from google.cloud import firestore
import os
from dotenv import load_dotenv
import logging
import json
import time

load_dotenv()

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

trade_project_id = os.getenv("TRADE_PROJECT_ID")
db = firestore.Client(project=trade_project_id)
logging.basicConfig(level=logging.INFO)
output_file = "data/companies_with_albums.json"


def load_existing_data():
    try:
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
            processed_company_ids = set(existing_data.keys())
            return existing_data, processed_company_ids
    except FileNotFoundError:
        return {}, set()


def append_data_to_file(data):
    with open(output_file, 'a') as f:
        for key, value in data.items():
            json.dump({key: value}, f)
            f.write('\n')


def get_company_albums(company_id):
    album_docs = db.collection("companies").document(company_id).collection("albums").stream()
    albums = {}
    for album in album_docs:
        album_data = album.to_dict()
        logging.debug(f"Album data for company {company_id}, album {album.id}: {album_data}")
        album_items = album_data.get("items", [])
        for item in album_items:
            if "imageId" in item:
                logging.debug(f"Found imageId {item['imageId']} in album {album.id} for company {company_id}")
                albums[item["imageId"]] = {
                    "company_id": company_id,
                    "album_id": album.id
                }
    return albums


def get_companies_with_albums(batch_size=1000):
    data, processed_company_ids = load_existing_data()
    docs_ref = db.collection("companies")
    last_doc = None
    while True:
        query = docs_ref.limit(batch_size)
        if last_doc:
            query = query.start_after(last_doc)

        docs_snap = query.stream()
        docs = list(docs_snap)
        if not docs:
            break

        batch_data = {}
        for doc in docs:
            company_id = doc.id
            if company_id in processed_company_ids:
                logging.info(f"Skipping already processed company {company_id}")
                continue
            logging.info(f"Processing company {company_id}")
            albums = get_company_albums(company_id)
            logging.debug(f"Albums for company {company_id}: {albums}")
            batch_data.update(albums)
            processed_company_ids.add(company_id)

        append_data_to_file(batch_data)  # Append batch data to file
        time.sleep(5)  # Sleep for 5 seconds

        last_doc = docs[-1]  # Set the last document for the next batch

    return data


# Retrieve the data
companies_with_albums = get_companies_with_albums()

# Print a message to indicate completion
logging.info(f"Data has been written to {output_file}")
