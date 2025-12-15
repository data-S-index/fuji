import requests
import time
import json

doi_api = "https://api.datacite.org/dois"
doi_list = []
while True:
    response = requests.get(doi_api)
    response.raise_for_status()
    data = response.json()
    for doi in data["data"]:
        doi_list.append(doi)
        print(f"Added {doi['id']} to the list")
    if "next" in data["links"]:
        doi_api = data["links"]["next"]
    else:
        break
    time.sleep(5)

with open("doi_list.json", "w") as f:
    json.dump(doi_list, f)
