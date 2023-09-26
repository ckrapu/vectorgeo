import fire
import os
import requests
import yaml
import constants as c
import json

from qdrant_client import QdrantClient


def check_health():
    """
    Attempts to get collection metadata from Qdrant using both the Python API and a manual request.
    """

    check_health_python_api()
    check_health_manual_request()


def check_health_python_api(collection="vectorgeo"):
    """
    Uses the Python API to get collection metadata from Qdrant.
    """

    secrets = yaml.load(
        open(os.path.join(c.BASE_DIR, "secrets.yml")), Loader=yaml.FullLoader
    )

    qdrant_client = QdrantClient(
        url=secrets["qdrant_url"], api_key=secrets["qdrant_api_key"]
    )
    health = qdrant_client.get_collection(collection)
    print(f"\n\nPYTHON API:\nHealth of collection {collection}: {health}")


def check_health_manual_request(collection="vectorgeo"):
    """
    Uses a manual request to get collection metadata from Qdrant.
    """

    secrets = yaml.load(
        open(os.path.join(c.BASE_DIR, "secrets.yml")), Loader=yaml.FullLoader
    )

    url = f"{secrets['qdrant_url']}/collections/{collection}"
    headers = {"api-key": secrets["qdrant_api_key"]}

    response = requests.get(url, headers=headers)
    json_formatted_str = json.dumps(response.json(), indent=2)

    print(
        f"\n\nMANUAL REQUEST:\nHealth of collection '{collection}': {json_formatted_str}"
    )


def main():
    fire.Fire(
        {
            "check_health_python_api": check_health_python_api,
            "check_health_manual_request": check_health_manual_request,
            "check_health": check_health,
        }
    )


if __name__ == "__main__":
    main()
