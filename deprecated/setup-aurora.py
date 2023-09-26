import psycopg2
import yaml
from vectorgeo import constants as c

# For IVFFlat
N_LISTS = 100

# Load secrets from the secrets.yml file
with open("secrets.yml", "r") as file:
    secrets = yaml.safe_load(file)

# Database connection parameters
params = {
    "user": secrets["aurora_user"],
    "password": secrets["aurora_password"],
    "host": secrets["aurora_url"],
}

# Connect to the database
print("Connecting to database...")
conn = psycopg2.connect(**params)
cur = conn.cursor()

# Drop the table if it already exists
print("Dropping table...")
cur.execute("DROP TABLE IF EXISTS vectorgeo;")

# Activate the PostGIS extension
print("Activating PostGIS...")
cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
print("Activating pgvector...")
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

# Create a new table with a geometric column to store the points
print("Creating table...")
cur.execute(
    f"""
    CREATE TABLE vectorgeo (
        id BIGSERIAL PRIMARY KEY,
        geom GEOMETRY(Point, 3857),
        embedding vector({c.EMBED_DIM})
    );
"""
)

# Create indices for and spatial coordinates;
# the vector index is created after the upload is finished
print("Creating indices...")
cur.execute("CREATE INDEX ON vectorgeo USING gist (geom)")

# Commit the changes
print("Committing changes...")
conn.commit()

print("\nIMPORTANT: Don't forget to rebuild the ivfflat index after upload!\n")
