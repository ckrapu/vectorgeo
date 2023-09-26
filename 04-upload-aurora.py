import boto3
import numpy as np
import os
import pandas as pd
import psycopg2
import yaml

from metaflow import FlowSpec, Parameter, step
from pyproj import Transformer

from vectorgeo.transfer import download_file
from vectorgeo import constants as c


class AuroraUploadFlow(FlowSpec):
    """
    Takes data artifacts from inference flow and uploads them into a fresh Aurora PostgreSQL table.
    """

    wipe_existing = Parameter(
        "wipe_existing",
        help="Whether to wipe the existing table before uploading",
        default=False,
    )

    max_rows_upload = Parameter(
        "max_rows_upload", help="Maximum number of rows to upload", default=100_000_000
    )

    maintenance_work_mem = Parameter(
        "maintenance_work_mem",
        help="Maintenance work memory (in kB) for the index rebuild",
        default=4_000_000,
    )

    @step
    def start(self):
        """
        Wipes the existing Aurora database and creates a new table for the vector data.
        """

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
        if self.wipe_existing:
            print("Dropping table...")
            cur.execute("DROP TABLE IF EXISTS vectorgeo;")

        # Activate the PostGIS extension
        print("Activating extensions...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Create a new table with a geometric column to store the points
        print("Creating table...")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS vectorgeo (
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
        conn.commit()

        print("\nIMPORTANT: Don't forget to rebuild the ivfflat index after upload!\n")

        self.next(self.upload)

    @step
    def upload(self):
        """
        Gets files from S3 bucket and then uploads them to Aurora.
        """

        # Load secrets (adjust the path as necessary)
        secrets = yaml.load(open("secrets.yml"), Loader=yaml.FullLoader)

        # Initialize S3 client
        s3 = boto3.client(
            "s3",
            aws_access_key_id=secrets["aws_access_key_id"],
            aws_secret_access_key=secrets["aws_secret_access_key"],
        )

        # Database connection parameters
        params = {
            "user": secrets["aurora_user"],
            "password": secrets["aurora_password"],
            "host": secrets["aurora_url"],
        }

        # Connect to the database
        conn = psycopg2.connect(**params)
        cur = conn.cursor()

        # Print out full schema for all tables matching the pattern
        cur.execute(
            """
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_name LIKE 'vectorgeo'
        """
        )
        print("Schema:")
        for row in cur.fetchall():
            print(row)

        # Specify your bucket name and prefix
        bucket_name = c.S3_BUCKET
        prefix = "vectors/"

        n_rows_uploaded = 0

        # List all Parquet files in the S3 bucket with the specified prefix
        contents = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)["Contents"]
        print(f"Found {len(contents)} files in S3")

        # Filter out the files that have already been run
        keys = [obj["Key"] for obj in contents if obj["Key"].endswith(".parquet")]
        print(f"Found {len(keys)} files to run")

        transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
        finished = False
        for key in keys:
            if finished:
                break

            print(f"...Downloading {key} from S3")
            basename = os.path.basename(key)
            local_path = os.path.join(c.TMP_DIR, basename)
            download_file(key, local_path)

            # Load the data into a Pandas DataFrame
            df = pd.read_parquet(local_path)

            # Extract vectors and other necessary information
            print(f"...Uploading {key} to Aurora")
            for df_piece in np.array_split(df, 10):
                (
                    df_piece["x"],
                    df_piece["y"],
                ) = transformer.transform(  # pylint: disable=unpacking-non-sequence
                    df_piece["lng"], df_piece["lat"]
                )

                values = [
                    (row["id"], row["x"], row["y"], row["vector"].tolist())
                    for _, row in df_piece.iterrows()
                ]
                args_str = ",".join(
                    cur.mogrify(
                        " (%s, ST_SetSRID(ST_MakePoint(%s, %s), 3857), %s)", x
                    ).decode("utf-8")
                    for x in values
                )
                cur.execute(
                    """INSERT INTO vectorgeo (id, geom, embedding) VALUES """
                    + args_str
                    + " ON CONFLICT (id) DO NOTHING;"
                )
                conn.commit()

                n_rows_uploaded += len(values)

                if n_rows_uploaded >= self.max_rows_upload:
                    print(
                        f"Reached maximum number of rows to upload ({self.max_rows_upload}); stopping"
                    )
                    finished = True
                    break

                print(f"...Uploaded {n_rows_uploaded} rows to {secrets['aurora_url']}")

        # Get the number of rows in the database
        print(f"Upload completed; preparing to rebuild the ivfflat index")
        cur.execute(f"SELECT COUNT(*) FROM vectorgeo;")
        n_rows = cur.fetchone()[0]
        n_lists = int(np.sqrt(n_rows))
        print(
            f"Found {n_rows} rows in the database, using {n_lists} lists for the IVFFlat index"
        )

        print(f"Beginning indexing operation - this can take 30 minutes or longer!")
        cur.execute(f"SET maintenance_work_mem TO {self.maintenance_work_mem};")
        cur.execute("DROP INDEX IF EXISTS vector_index;")
        cur.execute(
            f"CREATE INDEX vector_index ON vectorgeo USING ivfflat (embedding vector_cosine_ops) WITH (lists = {n_lists})"
        )
        cur.execute("SET maintenance_work_mem TO 1000000;")
        print(f"Indexing operation complete!")
        conn.commit()

        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.
        """
        pass


if __name__ == "__main__":
    AuroraUploadFlow()
