import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pgvector.sqlalchemy import Vector
from sqlalchemy import Index, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


# Define the models
class Base(DeclarativeBase):
    pass


class Video(Base):
    __tablename__ = "videos"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    videoId: Mapped[str] = mapped_column()
    title: Mapped[str] = mapped_column()
    description: Mapped[str] = mapped_column()
    embedding = mapped_column(Vector(1536))  # ada-002 is 1536-dimensional


# Connect to the database based on environment variables
load_dotenv(override=True)
DBUSER = os.environ["DBUSER"]
DBPASS = os.environ["DBPASS"]
DBHOST = os.environ["DBHOST"]
DBNAME = os.environ["DBNAME"]
DATABASE_URI = f"postgresql://{DBUSER}:{DBPASS}@{DBHOST}/{DBNAME}"
# Use SSL if not connecting to localhost
if DBHOST != "localhost":
    DATABASE_URI += "?sslmode=require"
engine = create_engine(DATABASE_URI, echo=False)

# Create tables in database
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

# Insert data and issue queries
with Session(engine) as session:
    session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    
    index = Index(
        "hnsw_index2",
        Video.embedding,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    index.create(engine)

    # Insert the videos from the JSON file
    current_directory = Path(__file__).parent
    data_path = current_directory / "videos_ada002.json"
    with open(data_path) as f:
        videos = json.load(f)
        for v in videos:
            video = Video(videoId=v['videoId'], title=v['title'], description=v['description'], embedding=v['embedding'])
            session.add(video)
        session.commit()

    # Query the database
    query = select(Video).where(Video.title == "Someone sent me this VS Code extension on Twitter")
    target_video = session.execute(query).scalars().first()
    if target_video is None:
        print("Video not found")
        exit(1)

    # Find the 5 most similar movies to "Winnie the Pooh"
    closest = session.scalars(
        select(Video).order_by(Video.embedding.cosine_distance(target_video.embedding)).limit(5)
    )
    for video in closest:
        print(video.title)
