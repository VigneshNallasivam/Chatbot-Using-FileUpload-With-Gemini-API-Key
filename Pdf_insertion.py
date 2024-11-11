import psycopg2
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

# Connect to your PostgreSQL database
conn = psycopg2.connect(
    dbname="PGVector",
    user="postgres",
    password="root",
    host="localhost",
    port="5433"
)
cur = conn.cursor()

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to create table if it doesn't exist
def create_table():
    create_table_query = """
    CREATE TABLE IF NOT EXISTS resume_table (
        id SERIAL PRIMARY KEY,
        pdf_text TEXT,
        vector FLOAT8[]
    );
    """
    cur.execute(create_table_query)
    conn.commit()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to convert text to vector
def text_to_vector(text):
    return model.encode(text).tolist()

# Function to insert PDF text into pgvector
def insert_pdf_to_pgvector(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    vector = text_to_vector(text)
    cur.execute("INSERT INTO resume_table (pdf_text, vector) VALUES (%s, %s)", (text, vector))
    conn.commit()

# Create table
create_table()

# Example usage
pdf_path = "D:\\Vigneshwaran'sData\\PDF\\Vigneshwaran_Resume.pdf"
insert_pdf_to_pgvector(pdf_path)

# Close the connection
cur.close()
conn.close()
