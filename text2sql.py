import os
import pandas as pd
import sqlite3
import openai

def setup_environment():
    """Set up environment variables."""
    os.environ['OPENAI_API_KEY'] = 'sk-your-openai-api-key'
    openai.api_key = os.environ['OPENAI_API_KEY']


def create_and_populate_table():
    """Create and populate the breast cancer data table."""
    # Read CSV file
    df = pd.read_csv("data.csv")

    # Connect to SQLite database
    conn = sqlite3.connect("breast_cancer_db.sqlite")
    cursor = conn.cursor()

    # Create the table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS breast_cancer_data (
        id INTEGER PRIMARY KEY,
        diagnosis TEXT,
        radius_mean REAL,
        texture_mean REAL,
        perimeter_mean REAL,
        area_mean REAL,
        smoothness_mean REAL,
        compactness_mean REAL,
        concavity_mean REAL,
        concave_points_mean REAL,
        symmetry_mean REAL,
        fractal_dimension_mean REAL,
        radius_se REAL,
        texture_se REAL,
        perimeter_se REAL,
        area_se REAL,
        smoothness_se REAL,
        compactness_se REAL,
        concavity_se REAL,
        concave_points_se REAL,
        symmetry_se REAL,
        fractal_dimension_se REAL,
        radius_worst REAL,
        texture_worst REAL,
        perimeter_worst REAL,
        area_worst REAL,
        smoothness_worst REAL,
        compactness_worst REAL,
        concavity_worst REAL,
        concave_points_worst REAL,
        symmetry_worst REAL,
        fractal_dimension_worst REAL
    );
    """)
    conn.commit()

    # Insert the data into the table
    df.to_sql("breast_cancer_data", conn, if_exists="replace", index=False)

    # Test retrieval
    print("Data inserted successfully!")
    data = pd.read_sql("SELECT * FROM breast_cancer_data LIMIT 5", conn)
    print(data)

    # Close connection
    conn.close()


def generate_sql_query_and_fetch_result(question: str, db_path: str) -> str:
    """
    Generate an SQL query for the given question and fetch the corresponding result from the database.
    """
    # Define the prompt
    prompt = f"""
    You are an expert in SQL. Generate a valid SQL query for the following question. Do not include explanations, comments, or additional text.
    Question: {question}
    """

    # Call OpenAI API to generate the SQL query
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # You can use "gpt-4" if available
            prompt=prompt,
            max_tokens=200,
            temperature=0
        )
        sql_query = response.choices[0].text.strip()
    except Exception as e:
        return f"Error generating SQL query: {str(e)}"

    # Execute the SQL query and fetch the result
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]  # Get column names
        conn.close()

        # Format the result as a table
        result_table = f"SQL Query:\n{sql_query}\n\nResults:\n"
        result_table += "\t".join(column_names) + "\n"
        result_table += "\n".join("\t".join(str(value) for value in row) for row in rows)
    except Exception as e:
        conn.close()
        return f"SQL Query:\n{sql_query}\n\nError:\n{str(e)}"

    return result_table


# Example usage
if __name__ == "__main__":
    setup_environment()
    create_and_populate_table()
    question = "Retrieve all columns and rows from the breast_cancer_data table where diagnosis is 'F.'"
    db_path = "breast_cancer_db.sqlite"  # Path to your SQLite database
    output = generate_sql_query_and_fetch_result(question, db_path)
    print(output)
