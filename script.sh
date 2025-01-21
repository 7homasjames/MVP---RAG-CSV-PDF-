#!/bin/bash

# Activate the virtual environment if needed
# source /path/to/your/venv/bin/activate

# Run the fastapi app
fastapi run api.py &

# Run the Streamlit app
streamlit run app.py