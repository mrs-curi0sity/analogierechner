#!/bin/bash
uvicorn api:app --host 0.0.0.0 --port 8081 & 
streamlit run streamlit_app.py --server.port 8080 --server.address 0.0.0.0