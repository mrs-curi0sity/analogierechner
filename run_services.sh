#!/bin/bash
export PYTHONPATH=/app/src:$PYTHONPATH
uvicorn src.api.api:app --host 0.0.0.0 --port 8081 &
streamlit run src/streamlit_app.py --server.port 8080 --server.address 0.0.0.0