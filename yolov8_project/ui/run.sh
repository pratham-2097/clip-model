#!/bin/bash
# Simple script to run the Streamlit UI

cd "$(dirname "$0")/.."
python3 -m streamlit run ui/app.py

