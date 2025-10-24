#!/bin/bash
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "Error: GOOGLE_API_KEY not set"
    echo "Set with: export GOOGLE_API_KEY='your_key'"
    exit 1
fi
streamlit run src/dynamic_r_analyst_v2.py
