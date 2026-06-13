#!/bin/bash 
# Quick start: seed sample data and launch the API 
set -e 
python scripts/seed_and_demo.py 
uvicorn app.api.main:app --reload
