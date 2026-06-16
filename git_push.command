#!/bin/bash
cd "/Users/syedhamzaali/Desktop/Masters/Intro to NLP/Final Project"

echo "=== Removing stale git lock ==="
rm -f .git/index.lock

echo "=== Staging all changes ==="
git add -A

echo "=== Committing ==="
git commit -m "Reorganize project for deployment readiness

- Move saved_model/ to project root (was webhook/saved_model/)
- Update main.py to load model from saved_model/ at root
- Update .gitattributes LFS path accordingly
- Update notebook: cell 21 loads from ../saved_model, cell 24 saves there
- Update .gitignore: cover notebook artifacts, webhook/, DS_Store recursively
- Update requirements.txt: add accelerate, uvicorn[standard]
- Add runtime.txt for Heroku Python version
- Write professional README with setup, training, API, and deployment docs"

echo "=== Pushing to GitHub ==="
git push origin main

echo ""
echo "Done! Press any key to close."
read -n 1
