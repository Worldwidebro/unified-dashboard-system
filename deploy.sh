#!/bin/bash

echo "🌐 Deploying Unified Dashboard..."
docker-compose up -d --build
echo "✅ Dashboard deployed at http://localhost:3000"
