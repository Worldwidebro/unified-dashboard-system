#!/bin/bash

# 🚀 Deployment Script for unified-dashboard-system
# Billionaire Consciousness Empire

set -e

echo "🚀 Deploying unified-dashboard-system..."

# Build and start services
docker-compose up -d --build

# Wait for services to be ready
sleep 10

# Health check
echo "🔍 Running health checks..."
docker-compose ps

# Show logs
echo "📋 Recent logs:"
docker-compose logs --tail=20

echo "✅ unified-dashboard-system deployed successfully!"
echo "🌐 Access: http://localhost:3000"
