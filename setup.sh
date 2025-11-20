#!/bin/bash

# Quick setup script for the Integer Factorization ML project

echo "🚀 Setting up Integer Factorization ML Project"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"
echo ""

# Build and start services
echo "📦 Building Docker images..."
docker-compose build

echo ""
echo "🚀 Starting services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check backend health
echo "🔍 Checking backend health..."
for i in {1..10}; do
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "✅ Backend is healthy"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "⚠️  Backend is not responding. Check logs with: docker-compose logs backend"
    fi
    sleep 2
done

echo ""
echo "📊 Generating initial dataset..."
curl -X POST http://localhost:8000/generate-dataset \
  -H "Content-Type: application/json" \
  -d '{"num_samples": 10000}' \
  -s | python3 -m json.tool || echo "⚠️  Dataset generation may have failed. Check backend logs."

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📍 Access points:"
echo "   - Frontend: http://localhost:3000"
echo "   - Backend API: http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo ""
echo "📝 Next steps:"
echo "   1. Train the model: curl -X POST http://localhost:8000/train -H 'Content-Type: application/json' -d '{\"epochs\": 50}'"
echo "   2. Or use the frontend at http://localhost:3000"
echo ""
echo "📋 Useful commands:"
echo "   - View logs: docker-compose logs -f"
echo "   - Stop services: docker-compose down"
echo "   - Restart services: docker-compose restart"

