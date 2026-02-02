# Frontend Dockerfile for Car-Psycho Next.js Application

FROM node:20-alpine

# Set working directory
WORKDIR /app

# Copy package files from frontend directory
COPY frontend/package*.json ./

# Install dependencies
RUN npm install

# Copy application files from frontend directory
COPY frontend/ .

# Expose port
EXPOSE 3000

# Start development server
CMD ["npm", "run", "dev"]
