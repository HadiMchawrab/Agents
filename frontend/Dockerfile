FROM node:18-alpine

WORKDIR /usr/src/app

# Install dependencies first (better caching)
COPY package*.json ./
RUN npm config set fetch-timeout 300000 && \
    npm config set registry https://registry.npmjs.org/ && \
    npm install --legacy-peer-deps

# Copy the rest of the application
COPY . .

# Build the application
RUN npm run build

EXPOSE 3000

CMD ["npm", "run", "dev", "--", "--host"] 