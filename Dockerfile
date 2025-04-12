FROM node:alpine

WORKDIR /app

COPY ./frontend/package*.json ./frontend

RUN npm install

RUN npm run build

WORKDIR /app

COPY ./requirements.txt ./backend/requirements.txt

RUN pip install --no-cache-dir -r ./backend/requirements.txt

COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev", "--", "--host"]