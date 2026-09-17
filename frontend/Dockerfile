# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Bangla & English ASR Studio — React/Vite frontend
#
# Targets:
#   dev  — Vite dev server with HMR (used by docker-compose.dev.yml)
#   prod — static build served by nginx, /api reverse-proxied to the backend
#
# Build context is frontend/.
# ---------------------------------------------------------------------------

ARG NODE_VERSION=22

# ------------------------------------------------------------------ deps stage
FROM node:${NODE_VERSION}-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# ------------------------------------------------------------------- dev stage
FROM node:${NODE_VERSION}-alpine AS dev
WORKDIR /app
ENV NODE_ENV=development
COPY --from=deps /app/node_modules ./node_modules
COPY . .
EXPOSE 5173
# Host/port already set by the "dev" script in package.json.
CMD ["npm", "run", "dev"]

# ----------------------------------------------------------------- build stage
FROM node:${NODE_VERSION}-alpine AS build
WORKDIR /app
# VITE_API_BASE is baked in at build time. Leave empty to use same-origin /api
# (the nginx proxy below), which is what the compose stack does.
ARG VITE_API_BASE=""
ENV VITE_API_BASE=${VITE_API_BASE}
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

# ------------------------------------------------------------------ prod stage
FROM nginx:1.27-alpine AS prod

# Upstream the /api proxy points at; substituted into the config at startup.
ENV BACKEND_UPSTREAM=http://backend:8000

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf.template /etc/nginx/templates/default.conf.template

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://127.0.0.1/healthz >/dev/null 2>&1 || exit 1
