######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
######################################################################################################

FROM node:20-alpine AS build

# Create working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

  # Configure npm for better reliability and install all dependencies
RUN npm config set fetch-retry-mintimeout 20000 && \
    npm config set fetch-retry-maxtimeout 120000 && \
    npm config set fetch-retries 5 && \
    npm config set fetch-timeout 300000 && \
    npm install

# Copy source code
COPY . .

# Build application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy build result to nginx directory
COPY --from=build /app/build /usr/share/nginx/html

# Copy nginx configuration template
COPY nginx.conf /etc/nginx/conf.d/default.conf.template

# Create a startup script to handle environment variable substitution.
# Dynamic substitution of environment variables.
# https://www.baeldung.com/linux/nginx-config-environment-variables
# Single-quoted heredoc keeps $VAR literal in the file so envsubst sees them.
RUN cat > /docker-entrypoint.d/30-substitute-env.sh <<'EOSCRIPT' && \
    chmod +x /docker-entrypoint.d/30-substitute-env.sh
#!/bin/sh
# \$VAR is escaped so envsubst — not /bin/sh — does the substitution.
envsubst "\
\$ANNOTATION_BACKEND_HOST \$ANNOTATION_BACKEND_PORT \
\$AUGMENTATION_BACKEND_HOST \$AUGMENTATION_BACKEND_PORT \
\$VLM_TRAINING_BACKEND_HOST \$VLM_TRAINING_BACKEND_PORT \
\$DDM_TRAINING_BACKEND_HOST \$DDM_TRAINING_BACKEND_PORT \
\$EVAL_BACKEND_HOST \$EVAL_BACKEND_PORT" \
< /etc/nginx/conf.d/default.conf.template \
> /etc/nginx/conf.d/default.conf
EOSCRIPT

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]