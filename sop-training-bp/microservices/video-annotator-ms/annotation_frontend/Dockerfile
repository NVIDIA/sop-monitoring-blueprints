######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
######################################################################################################

FROM node:18-alpine AS build

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

# Create a startup script to handle environment variable substitution
# Dynamic substitution of environment variables
# https://www.baeldung.com/linux/nginx-config-environment-variables
RUN echo '#!/bin/sh' > /docker-entrypoint.d/30-substitute-env.sh && \
    echo 'envsubst "\$ANNOTATION_BACKEND_HOST \$ANNOTATION_BACKEND_PORT \$AUGMENTATION_BACKEND_HOST \$AUGMENTATION_BACKEND_PORT \$VLM_TRAINING_BACKEND_HOST \$VLM_TRAINING_BACKEND_PORT \$DDM_TRAINING_BACKEND_HOST \$DDM_TRAINING_BACKEND_PORT" < /etc/nginx/conf.d/default.conf.template > /etc/nginx/conf.d/default.conf' >> /docker-entrypoint.d/30-substitute-env.sh && \
    chmod +x /docker-entrypoint.d/30-substitute-env.sh

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]