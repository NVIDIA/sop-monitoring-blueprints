FROM alpine:3.21.3

# Create a working directory
WORKDIR /opt/mdx/

# Copy the init scripts into the working directory
COPY ./kibana-dashboard ./

# Install bash and curl commands.
RUN apk update && apk add --no-cache bash curl