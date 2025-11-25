#!/bin/bash

# Usage function
usage() {
  echo "usage: $0 <service_name> [ros_domain_id]"
  echo "service_name:"
  echo "  dev             Development service (interactive shell)"
  echo "  fire            Fire detection service (run detector)"
  echo "ros_domain_id: (optional) ROS Domain ID (0-232, default: 0)"
  echo ""
  echo "Examples:"
  echo "  $0 dev          # Launch dev service with ROS_DOMAIN_ID=0"
  echo "  $0 fire 10      # Launch fire service with ROS_DOMAIN_ID=10"
  exit 1
}

# Check if at least one argument is provided
if [ $# -lt 1 ]; then
    echo "Error: Service name is required."
    usage
fi

# Parse arguments
SERVICE_NAME=$1
ROS_DOMAIN_ID=${2:-0}

# Validate service name
case "$SERVICE_NAME" in
    dev|fire)
        ;;
    *)
        echo "Error: Invalid service name '$SERVICE_NAME'."
        usage
        ;;
esac

# Validate ROS_DOMAIN_ID
if ! [[ "$ROS_DOMAIN_ID" =~ ^[0-9]+$ ]] || [ "$ROS_DOMAIN_ID" -lt 0 ] || [ "$ROS_DOMAIN_ID" -gt 232 ]; then
    echo "Error: Invalid ROS_DOMAIN_ID. Please provide a value between 0 and 232."
    usage
fi

# Configuration
COMPOSE_FILE="docker/compose.cpu.yml"

## 1. Clean container
echo "=== [FIRE DETECTION] Pull & Run ==="
echo "[FIRE DETECTION] Remove Containers ..."
docker compose -p fire_detection -f $COMPOSE_FILE down --volumes --remove-orphans

## 2. Environment setup
export DISPLAY=${DISPLAY:-:0}
export ROS_DOMAIN_ID

# Only run xhost for local displays (not SSH forwarded)
if [[ "$DISPLAY" != localhost:* ]]; then
  echo "[FIRE DETECTION] Allowing Docker containers to access X11..."
  xhost +local:docker 2>/dev/null || echo "[WARNING] xhost failed, X11 may not work"
else
  echo "[FIRE DETECTION] Detected SSH X11 forwarding (DISPLAY=$DISPLAY)"
  echo "[WARNING] X11 forwarding may not work in container. Consider using native display."
fi

## 3. Deployment
echo "[FIRE DETECTION] Deploying $SERVICE_NAME service on CPU..."
echo "[FIRE DETECTION] ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
docker compose -p fire_detection -f $COMPOSE_FILE up -d $SERVICE_NAME

echo "[FIRE DETECTION] Entering container..."
if [ "$SERVICE_NAME" = "dev" ]; then
    docker compose -p fire_detection -f $COMPOSE_FILE exec dev /bin/bash
elif [ "$SERVICE_NAME" = "fire" ]; then
    docker compose -p fire_detection -f $COMPOSE_FILE logs -f fire
fi