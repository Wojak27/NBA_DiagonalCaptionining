#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 /path/to/Dockerfile [gpu_type:gpus_requested] [account] [time_limit] [node_type] [bind_path]"
    echo "Example: $0 ./Dockerfile A100:2 project-account 00-01:00:00 thin /data:/app/data"
}

# Check for minimum arguments
if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

# Assign arguments
DOCKERFILE_PATH=$1
GPU_SPEC=${2:-A100:1}  # Default to 1 A100 GPU if not specified
ACCOUNT=${3:-default_account}  # Replace 'default_account' with your default project account
TIME_LIMIT=${4:-00-02:00:00}  # Default to 2 hours if not specified
NODE_TYPE=${5:-thin}  # Default to 'thin' node if not specified
BIND_PATH=${6:-""}  # Optional bind path

# Check if the Dockerfile exists
if [ ! -f "$DOCKERFILE_PATH" ]; then
    echo "Error: Dockerfile not found at $DOCKERFILE_PATH"
    exit 1
fi

# Extract directory and Dockerfile name
DOCKERFILE_DIR=$(dirname "$DOCKERFILE_PATH")
DOCKERFILE_NAME=$(basename "$DOCKERFILE_PATH")

# Generate a random name for the image
IMAGE_NAME="image_$(date +%s)"

# Build the Docker image and convert it to an Apptainer image
echo "Building Docker image and converting to Apptainer image..."
docker build -t "$IMAGE_NAME" -f "$DOCKERFILE_NAME" "$DOCKERFILE_DIR" &&
apptainer build "$IMAGE_NAME.sif" docker-daemon:"$IMAGE_NAME:latest"

# Check if the image was built successfully
if [ $? -ne 0 ]; then
    echo "Error: Failed to build the Apptainer image"
    exit 1
fi

# Create a batch job script
BATCH_SCRIPT="batch_script_$IMAGE_NAME.sh"
cat <<EOF > $BATCH_SCRIPT
#!/bin/bash
#SBATCH -A $ACCOUNT
#SBATCH --gpus $GPU_SPEC
#SBATCH -t $TIME_LIMIT
#SBATCH -C $NODE_TYPE

apptainer exec --nv ${BIND_PATH:+--bind $BIND_PATH} $IMAGE_NAME.sif python some_AI_model.py
EOF

# Submit the batch job
echo "Submitting batch job..."
sbatch $BATCH_SCRIPT

# Optional: Clean up Docker image after use
# echo "Cleaning up Docker image..."
# docker rmi "$IMAGE_NAME"

