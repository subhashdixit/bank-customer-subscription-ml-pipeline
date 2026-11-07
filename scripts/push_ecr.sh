#!/usr/bin/env bash
# Publishes an image to an EXISTING ECR repository; does not create infrastructure.
set -euo pipefail
region="${1:?Usage: bash scripts/push_ecr.sh REGION REPOSITORY IMAGE_TAG}"
repository="${2:?Repository required}"
tag="${3:?Immutable release tag required}"
account_id="$(aws sts get-caller-identity --query Account --output text)"
registry="$account_id.dkr.ecr.$region.amazonaws.com"
image="$registry/$repository:$tag"
aws ecr get-login-password --region "$region" | docker login --username AWS --password-stdin "$registry"
docker buildx build --platform linux/amd64 --load -t "$image" .
docker push "$image"
digest="$(aws ecr describe-images --region "$region" --repository-name "$repository" --image-ids "imageTag=$tag" --query 'imageDetails[0].imageDigest' --output text)"
printf 'Immutable image URI: %s/%s@%s\n' "$registry" "$repository" "$digest"
