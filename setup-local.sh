#!/bin/bash
set -e

docker compose up -d

until curl -s http://localhost:4566/_localstack/health | grep -q '"s3": "available"'; do
    sleep 2
done

cd terraform
terraform init -upgrade
terraform apply -auto-approve
cd ..



