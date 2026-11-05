# AWS architecture and honest deployment roadmap

## What exists today

A local Python training job, saved model, API, Docker instructions and CI. No AWS resources, S3 client, IAM policy, SageMaker job or cloud endpoint have been created. The `infra` folder is reserved for implementation in later lessons. Do not claim a cloud deployment from this scaffold.

## Local-to-cloud map

| Need | Current local solution | AWS service |
| --- | --- | --- |
| Dataset and model storage | Files under data and artifacts | S3 objects in buckets |
| Training compute | Python on laptop | SageMaker training job |
| Image storage | Local Docker images | ECR repository |
| Self-managed serving | Local Docker container | EC2 instance running our container |
| Managed ML serving | Not implemented | SageMaker endpoint |
| Permissions | Operating-system file access | IAM roles and policies |
| Logs and operational metrics | Terminal output | CloudWatch |
| Release automation | GitHub CI checks | GitHub workflow assuming an AWS role |

S3 stores bytes; it does not run the API. EC2 is a virtual machine whose application setup and patching we manage. SageMaker provides managed training and hosting, including underlying compute. We will practise EC2 and SageMaker hosting as two distinct deployment paths, not run duplicate services unnecessarily.

## Future cloud flow

1. GitHub checks the change and builds a versioned image.
2. An approved workflow obtains temporary AWS credentials through GitHub OIDC and a narrowly scoped IAM role.
3. The image goes into ECR; datasets and model artifacts go into S3.
4. A training job reads training data and writes a model artifact.
5. Evaluation checks data quality, performance and model metadata.
6. An explicit release selects the image and model versions.
7. An EC2-hosted container or SageMaker endpoint serves predictions.
8. Health checks, logs and metrics confirm service behaviour; rollback selects a known earlier version.

We will learn buckets, regions, roles, trust policies, instance types, security groups, ports and DNS before running these steps. Our existing `/health` and `/predict` API is not yet a SageMaker-compatible `/ping` and `/invocations` container. The managed service also has model artifact and training path conventions; those adapters are future implementation work.

## Free learning order

1. Finish the local handbook and demonstrate predictions.
2. Add Moto-based S3 tests with dummy credentials. They test API interactions, not actual IAM enforcement or AWS availability.
3. Add version manifests, local release switching and a rollback exercise.
4. Add batch predictions, input distribution reports and a reviewed retraining workflow.
5. Prepare infrastructure as code and inspect it without applying it. Syntax validation cannot prove an AWS deployment works.
6. Explore SageMaker local mode with custom images. Verify all SDK calls: some configurations still contact S3 or other AWS endpoints.
7. If a real cloud budget or provided training sandbox becomes available, perform short, bounded deployments with cleanup and record evidence.

## Cost boundary

The current application does not call AWS. Laptop electricity, internet and disk are your existing resources. Downloading packages or Docker base images requires internet. Private GitHub Actions uses account allowances and must be checked in your billing settings.

Without AWS credits or an applicable sandbox, EC2 and managed SageMaker should not be assumed free. Do not create resources to discover their price. Before an actual deployment, inspect current region-specific prices and your account's eligibility. A budget alert is a notification, not a guaranteed spending cap.

A later cleanup checklist must cover endpoints and endpoint configurations, compute, disks, snapshots, public IPs, load balancers, container images, stored objects and logs as applicable. Stopping a virtual machine does not delete its attached storage. We will document cleanup alongside each resource's creation, not after deployment.

## Primary references

- S3: https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html
- EC2: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/concepts.html
- SageMaker containers: https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers.html
- Local pipelines: https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-local-mode.html
- SageMaker pricing: https://aws.amazon.com/sagemaker/ai/pricing/
- GitHub Actions billing: https://docs.github.com/en/billing/concepts/product-billing/github-actions

Consult current official documentation before implementing each stage; these notes are a roadmap rather than tested AWS instructions.
