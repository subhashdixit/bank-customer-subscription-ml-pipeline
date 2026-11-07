"""Optional billable AWS inference exercise. Not run by CI or Render.

Deploys only the numeric classifier; the full web application stays on a web host.
Requires an existing ECR image, S3 release tar and narrowly scoped execution role.
"""
import argparse
import json


def main():
    import boto3
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["deploy", "delete"])
    parser.add_argument("--region", required=True)
    parser.add_argument("--name", required=True, help="Unique exercise resource prefix, max 40 characters")
    parser.add_argument("--role-arn")
    parser.add_argument("--image-uri")
    parser.add_argument("--artifact-uri")
    parser.add_argument("--instance-type", default="ml.m5.large")
    args = parser.parse_args()
    if not args.name or len(args.name) > 40 or not args.name.replace("-", "").isalnum():
        parser.error("Use a short alphanumeric/hyphen name")
    sm = boto3.client("sagemaker", region_name=args.region)
    model, config, endpoint = (args.name + suffix for suffix in ("-model", "-config", "-endpoint"))
    if args.action == "delete":
        # Explicit resource names only. Fail visibly on missing resources; rerun the
        # remaining individual deletions in the console after a partial cleanup.
        sm.delete_endpoint(EndpointName=endpoint)
        sm.get_waiter("endpoint_deleted").wait(EndpointName=endpoint)
        sm.delete_endpoint_config(EndpointConfigName=config)
        sm.delete_model(ModelName=model)
        print("Deleted named endpoint/config/model. S3 objects and ECR images remain.")
        return
    if not all((args.role_arn, args.image_uri, args.artifact_uri)):
        parser.error("Deployment requires --role-arn, --image-uri and --artifact-uri")
    print(json.dumps({"model": model, "config": config, "endpoint": endpoint,
                      "warning": "Creates billable managed inference resources"}))
    # Resources are intentionally not silently overwritten or deleted on failure.
    sm.create_model(ModelName=model, ExecutionRoleArn=args.role_arn, PrimaryContainer={
        "Image": args.image_uri, "ModelDataUrl": args.artifact_uri,
        "Environment": {"RELEASE_DIR": "/opt/ml/model", "PORT": "8080", "ALLOW_LLM_REQUESTS": "false"}
    })
    sm.create_endpoint_config(EndpointConfigName=config, ProductionVariants=[{
        "VariantName": "primary", "ModelName": model, "InitialInstanceCount": 1,
        "InstanceType": args.instance_type, "InitialVariantWeight": 1.0
    }])
    sm.create_endpoint(EndpointName=endpoint, EndpointConfigName=config)
    sm.get_waiter("endpoint_in_service").wait(EndpointName=endpoint)
    print("Ready: " + endpoint)


if __name__ == "__main__":
    main()
