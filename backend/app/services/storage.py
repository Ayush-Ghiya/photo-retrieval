from typing import Literal

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from app.config import Settings

Kind = Literal["originals", "thumbs"]


class Storage:
    """S3 access. Works against floci locally and real S3 when S3_ENDPOINT_URL is unset."""

    def __init__(self, settings: Settings):
        self._buckets: dict[str, str] = {
            "originals": settings.s3_bucket_originals,
            "thumbs": settings.s3_bucket_thumbs,
        }
        self._ttl = settings.presign_ttl_seconds
        self._region = settings.aws_region
        common = dict(
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region,
            config=BotoConfig(signature_version="s3v4", s3={"addressing_style": "path"}),
        )
        self.client = boto3.client("s3", endpoint_url=settings.s3_endpoint_url, **common)
        # Separate client so presigned URLs use the browser-reachable endpoint.
        self._signer = boto3.client("s3", endpoint_url=settings.s3_public_endpoint, **common)

    def bucket(self, kind: Kind) -> str:
        return self._buckets[kind]

    def ensure_buckets(self) -> None:
        for bucket in self._buckets.values():
            try:
                self.client.head_bucket(Bucket=bucket)
            except ClientError:
                kwargs = {}
                if self._region != "us-east-1":
                    kwargs["CreateBucketConfiguration"] = {"LocationConstraint": self._region}
                self.client.create_bucket(Bucket=bucket, **kwargs)

    def put(self, kind: Kind, key: str, data: bytes, content_type: str) -> None:
        self.client.put_object(Bucket=self.bucket(kind), Key=key, Body=data, ContentType=content_type)

    def get(self, kind: Kind, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket(kind), Key=key)["Body"].read()

    def delete(self, kind: Kind, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket(kind), Key=key)

    def presign(self, kind: Kind, key: str) -> str:
        return self._signer.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket(kind), "Key": key}, ExpiresIn=self._ttl
        )

    def ping(self) -> None:
        self.client.head_bucket(Bucket=self._buckets["originals"])
