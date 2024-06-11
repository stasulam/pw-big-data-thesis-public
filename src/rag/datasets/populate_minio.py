import logging
import json
import io
from datetime import datetime
from functools import partial
from multiprocessing import Pool, Manager, Lock
from typing import Any

from rag.clients import setup_minio_client


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def put_json_to_minio(
    json_as_str: str,
    bucket: str,
    path_to_env: str,
    counter: Any,
    lock: Any,
):
    """Put JSON to MinIO."""
    client = setup_minio_client(path_to_env)
    data: dict[str, Any] = json.loads(json_as_str)
    data["timestamp"] = datetime.now().isoformat()
    json_as_str = json.dumps(data)
    try:
        with lock:
            subdir = f"{(counter.value // 10_000 + 1) * 10:.0f}k"
            object_name = f"{subdir}/{data['id']}.json"
            counter.value += 1

        client.put_object(
            bucket_name=bucket,
            object_name=object_name,
            data=io.BytesIO(json_as_str.encode("utf-8")),
            length=len(json_as_str),
            content_type="application/json",
        )
        if counter.value % 10_000 == 0:
            LOGGER.info(f"Uploaded {counter.value} files to {subdir}.")
    except Exception as e:
        LOGGER.error(f"Error uploading {data['id']}: {e}")


def main(
    path_to_raw_data: str,
    bucket: str,
    processes: int,
    path_to_env: str,
):
    """Main."""
    client = setup_minio_client(path_to_env)
    # Create bucket if it does not exist
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        LOGGER.info(f"Created bucket: {bucket}")

    data = open(path_to_raw_data)

    with Manager() as manager:
        counter = manager.Value('i', 0)  # Shared counter
        lock = manager.Lock()  # Lock to synchronize access to the counter
        with Pool(processes) as pool:
            pool.map(
                partial(put_json_to_minio, bucket=bucket, path_to_env=path_to_env, counter=counter, lock=lock),
                data,
            )
