#!/usr/bin/env python3

import argparse

from cache_client import CacheClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7878, type=int)
    args = parser.parse_args()
    client = CacheClient(args.host, args.port)

    task_id = "task-python-client"
    parent_path = ["tool\nparent"]
    child_path = ["tool\nparent", "git\nstatus"]
    child_value = "value\nwith newline"
    parent_sandbox_id = "sandbox-1"

    parent_insert_response = client.insert(
        task_id, parent_path, None, parent_sandbox_id, runtime=10
    )
    print("parent insert response:", parent_insert_response)
    assert parent_insert_response == {
        "type": "insert_result",
        "value_inserted": True,
        "sandbox_inserted": True,
    }

    child_insert_response = client.insert(task_id, child_path, child_value, None, runtime=25)
    print("child insert response:", child_insert_response)
    assert child_insert_response == {
        "type": "insert_result",
        "value_inserted": True,
        "sandbox_inserted": False,
    }

    failed_extension_response = client.insert(
        task_id,
        ["missing\nparent", "child\nthat cannot be inserted"],
        "should not insert",
        "sandbox-that-should-not-insert",
        runtime=5,
    )
    print("failed extension response:", failed_extension_response)
    assert failed_extension_response == {
        "type": "insert_result",
        "value_inserted": False,
        "sandbox_inserted": False,
    }

    find_response = client.find(task_id, child_path)
    print("find child response:", find_response)
    assert find_response == {
        "type": "find_result",
        "value": child_value,
        "sandbox_path": parent_path,
        "sandbox_id": parent_sandbox_id,
    }

    missing_response = client.find(task_id, ["missing\npath"])
    print("missing response:", missing_response)
    assert missing_response == {
        "type": "find_result",
        "value": None,
        "sandbox_path": None,
        "sandbox_id": None,
    }

    dump_response = client.dump_all_response()
    print("dump all response:", dump_response["type"])
    assert dump_response["type"] == "json"


if __name__ == "__main__":
    main()
