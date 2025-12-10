#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2020 PANGAEA (https://www.pangaea.de/)
#
# SPDX-License-Identifier: MIT

"""
Example of using F-UJI's evaluate_fairness() function directly from Python.

This demonstrates how to call the FAIR assessment without making HTTP requests.
This is useful for batch processing, integration into other Python applications,
or when you want to avoid the overhead of HTTP requests.
"""

import json
import os

from fuji_server.controllers.fair_object_controller import evaluate_fairness


# Example: Evaluate a single object
def evaluate_single_object():
    """Example of evaluating a single object."""
    result = evaluate_fairness(
        object_identifier="https://doi.org/10.5281/zenodo.8347772",
        test_debug=True,
        use_datacite=True,
        use_github=False,
        metric_version="metrics_v0.8",
    )

    # The result is a FAIRResults object
    print(f"Assessment completed for: {result.request['object_identifier']}")
    print(f"Total metrics: {result.total_metrics}")
    print(f"FAIR score: {result.summary['score_percent']['FAIR']}%")
    print(f"Test ID: {result.test_id}")

    # You can convert it to a dict for JSON serialization
    result_dict = result.to_dict() if hasattr(result, "to_dict") else result
    return result


# Example: Batch evaluation
def evaluate_multiple_objects():
    """Example of evaluating multiple objects in batch."""
    pids = [
        "https://doi.org/10.5281/zenodo.8347772",
        "https://archive.materialscloud.org/record/2021.146",
    ]

    results_folder = "./results/"
    os.makedirs(results_folder, exist_ok=True)

    results = []
    for pid in pids:
        print(f"Evaluating: {pid}")
        try:
            result = evaluate_fairness(
                object_identifier=pid,
                test_debug=True,
                use_datacite=True,
                use_github=False,
            )
            results.append(result)

            # Save individual result
            res_filename = f"{pid.split('/')[-1]}.json"
            res_filename_path = os.path.join(results_folder, res_filename)

            # Convert to dict for JSON serialization
            result_dict = (
                result.to_dict()
                if hasattr(result, "to_dict")
                else {
                    "test_id": result.test_id,
                    "request": result.request,
                    "summary": result.summary,
                    "total_metrics": result.total_metrics,
                    "metric_version": result.metric_version,
                    "software_version": result.software_version,
                    "start_timestamp": (
                        result.start_timestamp.isoformat()
                        if hasattr(result.start_timestamp, "isoformat")
                        else str(result.start_timestamp)
                    ),
                    "end_timestamp": (
                        result.end_timestamp.isoformat()
                        if hasattr(result.end_timestamp, "isoformat")
                        else str(result.end_timestamp)
                    ),
                    "resolved_url": result.resolved_url,
                }
            )

            with open(res_filename_path, "w", encoding="utf-8") as fileo:
                json.dump(result_dict, fileo, ensure_ascii=False, indent=2)

            print(f"  Score: {result.summary['score_percent']['FAIR']}%")
        except Exception as e:
            print(f"  Error: {e}")

    return results


# Example: Using with authentication
def evaluate_with_auth():
    """Example of evaluating with authentication token."""
    result = evaluate_fairness(
        object_identifier="https://example.com/protected-resource",
        test_debug=True,
        auth_token="your-token-here",
        auth_token_type="Bearer",  # or "Basic"
        use_datacite=True,
    )
    return result


# Example: Using with metadata service
def evaluate_with_metadata_service():
    """Example of evaluating with a custom metadata service."""
    result = evaluate_fairness(
        object_identifier="https://doi.org/10.1594/PANGAEA.908011",
        test_debug=True,
        metadata_service_endpoint="http://ws.pangaea.de/oai/provider",
        metadata_service_type="oai_pmh",
        use_datacite=True,
    )
    return result


if __name__ == "__main__":
    # Run a simple example
    print("Running single object evaluation...")
    evaluate_single_object()

    # Uncomment to run batch evaluation
    # print("\nRunning batch evaluation...")
    # evaluate_multiple_objects()
