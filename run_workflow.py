import os
import yaml
import subprocess


def run_github_workflow(workflow_file):
    """Parses and executes GitHub Actions workflow steps locally."""
    with open(workflow_file, "r") as f:
        workflow = yaml.safe_load(f)

    for job_name, job in workflow.get("jobs", {}).items():
        print(f"\n🔹 Running job: {job_name}\n")

        for step in job.get("steps", []):
            if "run" in step:
                command = step["run"]
                print(f"➡️ Running: {command}")

                # Execute shell command
                process = subprocess.run(
                    command, shell=True, capture_output=True, text=True
                )

                # Print output
                if process.stdout:
                    print("✅ Output:\n", process.stdout)
                if process.stderr:
                    print("⚠️ Error:\n", process.stderr)

                if process.returncode != 0:
                    print(f"❌ Step failed: {step.get('name', 'Unnamed step')}")
                    break  # Stop execution on failure


run_github_workflow(".github/workflows/experiments_amplitudes.yaml")
run_github_workflow(".github/workflows/experiments_tagging.yaml")
run_github_workflow(".github/workflows/tests.yaml")
