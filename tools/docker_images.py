from __future__ import print_function

import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument(
    "--org", default="nmtwizard", help="Organization name on Docker Hub."
)
parser.add_argument(
    "--path", default="frameworks", help="Path to a directory or a Dockerfile to build."
)
parser.add_argument("--build", action="store_true", help="Build the image.")
parser.add_argument("--version", default="latest", help="Image version.")
parser.add_argument("--push", action="store_true", help="Push the image.")
parser.add_argument("--sudo", action="store_true", help="Run commands with sudo.")
args = parser.parse_args()


def run(cmd):
    print("+ %s" % " ".join(cmd))
    if args.sudo:
        cmd = ["sudo"] + cmd
    exit_code = subprocess.call(cmd)
    if exit_code != 0:
        exit(exit_code)


if os.path.isfile(args.path):
    dockerfiles = [args.path]
else:
    dockerfiles = []
    for filename in os.listdir(args.path):
        path = os.path.join(args.path, filename)
        if filename == "Dockerfile":
            dockerfiles.append(path)
        elif not os.path.isdir(path):
            continue
        else:
            dockerfile = os.path.join(path, "Dockerfile")
            if os.path.isfile(dockerfile):
                dockerfiles.append(dockerfile)

for dockerfile in dockerfiles:
    framework_dir = os.path.basename(os.path.split(dockerfile)[0])
    repo_name = framework_dir.replace("_", "-")
    image_name = "%s/%s" % (args.org, repo_name)
    image_latest = "%s:latest" % image_name
    image_full_name = "%s:%s" % (image_name, args.version)
    if args.build:
        run(["docker", "build", "--pull", "-t", image_latest, "-f", dockerfile, "."])
    run(["docker", "tag", image_latest, image_full_name])
    if args.push:
        run(["docker", "push", image_latest])
        run(["docker", "push", image_full_name])
