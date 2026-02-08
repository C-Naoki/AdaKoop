#!/bin/bash

folder=$(basename $(pwd))

root_path=$(pwd)
site_packages_path=$(find $root_path/.venv/lib/python*/site-packages -type d | head -n 1)

if [ -z "$site_packages_path" ]; then
  echo "Error: site-packages directory not found."
  exit 1
fi

pth_path=$(find "$site_packages_path" -name "$folder.pth" 2>/dev/null)

if [ -z "$pth_path" ]; then
  touch "$site_packages_path/$folder.pth"
  echo "$folder.pth created at $site_packages_path"
  pth_path="$site_packages_path/$folder.pth"
else
  echo "$folder.pth already exists"
fi

if ! grep -Fxq "$root_path" "$pth_path"; then
  echo "$root_path" >> "$pth_path"
  echo "Added '$root_path' to $folder.pth"
else
  echo "'$root_path' is already in $folder.pth"
fi
