#!/bin/bash

shopt -s nullglob

for file in *; do
  if [[ -f $file ]]; then
    newname=$(echo "$file" | sed 's/\.\{2,\}/./g')
    if [[ "$file" != "$newname" ]]; then
      mv "$file" "$newname"
      echo "Renamed: $file -> $newname"
    fi
  fi
done
