#!/usr/bin/env bash

# rename-files.sh
# Bulk rename: remove any "_digits(or dots)_" from filenames.

for file in *_[0-9.]*_*; do
  newname="$(echo "$file" | sed -E 's/_[0-9.]+_/_/')"

  if [ "$file" != "$newname" ]; then
    echo "Renaming: $file -> $newname"
    mv "$file" "$newname"
  fi
done
