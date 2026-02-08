#!/bin/sh

CURRENT_DIR="$(dirname "$0")"
CURRENT_DIR=$(realpath "$CURRENT_DIR/..")

USE_NOHUP=0
COMMAND=""

# Add modecast.pth if not exists
sh bin/add_pth.sh

# Parse options manually and rebuild ARGS without "-n"
for arg in "$@"; do
  # echo "arg: $arg"
  if [ "$arg" = "-n" ]; then
    USE_NOHUP=1
  else
    COMMAND="$COMMAND$arg"
  fi
done

# Check if COMMAND is empty
if [ -z "$COMMAND" ]; then
  echo "Error: COMMAND is required" >&2
  echo "Usage: $0 [-n] command" >&2
  exit 1
fi

# Check if COMMAND does not include invalid options
if [ "$(echo "$COMMAND" | cut -c1-6)" != "poetry" ]; then
  echo "Error: Invarid options" >&2
  echo "Use only: [-n]" >&2
  exit 1
fi

# extract model in COMMAND
MODEL_NAME=$(echo "$COMMAND" | sed -n 's/.*model=\([^ ]*\).*/\1/p')
[ -z "$MODEL_NAME" ] && MODEL_NAME="unknown_model"

# extract input_dir in COMMAND
INPUT_DIR=$(echo "$COMMAND" | sed -n 's/.*io.input_dir=\([^ ]*\).*/\1/p' | sed 's/[\/]/_/g')
[ -z "$INPUT_DIR" ] && INPUT_DIR="unknown_input"

# Judge whether to use nohup
if [ $USE_NOHUP -eq 1 ]; then
  if [ ! -d nohup ]; then
    mkdir -p nohup
  fi
  DATE_DIR=$(date +%Y%m%d)
  TARGET_DIR="nohup/$DATE_DIR"
  mkdir -p "$TARGET_DIR"

  OUTFILE="${TARGET_DIR}/${MODEL_NAME}_${INPUT_DIR}.out"
  # Check if the file already exists
  if [ -e "$OUTFILE" ]; then
    base="${OUTFILE%.out}"
    ext=".out"
    i=2
    while [ -e "${base}_${i}${ext}" ]; do
      i=$((i + 1))
    done
    OUTFILE="${base}_${i}${ext}"
  fi
  nohup sh -c "$COMMAND" > "$OUTFILE" 2>&1 &
else
  sh -c "$COMMAND"
fi
