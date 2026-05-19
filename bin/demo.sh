#!/bin/sh

model=adakoop
fn=Lorenz
input_dir=dysts

COMMAND="uv run python src/main.py --multirun \
  model=${model} \
  io.input_dir=${input_dir} \
  io.root_out_dir=out/ \
  data.fn=${fn} \
  save=False \
  verbose=False"

bash bin/run_wrapper.sh "$@" "$COMMAND"
