#!/bin/bash

source="$1"
target="a.out"
compiler="g++"
input="input.txt"
output="output.txt"

if [ ! -f "$input" ]; then
	echo "Error: no input.txt"
	exit 1
fi

if [ -f "$source" ]; then
	rm -f a.out
	$compiler -std=c++11 -I./ -o $target $source
	if  [ ! -f "$target" ]; then
		echo "Error: compile error"
		exit 2
	fi
	./$target 0<"$input" 1>"$output"
	echo "******input******"
	cat "$input"
	echo "******output******"
	cat "$output"
else
	echo "Usage:"
	echo "./test.sh source"
fi
