#!/bin/bash

MAIN_TEST_CONTAINER_TAG="20181123"

for os in linux; do
	for test_container in $MAIN_TEST_CONTAINER_TAG py37; do
		for test_env in prepro numerics models benchmark utils workflow graphics; do
			EXT="TEST_CONTAINER=$test_container OGGM_TEST_ENV=$test_env"
			if [[ "$test_env" == "workflow" ]] || [[ "$test_env" == "graphics" ]]; then
				EXT="$EXT MPL=--mpl"
			else
				EXT="$EXT MPL="
			fi
			if [[ "$test_env" == "workflow" ]]; then
				for mproc in True False; do
					echo "    - env: $EXT OGGM_TEST_MULTIPROC=$mproc"
					echo "      os: $os"
				done
			else
				echo "    - env: $EXT"
				echo "      os: $os"
			fi
		done
	done
done

