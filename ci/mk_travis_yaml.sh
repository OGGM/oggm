#!/bin/bash

for os in linux osx windows; do
	for py_ver in 3.6 3.7; do
		for test_env in prepro numerics models benchmark utils workflow graphics; do
			EXT="OGGM_TEST_ENV=$test_env PYTHON_VERSION=$py_ver"
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

