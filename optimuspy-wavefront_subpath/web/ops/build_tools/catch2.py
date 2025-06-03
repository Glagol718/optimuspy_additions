from pathlib import Path

from web.models import Task

from web.ops.compilers import Compiler, GenericCflags

TESTS = '''#include "../../../catch2/catch_amalgamated.hpp"
#include "__optimus_debug_hook.h"

TEST_CASE("Optimus")
{
    BENCHMARK("opt_bench")
    {
        return %s();
    };
}
'''

DEBUG_HOOK = '''#pragma once
%s;
'''


def setup(path: Path, c_files: list[Path], task: Task, c: Compiler, cf: GenericCflags):
    # https://learn.microsoft.com/en-us/cpp/build/reference/tc-tp-tc-tp-specify-source-file-type
    if c.name == 'cl':
        c_files = [f'/Tp {f}' for f in c_files]
    with open(path / 'Makefile', 'w', encoding='utf8') as f:
        f.write(  # MAKEFILE
            f'''CFLAGS={cf.value}
all: build test

build:
\t{c.name} $(CFLAGS) {' '.join(f for f in c_files)} __optimus_tests.cpp ../../../catch2/catch_amalgamated_{c.name}.o {c.out} __optimus_tests -I../../../catch2

test:
\t./__optimus_tests --benchmark-samples {task.tests} --out __optimus_tests.txt
''')

    with open(path / '__optimus_debug_hook.h', 'w', encoding='utf8') as f:
        f.write(DEBUG_HOOK
                % task.f_sign)

    with open(path / '__optimus_tests.cpp', 'w', encoding='utf8') as f:
        f.write(TESTS % task.f_name)


def parse_benchmark(path: Path):
    v, u = -1, 'err'
    try:
        with open(path / '__optimus_tests.txt', 'r', encoding='utf8') as f:
            for line in f:
                if line.startswith('opt_bench'):
                    tokens = [i for i in f.readline().split(' ') if i]
                    return float(tokens[0]), tokens[1]
     # pylint: disable=broad-exception-caught
    except Exception as exc:
        print(exc)
    return v, u


def cleanup(path: Path, files: list[Path]):
    _files = set(file.name for file in files)
    for file in path.iterdir():
        if file.name not in _files:
            file.unlink()
