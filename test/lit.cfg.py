# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

import lit.llvm
lit.llvm.initialize(lit_config, config)

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool


# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'QUANTUM_OPT'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir', '.qasm']


# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'Examples', 'Scratches', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']


if hasattr(config, 'quantum_obj_root'):
    # test_source_root: The root path where tests are located.
    config.test_source_root = os.path.dirname(__file__)

    # test_source_root: The root path where tests are located.
    config.test_source_root = os.path.dirname(__file__)


    config.substitutions.append(('%PATH%', config.environment['PATH']))
    config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

    llvm_config.with_system_environment(
        ['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

    llvm_config.use_default_substitutions()

    # test_exec_root: The root path where tests should be run.
    config.test_exec_root = os.path.join(config.quantum_obj_root, 'test')

    # test_exec_root: The root path where tests should be run.
    config.test_exec_root = os.path.join(config.quantum_obj_root, 'test')
    config.quantum_tools_dir = os.path.join(config.quantum_obj_root, 'bin')

    # Tweak the PATH to include the tools dir.
    llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

    tool_dirs = [config.quantum_tools_dir, config.llvm_tools_dir]
    tools = [
        'quantum-opt',
        'quantum-translate'
    ]
    llvm_config.add_tool_substitutions(tools, tool_dirs)
