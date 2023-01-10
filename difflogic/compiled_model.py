import torch
import math
from .difflogic import LogicLayer, GroupSum
import tempfile
import subprocess
import shutil
import ctypes
import numpy as np
import numpy.typing
import time
from typing import Union


ALL_OPERATIONS = [
    "zero",
    "and",
    "not_implies",
    "a",
    "not_implied_by",
    "b",
    "xor",
    "or",
    "not_or",
    "not_xor",
    "not_b",
    "implied_by",
    "not_a",
    "implies",
    "not_and",
    "one",
]

BITS_TO_DTYPE = {8: "char", 16: "short", 32: "int", 64: "long long"}
BITS_TO_ZERO_LITERAL = {8: "(char) 0",
                        16: "(short) 0", 32: "0", 64: "0LL"}
BITS_TO_ONE_LITERAL = {8: "(char) 1",
                        16: "(short) 1", 32: "1", 64: "1LL"}
BITS_TO_C_DTYPE = {8: ctypes.c_int8, 16: ctypes.c_int16,
                   32: ctypes.c_int32, 64: ctypes.c_int64}
BITS_TO_NP_DTYPE = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}


class CompiledLogicNet(torch.nn.Module):
    def __init__(
            self,
            model: torch.nn.Sequential,
            device='cpu',
            num_bits=64,
            cpu_compiler='gcc',
            verbose=False,
    ):
        super(CompiledLogicNet, self).__init__()
        self.model = model
        self.device = device
        self.num_bits = num_bits
        self.cpu_compiler = cpu_compiler

        assert cpu_compiler in ["clang", "gcc"], cpu_compiler
        assert num_bits in [8, 16, 32, 64]

        if self.model is not None:
            layers = []

            self.num_inputs = None

            assert isinstance(self.model[-1], GroupSum), 'The last layer of the model must be GroupSum, but it is {} / {}' \
                                                         ' instead.'.format(type(self.model[-1]), self.model[-1])
            self.num_classes = self.model[-1].k

            first = True
            for layer in self.model:
                if isinstance(layer, LogicLayer):
                    if first:
                        self.num_inputs = layer.in_dim
                        first = False
                    self.num_out_per_class = layer.out_dim // self.num_classes
                    layers.append((layer.indices[0], layer.indices[1], layer.weights.argmax(1)))
                elif isinstance(layer, torch.nn.Flatten):
                    if verbose:
                        print('Skipping torch.nn.Flatten layer ({}).'.format(type(layer)))
                elif isinstance(layer, GroupSum):
                    if verbose:
                        print('Skipping GroupSum layer ({}).'.format(type(layer)))
                else:
                    assert False, 'Error: layer {} / {} unknown.'.format(type(layer), layer)

            self.layers = layers

            if verbose:
                print('`layers` created and has {} layers.'.format(len(layers)))

        self.lib_fn = None

    def get_gate_code(self, var1, var2, gate_op):
        operation_name = ALL_OPERATIONS[gate_op]

        if operation_name == "zero":
            res = BITS_TO_ZERO_LITERAL[self.num_bits]
        elif operation_name == "and":
            res = f"{var1} & {var2}"
        elif operation_name == "not_implies":
            res = f"{var1} & ~{var2}"
        elif operation_name == "a":
            res = f"{var1}"
        elif operation_name == "not_implied_by":
            res = f"{var2} & ~{var1}"
        elif operation_name == "b":
            res = f"{var2}"
        elif operation_name == "xor":
            res = f"{var1} ^ {var2}"
        elif operation_name == "or":
            res = f"{var1} | {var2}"
        elif operation_name == "not_or":
            res = f"~({var1} | {var2})"
        elif operation_name == "not_xor":
            res = f"~({var1} ^ {var2})"
        elif operation_name == "not_b":
            res = f"~{var2}"
        elif operation_name == "implied_by":
            res = f"~{var2} | {var1}"
        elif operation_name == "not_a":
            res = f"~{var1}"
        elif operation_name == "implies":
            res = f"~{var1} | {var2}"
        elif operation_name == "not_and":
            res = f"~({var1} & {var2})"
        elif operation_name == "one":
            res = f"~{BITS_TO_ZERO_LITERAL[self.num_bits]}"
        else:
            assert False, 'Operator {} unknown.'.format(operation_name)

        if self.num_bits == 8:
            res = f"(char) ({res})"
        elif self.num_bits == 16:
            res = f"(short) ({res})"

        return res

    def get_layer_code(self, layer_a, layer_b, layer_op, layer_id, prefix_sums):
        code = []
        for var_id, (gate_a, gate_b, gate_op) in enumerate(zip(layer_a, layer_b, layer_op)):
            if self.device == 'cpu' and layer_id == len(prefix_sums) - 1:
                a = f"v{prefix_sums[layer_id - 1] + gate_a}"
                b = f"v{prefix_sums[layer_id - 1] + gate_b}"
                code.append(f"\tout[{var_id}] = {self.get_gate_code(a, b, gate_op)};")
            else:
                assert not (self.device == 'cpu' and layer_id >= len(prefix_sums) - 1), (layer_id, len(prefix_sums))
                if layer_id == 0:
                    a = f"inp[{gate_a}]"
                    b = f"inp[{gate_b}]"
                else:
                    a = f"v{prefix_sums[layer_id - 1] + gate_a}"
                    b = f"v{prefix_sums[layer_id - 1] + gate_b}"
                code.append(
                    f"\tconst {BITS_TO_DTYPE[self.num_bits]} v{prefix_sums[layer_id] + var_id} = {self.get_gate_code(a, b, gate_op)};"
                )
        return code

    def get_c_code(self):
        prefix_sums = [0]
        cur_count = 0
        for layer_a, layer_b, layer_op in self.layers[:-1]:
            cur_count += len(layer_a)
            prefix_sums.append(cur_count)

        code = [
            "#include <stddef.h>",
            "#include <stdlib.h>",
            "#include <stdbool.h>",
            "",
            f"void logic_gate_net({BITS_TO_DTYPE[self.num_bits]} const *inp, {BITS_TO_DTYPE[self.num_bits]} *out) {{",
        ]

        for layer_id, (layer_a, layer_b, layer_op) in enumerate(self.layers):
            code.extend(self.get_layer_code(layer_a, layer_b, layer_op, layer_id, prefix_sums))

        code.append("}")

        num_neurons_ll = self.layers[-1][0].shape[0]
        log2_of_num_neurons_per_class_ll = math.ceil(math.log2(num_neurons_ll / self.num_classes + 1))

        code.append(fr"""
void apply_logic_gate_net (bool const *inp, {BITS_TO_DTYPE[32]} *out, size_t len) {{
    {BITS_TO_DTYPE[self.num_bits]} *inp_temp = malloc({self.num_inputs}*sizeof({BITS_TO_DTYPE[self.num_bits]}));
    {BITS_TO_DTYPE[self.num_bits]} *out_temp = malloc({num_neurons_ll}*sizeof({BITS_TO_DTYPE[self.num_bits]}));
    {BITS_TO_DTYPE[self.num_bits]} *out_temp_o = malloc({log2_of_num_neurons_per_class_ll}*sizeof({BITS_TO_DTYPE[self.num_bits]}));
    
    for(size_t i = 0; i < len; ++i) {{
    
        // Converting the bool array into a bitpacked array
        for(size_t d = 0; d < {self.num_inputs}; ++d) {{
            {BITS_TO_DTYPE[self.num_bits]} res = {BITS_TO_ZERO_LITERAL[self.num_bits]};
            for(size_t b = 0; b < {self.num_bits}; ++b) {{
                res <<= 1;
                res += !!(inp[i * {self.num_inputs} * {self.num_bits} + ({self.num_bits} - b - 1) * {self.num_inputs} + d]);
            }}
            inp_temp[d] = res;
        }}
    
        // Applying the logic gate net
        logic_gate_net(inp_temp, out_temp);
        
        // GroupSum of the results via logic gate networks
        for(size_t c = 0; c < {self.num_classes}; ++c) {{  // for each class
            // Initialize the output bits
            for(size_t d = 0; d < {log2_of_num_neurons_per_class_ll}; ++d) {{
                out_temp_o[d] = {BITS_TO_ZERO_LITERAL[self.num_bits]};
            }}
            
            // Apply the adder logic gate network
            for(size_t a = 0; a < {self.layers[-1][0].shape[0] // self.num_classes}; ++a) {{
                {BITS_TO_DTYPE[self.num_bits]} carry = out_temp[c * {self.layers[-1][0].shape[0] // self.num_classes} + a];
                {BITS_TO_DTYPE[self.num_bits]} out_temp_o_d;
                for(int d = {log2_of_num_neurons_per_class_ll} - 1; d >= 0; --d) {{
                    out_temp_o_d  = out_temp_o[d];
                    out_temp_o[d] = carry ^ out_temp_o_d;
                    carry         = carry & out_temp_o_d;
                }}
            }}
            
            // Unpack the result bits
            for(size_t b = 0; b < {self.num_bits}; ++b) {{
                const {BITS_TO_DTYPE[self.num_bits]} bit_mask = {BITS_TO_ONE_LITERAL[self.num_bits]} << b;
                {BITS_TO_DTYPE[32]} res = 0;
                for(size_t d = 0; d < {log2_of_num_neurons_per_class_ll}; ++d) {{
                    res <<= 1;
                    res += !!(out_temp_o[d] & bit_mask);
                }}
                out[(i * {self.num_bits} + b) * {self.num_classes} + c] = res;
            }}
        }}
    }}
    free(inp_temp);
    free(out_temp);
    free(out_temp_o);
}}
""")

        return "\n".join(code)

    def compile(self, opt_level=1, save_lib_path=None, verbose=False):
        """
        Regarding the optimization level for C compiler:

        compilation time vs. call time for 48k lines of code
        -O0 -> 5.5s compiling -> 269ms call
        -O1 -> 190s compiling -> 125ms call
        -O2 -> 256s compiling -> 130ms call
        -O3 -> 346s compiling -> 124ms call

        :param opt_level: optimization level for C compiler
        :param save_lib_path: (optional) where to save the .so shared object library
        :param verbose:
        :return:
        """

        with tempfile.NamedTemporaryFile(suffix=".so") as lib_file:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".c" if self.device != "cuda" else ".cu"
            ) as c_file:
                if self.device == 'cpu':
                    code = self.get_c_code()
                else:
                    assert False, 'Device {} not supported.'.format(self.device)

                if verbose and len(code.split('\n')) <= 200:
                    print()
                    print()
                    print(code)
                    print()
                    print()

                c_file.write(code)
                c_file.flush()

                if verbose:
                    print('C code created and has {} lines. (temp location {})'.format(len(code.split('\n')), c_file.name))

                t_s = time.time()
                if self.device == 'cpu':
                    compiler_out = subprocess.run(
                        [
                            self.cpu_compiler,
                            "-shared",
                            "-fPIC",
                            "-O{}".format(opt_level),
                            "-march=native",
                            "-o",
                            lib_file.name,
                            c_file.name,
                        ]
                    )
                else:
                    assert False, 'Device {} not supported.'.format(self.device)

                if compiler_out.returncode != 0:
                    raise RuntimeError(
                        f'compilation exited with error code {compiler_out.returncode}')

                print('Compiling finished in {:.3f} seconds.'.format(time.time() - t_s))

            if save_lib_path is not None:
                shutil.copy(lib_file.name, save_lib_path)
                if verbose:
                    print('lib_file copied from {} to {} .'.format(lib_file.name, save_lib_path))

            lib = ctypes.cdll.LoadLibrary(lib_file.name)

            lib_fn = lib.apply_logic_gate_net
            lib_fn.restype = None
            lib_fn.argtypes = [
                np.ctypeslib.ndpointer(
                    ctypes.c_bool, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(
                    BITS_TO_C_DTYPE[32], flags="C_CONTIGUOUS"),
                ctypes.c_size_t,
            ]

        self.lib_fn = lib_fn

    @staticmethod
    def load(save_lib_path, num_classes, num_bits):

        self = CompiledLogicNet(None, num_bits=num_bits)
        self.num_classes = num_classes

        lib = ctypes.cdll.LoadLibrary(save_lib_path)

        lib_fn = lib.apply_logic_gate_net
        lib_fn.restype = None
        lib_fn.argtypes = [
            np.ctypeslib.ndpointer(
                ctypes.c_bool, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(
                BITS_TO_C_DTYPE[32], flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
        ]

        self.lib_fn = lib_fn
        return self

    def forward(
            self,
            x: Union[torch.BoolTensor, numpy.typing.NDArray[np.bool_]],
            verbose: bool = False
    ) -> torch.IntTensor:
        if isinstance(x, torch.Tensor):
            x = x.numpy()

        batch_size_div_bits = math.ceil(x.shape[0] / self.num_bits)
        pad_len = batch_size_div_bits * self.num_bits - x.shape[0]
        x = np.concatenate([x, np.zeros_like(x[:pad_len])])

        if verbose:
            print('x.shape', x.shape)

        out = np.zeros(
            x.shape[0] * self.num_classes, dtype=BITS_TO_NP_DTYPE[32]
        )
        x = x.reshape(-1)

        self.lib_fn(x, out, batch_size_div_bits)

        out = torch.tensor(out).view(batch_size_div_bits * self.num_bits, self.num_classes)
        if pad_len > 0:
            out = out[:-pad_len]
        if verbose:
            print('out.shape', out.shape)

        return out

