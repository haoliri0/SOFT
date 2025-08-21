import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable


def make_indented_writer(writer: Callable[[str], Any], indention: str = "    ") -> Callable[[str], Any]:
    first_indent = True

    def indented_writer(text: str):
        if not text:
            return

        nonlocal first_indent
        if first_indent:
            writer(indention)
            first_indent = False

        if text.endswith("\n"):
            text = text[:-1]
            text = text.replace("\n", "\n" + indention)
            text = text + "\n"
            first_indent = True
        else:
            text = text.replace("\n", "\n" + indention)

        writer(text)

    return indented_writer


@dataclass(kw_only=True)
class ParamSpec:
    name: str
    type: str


class TypeSpec:
    size: str | None
    align: str | None


@dataclass(kw_only=True)
class ValueTypeSpec(TypeSpec):
    name: str

    @property
    def size(self) -> str:
        return f"sizeof({self.name})"

    @property
    def align(self) -> str:
        return f"alignof({self.name})"


@dataclass(kw_only=True)
class DynamicStructTypeSpec(TypeSpec):
    spec: 'DynamicStructSpec'
    args: tuple[str, ...]

    @property
    def size(self) -> str:
        return f"{self.spec.name}Args{{{', '.join(self.args)}}}.get_size_bytes_n()"

    @property
    def align(self) -> str:
        return f"{self.spec.name}Args{{{', '.join(self.args)}}}.get_align_bytes_n()"


class FieldSpec:
    name: str


@dataclass(kw_only=True)
class ItemFieldSpec(FieldSpec):
    name: str
    type: TypeSpec


@dataclass(kw_only=True)
class ListFieldSpec(FieldSpec):
    name: str
    item_name: str
    item_type: TypeSpec
    index_name: str
    index_type: str
    count: str | int


@dataclass(kw_only=True)
class DynamicStructSpec:
    name: str
    params: tuple[ParamSpec, ...]
    fields: tuple[FieldSpec, ...]
    extra_args_body: str = ""
    extra_ptr_body: str = ""

    @property
    def size_expr(self) -> str:
        if len(self.fields) == 0:
            return "0"

        parts_size_expr = []
        for field in self.fields:
            parts_size_expr.append(f"get_{field.name}_bytes_n()")
            parts_size_expr.append(f"get_{field.name}_align_bytes_n()")
        return "+".join(parts_size_expr)


builtin_prefix = """
template<typename Item, typename... Args>
static __device__ __host__
Item max(Item item0, Args... args) {
    if constexpr (sizeof...(Args) == 0) {
        return item0;
    } else {
        Item item1 = max(args...);
        return item0 > item1 ? item0 : item1;
    }
}

static __device__ __host__
size_t compute_pad_bytes_n(const size_t offset_bytes_n, const size_t align_bytes_n) {
    return align_bytes_n - offset_bytes_n % align_bytes_n;
}
"""[1:]

builtin_postfix = ""


def write_get_field_item_size_bytes_n_method_body(writer: Callable[[str], Any], field: ListFieldSpec):
    writer(f"return {field.item_type.size};\n")


def write_get_field_item_size_bytes_n_method(writer: Callable[[str], Any], field: ListFieldSpec):
    writer("\n")
    writer("__device__ __host__\n")
    writer(f"size_t get_{field.item_name}_size_bytes_n() const {{\n")
    write_get_field_item_size_bytes_n_method_body(make_indented_writer(writer), field)
    writer("}\n")


def write_get_field_item_align_bytes_n_method_body(writer: Callable[[str], Any], field: ListFieldSpec):
    writer(f"return {field.item_type.align};\n")


def write_get_field_item_align_bytes_n_method(writer: Callable[[str], Any], field: ListFieldSpec):
    writer("\n")
    writer("__device__ __host__\n")
    writer(f"size_t get_{field.item_name}_align_bytes_n() const {{\n")
    write_get_field_align_bytes_n_method_body(make_indented_writer(writer), field)
    writer("}\n")


def write_get_field_item_pad_bytes_n_method_body(writer: Callable[[str], Any], field: ListFieldSpec):
    writer(f"return compute_pad_bytes_n(\n")
    indented_writer = make_indented_writer(writer)
    indented_writer(f"get_{field.item_name}_size_bytes_n(),\n")
    indented_writer(f"get_{field.item_name}_align_bytes_n());\n")


def write_get_field_item_pad_bytes_n_method(writer: Callable[[str], Any], field: ListFieldSpec):
    writer("\n")
    writer("__device__ __host__\n")
    writer(f"size_t get_{field.item_name}_pad_bytes_n() const {{\n")
    write_get_field_item_pad_bytes_n_method_body(make_indented_writer(writer), field)
    writer("}\n")


def write_get_field_size_bytes_n_method_body(writer: Callable[[str], Any], field: FieldSpec):
    if isinstance(field, ItemFieldSpec):
        writer(f"return {field.type.size};\n")
    elif isinstance(field, ListFieldSpec):
        writer(f"return \n")
        indented_writer = make_indented_writer(writer)
        indented_writer(f"{field.count} * get_{field.item_name}_size_bytes_n() +\n")
        indented_writer(f"{field.count} * get_{field.item_name}_pad_bytes_n();\n")
    else:
        raise TypeError(f"Unsupported field type: {field}")


def write_get_field_size_bytes_n_method(writer: Callable[[str], Any], field: FieldSpec):
    writer("\n")
    writer("__device__ __host__\n")
    writer(f"size_t get_{field.name}_size_bytes_n() const {{\n")
    write_get_field_size_bytes_n_method_body(make_indented_writer(writer), field)
    writer("}\n")


def write_get_field_align_bytes_n_method_body(writer: Callable[[str], Any], field: FieldSpec):
    if isinstance(field, ItemFieldSpec):
        writer(f"return {field.type.align};\n")
    elif isinstance(field, ListFieldSpec):
        writer(f"return {field.item_type.align};\n")
    else:
        raise TypeError(f"Unsupported field type: {field}")


def write_get_field_align_bytes_n_method(writer: Callable[[str], Any], field: FieldSpec):
    writer("\n")
    writer("__device__ __host__\n")
    writer(f"size_t get_{field.name}_align_bytes_n() const {{\n")
    write_get_field_align_bytes_n_method_body(make_indented_writer(writer), field)
    writer("}\n")


def write_get_field_pad_bytes_n_method_body(
    writer: Callable[[str], Any],
    field: FieldSpec,
    field_prev: FieldSpec | None,
):
    if field_prev is None:
        writer("return 0;\n")
    else:
        writer(f"return compute_pad_bytes_n(\n")
        indented_writer = make_indented_writer(writer)
        indented_writer(f"get_{field_prev.name}_offset_bytes_n() +\n")
        indented_writer(f"get_{field_prev.name}_size_bytes_n(),\n")
        indented_writer(f"get_{field.name}_align_bytes_n());\n")


def write_get_field_pad_bytes_n_method(
    writer: Callable[[str], Any],
    field: FieldSpec,
    field_prev: FieldSpec | None,
):
    writer("\n")
    writer("__device__ __host__\n")
    writer(f"size_t get_{field.name}_pad_bytes_n() const {{\n")
    write_get_field_pad_bytes_n_method_body(make_indented_writer(writer), field, field_prev)
    writer("}\n")


def write_get_field_offset_bytes_n_method_body(
    writer: Callable[[str], Any],
    field: FieldSpec,
    field_prev: FieldSpec | None,
):
    if field_prev is None:
        writer("return 0;\n")
    else:
        writer(f"return \n")
        indented_writer = make_indented_writer(writer)
        indented_writer(f"get_{field_prev.name}_offset_bytes_n() +\n")
        indented_writer(f"get_{field_prev.name}_size_bytes_n() +\n")
        indented_writer(f"get_{field.name}_pad_bytes_n();\n")


def write_get_field_offset_bytes_n_method(
    writer: Callable[[str], Any],
    field: FieldSpec,
    field_prev: FieldSpec | None,
):
    writer("\n")
    writer("__device__ __host__\n")
    writer(f"size_t get_{field.name}_offset_bytes_n() const {{\n")
    write_get_field_offset_bytes_n_method_body(make_indented_writer(writer), field, field_prev)
    writer("}\n")


def write_get_field_item_offset_bytes_n_method_body(writer: Callable[[str], Any], field: ListFieldSpec):
    writer(f"return get_{field.name}_offset_bytes_n() +\n")
    indented_writer = make_indented_writer(writer)
    indented_writer(f"{field.index_name} * get_{field.item_name}_size_bytes_n() +\n")
    indented_writer(f"{field.index_name} * get_{field.item_name}_pad_bytes_n();\n")


def write_get_field_item_offset_bytes_n_method(writer: Callable[[str], Any], field: ListFieldSpec):
    writer("\n")
    writer("__device__ __host__\n")
    writer(f"size_t get_{field.item_name}_offset_bytes_n({field.index_type} {field.index_name}) const {{\n")
    write_get_field_item_offset_bytes_n_method_body(make_indented_writer(writer), field)
    writer("}\n")


def write_get_size_bytes_n_method_body(writer: Callable[[str], Any], spec: DynamicStructSpec):
    if len(spec.fields) == 0:
        writer("return 0;\n")
        return
    writer("return \n")
    indented_writer = make_indented_writer(writer)
    for field_i, field in enumerate(spec.fields):
        is_last = field_i == len(spec.fields) - 1
        tail_str = ";" if is_last else " +"
        indented_writer(f"get_{field.name}_pad_bytes_n() +\n")
        indented_writer(f"get_{field.name}_size_bytes_n(){tail_str}\n")


def write_get_size_bytes_n_method(writer: Callable[[str], Any], spec: DynamicStructSpec):
    writer("\n")
    writer("__device__ __host__\n")
    writer(f"size_t get_size_bytes_n() const {{\n")
    write_get_size_bytes_n_method_body(make_indented_writer(writer), spec)
    writer("}\n")


def write_get_align_bytes_n_method_body(writer: Callable[[str], Any], spec: DynamicStructSpec):
    if len(spec.fields) == 0:
        writer("return 1;\n")
        return
    writer("return max(\n")
    indented_writer = make_indented_writer(writer)
    for field_i, field in enumerate(spec.fields):
        is_last = field_i == len(spec.fields) - 1
        tail_str = ");" if is_last else ","
        indented_writer(f"get_{field.name}_align_bytes_n(){tail_str}\n")


def write_get_align_bytes_n_method(writer: Callable[[str], Any], spec: DynamicStructSpec):
    writer("\n")
    writer("__device__ __host__\n")
    writer(f"size_t get_align_bytes_n() const {{\n")
    write_get_align_bytes_n_method_body(make_indented_writer(writer), spec)
    writer("}\n")


def write_dynamic_struct_args_define_body(writer: Callable[[str], Any], spec: DynamicStructSpec):
    for arg in spec.params:
        writer(f"{arg.type} {arg.name};\n")

    field_prev = None
    for field in spec.fields:
        if isinstance(field, ListFieldSpec):
            write_get_field_item_size_bytes_n_method(writer, field)
            write_get_field_item_align_bytes_n_method(writer, field)
            write_get_field_item_pad_bytes_n_method(writer, field)
        write_get_field_size_bytes_n_method(writer, field)
        write_get_field_align_bytes_n_method(writer, field)
        write_get_field_pad_bytes_n_method(writer, field, field_prev)
        write_get_field_offset_bytes_n_method(writer, field, field_prev)
        if isinstance(field, ListFieldSpec):
            write_get_field_item_offset_bytes_n_method(writer, field)
        field_prev = field

    write_get_size_bytes_n_method(writer, spec)
    write_get_align_bytes_n_method(writer, spec)

    writer(spec.extra_args_body)


def write_dynamic_struct_args_define(writer: Callable[[str], Any], spec: DynamicStructSpec):
    writer("\n")
    writer(f"struct {spec.name}Args {{\n")
    write_dynamic_struct_args_define_body(make_indented_writer(writer), spec)
    writer("};\n")


def write_get_field_ptr_method_body(writer: Callable[[str], Any], field: FieldSpec):
    writer(f"const size_t offset = get_{field.name}_offset_bytes_n();\n")

    if isinstance(field, ItemFieldSpec):
        value_type = field.type
    elif isinstance(field, ListFieldSpec):
        value_type = field.item_type
    else:
        raise ValueError(f"Unsupported field type: {field}")

    if isinstance(value_type, ValueTypeSpec):
        writer(f"return reinterpret_cast<{value_type.name} *>(ptr + offset);\n")
    elif isinstance(value_type, DynamicStructTypeSpec):
        writer(f"return {{{', '.join(value_type.args)}, ptr + offset}};\n")
    else:
        raise TypeError(f"Unsupported field type: {value_type}")


def write_get_field_ptr_method(writer: Callable[[str], Any], field: FieldSpec):
    writer("\n")
    writer("__device__ __host__\n")
    if isinstance(field, ItemFieldSpec):
        if isinstance(field.type, ValueTypeSpec):
            writer(f"{field.type.name} *get_{field.name}_ptr() const {{\n")
        elif isinstance(field.type, DynamicStructTypeSpec):
            writer(f"{field.type.spec.name}Ptr get_{field.name}_ptr() const {{\n")
        else:
            raise TypeError(f"Unsupported field type: {field.type}")
    elif isinstance(field, ListFieldSpec):
        if isinstance(field.item_type, ValueTypeSpec):
            writer(f"{field.item_type.name} *get_{field.name}_ptr() const {{\n")
        else:
            raise TypeError(f"Unsupported list field item type: {field.item_type}")
    else:
        raise TypeError(f"Unsupported field type: {field}")
    write_get_field_ptr_method_body(make_indented_writer(writer), field)
    writer("}\n")


def write_get_field_item_ptr_method_body(writer: Callable[[str], Any], field: ListFieldSpec):
    value_type = field.item_type

    if isinstance(value_type, ValueTypeSpec):
        writer(f"return get_{field.name}_ptr() + {field.index_name};\n")
    elif isinstance(value_type, DynamicStructTypeSpec):
        writer(f"const size_t offset = get_{field.item_name}_offset_bytes_n({field.index_name});\n")
        writer(f"return {{{', '.join(value_type.args)}, ptr + offset}};\n")
    else:
        raise TypeError(f"Unsupported field type: {value_type}")


def write_get_field_item_ptr_method(writer: Callable[[str], Any], field: ListFieldSpec):
    writer("\n")
    writer("__device__ __host__\n")
    if isinstance(field.item_type, ValueTypeSpec):
        writer(f"{field.item_type.name} *get_{field.item_name}_ptr("
               f"const {field.index_type} {field.index_name}) const {{\n")
    elif isinstance(field.item_type, DynamicStructTypeSpec):
        writer(f"{field.item_type.spec.name}Ptr get_{field.item_name}_ptr("
               f"const {field.index_type} {field.index_name}) const {{\n")
    else:
        raise TypeError(f"Unsupported list field item type: {field.item_type}")
    write_get_field_item_ptr_method_body(make_indented_writer(writer), field)
    writer("}\n")


def write_dynamic_struct_ptr_define_body(writer: Callable[[str], Any], spec: DynamicStructSpec):
    writer("char *ptr;\n")

    for field in spec.fields:
        if isinstance(field, ItemFieldSpec):
            write_get_field_ptr_method(writer, field)
        elif isinstance(field, ListFieldSpec):
            if isinstance(field.item_type, ValueTypeSpec):
                write_get_field_ptr_method(writer, field)
                write_get_field_item_ptr_method(writer, field)
            elif isinstance(field.item_type, DynamicStructTypeSpec):
                write_get_field_item_ptr_method(writer, field)
            else:
                raise TypeError(f"Unsupported list field item type: {field.item_type}")
        else:
            raise TypeError(f"Unsupported field type: {field}")

    writer(spec.extra_ptr_body)


def write_dynamic_struct_ptr_define(writer: Callable[[str], Any], spec: DynamicStructSpec):
    writer("\n")
    writer(f"struct {spec.name}Ptr : {spec.name}Args {{\n")
    write_dynamic_struct_ptr_define_body(make_indented_writer(writer), spec)
    writer("};\n")


def iter_dynamic_struct_spec(spec: DynamicStructSpec | FieldSpec | TypeSpec):
    if isinstance(spec, DynamicStructSpec):
        for field in spec.fields:
            yield from iter_dynamic_struct_spec(field)
        yield spec
    if isinstance(spec, ItemFieldSpec):
        yield from iter_dynamic_struct_spec(spec.type)
    if isinstance(spec, ListFieldSpec):
        yield from iter_dynamic_struct_spec(spec.item_type)
    if isinstance(spec, ValueTypeSpec):
        pass
    if isinstance(spec, DynamicStructTypeSpec):
        yield from iter_dynamic_struct_spec(spec.spec)


def write_entire_code(writer: Callable[[str], Any], spec: DynamicStructSpec, prefix: str, postfix: str):
    if prefix:
        writer(prefix)
        writer("\n")
    if builtin_prefix:
        writer(builtin_prefix)
        writer("\n")

    defined_specs = []
    for spec in iter_dynamic_struct_spec(spec):
        if spec not in defined_specs:
            write_dynamic_struct_args_define(writer, spec)
            write_dynamic_struct_ptr_define(writer, spec)
            defined_specs.append(spec)

    if builtin_postfix:
        writer("\n")
        writer(builtin_postfix)
    if postfix:
        writer("\n")
        writer(postfix)


def main(code_file_path: str | None = None):
    prefix = (
        "#ifndef STN_CUDA_DATASTRUCT_CUH\n"
        "#define STN_CUDA_DATASTRUCT_CUH\n"
        "\n"
        '#include "./datatype.cuh"\n'
        "\n"
        "namespace StnCuda {\n")

    postfix = (
        "}\n"
        "\n"
        "#endif\n")

    pauli_row_spec = DynamicStructSpec(
        name="PauliRow",
        params=(ParamSpec(name="qubits_n", type="Qid"),),
        fields=(
            ListFieldSpec(
                name="bits",
                item_name="bit",
                item_type=ValueTypeSpec(name="Bit"),
                index_name="bit_i",
                index_type="Qid",
                count="2 * qubits_n"),),
        extra_ptr_body="""
__device__ __host__
Bit *get_x_ptr(const Qid qubit_i) const {
    return get_bit_ptr(qubit_i);
}

__device__ __host__
Bit *get_z_ptr(const Qid qubit_i) const {
    return get_bit_ptr(qubits_n + qubit_i);
}\n""")
    table_row_spec = DynamicStructSpec(
        name="TableRow",
        params=(ParamSpec(name="qubits_n", type="Qid"),),
        fields=(
            ItemFieldSpec(
                name="pauli",
                type=DynamicStructTypeSpec(
                    spec=pauli_row_spec,
                    args=("qubits_n",))),
            ItemFieldSpec(
                name="sign",
                type=ValueTypeSpec(name="Bit")),
        ))
    table_spec = DynamicStructSpec(
        name="Table",
        params=(ParamSpec(name="qubits_n", type="Qid"),),
        fields=(
            ListFieldSpec(
                name="rows",
                item_name="row",
                item_type=DynamicStructTypeSpec(
                    spec=table_row_spec,
                    args=("qubits_n",)),
                index_name="row_i",
                index_type="Qid",
                count="2 * qubits_n"),
        ))
    decomp_spec = DynamicStructSpec(
        name="Decomp",
        params=(ParamSpec(name="qubits_n", type="Qid"),),
        fields=(
            ListFieldSpec(
                name="bits",
                item_name="bit",
                item_type=ValueTypeSpec(name="Bit"),
                index_name="bit_i",
                index_type="Qid",
                count="2 * qubits_n"),
            ItemFieldSpec(
                name="pauli",
                type=DynamicStructTypeSpec(
                    spec=pauli_row_spec,
                    args=("qubits_n",))),
            ItemFieldSpec(
                name="phase",
                type=ValueTypeSpec(name="Phs")),
            ItemFieldSpec(
                name="pivot",
                type=ValueTypeSpec(name="Qid"))),
        extra_ptr_body="""
__device__ __host__
Bit *get_destab_bits_ptr() const {
    return get_bits_ptr();
}

__device__ __host__
Bit *get_destab_bit_ptr(const Qid qubit_i) const {
    return get_destab_bits_ptr() + qubit_i;
}

__device__ __host__
Bit *get_stab_bits_ptr() const {
    return get_bits_ptr() + qubits_n;
}

__device__ __host__
Bit *get_stab_bit_ptr(const Qid qubit_i) const {
    return get_stab_bits_ptr() + qubit_i;
}\n""")
    entries_spec = DynamicStructSpec(
        name="Entries",
        params=(
            ParamSpec(name="qubits_n", type="Qid"),
            ParamSpec(name="entries_m", type="Eid")),
        fields=(
            ItemFieldSpec(
                name="entries_n",
                type=ValueTypeSpec(name="Eid")),
            ListFieldSpec(
                name="bsts",
                item_name="bst",
                item_type=ValueTypeSpec(name="Bst"),
                index_name="entry_i",
                index_type="Eid",
                count="entries_m"),
            ListFieldSpec(
                name="amps",
                item_name="amp",
                item_type=ValueTypeSpec(name="Amp"),
                index_name="entry_i",
                index_type="Eid",
                count="entries_m"),
            ItemFieldSpec(
                name="half0_entries_n",
                type=ValueTypeSpec(name="Eid")),
            ItemFieldSpec(
                name="half1_entries_n",
                type=ValueTypeSpec(name="Eid")),
            ItemFieldSpec(
                name="half0_prob",
                type=ValueTypeSpec(name="Flt")),
            ItemFieldSpec(
                name="half1_prob",
                type=ValueTypeSpec(name="Flt"))),
        extra_ptr_body="""
__device__ __host__
Bst *get_half0_bsts_ptr() const {
    return get_bsts_ptr();
}

__device__ __host__
Bst *get_half0_bst_ptr(const Eid entry_i) const {
    return get_half0_bsts_ptr() + entry_i;
}

__device__ __host__
Amp *get_half0_amps_ptr() const {
    return get_amps_ptr();
}

__device__ __host__
Amp *get_half0_amp_ptr(const Eid entry_i) const {
    return get_half0_amps_ptr() + entry_i;
}

__device__ __host__
Bst *get_half1_bsts_ptr() const {
    return get_bsts_ptr() + entries_m / 2;
}

__device__ __host__
Bst *get_half1_bst_ptr(const Eid entry_i) const {
    return get_half1_bsts_ptr() + entry_i;
}

__device__ __host__
Amp *get_half1_amps_ptr() const {
    return get_amps_ptr() + entries_m / 2;
}

__device__ __host__
Amp *get_half1_amp_ptr(const Eid entry_i) const {
    return get_half1_amps_ptr() + entry_i;
}\n""")
    results_spec = DynamicStructSpec(
        name="Results",
        params=(
            ParamSpec(name="results_m", type="Rid"),),
        fields=(
            ItemFieldSpec(
                name="error",
                type=ValueTypeSpec(name="Err")),
            ItemFieldSpec(
                name="rand_state",
                type=ValueTypeSpec(name="curandState")),
            ItemFieldSpec(
                name="results_n",
                type=ValueTypeSpec(name="Rid")),
            ListFieldSpec(
                name="probs",
                item_name="prob",
                item_type=ValueTypeSpec(name="Flt"),
                index_name="result_i",
                index_type="Rid",
                count="results_m"),
            ListFieldSpec(
                name="values",
                item_name="value",
                item_type=ValueTypeSpec(name="Rvl"),
                index_name="result_i",
                index_type="Rid",
                count="results_m"),
        ))
    shot_state_spec = DynamicStructSpec(
        name="ShotState",
        params=(
            ParamSpec(name="qubits_n", type="Qid"),
            ParamSpec(name="entries_m", type="Eid"),
            ParamSpec(name="results_m", type="Rid")),
        fields=(
            ItemFieldSpec(
                name="table",
                type=DynamicStructTypeSpec(
                    spec=table_spec,
                    args=("qubits_n",))),
            ItemFieldSpec(
                name="decomp",
                type=DynamicStructTypeSpec(
                    spec=decomp_spec,
                    args=("qubits_n",))),
            ItemFieldSpec(
                name="entries",
                type=DynamicStructTypeSpec(
                    spec=entries_spec,
                    args=("qubits_n", "entries_m"))),
            ItemFieldSpec(
                name="results",
                type=DynamicStructTypeSpec(
                    spec=results_spec,
                    args=("results_m",))),
        ))
    shots_state_spec = DynamicStructSpec(
        name="ShotsState",
        params=(
            ParamSpec(name="shots_n", type="Sid"),
            ParamSpec(name="qubits_n", type="Qid"),
            ParamSpec(name="entries_m", type="Eid"),
            ParamSpec(name="results_m", type="Rid")),
        fields=(
            ListFieldSpec(
                name="shots",
                item_name="shot",
                item_type=DynamicStructTypeSpec(
                    spec=shot_state_spec,
                    args=("qubits_n", "entries_m", "results_m")),
                index_name="shot_i",
                index_type="Sid",
                count="shots_n"),
        ))

    if code_file_path is None:
        write_entire_code(partial(print, end=""), shots_state_spec, prefix, postfix)
    else:
        with open(code_file_path, "wt") as fp:
            write_entire_code(fp.write, shots_state_spec, prefix, postfix)


if __name__ == '__main__':
    project_dir_path = os.path.join(os.path.dirname(__file__), "../..")
    code_file_path = os.path.join(project_dir_path, "source/datastruct.cuh")
    main(code_file_path)
    # main()
