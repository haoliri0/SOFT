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
    args: str

    @property
    def size(self) -> str:
        return f"{self.spec.name}Ptr::get_size_bytes_n({self.args})"

    @property
    def align(self) -> str:
        return f"{self.spec.name}Ptr::get_align_bytes_n({self.args})"


class FieldSpec:
    name: str
    size: str
    align: str


@dataclass(kw_only=True)
class ItemFieldSpec(FieldSpec):
    name: str
    type: TypeSpec

    @property
    def size(self) -> str:
        return self.type.size

    @property
    def align(self) -> str:
        return self.type.align


@dataclass(kw_only=True)
class ListFieldSpec(FieldSpec):
    name: str
    item_name: str
    item_type: TypeSpec
    index_name: str
    index_type: str
    count: str | int

    @property
    def size(self) -> str:
        return f"{self.count} * get_{self.item_name}_size_bytes_n(args)"

    @property
    def align(self) -> str:
        return self.item_type.align


@dataclass(kw_only=True)
class DynamicStructSpec:
    name: str
    params: tuple[ParamSpec, ...]
    fields: tuple[FieldSpec, ...]
    extras: str = ""

    @property
    def size_expr(self) -> str:
        if len(self.fields) == 0:
            return "0"

        parts_size_expr = []
        for field in self.fields:
            parts_size_expr.append(f"get_{field.name}_bytes_n(args)")
            parts_size_expr.append(f"get_{field.name}_align_bytes_n(args)")
        return "+".join(parts_size_expr)


builtin_prefix = """template<typename Item>
static __device__ __host__
Item max() {
    return 0;
}

template<typename Item, typename... Args>
static __device__ __host__
Item max(Item item, Args... args) {
    return item + max<Item>(args...);
}

static __device__ __host__
size_t compute_pad_bytes_n(const size_t offset_bytes_n, const size_t align_bytes_n) {
    return align_bytes_n - offset_bytes_n % align_bytes_n;
}
"""

builtin_postfix = ""


def write_args_define_body(writer: Callable[[str], Any], spec: DynamicStructSpec):
    for arg in spec.params:
        writer(f"{arg.type} {arg.name};\n")


def write_args_define_expr(writer: Callable[[str], Any], spec: DynamicStructSpec):
    writer("struct Args {\n")
    write_args_define_body(make_indented_writer(writer), spec)
    writer("};\n")


def write_get_field_size_bytes_n_method_body(writer: Callable[[str], Any], field: FieldSpec):
    writer(f"return {field.size};\n")


def write_get_field_size_bytes_n_method(writer: Callable[[str], Any], field: FieldSpec):
    writer("\n")
    writer("static __device__ __host__\n")
    writer(f"size_t get_{field.name}_size_bytes_n(const Args args) {{\n")
    write_get_field_size_bytes_n_method_body(make_indented_writer(writer), field)
    writer("}\n")


def write_get_field_item_size_bytes_n_method_body(writer: Callable[[str], Any], field: ListFieldSpec):
    writer(f"return {field.item_type.size};\n")


def write_get_field_item_size_bytes_n_method(writer: Callable[[str], Any], field: ListFieldSpec):
    writer("\n")
    writer("static __device__ __host__\n")
    writer(f"size_t get_{field.item_name}_size_bytes_n(const Args args) {{\n")
    write_get_field_item_size_bytes_n_method_body(make_indented_writer(writer), field)
    writer("}\n")


def write_get_field_align_bytes_n_method_body(writer: Callable[[str], Any], field: FieldSpec):
    writer(f"return {field.align};\n")


def write_get_field_align_bytes_n_method(writer: Callable[[str], Any], field: FieldSpec):
    writer("\n")
    writer("static __device__ __host__\n")
    writer(f"size_t get_{field.name}_align_bytes_n(const Args args) {{\n")
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
        writer(f"const size_t offset_bytes_n = get_{field_prev.name}_offset_bytes_n(args);\n")
        writer(f"const size_t align_bytes_n = get_{field.name}_align_bytes_n(args);\n")
        writer(f"return compute_pad_bytes_n(offset_bytes_n, align_bytes_n);\n")


def write_get_field_pad_bytes_n_method(
    writer: Callable[[str], Any],
    field: FieldSpec,
    field_prev: FieldSpec | None,
):
    writer("\n")
    writer("static __device__ __host__\n")
    writer(f"size_t get_{field.name}_pad_bytes_n(const Args args) {{\n")
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
        indented_writer(f"get_{field_prev.name}_offset_bytes_n(args) +\n")
        indented_writer(f"get_{field_prev.name}_size_bytes_n(args) +\n")
        indented_writer(f"get_{field.name}_pad_bytes_n(args);\n")


def write_get_field_offset_bytes_n_method(
    writer: Callable[[str], Any],
    field: FieldSpec,
    field_prev: FieldSpec | None,
):
    writer("\n")
    writer("static __device__ __host__\n")
    writer(f"size_t get_{field.name}_offset_bytes_n(const Args args) {{\n")
    write_get_field_offset_bytes_n_method_body(make_indented_writer(writer), field, field_prev)
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
        indented_writer(f"get_{field.name}_pad_bytes_n(args) +\n")
        indented_writer(f"get_{field.name}_size_bytes_n(args){tail_str}\n")


def write_get_size_bytes_n_method(writer: Callable[[str], Any], spec: DynamicStructSpec):
    writer("\n")
    writer("static __device__ __host__\n")
    writer(f"size_t get_size_bytes_n(const Args args) {{\n")
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
        indented_writer(f"get_{field.name}_align_bytes_n(args){tail_str}\n")


def write_get_align_bytes_n_method(writer: Callable[[str], Any], spec: DynamicStructSpec):
    writer("\n")
    writer("static __device__ __host__\n")
    writer(f"size_t get_align_bytes_n(const Args args) {{\n")
    write_get_align_bytes_n_method_body(make_indented_writer(writer), spec)
    writer("}\n")


def write_get_field_ptr_method_body(writer: Callable[[str], Any], field: FieldSpec):
    if isinstance(field, ItemFieldSpec):
        writer(f"const size_t offset = get_{field.name}_offset_bytes_n(args);\n")
        value_type = field.type
    elif isinstance(field, ListFieldSpec):
        writer(f"const size_t {field.item_name}_size = get_{field.item_name}_size_bytes_n(args);\n")
        writer(f"const size_t offset0 = get_{field.name}_offset_bytes_n(args);\n")
        writer(f"const size_t offset = offset0 + {field.index_name} * {field.item_name}_size;\n")
        value_type = field.item_type
    else:
        raise ValueError(f"Unsupported field type: {field}")

    if isinstance(value_type, ValueTypeSpec):
        writer(f"return reinterpret_cast<{value_type.name} *>(ptr + offset);\n")
    elif isinstance(value_type, DynamicStructTypeSpec):
        writer(f"return {{{value_type.args}, ptr + offset}};\n")
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
            writer(f"{field.item_type.name} *get_{field.item_name}_ptr("
                   f"const {field.index_type} {field.index_name}) const {{\n")
        elif isinstance(field.item_type, DynamicStructTypeSpec):
            writer(f"{field.item_type.spec.name}Ptr get_{field.item_name}_ptr("
                   f"const {field.index_type} {field.index_name}) const {{\n")
        else:
            raise TypeError(f"Unsupported list field item type: {field.item_type}")
    else:
        raise TypeError(f"Unsupported field type: {field}")
    write_get_field_ptr_method_body(make_indented_writer(writer), field)
    writer("}\n")


def write_dynamic_struct_define_body(writer: Callable[[str], Any], spec: DynamicStructSpec):
    write_args_define_expr(writer, spec)

    field_prev = None
    for field in spec.fields:
        if isinstance(field, ListFieldSpec):
            write_get_field_item_size_bytes_n_method(writer, field)
        write_get_field_size_bytes_n_method(writer, field)
        write_get_field_align_bytes_n_method(writer, field)
        write_get_field_pad_bytes_n_method(writer, field, field_prev)
        write_get_field_offset_bytes_n_method(writer, field, field_prev)
        field_prev = field

    write_get_size_bytes_n_method(writer, spec)
    write_get_align_bytes_n_method(writer, spec)

    writer("\n")
    writer("Args args;\n")
    writer("char *ptr;\n")

    for field in spec.fields:
        write_get_field_ptr_method(writer, field)

    writer(spec.extras)


def write_dynamic_struct_define(writer: Callable[[str], Any], spec: DynamicStructSpec):
    writer("\n")
    writer(f"struct {spec.name}Ptr {{\n")
    write_dynamic_struct_define_body(make_indented_writer(writer), spec)
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
            write_dynamic_struct_define(writer, spec)
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
                count="2 * args.qubits_n"),),
        extras="""
__device__ __host__
Bit *get_x_ptr(const Qid qubit_i) const {
    return get_bit_ptr(qubit_i);
}

__device__ __host__
Bit *get_z_ptr(const Qid qubit_i) const {
    return get_bit_ptr(args.qubits_n + qubit_i);
}\n""")
    table_row_spec = DynamicStructSpec(
        name="TableRow",
        params=(ParamSpec(name="qubits_n", type="Qid"),),
        fields=(
            ItemFieldSpec(
                name="pauli",
                type=DynamicStructTypeSpec(
                    spec=pauli_row_spec,
                    args="{args.qubits_n}")),
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
                    args="{args.qubits_n}"),
                index_name="row_i",
                index_type="Qid",
                count="2 * args.qubits_n"),
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
                count="2 * args.qubits_n"),
            ItemFieldSpec(
                name="pauli",
                type=DynamicStructTypeSpec(
                    spec=pauli_row_spec,
                    args="{args.qubits_n}")),
            ItemFieldSpec(
                name="phase",
                type=ValueTypeSpec(name="Phs")),
            ItemFieldSpec(
                name="pivot",
                type=ValueTypeSpec(name="Qid")),
        ))
    amps_spec = DynamicStructSpec(
        name="AmpsMap",
        params=(
            ParamSpec(name="qubits_n", type="Qid"),
            ParamSpec(name="amps_m", type="Kid")),
        fields=(
            ItemFieldSpec(
                name="amps_n",
                type=ValueTypeSpec(name="Kid")),
            ListFieldSpec(
                name="amps",
                item_name="amp",
                item_type=ValueTypeSpec(name="Amp"),
                index_name="amp_i",
                index_type="Kid",
                count="args.amps_m"),
            ListFieldSpec(
                name="aids",
                item_name="aid",
                item_type=ValueTypeSpec(name="Aid"),
                index_name="amp_i",
                index_type="Kid",
                count="args.amps_m"),
        ))
    shot_state_spec = DynamicStructSpec(
        name="ShotState",
        params=(
            ParamSpec(name="qubits_n", type="Qid"),
            ParamSpec(name="amps_m", type="Kid")),
        fields=(
            ItemFieldSpec(
                name="table",
                type=DynamicStructTypeSpec(
                    spec=table_spec,
                    args="{args.qubits_n}")),
            ItemFieldSpec(
                name="decomp",
                type=DynamicStructTypeSpec(
                    spec=decomp_spec,
                    args="{args.qubits_n}")),
            ItemFieldSpec(
                name="amps",
                type=DynamicStructTypeSpec(
                    spec=amps_spec,
                    args="{args.qubits_n, args.amps_m}")),
        ))
    shots_state_spec = DynamicStructSpec(
        name="ShotsState",
        params=(
            ParamSpec(name="shots_n", type="Sid"),
            ParamSpec(name="qubits_n", type="Qid"),
            ParamSpec(name="amps_m", type="Kid")),
        fields=(
            ListFieldSpec(
                name="shots",
                item_name="shot",
                item_type=DynamicStructTypeSpec(
                    spec=shot_state_spec,
                    args="{args.qubits_n, args.amps_m}"),
                index_name="shot_i",
                index_type="Sid",
                count="args.shots_n"),
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
