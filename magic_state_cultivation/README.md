# Magic State Cultivation

This folder contains resources for the Magic State Cultivation experiments.

* [📂circuits](./circuits) - Contains quantum circuits for magic state cultivation.
* [📂erroneous](./erroneous) - Contains full-process records of erroneous cases in magic state cultivation.

## Running Circuits

To run the circuits, you can run the command-line interface with proper arguments,
send a `*.compiled` file through stdin and receive outputs from stdout.

Here is an example:

```bash
cat ./magic_state_cultivation/circuits/circuit_d5_p0.001.stim.compiled | \
./cmake-build-release/soft-cuda-cli/soft-cuda-cli \
  --qubits_n 42 \
  --entries_m 2048 \
  --mem_ints_n 112 \
  --shots_n 4 \
  --seed 42
```

**Note:** the command-line arguments `qubits_n`, `entries_m`, `mem_ints_n`, `mem_flts_n`
should be configured according to the circuit,
ensuring sufficient memory allocation, or else an overflow error will occur.
Here we provide a set of feasible arguments for different circuits:

* for d=3 circuits, use `--qubits_n 15 --entries_m 32   --mem_ints_n 21`
* for d=5 circuits, use `--qubits_n 42 --entries_m 2048 --mem_ints_n 112`

**Note:** the command-line arguments `seed` is default to be constant `0`.
Do not omit it if you want to sample random results in different runs.

**Note:** the command-line argument `shots_n` should be adjusted according to your GPU device.
The larger `shots_n` is usually more efficient, but requires more memory.

## Processing Outputs

The output is produced through stdout in a YAML format,
containing the result of every shot from every print operations,
as the following example:

``` yaml
print_0:
  shot_0: 0
  shot_1: 96
  shot_2: 20
  shot_3: 20
print_1:
  shot_0: 0
  shot_1: 0
  shot_2: 0
  shot_3: 0
print_2:
  shot_0: 0
  shot_1: 0
  shot_2: 0
  shot_3: 0
```

In our circuits, the first print is the error code,
indicating whether the shot succeeded or not.

- the int `0` means success;
- positive int `i` means that it was discarded by the `i`-th detector;
- negative int stands for an internal error like overflowing.

The following prints correspond to result of observables (some measurements that matters) in the circuit.
For magic state cultivation, the last print indicates whether the cultivation succeeded:

- `0` stands for a successful result;
- `1` stands for an erroneous result.

## Inspecting Circuits

Any circuit must be compiled to a specific format to run with our simulator.
But the compiled circuit is usually not human-readable.
So we also provide the original circuit (the `*.stim` files).

Here are the naming rules for the circuit files:

* `d3`, `d5` stand for distance 3 and distance 5 circuits.
* `p0.001`, `p0.01` stand for physical error rates.
* `.stim` stands for the original circuit file.
* `.stim.compiled` stands for the compiled circuit file.

## Inspecting Erroneous Records

The magic state cultivation usually succeed with a high probability,
but it still fails in some rare cases.
To figure what happened in these erroneous cases,
we can record all events in the whole process.

We provided such full-process records
for serval captured erroneous cases in `*.reveal` files under folder `erroneous`,
where a bunch of extra information is recorded with prefix `#`.

Here shows an example snippet from a `*.reveal`, 
indicating an occurrence of a depolarizing error.

```text
...

# line 378: DEPOLARIZE1(0.001) 41 35 33 31 22 20 18 8 6 37 28 26 24 16 14 12 4 2 0 1 3 5 7 9 10 11 13 15 17 19 21 23 25 27 29 30 32 34 36 38 39 40
# record 2907: result=0, prob=0.999
# nothing
# record 2908: result=1, prob=0.000333333
X 35
# record 2909: result=0, prob=0.999
# nothing
# record 2910: result=0, prob=0.999

...
```