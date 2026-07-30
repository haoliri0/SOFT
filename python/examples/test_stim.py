import argparse

import symft


CUDA_MODES = (
    "gpu",
    "cpu_presampled",
    "gpu_presample_expressions",
    "gpu_on_demand_expressions",
    "gpu_lazy",
)


def main():
    parser = argparse.ArgumentParser(description="Sample logical-error counts from a Stim file.")
    parser.add_argument("circuit", help="path to a .stim circuit")
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--stream-id", type=int, default=None)
    parser.add_argument("--observable", type=int, default=0)
    parser.add_argument("--postselect-detectors", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch", dest="batch", action="store_true", default=True)
    parser.add_argument("--single", dest="batch", action="store_false")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--sample-chunk-shots", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--cuda-mode", choices=CUDA_MODES, default="gpu")
    parser.add_argument("--shots-per-launch", type=int, default=0)
    parser.add_argument("--threads-per-block", type=int, default=0)
    args = parser.parse_args()

    if args.cuda and not symft.cuda_enabled():
        parser.error("--cuda requested, but this symft extension was not built with CUDA support")

    try:
        circuit = symft.read_stim_file(args.circuit)
        sampler = circuit.compile_counts_sampler(
            batch=args.batch,
            observable=args.observable,
            postselect_detectors=args.postselect_detectors,
            threads=args.threads,
            batch_size=args.batch_size,
            sample_chunk_shots=args.sample_chunk_shots,
            cuda=args.cuda,
            cuda_mode=args.cuda_mode,
            shots_per_launch=args.shots_per_launch,
            threads_per_block=args.threads_per_block,
        )
        result = sampler.sample(args.shots, stream_id=args.stream_id)
        
    except symft.SymFTError as exc:
        parser.exit(1, f"symft error: {exc}\n")

    print(circuit)
    print("simd_backend", symft.simd_backend())
    print("cuda_enabled", symft.cuda_enabled())
    print("cuda_backend", symft.active_cuda_backend())
    print(sampler.info)
    print(result)


if __name__ == "__main__":
    main()
