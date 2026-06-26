"""`embers` CLI — the easy shell where users live.

    embers serve <model> [--port 8000] [--mock] [--no-eager-load]
"""
import argparse


def main() -> None:
    ap = argparse.ArgumentParser(prog="embers")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="serve one model behind an HTTP endpoint")
    s.add_argument("model", help="HF model id or local path")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--mock", action="store_true",
                   help="no GPU/vLLM — fake completions to test the loop")
    s.add_argument("--no-eager-load", dest="eager", action="store_false",
                   help="defer model load to first request (cold-on-demand)")
    s.add_argument("--max-model-len", type=int, default=4096)
    s.add_argument("--dtype", default="auto")
    s.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    s.add_argument("--tensor-parallel-size", type=int, default=1,
                   help="shard one model across N GPUs (tensor parallelism)")
    s.add_argument("--lora", action="append", default=[], metavar="NAME=PATH",
                   help="serve a LoRA adapter off the base, e.g. sql=/adapters/sql "
                        "(repeat); request it via the model field")

    g = sub.add_parser("gateway", help="front multiple serving units behind one URL")
    g.add_argument("--backend", action="append", default=[], metavar="MODEL=URL",
                   help="register a replica, e.g. Qwen/Qwen2.5-3B=http://host:8000 "
                        "(repeat for more models/replicas)")
    g.add_argument("--host", default="0.0.0.0")
    g.add_argument("--port", type=int, default=8080)
    g.add_argument("--api-key", action="append", default=[],
                   help="require this bearer key (repeat to allow several)")

    sc = sub.add_parser("schedule", help="plan model placement across GPUs (dry run)")
    sc.add_argument("--gpu", action="append", default=[], metavar="ID:VRAM_MB",
                    help="a GPU, e.g. g0:24000 (repeat)")
    sc.add_argument("--model", action="append", default=[], metavar="NAME:VRAM_MB[:REPLICAS]",
                    help="a model to place, e.g. Qwen2.5-3B:6000:2 (repeat)")
    sc.add_argument("--policy", default="best-fit",
                    choices=["first-fit", "best-fit", "worst-fit"])

    d = sub.add_parser("dashboard", help="render a running gateway's /stats view")
    d.add_argument("--url", default="http://127.0.0.1:8080",
                   help="gateway base URL (needs /stats enabled)")

    up = sub.add_parser("up", help="run the whole platform from a config file")
    up.add_argument("--config", "-c", default="platform.yaml",
                    help="path to the platform config (see `embers init`)")
    up.add_argument("--mock", action="store_true",
                    help="no GPU/vLLM — in-process mock units (test the platform)")

    nd = sub.add_parser("node", help="run a node server (one GPU box in a fleet, "
                                     "driven by a remote control plane)")
    nd.add_argument("--config", "-c", default="node.yaml",
                    help="this box's GPUs (+ adapters); see platform.yaml format")
    nd.add_argument("--id", default="node0", help="this node's id in the fleet")
    nd.add_argument("--host", default="0.0.0.0")
    nd.add_argument("--port", type=int, default=8090, help="control API port")
    nd.add_argument("--advertise-host", default="127.0.0.1",
                    help="host other boxes reach this node's serving units at")
    nd.add_argument("--serve-host", default="127.0.0.1",
                    help="host serving units bind (0.0.0.0 to be reachable off-box)")
    nd.add_argument("--mock", action="store_true", help="no GPU/vLLM")

    ini = sub.add_parser("init", help="write a starter platform.yaml")
    ini.add_argument("--out", "-o", default="platform.yaml")
    ini.add_argument("--force", action="store_true", help="overwrite if it exists")

    args = ap.parse_args()
    if args.cmd in ("serve", "gateway", "up", "node", "dashboard"):
        try:                                # these need the serving deps
            import fastapi  # noqa: F401
        except ModuleNotFoundError:
            ap.error(f"`embers {args.cmd}` needs the serving extras — install with: "
                     f"pip install 'embers[serve]'  (and [gpu] on a CUDA box)")
    if args.cmd == "serve":
        from embers import server
        kwargs = {} if args.mock else {
            "max_model_len": args.max_model_len, "dtype": args.dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tensor_parallel_size": args.tensor_parallel_size,
        }
        adapters = {}
        for spec in args.lora:
            if "=" not in spec:
                ap.error(f"--lora must be NAME=PATH, got {spec!r}")
            name, path = spec.split("=", 1)
            adapters[name] = path
        server.serve(args.model, host=args.host, port=args.port, mock=args.mock,
                     eager_load=args.eager, adapters=adapters or None, **kwargs)
    elif args.cmd == "gateway":
        from embers import gateway
        backends = []
        for spec in args.backend:
            if "=" not in spec:
                ap.error(f"--backend must be MODEL=URL, got {spec!r}")
            model, url = spec.split("=", 1)
            backends.append((model, url))
        gateway.serve_gateway(backends, host=args.host, port=args.port,
                              api_keys=set(args.api_key) or None)
    elif args.cmd == "schedule":
        run_schedule(args, ap.error)
    elif args.cmd == "dashboard":
        import httpx

        from embers.metrics import render_dashboard
        snap = httpx.get(f"{args.url.rstrip('/')}/stats", timeout=5).json()
        print(render_dashboard(snap))
    elif args.cmd == "up":
        from embers.platform import Platform, load_config
        Platform(load_config(args.config), mock=args.mock).serve()
    elif args.cmd == "node":
        from embers.platform import load_config, run_node
        run_node(load_config(args.config), node_id=args.id, host=args.host,
                 port=args.port, advertise_host=args.advertise_host,
                 serve_host=args.serve_host, mock=args.mock)
    elif args.cmd == "init":
        import os

        from embers.starter import STARTER_CONFIG
        if os.path.exists(args.out) and not args.force:
            ap.error(f"{args.out} exists (use --force to overwrite)")
        with open(args.out, "w") as f:
            f.write(STARTER_CONFIG)
        print(f"wrote {args.out} — edit it, then: embers up --config {args.out}")


def run_schedule(args, fail) -> None:
    """Parse --gpu / --model specs, place them, print the plan. Pure dry run."""
    from embers.scheduler import GPU, NoCapacity, Scheduler

    gpus = []
    for spec in args.gpu:
        try:
            gid, vram = spec.split(":")
            gpus.append(GPU(gid, int(vram)))
        except ValueError:
            fail(f"--gpu must be ID:VRAM_MB, got {spec!r}")
    if not gpus:
        fail("need at least one --gpu")

    sched = Scheduler(gpus, policy=args.policy)
    for spec in args.model:
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            fail(f"--model must be NAME:VRAM_MB[:REPLICAS], got {spec!r}")
        name, vram = parts[0], int(parts[1])
        replicas = int(parts[2]) if len(parts) == 3 else 1
        try:
            sched.place(name, vram, replicas=replicas)
        except NoCapacity as e:
            print(f"[schedule] UNPLACED: {e}")
    print(sched.render_plan())


if __name__ == "__main__":
    main()
